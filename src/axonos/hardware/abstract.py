#!/usr/bin/env python3
"""
Hardware Abstraction Layer for AxonOS
Production-ready device interfaces with proper real-time considerations

КРИТИЧЕСКИ ВАЖНО v2.2:
- Python не подходит для "жёсткого" real-time из-за GIL
- Этот модуль для "мягкого" real-time и оркестрации
- Критичные операции выносятся в C/C++/Rust через FFI
- Используем asyncio для конкурентности без блокировок
- Чёткое разделение: Python для логики, C для железа
"""

import asyncio
import threading
import queue
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np
import logging


# ============================================================================
# ТИПЫ И КОНФИГУРАЦИИ
# ============================================================================

class DeviceType(Enum):
    """Типы BCI устройств"""
    EEG = "eeg"
    EMG = "emg"
    ECG = "ecg"
    FNIRS = "fnirs"
    SPIKE = "spike"
    HYBRID = "hybrid"


class SamplingRate(Enum):
    """Стандартные частоты дискретизации"""
    HZ_128 = 128
    HZ_256 = 256
    HZ_512 = 512
    HZ_1000 = 1000
    HZ_2000 = 2000
    HZ_30000 = 30000  # Для спайков


@dataclass
class DeviceConfig:
    """Конфигурация устройства"""
    device_type: DeviceType
    sampling_rate: int
    num_channels: int
    buffer_size: int = 1000
    timeout_ms: int = 100
    realtime_mode: bool = True  # True = мягкий real-time
    
    # Для критичных к задержкам операций
    use_native_driver: bool = False  # Использовать C/C++ драйвер
    callback_mode: bool = True  # Callback-based чтение
    
    def __post_init__(self):
        if self.realtime_mode and self.sampling_rate > 1000:
            import warnings
            warnings.warn(
                f"High sampling rate ({self.sampling_rate} Hz) with Python "
                f"realtime mode may cause latency. Consider using native driver.",
                RuntimeWarning
            )


@dataclass
class DataPacket:
    """Пакет данных от устройства"""
    timestamp: float
    data: np.ndarray
    channel_names: List[str]
    device_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)


# ============================================================================
# ИСКЛЮЧЕНИЯ
# ============================================================================

class HardwareError(Exception):
    """Базовое исключение оборудования"""
    pass


class DeviceConnectionError(HardwareError):
    """Ошибка подключения к устройству"""
    pass


class RealtimeViolationError(HardwareError):
    """Нарушение real-time требований"""
    pass


class BufferOverflowError(HardwareError):
    """Переполнение буфера"""
    pass


# ============================================================================
# АБСТРАКТНЫЙ БАЗОВЫЙ КЛАСС
# ============================================================================

class AbstractBCIDevice(ABC):
    """
    Абстрактный базовый класс для BCI устройств
    
    ВАЖНО:
    - Этот класс для оркестрации, НЕ для прямого управления железом
    - Для критичных к задержкам операций используйте native драйверы
    - Python = логика высокого уровня, C/C++ = низкоуровневое управление
    """
    
    def __init__(self, config: DeviceConfig):
        self.config = config
        self.device_id = f"device_{int(time.time() * 1000)}"
        self.is_connected = False
        self.is_streaming = False
        
        # Буферы и очереди
        self.data_buffer: queue.Queue = queue.Queue(maxsize=config.buffer_size)
        self.callbacks: List[Callable] = []
        
        # Логирование
        self.logger = self._setup_logger()
        
        # Статистика
        self.stats = {
            'packets_received': 0,
            'packets_dropped': 0,
            'last_timestamp': 0.0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0
        }
        
        # Real-time monitoring
        self._latency_monitor = LatencyMonitor(
            callback=self._on_latency_violation,
            threshold_ms=config.timeout_ms
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger(f'axonos.hardware.{self.device_id}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - HARDWARE - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    # ------------------------------------------------------------------------
    # АБСТРАКТНЫЕ МЕТОДЫ (ДОЛЖНЫ БЫТЬ РЕАЛИЗОВАНЫ ПОДКЛАССАМИ)
    # ------------------------------------------------------------------------
    
    @abstractmethod
    def _connect_impl(self) -> bool:
        """
        Реализация подключения к устройству
        
        Returns:
            True если подключение успешно
        """
        pass
    
    @abstractmethod
    def _disconnect_impl(self) -> bool:
        """
        Реализация отключения от устройства
        
        Returns:
            True если отключение успешно
        """
        pass
    
    @abstractmethod
    def _start_streaming_impl(self) -> bool:
        """
        Реализация начала потокового чтения
        
        Returns:
            True если поток запущен
        """
        pass
    
    @abstractmethod
    def _stop_streaming_impl(self) -> bool:
        """
        Реализация остановки потокового чтения
        
        Returns:
            True если поток остановлен
        """
        pass
    
    @abstractmethod
    def _read_data_impl(self) -> Optional[DataPacket]:
        """
        Реализация чтения данных с устройства
        
        Returns:
            DataPacket или None если данных нет
            
        ВАЖНО:
        - Этот метод НЕ должен блокировать надолго
        - Должен возвращать None если данных нет
        - Для real-time используйте callback вместо polling
        """
        pass
    
    # ------------------------------------------------------------------------
    # ПУБЛИЧНЫЕ МЕТОДЫ
    # ------------------------------------------------------------------------
    
    def connect(self) -> bool:
        """Подключение к устройству"""
        try:
            self.logger.info(f"Connecting to {self.config.device_type.value} device...")
            success = self._connect_impl()
            
            if success:
                self.is_connected = True
                self.logger.info("Device connected successfully")
            else:
                self.logger.error("Failed to connect to device")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            raise DeviceConnectionError(f"Failed to connect: {e}")
    
    def disconnect(self) -> bool:
        """Отключение от устройства"""
        try:
            self.logger.info("Disconnecting from device...")
            
            # Останавливаем стриминг если активен
            if self.is_streaming:
                self.stop_streaming()
            
            success = self._disconnect_impl()
            
            if success:
                self.is_connected = False
                self.logger.info("Device disconnected successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Disconnection error: {e}")
            raise HardwareError(f"Failed to disconnect: {e}")
    
    def start_streaming(self) -> bool:
        """Начало потокового чтения данных"""
        if not self.is_connected:
            raise DeviceConnectionError("Device not connected")
        
        try:
            self.logger.info("Starting data streaming...")
            
            success = self._start_streaming_impl()
            
            if success:
                self.is_streaming = True
                self._start_data_thread()
                self.logger.info("Data streaming started")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Streaming start error: {e}")
            raise HardwareError(f"Failed to start streaming: {e}")
    
    def stop_streaming(self) -> bool:
        """Остановка потокового чтения данных"""
        try:
            self.logger.info("Stopping data streaming...")
            
            self.is_streaming = False
            success = self._stop_streaming_impl()
            
            if success:
                self.logger.info("Data streaming stopped")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Streaming stop error: {e}")
            raise HardwareError(f"Failed to stop streaming: {e}")
    
    def read_data(self, timeout: Optional[float] = None) -> Optional[DataPacket]:
        """
        Чтение данных из буфера
        
        Args:
            timeout: Таймаут в секундах
            
        Returns:
            DataPacket или None
        """
        try:
            if self.config.callback_mode:
                # Callback mode - ждём данные из очереди
                try:
                    packet = self.data_buffer.get(timeout=timeout)
                    return packet
                except queue.Empty:
                    return None
            else:
                # Polling mode - читаем напрямую
                return self._read_data_impl()
                
        except Exception as e:
            self.logger.error(f"Read error: {e}")
            raise HardwareError(f"Failed to read data: {e}")
    
    def add_callback(self, callback: Callable[[DataPacket], None]):
        """Добавление callback для обработки данных"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[DataPacket], None]):
        """Удаление callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики устройства"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Сброс статистики"""
        self.stats = {
            'packets_received': 0,
            'packets_dropped': 0,
            'last_timestamp': 0.0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0
        }
    
    # ------------------------------------------------------------------------
    # ВНУТРЕННИЕ МЕТОДЫ
    # ------------------------------------------------------------------------
    
    def _start_data_thread(self):
        """Запуск потока для сбора данных"""
        if self.config.callback_mode:
            data_thread = threading.Thread(
                target=self._data_collection_loop,
                daemon=True,
                name=f"data_thread_{self.device_id}"
            )
            data_thread.start()
    
    def _data_collection_loop(self):
        """Цикл сбора данных (в отдельном потоке)"""
        while self.is_streaming:
            try:
                # Читаем данные
                packet = self._read_data_impl()
                
                if packet is not None:
                    # Обновляем статистику
                    self._update_stats(packet)
                    
                    # Отправляем в очередь
                    if not self.data_buffer.full():
                        self.data_buffer.put(packet)
                    else:
                        self.stats['packets_dropped'] += 1
                        self.logger.warning("Buffer overflow - packet dropped")
                    
                    # Вызываем callbacks
                    for callback in self.callbacks:
                        try:
                            callback(packet)
                        except Exception as e:
                            self.logger.error(f"Callback error: {e}")
                
                # Мониторинг задержек
                self._latency_monitor.check(packet.timestamp if packet else None)
                
            except Exception as e:
                self.logger.error(f"Data collection error: {e}")
                
            # Небольшая пауза чтобы не нагружать CPU
            time.sleep(0.001)  # 1ms
    
    def _update_stats(self, packet: DataPacket):
        """Обновление статистики"""
        current_time = time.time()
        
        # Задержка
        if self.stats['last_timestamp'] > 0:
            latency_ms = (current_time - packet.timestamp) * 1000
            
            # Обновляем среднюю задержку
            n = self.stats['packets_received']
            self.stats['avg_latency_ms'] = (
                (self.stats['avg_latency_ms'] * n + latency_ms) / (n + 1)
            )
            
            # Обновляем максимальную задержку
            self.stats['max_latency_ms'] = max(
                self.stats['max_latency_ms'],
                latency_ms
            )
        
        self.stats['packets_received'] += 1
        self.stats['last_timestamp'] = current_time
    
    def _on_latency_violation(self, latency_ms: float):
        """Обработка нарушения real-time требований"""
        self.logger.warning(
            f"Real-time violation: latency {latency_ms:.2f}ms exceeds threshold"
        )
        
        # В production: предпринять действия
        # - Переключиться на native драйвер
        # - Увеличить приоритет потока
        # - Сообщить в monitoring


# ============================================================================
# МОНИТОРИНГ ЗАДЕРЖЕК
# ============================================================================

class LatencyMonitor:
    """
    Мониторинг задержек для real-time требований
    """
    
    def __init__(self, callback: Callable[[float], None], threshold_ms: float):
        self.callback = callback
        self.threshold_ms = threshold_ms
        self.last_timestamp = None
        self.violations = 0
    
    def check(self, timestamp: Optional[float]):
        """Проверка задержки"""
        if timestamp is None or self.last_timestamp is None:
            self.last_timestamp = timestamp
            return
        
        current_time = time.time()
        latency_ms = (current_time - timestamp) * 1000
        
        if latency_ms > self.threshold_ms:
            self.violations += 1
            self.callback(latency_ms)
        
        self.last_timestamp = timestamp


# ============================================================================
# АСИНХРОННЫЙ ИНТЕРФЕЙС (ДЛЯ КОНКУРЕНТНОСТИ)
# ============================================================================

class AsyncBCIDevice:
    """
    Асинхронный интерфейс для BCI устройств
    Использует asyncio для конкурентности без блокировок
    """
    
    def __init__(self, device: AbstractBCIDevice):
        self.device = device
        self._loop = None
        self._stream_task = None
    
    async def connect_async(self) -> bool:
        """Асинхронное подключение"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.device.connect
        )
    
    async def start_streaming_async(self) -> bool:
        """Асинхронный старт стриминга"""
        success = await asyncio.get_event_loop().run_in_executor(
            None, self.device.start_streaming
        )
        
        if success:
            self._loop = asyncio.get_event_loop()
            self._stream_task = asyncio.create_task(self._streaming_loop())
        
        return success
    
    async def stop_streaming_async(self) -> bool:
        """Асинхронная остановка стриминга"""
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        
        return await asyncio.get_event_loop().run_in_executor(
            None, self.device.stop_streaming
        )
    
    async def read_data_async(self, timeout: Optional[float] = None) -> Optional[DataPacket]:
        """Асинхронное чтение данных"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.device.read_data, timeout
        )
    
    async def _streaming_loop(self):
        """Асинхронный цикл стриминга"""
        while True:
            try:
                packet = await self.read_data_async(timeout=0.1)
                if packet:
                    await self._process_packet_async(packet)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.device.logger.error(f"Streaming error: {e}")
    
    async def _process_packet_async(self, packet: DataPacket):
        """Асинхронная обработка пакета"""
        # Здесь можно добавить асинхронную обработку
        # Например: запись в БД, отправка по сети, ML inference
        pass


# ============================================================================
# NATIVE ДРАЙВЕРЫ (PLACEHOLDER ДЛЯ C/C++ ИНТЕГРАЦИИ)
# ============================================================================

class NativeDriverManager:
    """
    Менеджер native драйверов для критичных к задержкам операций
    
    ВАЖНО:
    - Python вызывает C/C++ код через FFI (ctypes, cffi, pybind11)
    - Native код выполняет критичные операции
    - Python только оркестрирует и обрабатывает результаты
    """
    
    def __init__(self):
        self.native_lib = None
        self._load_native_library()
    
    def _load_native_library(self):
        """Загрузка native библиотеки"""
        try:
            # Placeholder для загрузки native библиотеки
            # import ctypes
            # self.native_lib = ctypes.CDLL('libaxonos_hw.so')
            
            # Пока симулируем отсутствие native драйвера
            self.native_lib = None
            
        except Exception as e:
            print(f"Native driver not available: {e}")
            self.native_lib = None
    
    def is_available(self) -> bool:
        """Доступен ли native драйвер"""
        return self.native_lib is not None
    
    def read_data_native(self, device_handle: int) -> Optional[np.ndarray]:
        """
        Чтение данных через native драйвер
        
        Args:
            device_handle: Хэндл устройства
            
        Returns:
            Массив данных или None
        
        ПРИМЕЧАНИЕ:
        - Это placeholder для реальной C/C++ функции
        - Native код должен быть предварительно скомпилирован
        - Используйте ctypes, cffi, или pybind11 для интеграции
        """
        if not self.is_available():
            return None
        
        # Placeholder для вызова native функции
        # data = self.native_lib.read_data(device_handle)
        # return np.array(data)
        
        return None


# ============================================================================
# ЭКСПОРТ
# ============================================================================

__all__ = [
    'AbstractBCIDevice',
    'AsyncBCIDevice',
    'DeviceConfig',
    'DataPacket',
    'DeviceType',
    'SamplingRate',
    'HardwareError',
    'DeviceConnectionError',
    'RealtimeViolationError',
    'BufferOverflowError',
    'NativeDriverManager'
]
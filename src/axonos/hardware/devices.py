#!/usr/bin/env python3
"""
Device Implementations for AxonOS Hardware Layer
Production-ready drivers for specific BCI devices

ВАЖНО:
- Эти реализации для "мягкого" real-time
- Для "жёсткого" real-time используйте native драйверы
- Python = оркестрация, C/C++ = критичные операции
"""

import time
import random
import numpy as np
from typing import Optional, List
from pathlib import Path

from .abstract import (
    AbstractBCIDevice,
    DeviceConfig,
    DataPacket,
    DeviceType,
    DeviceConnectionError,
    HardwareError
)


# ============================================================================
# ЭМУЛЯТОР УСТРОЙСТВА (ДЛЯ ТЕСТИРОВАНИЯ)
# ============================================================================

class EmulatedBCIDevice(AbstractBCIDevice):
    """
    Эмулированное BCI устройство для тестирования
    Генерирует синтетические нейросигналы
    """
    
    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        
        # Параметры эмуляции
        self._emulation_running = False
        self._emulation_thread = None
        self._sample_counter = 0
        
        # Синтетические параметры
        self.frequencies = [10, 20, 40]  # Генерируемые частоты
        self.noise_level = 0.1
        
        self.logger.info(f"Initialized emulated {config.device_type.value} device")
    
    def _connect_impl(self) -> bool:
        """Эмуляция подключения"""
        self.logger.info("Connecting to emulated device...")
        time.sleep(0.1)  # Имитация задержки подключения
        
        self.logger.info("Emulated device connected")
        return True
    
    def _disconnect_impl(self) -> bool:
        """Эмуляция отключения"""
        self.logger.info("Disconnecting from emulated device...")
        return True
    
    def _start_streaming_impl(self) -> bool:
        """Запуск эмуляции стриминга"""
        self.logger.info("Starting emulated streaming...")
        self._emulation_running = True
        return True
    
    def _stop_streaming_impl(self) -> bool:
        """Остановка эмуляции стриминга"""
        self.logger.info("Stopping emulated streaming...")
        self._emulation_running = False
        return True
    
    def _read_data_impl(self) -> Optional[DataPacket]:
        """Генерация эмулированных данных"""
        if not self._emulation_running:
            return None
        
        # Генерируем синтетический сигнал
        num_samples = int(self.config.sampling_rate * 0.001)  # 1ms worth of data
        time_points = np.arange(num_samples) / self.config.sampling_rate
        
        # Синтезируем сигнал с несколькими частотами
        data = np.zeros((self.config.num_channels, num_samples))
        
        for ch in range(self.config.num_channels):
            for freq in self.frequencies:
                amplitude = random.uniform(0.1, 1.0)
                phase = random.uniform(0, 2 * np.pi)
                data[ch] += amplitude * np.sin(2 * np.pi * freq * time_points + phase)
            
            # Добавляем шум
            data[ch] += self.noise_level * np.random.randn(num_samples)
        
        # Создаём пакет
        packet = DataPacket(
            timestamp=time.time(),
            data=data,
            channel_names=[f"Ch{i}" for i in range(self.config.num_channels)],
            device_id=self.device_id,
            metadata={
                'sample_counter': self._sample_counter,
                'emulated': True,
                'device_type': self.config.device_type.value
            }
        )
        
        self._sample_counter += num_samples
        return packet


# ============================================================================
# OPENBCI CYTON (ЧЕРЕЗ SERIAL)
# ============================================================================

class OpenBCICytonDevice(AbstractBCIDevice):
    """
    Драйвер для OpenBCI Cyton board
    
    ПРИМЕЧАНИЕ:
    - Для production используйте официальный OpenBCI Python SDK
    - Это упрощённая реализация для демонстрации
    """
    
    def __init__(self, config: DeviceConfig, port: str = None):
        super().__init__(config)
        
        # Serial порт
        self.port = port or self._detect_openbci_port()
        self.serial_connection = None
        
        # OpenBCI специфика
        self._board_mode = 'default'
        self._impedance_check = False
        
        self.logger.info(f"Initialized OpenBCI Cyton on {self.port}")
    
    def _detect_openbci_port(self) -> str:
        """Автоопределение порта OpenBCI"""
        import serial.tools.list_ports
        
        # Ищем известные VID/PID OpenBCI
        for port in serial.tools.list_ports.comports():
            if 'OpenBCI' in port.description or 'FTDI' in port.description:
                return port.device
        
        # Fallback на дефолтный порт
        return '/dev/ttyUSB0' if Path('/dev/ttyUSB0').exists() else 'COM3'
    
    def _connect_impl(self) -> bool:
        """Подключение к OpenBCI"""
        try:
            import serial
            
            self.logger.info(f"Connecting to OpenBCI on {self.port}...")
            
            # Настройка serial соединения
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=115200,
                timeout=1
            )
            
            # Инициализация платы
            time.sleep(2)  # Ждём загрузки
            
            # Отправляем команды инициализации
            self._send_command('v')  # Version
            self._send_command('~')  # Default settings
            
            # Настройка каналов
            self._configure_channels()
            
            self.logger.info("OpenBCI connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"OpenBCI connection error: {e}")
            return False
    
    def _disconnect_impl(self) -> bool:
        """Отключение от OpenBCI"""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
                self.logger.info("OpenBCI disconnected")
            return True
        except Exception as e:
            self.logger.error(f"OpenBCI disconnection error: {e}")
            return False
    
    def _start_streaming_impl(self) -> bool:
        """Запуск стриминга OpenBCI"""
        try:
            self._send_command('b')  # Begin streaming
            self.logger.info("OpenBCI streaming started")
            return True
        except Exception as e:
            self.logger.error(f"OpenBCI streaming start error: {e}")
            return False
    
    def _stop_streaming_impl(self) -> bool:
        """Остановка стриминга OpenBCI"""
        try:
            self._send_command('s')  # Stop streaming
            self.logger.info("OpenBCI streaming stopped")
            return True
        except Exception as e:
            self.logger.error(f"OpenBCI streaming stop error: {e}")
            return False
    
    def _read_data_impl(self) -> Optional[DataPacket]:
        """Чтение данных с OpenBCI"""
        if not self.serial_connection or not self.serial_connection.is_open:
            return None
        
        try:
            # Читаем строку данных
            line = self.serial_connection.readline()
            
            if line and line.startswith(b'$$$'):
                # Парсим данные OpenBCI
                data = self._parse_openbci_packet(line)
                
                if data is not None:
                    return DataPacket(
                        timestamp=time.time(),
                        data=data,
                        channel_names=[f"EEG{i}" for i in range(self.config.num_channels)],
                        device_id=self.device_id,
                        metadata={
                            'board_mode': self._board_mode,
                            'impedance_check': self._impedance_check
                        }
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"OpenBCI read error: {e}")
            return None
    
    def _send_command(self, command: str):
        """Отправка команды на плату"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.write(command.encode())
            time.sleep(0.1)  # Небольшая задержка
    
    def _configure_channels(self):
        """Настройка каналов OpenBCI"""
        # Включаем нужное количество каналов
        for i in range(self.config.num_channels):
            self._send_command(str(i + 1))  # Команды 1-8 включают каналы
    
    def _parse_openbci_packet(self, raw_packet: bytes) -> Optional[np.ndarray]:
        """Парсинг пакета OpenBCI"""
        try:
            # Упрощённый парсинг - в реальности сложнее
            parts = raw_packet.strip().split(b',')
            
            if len(parts) >= self.config.num_channels + 1:
                # Пропускаем заголовок, извлекаем данные каналов
                channel_data = []
                for i in range(self.config.num_channels):
                    value = float(parts[i + 1])
                    channel_data.append(value)
                
                return np.array(channel_data).reshape(-1, 1)
            
            return None
            
        except (ValueError, IndexError):
            return None


# ============================================================================
# LSL (LAB STREAMING LAYER) УСТРОЙСТВО
# ============================================================================

class LSLDevice(AbstractBCIDevice):
    """
    Устройство через Lab Streaming Layer (LSL)
    
    LSL - стандарт для потоковой передачи временных рядов
    в исследовательской нейронауке
    """
    
    def __init__(self, config: DeviceConfig, stream_name: str = None):
        super().__init__(config)
        
        self.stream_name = stream_name or "AxonOS_Stream"
        self.inlet = None
        self._resolve_timeout = 5.0  # секунды
        
        self.logger.info(f"Initialized LSL device for stream '{self.stream_name}'")
    
    def _connect_impl(self) -> bool:
        """Подключение к LSL потоку"""
        try:
            import pylsl
            
            self.logger.info(f"Resolving LSL stream '{self.stream_name}'...")
            
            # Ищем потоки
            streams = pylsl.resolve_streams(self._resolve_timeout)
            
            if not streams:
                self.logger.error("No LSL streams found")
                return False
            
            # Выбираем нужный поток
            target_stream = None
            for stream in streams:
                if self.stream_name in stream.name():
                    target_stream = stream
                    break
            
            if target_stream is None:
                # Используем первый доступный
                target_stream = streams[0]
                self.logger.warning(f"Using first available stream: {target_stream.name()}")
            
            # Создаём inlet
            self.inlet = pylsl.StreamInlet(target_stream)
            
            # Получаем информацию о потоке
            stream_info = self.inlet.info()
            self.logger.info(f"Connected to LSL stream:")
            self.logger.info(f"  Name: {stream_info.name()}")
            self.logger.info(f"  Type: {stream_info.type()}")
            self.logger.info(f"  Channels: {stream_info.channel_count()}")
            self.logger.info(f"  Rate: {stream_info.nominal_srate()}")
            
            # Обновляем конфигурацию
            self.config.sampling_rate = int(stream_info.nominal_srate())
            self.config.num_channels = stream_info.channel_count()
            
            return True
            
        except Exception as e:
            self.logger.error(f"LSL connection error: {e}")
            return False
    
    def _disconnect_impl(self) -> bool:
        """Отключение от LSL"""
        try:
            if self.inlet:
                self.inlet.close_stream()
                self.logger.info("LSL stream closed")
            return True
        except Exception as e:
            self.logger.error(f"LSL disconnection error: {e}")
            return False
    
    def _start_streaming_impl(self) -> bool:
        """Запуск LSL стриминга"""
        # Для LSL не нужен отдельный старт - inlet уже готов
        self.logger.info("LSL streaming ready")
        return True
    
    def _stop_streaming_impl(self) -> bool:
        """Остановка LSL стриминга"""
        # Всё сделано в disconnect
        return True
    
    def _read_data_impl(self) -> Optional[DataPacket]:
        """Чтение данных с LSL"""
        if not self.inlet:
            return None
        
        try:
            # Читаем sample
            sample, timestamp = self.inlet.pull_sample(timeout=0.0)
            
            if sample is not None:
                # Конвертируем в numpy array
                data = np.array(sample).reshape(-1, 1)
                
                return DataPacket(
                    timestamp=timestamp,
                    data=data,
                    channel_names=[f"Ch{i}" for i in range(len(sample))],
                    device_id=self.device_id,
                    metadata={
                        'lsl_source': self.inlet.info().name(),
                        'sample_format': 'float32'
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"LSL read error: {e}")
            return None


# ============================================================================
# ФАБРИКА УСТРОЙСТВ
# ============================================================================

class DeviceFactory:
    """
    Фабрика для создания экземпляров BCI устройств
    """
    
    @staticmethod
    def create_emulator(device_type: DeviceType = DeviceType.EEG,
                       num_channels: int = 8,
                       sampling_rate: int = 256) -> EmulatedBCIDevice:
        """Создание эмулированного устройства"""
        config = DeviceConfig(
            device_type=device_type,
            sampling_rate=sampling_rate,
            num_channels=num_channels,
            realtime_mode=True
        )
        
        return EmulatedBCIDevice(config)
    
    @staticmethod
    def create_openbci(port: str = None,
                      num_channels: int = 8,
                      sampling_rate: int = 250) -> OpenBCICytonDevice:
        """Создание OpenBCI устройства"""
        config = DeviceConfig(
            device_type=DeviceType.EEG,
            sampling_rate=sampling_rate,
            num_channels=num_channels,
            realtime_mode=True,
            use_native_driver=False  # True для production
        )
        
        return OpenBCICytonDevice(config, port)
    
    @staticmethod
    def create_lsl(stream_name: str = None,
                  num_channels: int = None,
                  sampling_rate: int = None) -> LSLDevice:
        """Создание LSL устройства"""
        config = DeviceConfig(
            device_type=DeviceType.EEG,
            sampling_rate=sampling_rate or 256,  # Будет обновлено при подключении
            num_channels=num_channels or 8,      # Будет обновлено при подключении
            realtime_mode=True
        )
        
        return LSLDevice(config, stream_name)
    
    @staticmethod
    def auto_detect() -> Optional[AbstractBCIDevice]:
        """Автоопределение доступных устройств"""
        # Проверяем LSL потоки
        try:
            import pylsl
            streams = pylsl.resolve_streams(1.0)
            if streams:
                return DeviceFactory.create_lsl()
        except ImportError:
            pass
        
        # Проверяем serial порты для OpenBCI
        try:
            import serial.tools.list_ports
            for port in serial.tools.list_ports.comports():
                if 'OpenBCI' in port.description or 'FTDI' in port.description:
                    return DeviceFactory.create_openbci(port.device)
        except ImportError:
            pass
        
        # Возвращаем эмулятор если ничего не найдено
        return DeviceFactory.create_emulator()


# ============================================================================
# ЭКСПОРТ
# ============================================================================

__all__ = [
    'EmulatedBCIDevice',
    'OpenBCICytonDevice',
    'LSLDevice',
    'DeviceFactory'
]
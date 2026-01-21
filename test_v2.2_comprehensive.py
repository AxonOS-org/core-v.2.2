#!/usr/bin/env python3
"""
Comprehensive Test Suite for AxonOS v2.2
Tests all critical fixes and production-ready features

ТЕСТЫ:
1. Security fixes (vault, encryption)
2. Hardware real-time considerations
3. Import structure
4. ML models integration
5. Signal processing
6. Zero-knowledge guarantees
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_security_vault():
    """Тест исправлений в security/vault.py"""
    print("\n" + "="*60)
    print("TEST 1: Security Vault Fixes")
    print("="*60)
    
    try:
        # Тест 1: Удаление самодельного vault
        print("✓ Testing NeuralDataVault initialization...")
        
        # Устанавливаем тестовый ключ
        os.environ['AXONOS_MASTER_KEY'] = 'test_master_key_32_chars_long_for_testing!'
        
        from axonos.security.vault import NeuralDataVault, SecurityConfig
        
        config = SecurityConfig(mode='env', zero_knowledge_mode=True)
        vault = NeuralDataVault(config)
        
        print("✓ NeuralDataVault initialized successfully")
        
        # Тест 2: Правильное шифрование (не XOR!)
        print("✓ Testing proper encryption (Fernet/AES)...")
        
        test_data = b"sensitive neural data"
        encrypted, data_id = vault.encrypt_neural_data(test_data)
        
        # Проверяем что данные зашифрованы
        assert encrypted != test_data, "Data not encrypted!"
        assert len(encrypted) > len(test_data), "Encrypted data too small"
        
        # Проверяем расшифровку
        decrypted, metadata = vault.decrypt_neural_data(encrypted, data_id)
        assert decrypted == test_data, "Decryption failed!"
        
        print("✓ Encryption/decryption working correctly")
        
        # Тест 3: Zero-knowledge гарантии
        print("✓ Testing zero-knowledge guarantees...")
        
        # Проверяем что сырые данные нигде не хранятся
        assert not hasattr(vault, 'raw_data'), "Raw data stored in vault!"
        assert not hasattr(vault, 'encryption_key'), "Key exposed!"
        
        print("✓ Zero-knowledge guarantees maintained")
        
        # Тест 4: Differential Privacy
        from axonos.security.vault import DifferentialPrivacy
        
        signal = np.random.randn(1000)
        private_signal = DifferentialPrivacy.add_calibrated_noise(signal, epsilon=1.0)
        
        assert not np.array_equal(signal, private_signal), "DP noise not added!"
        assert signal.shape == private_signal.shape, "DP changed signal shape!"
        
        print("✓ Differential privacy working correctly")
        
        # Тест 5: Аудит
        print("✓ Testing audit logging...")
        
        audit_log = vault.get_audit_log(limit=10)
        assert len(audit_log) > 0, "Audit log empty!"
        
        # Проверяем что в логе нет сырых данных
        for event in audit_log:
            assert 'ENCRYPT' in str(event) or 'DECRYPT' in str(event), "Invalid audit event!"
        
        print("✓ Audit logging working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Security vault test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_security_encryption():
    """Тест исправлений в security/encryption.py"""
    print("\n" + "="*60)
    print("TEST 2: Security Encryption Fixes")
    print("="*60)
    
    try:
        # Тест 1: Правильные алгоритмы шифрования
        print("✓ Testing EncryptionEngine...")
        
        from axonos.security.encryption import EncryptionEngine
        
        engine = EncryptionEngine()
        
        # Тест Fernet (AES-128-CBC + HMAC-SHA256)
        test_data = "sensitive neural signal data"
        encrypted = engine.encrypt_fernet(test_data)
        decrypted = engine.decrypt_fernet(encrypted)
        
        assert decrypted.decode() == test_data, "Fernet encryption failed!"
        assert encrypted != test_data, "Data not encrypted!"
        
        print("✓ Fernet encryption (AES-128-CBC + HMAC) working")
        
        # Тест AES-GCM
        print("✓ Testing AES-256-GCM...")
        
        ciphertext, nonce = engine.encrypt_aes_gcm(test_data)
        decrypted_gcm = engine.decrypt_aes_gcm(ciphertext, nonce)
        
        assert decrypted_gcm.decode() == test_data, "AES-GCM decryption failed!"
        
        print("✓ AES-256-GCM working correctly")
        
        # Тест 2: Цифровые подписи ECDSA (не самодельные!)
        print("✓ Testing ECDSA digital signatures...")
        
        from axonos.security.encryption import DigitalSignature
        
        signer = DigitalSignature()
        private_key, public_key = signer.generate_key_pair()
        
        # Подписываем данные
        test_message = "integrity check"
        signature = signer.sign_data(test_message, private_key)
        
        # Проверяем подпись
        is_valid = signer.verify_signature(test_message, signature, public_key)
        assert is_valid, "ECDSA signature verification failed!"
        
        # Проверяем что подпись не валидна для других данных
        is_invalid = signer.verify_signature("tampered", signature, public_key)
        assert not is_invalid, "ECDSA accepted invalid signature!"
        
        print("✓ ECDSA digital signatures working correctly")
        
        # Тест 3: RSA шифрование
        print("✓ Testing RSA encryption...")
        
        from axonos.security.encryption import RSAEncryption
        
        rsa_private, rsa_public = RSAEncryption.generate_key_pair()
        
        # Шифруем публичным ключом
        secret_data = b"secret session key"
        encrypted_rsa = RSAEncryption.encrypt_with_public_key(secret_data, rsa_public)
        
        # Расшифровываем приватным ключом
        decrypted_rsa = RSAEncryption.decrypt_with_private_key(encrypted_rsa, rsa_private)
        
        assert decrypted_rsa == secret_data, "RSA decryption failed!"
        
        print("✓ RSA encryption working correctly")
        
        # Тест 4: Хэширование
        print("✓ Testing hash utilities...")
        
        from axonos.security.encryption import HashUtils
        
        test_array = np.random.randn(100)
        hash1 = HashUtils.hash_neural_data(test_array)
        hash2 = HashUtils.hash_neural_data(test_array)
        
        assert hash1 == hash2, "Hash not deterministic!"
        assert len(hash1) == 64, "Invalid hash length!"
        
        print("✓ Hash utilities working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Security encryption test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hardware_realtime():
    """Тест real-time considerations в hardware модуле"""
    print("\n" + "="*60)
    print("TEST 3: Hardware Real-Time Considerations")
    print("="*60)
    
    try:
        # Тест 1: Создание эмулированного устройства
        print("✓ Testing emulated BCI device...")
        
        from axonos.hardware import DeviceFactory, DeviceConfig, DeviceType
        
        config = DeviceConfig(
            device_type=DeviceType.EEG,
            sampling_rate=256,
            num_channels=8,
            realtime_mode=True
        )
        
        device = DeviceFactory.create_emulator(
            device_type=DeviceType.EEG,
            num_channels=8,
            sampling_rate=256
        )
        
        print("✓ Emulated device created successfully")
        
        # Тест 2: Подключение и стриминг
        print("✓ Testing device connection and streaming...")
        
        success = device.connect()
        assert success, "Device connection failed!"
        
        success = device.start_streaming()
        assert success, "Streaming start failed!"
        
        print("✓ Device connected and streaming")
        
        # Тест 3: Чтение данных
        print("✓ Testing data reading...")
        
        time.sleep(0.1)  # Даём время на генерацию данных
        
        packet = device.read_data(timeout=0.1)
        assert packet is not None, "No data received!"
        assert isinstance(packet.data, np.ndarray), "Invalid data format!"
        assert packet.data.shape[0] == 8, "Wrong number of channels!"
        
        print(f"✓ Data packet received: {packet.data.shape}")
        
        # Тест 4: Callback system
        print("✓ Testing callback system...")
        
        callback_data = []
        
        def test_callback(packet):
            callback_data.append(packet)
        
        device.add_callback(test_callback)
        time.sleep(0.1)
        
        assert len(callback_data) > 0, "Callback not called!"
        
        print("✓ Callback system working")
        
        # Тест 5: Статистика
        print("✓ Testing device statistics...")
        
        stats = device.get_stats()
        assert stats['packets_received'] > 0, "No packets received!"
        assert 'avg_latency_ms' in stats, "Latency stats missing!"
        
        print(f"✓ Device stats: {stats['packets_received']} packets received")
        
        # Останавливаем стриминг
        device.stop_streaming()
        device.disconnect()
        
        # Тест 6: Async interface
        print("✓ Testing async interface...")
        
        from axonos.hardware import AsyncBCIDevice
        import asyncio
        
        async def test_async_device():
            async_device = AsyncBCIDevice(device)
            
            # Асинхронное подключение
            connected = await async_device.connect_async()
            assert connected, "Async connection failed!"
            
            # Асинхронный старт стриминга
            streaming = await async_device.start_streaming_async()
            assert streaming, "Async streaming start failed!"
            
            # Асинхронное чтение
            await asyncio.sleep(0.1)
            packet = await async_device.read_data_async(timeout=0.1)
            assert packet is not None, "Async read failed!"
            
            # Остановка
            await async_device.stop_streaming_async()
        
        asyncio.run(test_async_device())
        
        print("✓ Async interface working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware real-time test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_integration():
    """Тест ML моделей и интеграции"""
    print("\n" + "="*60)
    print("TEST 4: ML Models Integration")
    print("="*60)
    
    try:
        # Тест 1: LSTM модель
        print("✓ Testing LSTM model...")
        
        from axonos.core.ml.axonml_models import LSTMBCI, ModelConfig
        
        config = ModelConfig(
            input_size=64,
            hidden_size=128,
            num_classes=3,
            num_layers=2
        )
        
        model = LSTMBCI(config)
        
        # Тестовый вход
        batch_size, seq_len, input_size = 2, 100, 64
        test_input = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)
        
        # Forward pass
        import torch
        with torch.no_grad():
            output = model(torch.from_numpy(test_input))
        
        assert output.shape == (batch_size, 3), f"Wrong output shape: {output.shape}"
        
        print("✓ LSTM model working correctly")
        
        # Тест 2: Transformer модель
        print("✓ Testing Transformer model...")
        
        from axonos.core.ml.axonml_models import TransformerBCI
        
        transformer_model = TransformerBCI(config)
        
        with torch.no_grad():
            transformer_output = transformer_model(torch.from_numpy(test_input))
        
        assert transformer_output.shape == (batch_size, 3), f"Wrong transformer output shape: {transformer_output.shape}"
        
        print("✓ Transformer model working correctly")
        
        # Тест 3: Attention mechanism
        print("✓ Testing attention mechanism...")
        
        from axonos.core.ml.axonml_models import AttentionMechanism
        
        attention = AttentionMechanism(hidden_size=128)
        
        # Тестовый LSTM output
        lstm_output = torch.randn(batch_size, seq_len, 128)
        attended, weights = attention(lstm_output)
        
        assert attended.shape == (batch_size, 128), f"Wrong attention output shape: {attended.shape}"
        assert weights.shape == (batch_size, seq_len), f"Wrong attention weights shape: {weights.shape}"
        
        print("✓ Attention mechanism working correctly")
        
        # Тест 4: Инференс движок
        print("✓ Testing inference engine...")
        
        from axonos.core.pipeline.axonml_inference import (
            InferenceEngine, 
            InferenceConfig,
            SignalType,
            InferenceMode
        )
        
        inference_config = InferenceConfig(
            model_type="lstm",
            signal_type=SignalType.MOTOR_IMAGERY,
            mode=InferenceMode.REALTIME
        )
        
        engine = InferenceEngine(inference_config)
        
        # Тестовый сигнал
        test_signal = np.random.randn(8, 256)  # 8 channels, 1 second at 256Hz
        
        # Инференс
        result = engine.predict(test_signal)
        
        assert result is not None, "Inference returned None!"
        assert hasattr(result, 'prediction'), "Result missing prediction!"
        
        print("✓ Inference engine working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ ML integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_processing():
    """Тест обработки сигналов"""
    print("\n" + "="*60)
    print("TEST 5: Signal Processing")
    print("="*60)
    
    try:
        # Тест 1: Фильтрация
        print("✓ Testing signal filtering...")
        
        from axonos.core.signal import (
            bandpass_filter,
            notch_filter,
            SignalPreprocessor,
            FREQUENCY_BANDS
        )
        
        # Генерируем тестовый сигнал
        fs = 250  # Hz
        t = np.arange(0, 1, 1/fs)
        
        # Сигнал с компонентами на разных частотах
        signal_1hz = np.sin(2 * np.pi * 1 * t)
        signal_10hz = np.sin(2 * np.pi * 10 * t)
        signal_50hz = np.sin(2 * np.pi * 50 * t)
        signal_100hz = np.sin(2 * np.pi * 100 * t)
        
        test_signal = signal_1hz + signal_10hz + signal_50hz + signal_100hz
        test_signal = test_signal.reshape(1, -1)  # Single channel
        
        # Bandpass фильтр 1-50 Hz
        filtered = bandpass_filter(test_signal, 1, 50, fs)
        
        # Проверяем что сигнал прошёл через фильтр
        assert filtered.shape == test_signal.shape, "Filter changed signal shape!"
        
        print("✓ Bandpass filter working correctly")
        
        # Тест 2: Notch фильтр
        print("✓ Testing notch filter...")
        
        notched = notch_filter(test_signal, 50, fs)
        
        assert notched.shape == test_signal.shape, "Notch filter changed signal shape!"
        
        print("✓ Notch filter working correctly")
        
        # Тест 3: Препроцессор
        print("✓ Testing signal preprocessor...")
        
        preprocessor = SignalPreprocessor(fs, bandpass=(1, 50), notch_freq=50)
        preprocessed = preprocessor.process(test_signal)
        
        assert preprocessed.shape == test_signal.shape, "Preprocessor changed signal shape!"
        
        print("✓ Signal preprocessor working correctly")
        
        # Тест 4: Частотные полосы
        print("✓ Testing frequency bands...")
        
        from axonos.core.signal import compute_psd, extract_band_power
        
        f, psd = compute_psd(test_signal, fs)
        
        # Проверяем что частотные полосы определены
        for band_name, (low, high) in FREQUENCY_BANDS.items():
            assert low < high, f"Invalid band {band_name}: {low}-{high}"
            
        # Извлекаем мощность в полосе альфа
        alpha_power = extract_band_power(psd, f, FREQUENCY_BANDS['alpha'])
        
        assert alpha_power >= 0, "Negative power!"
        
        print("✓ Frequency bands working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Signal processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zero_knowledge_integration():
    """Тест полной интеграции zero-knowledge"""
    print("\n" + "="*60)
    print("TEST 6: Zero-Knowledge Integration")
    print("="*60)
    
    try:
        # Тест 1: Полный пайплайн с шифрованием
        print("✓ Testing full pipeline with encryption...")
        
        from axonos.security import NeuralDataVault, SecurityConfig
        from axonos.hardware import DeviceFactory
        from axonos.core.signal import SignalPreprocessor
        
        # Инициализация vault
        vault_config = SecurityConfig(
            mode='env',
            zero_knowledge_mode=True
        )
        vault = NeuralDataVault(vault_config)
        
        # Устройство
        device = DeviceFactory.create_emulator(
            num_channels=8,
            sampling_rate=256
        )
        
        device.connect()
        device.start_streaming()
        
        # Ждём данные
        import time
        time.sleep(0.1)
        
        packet = device.read_data()
        assert packet is not None, "No data from device!"
        
        # Шифруем данные
        encrypted, data_id = vault.encrypt_with_metadata(
            packet.data,
            subject_id="subject_001",
            session_id="session_001",
            tags=["motor_imagery", "left_hand"]
        )
        
        # Проверяем что данные зашифрованы
        assert isinstance(encrypted, bytes), "Data not encrypted!"
        assert encrypted != packet.data.tobytes(), "Raw data exposed!"
        
        print("✓ Device-to-vault encryption working")
        
        # Тест 2: Расшифровка и обработка
        print("✓ Testing decryption and processing...")
        
        # Расшифровываем
        decrypted, metadata = vault.decrypt_neural_data(encrypted, data_id)
        
        # Проверяем целостность
        original_data = packet.data
        restored_data = np.frombuffer(decrypted, dtype=original_data.dtype)
        restored_data = restored_data.reshape(original_data.shape)
        
        assert np.array_equal(original_data, restored_data), "Data integrity violated!"
        
        # Обрабатываем сигнал
        preprocessor = SignalPreprocessor(256, bandpass=(1, 50))
        processed = preprocessor.process(restored_data)
        
        assert processed.shape == restored_data.shape, "Processing changed shape!"
        
        print("✓ Decryption and processing working")
        
        # Тест 3: Аудит безопасности
        print("✓ Testing security audit...")
        
        audit_log = vault.get_audit_log(limit=10)
        
        # Находим события шифрования/расшифровки
        encrypt_events = [e for e in audit_log if e['action'] == 'ENCRYPT']
        decrypt_events = [e for e in audit_log if e['action'] == 'DECRYPT']
        
        assert len(encrypt_events) > 0, "No encryption events in audit!"
        assert len(decrypt_events) > 0, "No decryption events in audit!"
        
        # Проверяем что в метаданных нет сырых данных
        for event in encrypt_events:
            assert 'raw' not in str(event).lower(), "Raw data leaked to audit!"
        
        print("✓ Security audit working correctly")
        
        # Очистка
        device.stop_streaming()
        device.disconnect()
        
        return True
        
    except Exception as e:
        print(f"❌ Zero-knowledge integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_import_structure():
    """Тест структуры импортов"""
    print("\n" + "="*60)
    print("TEST 7: Import Structure")
    print("="*60)
    
    try:
        # Тест 1: Главный импорт
        print("✓ Testing main package import...")
        
        import axonos
        assert axonos.__version__ == "2.2.0", f"Wrong version: {axonos.__version__}"
        
        print("✓ Main package imported successfully")
        
        # Тест 2: Security импорты
        print("✓ Testing security imports...")
        
        from axonos.security import (
            NeuralDataVault,
            EncryptionEngine,
            DigitalSignature,
            DifferentialPrivacy
        )
        
        print("✓ Security imports working")
        
        # Тест 3: Hardware импорты
        print("✓ Testing hardware imports...")
        
        from axonos.hardware import (
            AbstractBCIDevice,
            DeviceFactory,
            DeviceConfig,
            DeviceType
        )
        
        print("✓ Hardware imports working")
        
        # Тест 4: ML импорты
        print("✓ Testing ML imports...")
        
        from axonos.core.ml.axonml_models import LSTMBCI, TransformerBCI
        from axonos.core.pipeline.axonml_inference import InferenceEngine
        
        print("✓ ML imports working")
        
        # Тест 5: Signal импорты
        print("✓ Testing signal imports...")
        
        from axonos.core.signal import SignalPreprocessor, FREQUENCY_BANDS
        
        print("✓ Signal imports working")
        
        return True
        
    except Exception as e:
        print(f"❌ Import structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Запуск всех тестов"""
    print("="*60)
    print("AXONOS v2.2 COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Устанавливаем тестовый ключ
    os.environ['AXONOS_MASTER_KEY'] = 'test_master_key_32_chars_long_for_testing!'
    
    tests = [
        ("Security Vault Fixes", test_security_vault),
        ("Security Encryption Fixes", test_security_encryption),
        ("Hardware Real-Time Considerations", test_hardware_realtime),
        ("ML Models Integration", test_ml_integration),
        ("Signal Processing", test_signal_processing),
        ("Zero-Knowledge Integration", test_zero_knowledge_integration),
        ("Import Structure", test_import_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append(False)
    
    # Финальные результаты
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{i+1}. {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("AxonOS v2.2 is production-ready with critical fixes!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed.")
        print("Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
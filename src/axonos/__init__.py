"""
AxonOS v2.2 - Secure Privacy-First BCI Protocol
Production-ready platform for neurointerfaces with enterprise-grade security

CRITICAL FIXES v2.2:
- Removed custom vault implementation
- Proper encryption through Fernet/AES-GCM
- Hardware real-time considerations
- No keys in code - environment/KMS only
"""

__version__ = "2.2.0"
__author__ = "AxonOS Team"
__email__ = "team@axonos.org"

# Security (import first - security is core)
from .security.vault import (
    NeuralDataVault,
    SecurityConfig,
    KeyManager,
    DifferentialPrivacy,
    SecurityUtils
)
from .security.encryption import (
    EncryptionEngine,
    DigitalSignature,
    RSAEncryption,
    HybridEncryption,
    HashUtils
)

# Core ML and signal processing
from .core.ml.axonml_models import (
    LSTMBCI,
    TransformerBCI,
    AttentionMechanism,
    ModelConfig
)
from .core.pipeline.axonml_inference import (
    InferenceEngine,
    SignalType,
    InferenceMode,
    InferenceConfig
)
from .core.signal.processing import (
    SignalPreprocessor,
    bandpass_filter,
    notch_filter,
    preprocess_eeg,
    FREQUENCY_BANDS
)

# Hardware abstraction
from .hardware.abstract import (
    AbstractBCIDevice,
    AsyncBCIDevice,
    DeviceConfig,
    DataPacket,
    DeviceType,
    SamplingRate
)
from .hardware.devices import (
    EmulatedBCIDevice,
    OpenBCICytonDevice,
    LSLDevice,
    DeviceFactory
)

# Protocol schemas
from .protocol.schemas import (
    NeuralPacket,
    DeviceInfo,
    SignalData,
    SecurityLevel
)

__all__ = [
    # Security
    "NeuralDataVault",
    "SecurityConfig",
    "KeyManager",
    "DifferentialPrivacy",
    "SecurityUtils",
    "EncryptionEngine",
    "DigitalSignature",
    "RSAEncryption",
    "HybridEncryption",
    "HashUtils",
    
    # ML and Signal Processing
    "LSTMBCI",
    "TransformerBCI",
    "AttentionMechanism",
    "ModelConfig",
    "InferenceEngine",
    "SignalType",
    "InferenceMode",
    "InferenceConfig",
    "SignalPreprocessor",
    "FREQUENCY_BANDS",
    
    # Hardware
    "AbstractBCIDevice",
    "AsyncBCIDevice",
    "DeviceConfig",
    "DataPacket",
    "DeviceType",
    "SamplingRate",
    "EmulatedBCIDevice",
    "OpenBCICytonDevice",
    "LSLDevice",
    "DeviceFactory",
    
    # Protocol
    "NeuralPacket",
    "DeviceInfo",
    "SignalData",
    "SecurityLevel"
]
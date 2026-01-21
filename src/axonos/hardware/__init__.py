"""
Hardware Module - Device drivers and interfaces
Production-ready hardware abstraction with real-time considerations
"""

from .abstract import (
    AbstractBCIDevice,
    AsyncBCIDevice,
    DeviceConfig,
    DataPacket,
    DeviceType,
    SamplingRate,
    HardwareError,
    DeviceConnectionError,
    RealtimeViolationError,
    BufferOverflowError,
    NativeDriverManager
)

from .devices import (
    EmulatedBCIDevice,
    OpenBCICytonDevice,
    LSLDevice,
    DeviceFactory
)

__all__ = [
    "AbstractBCIDevice",
    "AsyncBCIDevice",
    "DeviceConfig",
    "DataPacket",
    "DeviceType",
    "SamplingRate",
    "HardwareError",
    "DeviceConnectionError",
    "RealtimeViolationError",
    "BufferOverflowError",
    "NativeDriverManager",
    "EmulatedBCIDevice",
    "OpenBCICytonDevice",
    "LSLDevice",
    "DeviceFactory"
]
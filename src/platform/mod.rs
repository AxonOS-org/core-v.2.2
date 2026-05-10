//! Platform abstraction layer
//!
//! DWT cycle counter, GPIO stubs, and core-local peripherals.
//!
//! See STMicroelectronics RM0090 (2024) for DWT and NVIC registers.

pub mod dwt;
pub mod gpio;

pub use dwt::Dwt;
pub use gpio::{GpioPin, GPIO_PC13};

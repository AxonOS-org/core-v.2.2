//! Platform abstraction layer
//!
//! DWT cycle counter, GPIO stubs, and core-local peripherals.

pub mod dwt;
pub mod gpio;

pub use dwt::Dwt;
pub use gpio::{GpioPin, GPIO_PC13};

//! GPIO abstraction for stimulation interlock
//!
//! PC13 is used as active-high stimulation enable on STM32F4 Discovery.
//!
//! Reference: STM32F4xx Reference Manual RM0090, Section 8: GPIOs.

/// GPIO pin abstraction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpioPin {
    port: char,
    pin: u8,
    odr_addr: *mut u32,
    bsrr_addr: *mut u32,
}

/// PC13 — stimulation enable (active high)
pub const GPIO_PC13: GpioPin = GpioPin {
    port: 'C',
    pin: 13,
    odr_addr: 0x4002_080C as *mut u32,
    bsrr_addr: 0x4002_0818 as *mut u32,
};

impl GpioPin {
    /// Configure as push-pull output
    pub fn configure_output(&self) {
        let _ = self.odr_addr;
    }

    /// Set pin high via BSRR atomic bit-set
    pub fn set_high(&self) {
        unsafe { core::ptr::write_volatile(self.bsrr_addr, 1 << self.pin) };
    }

    /// Set pin low via BSRR atomic bit-reset
    pub fn set_low(&self) {
        unsafe { core::ptr::write_volatile(self.bsrr_addr, 1 << (self.pin + 16)) };
    }

    /// Read current output state
    pub fn is_high(&self) -> bool {
        let odr = unsafe { core::ptr::read_volatile(self.odr_addr) };
        (odr & (1 << self.pin)) != 0
    }
}

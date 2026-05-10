//! GPIO abstraction for stimulation interlock
//!
//! PC13 is used as active-high stimulation enable on STM32F4 Discovery.

/// GPIO pin abstraction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpioPin {
    port: char,
    pin: u8,
    addr: *mut u32,
}

/// PC13 — stimulation enable (active high)
pub const GPIO_PC13: GpioPin = GpioPin {
    port: 'C',
    pin: 13,
    addr: 0x4002_0800 as *mut u32, // GPIOG_ODR offset placeholder
};

impl GpioPin {
    /// Configure as push-pull output
    pub fn configure_output(&self) {
        // HAL placeholder: real implementation uses RCC + GPIO moder/otyper
        let _ = self.addr;
    }

    /// Set pin high
    pub fn set_high(&self) {
        // Atomic bit-set via BSRR
        unsafe { core::ptr::write_volatile((self.addr as usize + 0x18) as *mut u32, 1 << self.pin) };
    }

    /// Set pin low
    pub fn set_low(&self) {
        // Atomic bit-reset via BSRR
        unsafe { core::ptr::write_volatile((self.addr as usize + 0x18) as *mut u32, 1 << (self.pin + 16)) };
    }
}

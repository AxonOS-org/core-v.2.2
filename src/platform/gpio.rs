//! GPIO abstraction for stimulation interlock
//!
//! PC13 is used as active-high stimulation enable on STM32F4 Discovery.
//!
//! Reference: STM32F4xx Reference Manual RM0090, Section 8: GPIOs.

#![allow(unsafe_code)]

/// GPIO pin abstraction
///
/// Uses memory-mapped register access for deterministic timing.
/// Word-sized stores to BSRR are atomic and require no critical section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpioPin {
    /// Port identifier (for documentation)
    port: char,
    /// Pin number (0-15)
    pin: u8,
    /// Output Data Register (ODR) address
    odr_addr: *mut u32,
    /// Bit Set/Reset Register (BSRR) address
    bsrr_addr: *mut u32,
}

/// PC13 — stimulation enable (active high)
///
/// On STM32F4 Discovery, PC13 drives the user LED.
/// On production boards, this drives a safety relay or MOSFET.
pub const GPIO_PC13: GpioPin = GpioPin {
    port: 'C',
    pin: 13,
    odr_addr: 0x4002_080C as *mut u32,  // GPIOC_ODR
    bsrr_addr: 0x4002_0818 as *mut u32, // GPIOC_BSRR
};

impl GpioPin {
    /// Configure as push-pull output
    ///
    /// # Safety
    /// This is a HAL placeholder. Real implementation configures
    /// RCC clock enable, MODER, OTYPER, OSPEEDR, and PUPDR registers.
    pub fn configure_output(&self) {
        // HAL placeholder: real implementation uses RCC + GPIO moder/otyper
        let _ = self.odr_addr;
    }

    /// Set pin high via BSRR atomic bit-set
    ///
    /// BSRR bit n = 1 sets ODR[n]; BSRR bit n+16 = 1 resets ODR[n].
    /// This is a single atomic write — no read-modify-write needed.
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

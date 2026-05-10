//! NVIC (Nested Vectored Interrupt Controller) helpers

/// NVIC abstraction
pub struct Nvic;

/// Interrupt priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(pub u8);

impl Nvic {
    /// Enable interrupt
    pub fn enable_irq(irqn: u8) {
        let nvic_iser = 0xE000_E100 as *mut u32;
        unsafe {
            core::ptr::write_volatile(nvic_iser.offset((irqn / 32) as isize),
                core::ptr::read_volatile(nvic_iser.offset((irqn / 32) as isize))
                | (1 << (irqn % 32)));
        }
    }

    /// Set priority (preemptive)
    pub fn set_priority(irqn: u8, prio: Priority) {
        let nvic_ipr = 0xE000_E400 as *mut u8;
        unsafe {
            core::ptr::write_volatile(nvic_ipr.offset(irqn as isize), prio.0);
        }
    }
}

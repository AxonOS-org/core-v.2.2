//! Critical section implementation for single-core Cortex-M

use core::arch::asm;

pub struct CriticalSection;

impl CriticalSection {
    /// Enter critical section (disable interrupts)
    pub fn enter() {
        unsafe {
            asm!("cpsid i");
        }
    }

    /// Exit critical section (enable interrupts)
    pub fn exit() {
        unsafe {
            asm!("cpsie i");
        }
    }

    /// Execute closure in critical section
    pub fn with<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        Self::enter();
        let result = f();
        Self::exit();
        result
    }
}

//! Critical section primitives

use core::sync::atomic::{compiler_fence, Ordering};

/// Enter critical section (disable interrupts)
pub fn enter_critical() {
    unsafe { core::arch::asm!("cpsid i") };
    compiler_fence(Ordering::SeqCst);
}

/// Exit critical section (enable interrupts)
pub fn exit_critical() {
    compiler_fence(Ordering::SeqCst);
    unsafe { core::arch::asm!("cpsie i") };
}

/// RAII critical section guard
pub struct CriticalSection;

impl CriticalSection {
    /// Create and enter critical section
    pub fn new() -> Self {
        enter_critical();
        Self
    }
}

impl Drop for CriticalSection {
    fn drop(&mut self) {
        exit_critical();
    }
}

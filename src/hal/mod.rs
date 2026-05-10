//! Hardware abstraction layer
//!
//! Critical sections, memory barriers, and NVIC helpers.

use core::sync::atomic::{compiler_fence, Ordering};

/// Enter critical section (disable interrupts)
#[inline]
pub fn critical_section<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let primask = cortex_m::register::primask::read();
    cortex_m::interrupt::disable();
    let r = f();
    if primask.is_active() {
        cortex_m::interrupt::enable();
    }
    compiler_fence(Ordering::SeqCst);
    r
}

/// Data memory barrier (for dual-core shared memory)
#[inline]
pub fn dmb() {
    compiler_fence(Ordering::SeqCst);
}

/// Instruction synchronization barrier
#[inline]
pub fn isb() {
    cortex_m::asm::isb();
}

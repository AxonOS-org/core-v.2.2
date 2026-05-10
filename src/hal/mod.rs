//! Hardware abstraction layer
//!
//! Critical sections, memory barriers, and NVIC helpers.
//!
//! Reference: ARM Cortex-M4 Technical Reference Manual DDI0439B,
//!            Chapter 5: Nested Vectored Interrupt Controller.

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

/// Data memory barrier
#[inline]
pub fn dmb() {
    compiler_fence(Ordering::SeqCst);
}

/// Instruction synchronization barrier
#[inline]
pub fn isb() {
    cortex_m::asm::isb();
}

/// Memory-mapped I/O write with volatile semantics
#[inline]
pub unsafe fn mmio_write(addr: *mut u32, value: u32) {
    core::ptr::write_volatile(addr, value);
}

/// Memory-mapped I/O read with volatile semantics
#[inline]
pub unsafe fn mmio_read(addr: *const u32) -> u32 {
    core::ptr::read_volatile(addr)
}

//! Memory barriers

/// Data Synchronization Barrier
pub fn dsb() {
    unsafe { core::arch::asm!("dsb") };
}

/// Instruction Synchronization Barrier
pub fn isb() {
    unsafe { core::arch::asm!("isb") };
}

/// Data Memory Barrier
pub fn dmb() {
    unsafe { core::arch::asm!("dmb") };
}

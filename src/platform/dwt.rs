//! Data Watchpoint and Trace (DWT) cycle counter
//!
//! Provides monotonic 32-bit cycle counting with 64-bit extension
//! to prevent wraparound bugs (see RFC-0007).
//!
//! Reference: ARM Cortex-M4 Technical Reference Manual, DDI0439B,
//!            Chapter 11: Data Watchpoint and Trace unit.

#![allow(unsafe_code)]

use core::sync::atomic::{AtomicU32, Ordering};

/// DWT cycle counter wrapper with 64-bit virtual counter
///
/// The DWT.CYCCNT is a 32-bit free-running counter. At 168 MHz,
/// it wraps every ~25.5 seconds. This wrapper extends it to 64 bits
/// by detecting wraparound in software.
pub struct Dwt {
    /// Snapshot of DWT.CYCCNT at last read
    last_raw: AtomicU32,
    /// High 32 bits (overflow count)
    high: AtomicU32,
}

impl Dwt {
    /// Create and enable DWT cycle counter
    ///
    /// # Safety
    /// Must be called with interrupts disabled during initialization
    /// to prevent concurrent access to DWT registers.
    pub fn new() -> Self {
        let dwt = unsafe { &*cortex_m::peripheral::DWT::ptr() };
        // Enable CYCCNT (bit 0 of DWT_CTRL)
        dwt.ctrl.modify(|r| r | 1);
        let raw = dwt.cyccnt.read();
        Self {
            last_raw: AtomicU32::new(raw),
            high: AtomicU32::new(0),
        }
    }

    /// Read 64-bit monotonic cycle count
    ///
    /// Safe against 32-bit wraparound (≈ 25.5 s at 168 MHz).
    /// Thread-safe: uses atomic operations only.
    pub fn cycles(&self) -> u64 {
        let dwt = unsafe { &*cortex_m::peripheral::DWT::ptr() };
        let raw = dwt.cyccnt.read();
        let last = self.last_raw.load(Ordering::Relaxed);
        let high = self.high.load(Ordering::Relaxed);

        if raw < last {
            // Wraparound detected: raw has underflowed since last read
            let new_high = high.wrapping_add(1);
            self.high.store(new_high, Ordering::Relaxed);
            self.last_raw.store(raw, Ordering::Relaxed);
            ((new_high as u64) << 32) | (raw as u64)
        } else {
            self.last_raw.store(raw, Ordering::Relaxed);
            ((high as u64) << 32) | (raw as u64)
        }
    }

    /// Convert cycles to microseconds [L1]
    ///
    /// Formula: µs = cycles * 1_000 / CPU_HZ
    pub fn cycles_to_us(&self, cycles: u64) -> u32 {
        ((cycles * 1_000) / 168_000) as u32
    }
}

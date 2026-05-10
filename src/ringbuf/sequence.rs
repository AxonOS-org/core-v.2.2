//! Sequence numbers for SPSC protocol
//!
//! Theorem 6.3: SPSC sequence-number correctness (Release-Acquire).

use core::sync::atomic::{AtomicU32, Ordering};

/// Sequence number state machine:
/// - seq == index      → Free (producer may write)
/// - seq == index + 1  → Published (consumer may read)
/// - seq == index + N  → Consumed (where N = capacity)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceNumber(pub u32);

/// Atomic sequence number with Release-Acquire semantics
pub struct AtomicSequence {
    inner: AtomicU32,
}

impl AtomicSequence {
    /// Create new atomic sequence with initial value
    pub const fn new(val: u32) -> Self {
        Self { inner: AtomicU32::new(val) }
    }

    /// Load with Acquire ordering
    #[inline]
    pub fn load_acquire(&self) -> SequenceNumber {
        SequenceNumber(self.inner.load(Ordering::Acquire))
    }

    /// Store with Release ordering
    #[inline]
    pub fn store_release(&self, seq: SequenceNumber) {
        self.inner.store(seq.0, Ordering::Release);
    }
}

impl SequenceNumber {
    /// Check if slot is free for producer index `w`
    #[inline]
    pub fn is_free(&self, w: u32) -> bool {
        self.0 == w
    }

    /// Check if slot is published for consumer index `r`
    #[inline]
    pub fn is_published(&self, r: u32) -> bool {
        self.0 == r.wrapping_add(1)
    }

    /// Check if slot is consumed (available for reuse)
    #[inline]
    pub fn is_consumed(&self, r: u32, capacity: u32) -> bool {
        self.0 == r.wrapping_add(capacity)
    }
}

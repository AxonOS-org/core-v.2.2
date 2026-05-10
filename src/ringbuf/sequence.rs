//! Sequence numbers for SPSC protocol
//!
//! Theorem 6.3 (AxonOS RFC-0007): SPSC sequence-number correctness
//! under Release-Acquire memory ordering.
//!
//! Proof sketch:
//! 1. W --sb--> S (program order, producer thread)
//! 2. S --sw--> L (Release-Acquire synchronizes-with pair)
//! 3. L --sb--> R (program order, consumer thread)
//! 4. By transitivity: W --hb--> R, so R observes W.
//!
//! See Batty et al. (2011) "Mathematizing C++ concurrency" POPL 2011
//! for C11 memory model formalism.

use core::sync::atomic::{AtomicU32, Ordering};

/// Sequence number state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceNumber(pub u32);

/// Atomic sequence number with Release-Acquire semantics
pub struct AtomicSequence {
    inner: AtomicU32,
}

impl AtomicSequence {
    pub const fn new(val: u32) -> Self {
        Self { inner: AtomicU32::new(val) }
    }

    #[inline]
    pub fn load_acquire(&self) -> SequenceNumber {
        SequenceNumber(self.inner.load(Ordering::Acquire))
    }

    #[inline]
    pub fn store_release(&self, seq: SequenceNumber) {
        self.inner.store(seq.0, Ordering::Release);
    }
}

impl SequenceNumber {
    #[inline]
    pub fn is_free(&self, w: u32) -> bool {
        self.0 == w
    }

    #[inline]
    pub fn is_published(&self, r: u32) -> bool {
        self.0 == r.wrapping_add(1)
    }

    #[inline]
    pub fn is_consumed(&self, r: u32, capacity: u32) -> bool {
        self.0 == r.wrapping_add(capacity)
    }
}

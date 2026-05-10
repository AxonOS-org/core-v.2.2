//! Ring buffer module
//!
//! Generic SPSC ring buffer with sequence-number protocol (Theorem 6.3).
//!
//! Reference: Vyukov, D. (2010). "Lock-free algorithms: The queue and
//! the ring buffer." See RFC-0007 for formal proof.

pub mod sequence;
pub mod spsc;

pub use spsc::{SpscRingBuffer, RingBufferConfig, SpscError};

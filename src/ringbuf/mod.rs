//! Ring buffer module
//!
//! Generic SPSC ring buffer with sequence-number protocol (Theorem 6.3).

pub mod sequence;
pub mod spsc;

pub use spsc::{SpscRingBuffer, RingBufferConfig, SpscError};

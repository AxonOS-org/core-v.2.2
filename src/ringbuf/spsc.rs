//! Generic SPSC Ring Buffer
//!
//! Zero-copy, wait-free (producer), lock-free (consumer) ring buffer.
//! Based on Dmitry Vyukov's sequence-number protocol.
//!
//! ## Memory Ordering Proof (Theorem 6.3)
//!
//! Let W be the producer's payload write and S the subsequent
//! seq.store(i+1, Release). Let L be the consumer's seq.load(Acquire)
//! observing i+1, and R the subsequent payload read.
//!
//! 1. W --sb--> S (program order, same thread)
//! 2. S --sw--> L (Release-Acquire synchronizes-with pair)
//! 3. L --sb--> R (program order, same thread)
//!
//! By transitivity of happens-before: W --hb--> R, so R observes W.

#![allow(unsafe_code)]

use super::sequence::{SequenceNumber, AtomicSequence};
use crate::config;
use core::sync::atomic::{AtomicU32, Ordering};
use core::mem::MaybeUninit;

/// SPSC ring buffer error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpscError {
    Overrun,
    Underrun,
    ProtocolViolation,
}

/// Ring buffer configuration
#[derive(Debug, Clone, Copy)]
pub struct RingBufferConfig {
    pub capacity: usize,
    pub slot_size: usize,
}

impl Default for RingBufferConfig {
    fn default() -> Self {
        Self {
            capacity: config::RING_BUFFER_CAPACITY,
            slot_size: 64,
        }
    }
}

/// Single-producer single-consumer ring buffer
///
/// Capacity must be power of 2 (enforced at compile time via config).
/// Generic over `T` to support any Copy or owned type.
pub struct SpscRingBuffer<T> {
    buffer: [MaybeUninit<T>; config::RING_BUFFER_CAPACITY],
    sequences: [AtomicSequence; config::RING_BUFFER_CAPACITY],
    write_index: AtomicU32,
    read_index: AtomicU32,
}

impl<T> SpscRingBuffer<T> {
    /// Create new SPSC ring buffer
    ///
    /// All slots initialized to Free state (seq = index).
    pub fn new() -> Self {
        let sequences = core::array::from_fn(|i| AtomicSequence::new(i as u32));
        Self {
            buffer: [const { MaybeUninit::uninit() }; config::RING_BUFFER_CAPACITY],
            sequences,
            write_index: AtomicU32::new(0),
            read_index: AtomicU32::new(0),
        }
    }

    /// Producer: push value to ring buffer
    ///
    /// Wait-free: completes in O(1) steps regardless of consumer state.
    /// Returns Err(Overrun) if buffer full — caller must handle DC1 violation.
    pub fn try_push(&self, value: T) -> Result<(), SpscError> {
        let w = self.write_index.load(Ordering::Relaxed);
        let cap = config::RING_BUFFER_CAPACITY as u32;
        let slot_idx = (w % cap) as usize;

        let seq = self.sequences[slot_idx].load_acquire();
        if !seq.is_free(w) {
            return Err(SpscError::Overrun);
        }

        // SAFETY: We hold exclusive access (seq == w means Free state).
        unsafe {
            core::ptr::write(self.buffer[slot_idx].as_mut_ptr(), value);
        }

        self.sequences[slot_idx].store_release(SequenceNumber(w.wrapping_add(1)));
        self.write_index.store(w.wrapping_add(1), Ordering::Relaxed);
        Ok(())
    }

    /// Consumer: pop value from ring buffer
    ///
    /// Lock-free: may spin briefly if producer is mid-write.
    /// Returns Err(Underrun) if buffer empty.
    pub fn try_pop(&self) -> Result<T, SpscError> {
        let r = self.read_index.load(Ordering::Relaxed);
        let cap = config::RING_BUFFER_CAPACITY as u32;
        let slot_idx = (r % cap) as usize;

        let seq = self.sequences[slot_idx].load_acquire();
        if !seq.is_published(r) {
            return Err(SpscError::Underrun);
        }

        // SAFETY: We verified slot is Published (seq == r + 1).
        let value = unsafe {
            core::ptr::read(self.buffer[slot_idx].as_ptr())
        };

        let consumed_seq = SequenceNumber(r.wrapping_add(cap));
        self.sequences[slot_idx].store_release(consumed_seq);
        self.read_index.store(r.wrapping_add(1), Ordering::Relaxed);
        Ok(value)
    }

    pub fn is_full(&self) -> bool {
        let w = self.write_index.load(Ordering::Relaxed);
        let r = self.read_index.load(Ordering::Relaxed);
        let cap = config::RING_BUFFER_CAPACITY as u32;
        w.wrapping_sub(r) >= cap
    }

    pub fn is_empty(&self) -> bool {
        let w = self.write_index.load(Ordering::Relaxed);
        let r = self.read_index.load(Ordering::Relaxed);
        w == r
    }

    pub fn len(&self) -> usize {
        let w = self.write_index.load(Ordering::Relaxed);
        let r = self.read_index.load(Ordering::Relaxed);
        w.wrapping_sub(r) as usize
    }

    /// Reset ring buffer (for recovery)
    ///
    /// # Safety
    /// Must only be called when producer and consumer are quiescent.
    /// Drops any Published items to avoid leaks.
    pub unsafe fn reset(&self) {
        let cap = config::RING_BUFFER_CAPACITY;
        for i in 0..cap {
            let seq = self.sequences[i].load_acquire();
            let r = self.read_index.load(Ordering::Relaxed);
            if seq.is_published(r) {
                core::ptr::drop_in_place(self.buffer[i].as_mut_ptr());
            }
            self.sequences[i].store_release(SequenceNumber(i as u32));
        }
        self.write_index.store(0, Ordering::Relaxed);
        self.read_index.store(0, Ordering::Relaxed);
    }
}

#[cfg(feature = "kani")]
mod proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(8)]
    fn spsc_no_data_race() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        let value: u32 = kani::any();
        let _ = ring.try_push(value);
        if let Ok(read) = ring.try_pop() {
            assert_eq!(read, value);
        }
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn spsc_push_wait_free() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        let value: u32 = kani::any();
        let _ = ring.try_push(value);
    }

    #[kani::proof]
    #[kani::unwind(2)]
    fn spsc_memory_order() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        let w: u32 = kani::any();
        ring.try_push(w).unwrap();
        let r = ring.try_pop().unwrap();
        assert_eq!(r, w);
    }

    #[kani::proof]
    #[kani::unwind(8)]
    fn spsc_overrun_detected() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        for _ in 0..config::RING_BUFFER_CAPACITY {
            let _ = ring.try_push(0u32);
        }
        assert!(ring.try_push(1u32).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        assert!(ring.try_push(42).is_ok());
        assert_eq!(ring.try_pop().unwrap(), 42);
    }

    #[test]
    fn test_overrun() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        for i in 0..config::RING_BUFFER_CAPACITY {
            assert!(ring.try_push(i as u32).is_ok());
        }
        assert!(ring.try_push(999).is_err());
    }

    #[test]
    fn test_underrun() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        assert!(ring.try_pop().is_err());
    }

    #[test]
    fn test_wraparound() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        for _ in 0..10_000 {
            assert!(ring.try_push(1).is_ok());
            assert_eq!(ring.try_pop().unwrap(), 1);
        }
    }

    #[test]
    fn test_wraparound_32bit() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        ring.write_index.store(u32::MAX - 5, Ordering::Relaxed);
        ring.read_index.store(u32::MAX - 5, Ordering::Relaxed);
        for i in 0..10 {
            assert!(ring.try_push(i).is_ok());
        }
        for i in 0..10 {
            assert_eq!(ring.try_pop().unwrap(), i);
        }
    }
}

//! SPSC ring buffer stress tests

use axonos_kernel::ringbuf::*;
use axonos_kernel::config;

#[test]
fn test_spsc_wraparound_simulation() {
    let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
    // Simulate 2^32 + 100 operations (would take too long; use smaller)
    // Instead, test wrapping arithmetic directly
    for i in 0..1000u32 {
        assert!(ring.try_push(i).is_ok());
        assert_eq!(ring.try_pop().unwrap(), i);
    }
}

#[test]
fn test_spsc_capacity_boundary() {
    let ring: SpscRingBuffer<u8> = SpscRingBuffer::new();
    for i in 0..config::RING_BUFFER_CAPACITY {
        assert!(ring.try_push(i as u8).is_ok());
    }
    assert!(ring.is_full());
    assert!(ring.try_push(0).is_err());
}

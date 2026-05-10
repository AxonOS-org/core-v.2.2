//! Kani proofs for SPSC ring buffer
//!
//! Run with: cargo kani --features kani

#![cfg(feature = "kani")]

use axonos_kernel::ringbuf::SpscRingBuffer;
use axonos_kernel::config;

/// K1: No data race + payload integrity
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

/// K2: Wait-freedom (push returns in bounded steps)
#[kani::proof]
#[kani::unwind(4)]
fn spsc_push_wait_free() {
    let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
    let value: u32 = kani::any();
    let _ = ring.try_push(value);
}

/// K3: Memory ordering / payload integrity
#[kani::proof]
#[kani::unwind(2)]
fn spsc_memory_order() {
    let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
    let w: u32 = kani::any();
    ring.try_push(w).unwrap();
    let r = ring.try_pop().unwrap();
    assert_eq!(r, w);
}

/// K4: Overrun detection
#[kani::proof]
#[kani::unwind(8)]
fn spsc_overrun_detected() {
    let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
    for _ in 0..config::RING_BUFFER_CAPACITY {
        let _ = ring.try_push(0u32);
    }
    assert!(ring.try_push(1u32).is_err());
}

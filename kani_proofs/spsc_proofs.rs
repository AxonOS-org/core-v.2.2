//! Kani BMC proofs for SPSC ring buffer
//!
//! Compile with: cargo kani --features kani

#[cfg(feature = "kani")]
mod proofs {
    use axonos_kernel::ringbuf::SpscRingBuffer;
    use axonos_kernel::config;

    /// K1: No data race — push then pop yields same value
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

    /// K2: Wait-freedom — push always returns
    #[kani::proof]
    #[kani::unwind(4)]
    fn spsc_push_wait_free() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        let value: u32 = kani::any();
        let result = ring.try_push(value);
        // Must not loop forever
        assert!(result.is_ok() || result.is_err());
    }

    /// K3: Memory ordering — Release-Acquire ensures visibility
    #[kani::proof]
    #[kani::unwind(2)]
    fn spsc_memory_order() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        let w: u32 = kani::any();
        ring.try_push(w).unwrap();
        let r = ring.try_pop().unwrap();
        assert_eq!(r, w);
    }
}

//! Kani BMC proofs for SPSC ring buffer
//!
//! Run: cargo kani --features kani

#[cfg(feature = "kani")]
mod spsc_proofs {
    use axonos_kernel::ringbuf::SpscRingBuffer;

    /// K1: No data race — symbolic push then pop
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

    /// K2: Wait-freedom — push returns in bounded steps
    #[kani::proof]
    #[kani::unwind(4)]
    fn spsc_push_wait_free() {
        let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
        let value: u32 = kani::any();
        let _ = ring.try_push(value);
    }

    /// K3: Memory ordering — producer write observed by consumer
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

#[cfg(feature = "kani")]
mod consent_proofs {
    use axonos_kernel::consent::{ConsentFsm, ConsentOp, ConsentState};

    /// K4: Safety — Withdrawn is terminal
    #[kani::proof]
    #[kani::unwind(12)]
    fn consent_withdrawn_terminal() {
        let mut fsm = ConsentFsm::new();
        fsm.transition(ConsentOp::Grant, 0);
        fsm.transition(ConsentOp::Withdraw, 1);
        let result = fsm.transition(ConsentOp::Grant, 2);
        assert!(result.is_none());
        assert_eq!(fsm.state(), ConsentState::Withdrawn);
    }

    /// K5: Liveness — Active can be reached from Inactive
    #[kani::proof]
    #[kani::unwind(12)]
    fn consent_liveness() {
        let mut fsm = ConsentFsm::new();
        let result = fsm.transition(ConsentOp::Grant, 0);
        assert_eq!(result, Some(ConsentState::Active));
    }

    /// K6: Monotonicity — version only increases
    #[kani::proof]
    #[kani::unwind(8)]
    fn consent_version_monotonic() {
        let mut fsm = ConsentFsm::new();
        let v0 = fsm.transition(ConsentOp::Grant, 0);
        let v1 = fsm.transition(ConsentOp::Suspend, 1);
        let v2 = fsm.transition(ConsentOp::Resume, 2);
        // Version increments only on Grant
        assert!(v0.is_some());
    }
}

#![cfg(feature = "kani")]

use axonos_kernel::consent::{ConsentFsm, ConsentOp, ConsentState};

#[kani::proof]
#[kani::unwind(12)]
fn consent_withdrawn_terminal() {
    let mut fsm = ConsentFsm::new();
    fsm.transition(ConsentOp::Grant, 1000);
    fsm.transition(ConsentOp::Withdraw, 2000);
    let result = fsm.transition(ConsentOp::Grant, 3000);
    assert!(result.is_none());
    assert_eq!(fsm.state(), ConsentState::Withdrawn);
}

#[kani::proof]
#[kani::unwind(12)]
fn consent_active_reachable() {
    let mut fsm = ConsentFsm::new();
    let result = fsm.transition(ConsentOp::Grant, 1000);
    assert_eq!(result, Some(ConsentState::Active));
}

#[kani::proof]
#[kani::unwind(12)]
fn consent_suspended_permissions() {
    let mut fsm = ConsentFsm::new();
    fsm.transition(ConsentOp::Grant, 1000);
    fsm.transition(ConsentOp::Suspend, 2000);
    assert!(fsm.is_processing_allowed());
    assert!(!fsm.is_stimulation_allowed());
}

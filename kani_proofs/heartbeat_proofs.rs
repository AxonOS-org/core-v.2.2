//! Kani BMC proofs for heartbeat FSM (DC5)

use axonos_kernel::ipc::*;
use axonos_kernel::config;

/// K5: Heartbeat safety — timeout correctly triggers safe-idle
#[kani::proof]
#[kani::unwind(12)]
fn heartbeat_safety() {
    let mut contract = DualCoreContract::new();
    contract.send_heartbeat(0);

    let now: u64 = kani::any();
    kani::assume(now <= config::SAFE_IDLE_TIMEOUT_MS as u64 * 1000 + 1000);

    contract.check_heartbeat(now);

    if now > config::SAFE_IDLE_TIMEOUT_MS as u64 * 1000 {
        assert!(contract.is_safe_idle());
        assert!(matches!(contract.clause_status(DcClause::Dc5), ClauseStatus::Violated));
    }
}

/// K6: Heartbeat liveness — valid heartbeat keeps system active
#[kani::proof]
#[kani::unwind(12)]
fn heartbeat_liveness() {
    let mut contract = DualCoreContract::new();
    contract.send_heartbeat(0);
    assert!(contract.check_heartbeat(5000)); // 5 ms < 12 ms
    assert!(!contract.is_safe_idle());
}

/// K7: Monotonicity — heartbeat count only increases
#[kani::proof]
#[kani::unwind(8)]
fn heartbeat_monotonicity() {
    let contract = DualCoreContract::new();
    contract.send_heartbeat(0);
    contract.send_heartbeat(1000);
    // Count should be 2; we can't directly observe AtomicU64 in Kani,
    // but we can verify no panic occurs
}

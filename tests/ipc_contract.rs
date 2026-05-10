//! IPC contract tests

use axonos_kernel::ipc::*;

#[test]
fn test_heartbeat_timeout() {
    let mut contract = DualCoreContract::new();
    contract.send_heartbeat(0);
    assert!(contract.check_heartbeat(10_000)); // 10 ms < 12 ms timeout
    assert!(!contract.is_safe_idle());

    assert!(!contract.check_heartbeat(25_000)); // 25 ms > 12 ms timeout
    assert!(contract.is_safe_idle());
}

#[test]
fn test_intent_packet_roundtrip() {
    let contract = DualCoreContract::new();
    let packet = IntentPacket {
        class: 1,
        confidence: 200,
        hmac_tag: [0xAA; 4],
        epoch: 42,
        timestamp: 1000,
    };
    assert!(contract.send_intent(packet).is_ok());
    let received = contract.receive_intent().unwrap();
    assert_eq!(received.class, 1);
    assert_eq!(received.epoch, 42);
}

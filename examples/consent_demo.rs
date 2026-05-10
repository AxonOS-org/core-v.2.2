//! Consent + TrustZone demo — STM32H573

#![no_std]
#![no_main]

use axonos_kernel::consent::{ConsentFsm, ConsentOp, Interlock};
use cortex_m_rt::entry;
use panic_halt as _;

#[entry]
fn main() -> ! {
    let mut fsm = ConsentFsm::new();
    let mut il = Interlock::new();
    il.init_gpio();

    fsm.transition(ConsentOp::Grant, 1000);
    il.update(&fsm, true);

    // Simulate heartbeat loss
    il.update(&fsm, false);
    il.update(&fsm, false);
    il.update(&fsm, false);

    assert!(!il.is_stimulating());
    loop {}
}

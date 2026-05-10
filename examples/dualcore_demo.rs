//! Dual-core contract demo

#![no_std]
#![no_main]

use axonos_kernel::ipc::{DualCoreContract, IntentPacket};
use cortex_m_rt::entry;
use panic_halt as _;

#[entry]
fn main() -> ! {
    let mut contract = DualCoreContract::new();
    let packet = IntentPacket {
        class: 1,
        confidence: 200,
        hmac_tag: [0u8; 4],
        epoch: 0,
        timestamp: 0,
    };
    let _ = contract.send_intent(packet);
    loop {}
}

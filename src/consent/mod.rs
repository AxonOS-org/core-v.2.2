//! Consent management
//!
//! FSM for user consent and stimulation interlock (DC5).

pub mod fsm;
pub mod interlock;

pub use fsm::{ConsentFsm, ConsentState, ConsentOp, ConsentEvent};
pub use interlock::{Interlock, InterlockState};

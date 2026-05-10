//! Consent management
//!
//! FSM for user consent and stimulation interlock (DC5).
//!
//! References:
//! - AxonOS RFC-0002: Mental Privacy Protocol (MMP) Consent Model.
//! - IEC 62304:2006+AMD1:2015, Section 5.3.5: Safety requirements.

pub mod fsm;
pub mod interlock;

pub use fsm::{ConsentFsm, ConsentState, ConsentOp, ConsentEvent};
pub use interlock::{Interlock, InterlockState};

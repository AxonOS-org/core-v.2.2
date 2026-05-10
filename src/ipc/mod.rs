//! Inter-processor communication
//!
//! DC1-DC6 dual-core contract between M4F and A53.

pub mod contract;
pub mod dualcore;

pub use contract::{DcClause, IpcLatency, ContractViolation};
pub use dualcore::{DualCoreContract, IntentPacket, ClauseStatus};

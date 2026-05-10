//! Dual-core contract clauses
//!
//! DC1-DC6 define the formal interface between M4F (hard real-time)
//! and A53 (soft real-time) cores.
//!
//! Reference: AxonOS RFC-0006.

/// Contract clause identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcClause {
    Dc1,
    Dc2,
    Dc3,
    Dc4,
    Dc5,
    Dc6,
}

/// IPC latency budget [µs]
#[derive(Debug, Clone, Copy)]
pub struct IpcLatency {
    pub budget_us: u32,
    pub observed_us: u32,
}

/// Contract violation record
#[derive(Debug, Clone, Copy)]
pub struct ContractViolation {
    pub clause: DcClause,
    pub timestamp: u64,
    pub observed: Option<f32>,
    pub expected: Option<f32>,
}

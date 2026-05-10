//! Dual-core contract clauses

/// Contract clause identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcClause {
    /// DC1: Signal pipeline deadline guarantee
    Dc1,
    /// DC2: Intent packet integrity
    Dc2,
    /// DC3: Capability isolation
    Dc3,
    /// DC4: Mutual information bound
    Dc4,
    /// DC5: Heartbeat / safe-idle
    Dc5,
    /// DC6: Attestation / HMAC
    Dc6,
}

/// IPC latency budget [µs]
#[derive(Debug, Clone, Copy)]
pub struct IpcLatency {
    /// Budget [µs]
    pub budget_us: u32,
    /// Observed [µs]
    pub observed_us: u32,
}

/// Contract violation record
#[derive(Debug, Clone, Copy)]
pub struct ContractViolation {
    /// Violated clause
    pub clause: DcClause,
    /// Timestamp [µs]
    pub timestamp: u64,
    /// Observed value
    pub observed: Option<f32>,
    /// Expected value
    pub expected: Option<f32>,
}

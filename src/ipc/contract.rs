//! Dual-core contract clauses

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcClause {
    Dc1, Dc2, Dc3, Dc4, Dc5, Dc6,
}

#[derive(Debug, Clone, Copy)]
pub struct IpcLatency {
    pub budget_us: u32,
    pub observed_us: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ContractViolation {
    pub clause: DcClause,
    pub timestamp: u64,
    pub observed: Option<f32>,
    pub expected: Option<f32>,
}

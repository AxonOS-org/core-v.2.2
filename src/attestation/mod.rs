//! Attestation module
//!
//! ATECC608B HMAC-SHA256 for secure logging and manifest signing.

use crate::scheduler::TaskId;
use crate::ipc::DcClause;

pub struct Attestation;

pub type HmacSha256 = [u8; 32];

impl Attestation {
    pub fn log_violation(clause: DcClause, task_id: TaskId, timestamp: u32) {
        let _ = (clause, task_id, timestamp);
    }

    pub fn sign_manifest(_manifest_bytes: &[u8]) -> HmacSha256 {
        [0u8; 32]
    }
}

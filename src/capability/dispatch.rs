//! Capability dispatch

use super::{Capability, Manifest};

/// Dispatch table for capability-checked system calls
pub struct Dispatch;

impl Dispatch {
    /// Verify manifest against catalogue and dispatch
    pub fn verify_and_dispatch(manifest: &Manifest) -> Result<(), super::ManifestError> {
        for (cap, rate) in &manifest.capabilities {
            if !super::Catalogue::contains(cap) {
                return Err(super::ManifestError::ProhibitedCapability);
            }
            if *rate > cap.max_rate_hz() {
                return Err(super::ManifestError::RateLimitExceeded);
            }
        }
        Ok(())
    }
}

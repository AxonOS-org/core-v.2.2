//! Artifact rejection (Stage 4)
//!
//! ±120 µV threshold rejection.

use super::EegFrame;

/// Artifact detector
pub struct ArtifactRejection {
    threshold_uv: i32,
}

impl ArtifactRejection {
    /// Create with threshold [µV]
    pub fn new(threshold_uv: i32) -> Self {
        Self { threshold_uv }
    }

    /// Check if frame contains artifact
    pub fn reject(&mut self, frame: EegFrame) -> bool {
        for ch in 0..crate::config::EEG_CHANNELS {
            if frame.channels[ch].abs() > self.threshold_uv {
                return true;
            }
        }
        false
    }

    /// Reset (no state)
    pub fn reset(&mut self) {}
}

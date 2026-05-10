//! Artifact rejection (Stage 4)
//!
//! ±120 µV threshold rejection.
//!
//! Reference: Jung, T. P., et al. (2000). "Removing electroencephalographic
//! artifacts by blind source separation." Psychophysiology 37(2), 163–178.

use super::EegFrame;

pub struct ArtifactRejection {
    threshold_uv: i32,
}

impl ArtifactRejection {
    pub fn new(threshold_uv: i32) -> Self {
        Self { threshold_uv }
    }

    pub fn reject(&mut self, frame: EegFrame) -> bool {
        for ch in 0..crate::config::EEG_CHANNELS {
            if frame.channels[ch].abs() > self.threshold_uv {
                return true;
            }
        }
        false
    }

    pub fn reset(&mut self) {}
}

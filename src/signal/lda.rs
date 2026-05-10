//! LDA classifier (Stage 6)
//!
//! Linear discriminant analysis for 4-class motor imagery.

use super::{EegFrame, MotorImageryClass};

/// LDA classifier
pub struct LdaClassifier {
    /// Class means (4 classes × 8 features)
    means: [[f32; crate::config::EEG_CHANNELS]; 4],
    /// Shared covariance inverse (diagonal approximation)
    cov_inv: [f32; crate::config::EEG_CHANNELS],
}

impl LdaClassifier {
    /// Create with placeholder parameters
    pub fn new(_num_classes: usize) -> Self {
        let means = [[0.0f32; crate::config::EEG_CHANNELS]; 4];
        let cov_inv = [1.0f32; crate::config::EEG_CHANNELS];
        Self { means, cov_inv }
    }

    /// Classify frame
    pub fn classify(&mut self, frame: EegFrame) -> MotorImageryClass {
        let mut best_class = 0usize;
        let mut best_score = f32::MIN;
        for c in 0..4 {
            let mut score = 0.0f32;
            for i in 0..crate::config::EEG_CHANNELS {
                let diff = frame.channels[i] as f32 - self.means[c][i];
                score -= diff * diff * self.cov_inv[i];
            }
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }
        match best_class {
            0 => MotorImageryClass::Rest,
            1 => MotorImageryClass::LeftHand,
            2 => MotorImageryClass::RightHand,
            3 => MotorImageryClass::Feet,
            _ => MotorImageryClass::Rest,
        }
    }

    /// Reset (no adaptive state)
    pub fn reset(&mut self) {}
}

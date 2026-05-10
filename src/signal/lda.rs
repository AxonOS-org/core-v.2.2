//! LDA classifier (Stage 6)
//!
//! Linear discriminant analysis for 4-class motor imagery.
//!
//! Reference: Fukunaga, K. (1990). *Introduction to Statistical Pattern
//! Recognition* (2nd ed.). Academic Press. Chapter 4: Linear Discriminant Functions.

use super::{EegFrame, MotorImageryClass};

pub struct LdaClassifier {
    means: [[f32; crate::config::EEG_CHANNELS]; 4],
    cov_inv: [f32; crate::config::EEG_CHANNELS],
}

impl LdaClassifier {
    pub fn new(_num_classes: usize) -> Self {
        let means = [[0.0f32; crate::config::EEG_CHANNELS]; 4];
        let cov_inv = [1.0f32; crate::config::EEG_CHANNELS];
        Self { means, cov_inv }
    }

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

    pub fn reset(&mut self) {}
}

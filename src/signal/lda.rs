//! LDA classifier (Stage 6)
use super::{EegFrame, MotorImageryClass};
pub struct LdaClassifier {
    means: [[f32; crate::config::EEG_CHANNELS]; 4],
    cov_inv: [f32; crate::config::EEG_CHANNELS],
}
impl LdaClassifier {
    pub fn new(_: usize) -> Self {
        Self { means: [[0.0f32; 8]; 4], cov_inv: [1.0f32; 8] }
    }
    pub fn classify(&mut self, frame: EegFrame) -> MotorImageryClass {
        let mut best = 0usize;
        let mut best_score = f32::MIN;
        for c in 0..4 {
            let mut score = 0.0f32;
            for i in 0..crate::config::EEG_CHANNELS {
                let diff = frame.channels[i] as f32 - self.means[c][i];
                score -= diff * diff * self.cov_inv[i];
            }
            if score > best_score { best_score = score; best = c; }
        }
        match best {
            0 => MotorImageryClass::Rest,
            1 => MotorImageryClass::LeftHand,
            2 => MotorImageryClass::RightHand,
            3 => MotorImageryClass::Feet,
            _ => MotorImageryClass::Rest,
        }
    }
    pub fn reset(&mut self) {}
}

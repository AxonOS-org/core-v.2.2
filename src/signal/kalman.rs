//! Kalman state estimator (Stage 1)
use super::EegFrame;
pub struct KalmanEstimator {
    x: [f32; crate::config::EEG_CHANNELS],
    p: [f32; crate::config::EEG_CHANNELS],
    q: f32,
    r: f32,
}
impl KalmanEstimator {
    pub fn new(_: usize) -> Self {
        Self { x: [0.0; 8], p: [1.0; 8], q: 0.01, r: 100.0 }
    }
    pub fn update(&mut self, frame: EegFrame) -> EegFrame {
        let mut out = EegFrame::zero();
        for i in 0..crate::config::EEG_CHANNELS {
            let z = frame.channels[i] as f32;
            let p_pred = self.p[i] + self.q;
            let k = p_pred / (p_pred + self.r);
            self.x[i] += k * (z - self.x[i]);
            self.p[i] = (1.0 - k) * p_pred;
            out.channels[i] = self.x[i] as i32;
        }
        out
    }
    pub fn reset(&mut self) {
        self.x = [0.0; 8];
        self.p = [1.0; 8];
    }
}

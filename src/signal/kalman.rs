//! Kalman state estimator (Stage 1)
//!
//! Simple scalar Kalman filter per channel.
//!
//! Reference: Welch, G., & Bishop, G. (2006). "An Introduction to the
//! Kalman Filter." UNC-Chapel Hill Technical Report TR 95-041.

use super::EegFrame;

pub struct KalmanEstimator {
    x: [f32; crate::config::EEG_CHANNELS],
    p: [f32; crate::config::EEG_CHANNELS],
    q: f32,
    r: f32,
}

impl KalmanEstimator {
    pub fn new(_channels: usize) -> Self {
        Self {
            x: [0.0; crate::config::EEG_CHANNELS],
            p: [1.0; crate::config::EEG_CHANNELS],
            q: 0.01,
            r: 100.0,
        }
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
        self.x = [0.0; crate::config::EEG_CHANNELS];
        self.p = [1.0; crate::config::EEG_CHANNELS];
    }
}

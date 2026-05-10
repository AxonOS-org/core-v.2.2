//! Kalman state estimator (Stage 1)
//!
//! Simple scalar Kalman filter per channel.

use super::EegFrame;

/// Kalman estimator per channel
pub struct KalmanEstimator {
    /// State estimate per channel [µV]
    x: [f32; crate::config::EEG_CHANNELS],
    /// Error covariance per channel
    p: [f32; crate::config::EEG_CHANNELS],
    /// Process noise covariance
    q: f32,
    /// Measurement noise covariance
    r: f32,
}

impl KalmanEstimator {
    /// Create new estimator
    pub fn new(_channels: usize) -> Self {
        Self {
            x: [0.0; crate::config::EEG_CHANNELS],
            p: [1.0; crate::config::EEG_CHANNELS],
            q: 0.01,
            r: 100.0,
        }
    }

    /// Update with new frame
    pub fn update(&mut self, frame: EegFrame) -> EegFrame {
        let mut out = EegFrame::zero();
        for i in 0..crate::config::EEG_CHANNELS {
            let z = frame.channels[i] as f32;
            // Prediction
            let p_pred = self.p[i] + self.q;
            // Update
            let k = p_pred / (p_pred + self.r);
            self.x[i] += k * (z - self.x[i]);
            self.p[i] = (1.0 - k) * p_pred;
            out.channels[i] = self.x[i] as i32;
        }
        out
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.x = [0.0; crate::config::EEG_CHANNELS];
        self.p = [1.0; crate::config::EEG_CHANNELS];
    }
}

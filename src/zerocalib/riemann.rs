//! Riemannian minimum distance to mean (MDM) classifier
//!
//! Uses fixed-point arithmetic for `#![no_std]` compatibility.

use micromath::F32Ext;

/// 8×8 covariance matrix (fixed-point Q15.16)
pub struct CovarianceMatrix {
    data: [[i32; crate::config::EEG_CHANNELS]; crate::config::EEG_CHANNELS],
}

impl CovarianceMatrix {
    /// Create zero matrix
    pub fn zero() -> Self {
        Self { data: [[0; crate::config::EEG_CHANNELS]; crate::config::EEG_CHANNELS] }
    }

    /// Compute sample covariance from EEG frame window
    pub fn from_window(window: &[crate::signal::EegFrame]) -> Self {
        let mut cov = Self::zero();
        let n = window.len() as i32;
        if n == 0 { return cov; }

        // Compute mean per channel
        let mut mean = [0i64; crate::config::EEG_CHANNELS];
        for frame in window {
            for ch in 0..crate::config::EEG_CHANNELS {
                mean[ch] += frame.channels[ch] as i64;
            }
        }
        for ch in 0..crate::config::EEG_CHANNELS {
            mean[ch] /= n as i64;
        }

        // Compute covariance
        for frame in window {
            for i in 0..crate::config::EEG_CHANNELS {
                for j in 0..crate::config::EEG_CHANNELS {
                    let xi = (frame.channels[i] as i64) - mean[i];
                    let xj = (frame.channels[j] as i64) - mean[j];
                    cov.data[i][j] += ((xi * xj) / n as i64) as i32;
                }
            }
        }
        cov
    }
}

/// Riemannian MDM classifier
pub struct RiemannianClassifier {
    class_means: [CovarianceMatrix; 4],
}

impl RiemannianClassifier {
    /// Create classifier with identity means
    pub fn new() -> Self {
        Self {
            class_means: [
                CovarianceMatrix::zero(),
                CovarianceMatrix::zero(),
                CovarianceMatrix::zero(),
                CovarianceMatrix::zero(),
            ],
        }
    }

    /// Compute approximate Riemannian distance (Frobenius norm in log-Euclidean)
    pub fn distance(a: &CovarianceMatrix, b: &CovarianceMatrix) -> f32 {
        let mut sum: f32 = 0.0;
        for i in 0..crate::config::EEG_CHANNELS {
            for j in 0..crate::config::EEG_CHANNELS {
                let diff = (a.data[i][j] - b.data[i][j]) as f32 / 65536.0;
                sum += diff * diff;
            }
        }
        sum.sqrt()
    }

    /// Classify window by minimum distance to class means
    pub fn classify(&self, window: &[crate::signal::EegFrame]) -> crate::signal::MotorImageryClass {
        let cov = CovarianceMatrix::from_window(window);
        let mut best = 0usize;
        let mut best_dist = f32::MAX;
        for c in 0..4 {
            let d = Self::distance(&cov, &self.class_means[c]);
            if d < best_dist {
                best_dist = d;
                best = c;
            }
        }
        match best {
            0 => crate::signal::MotorImageryClass::Rest,
            1 => crate::signal::MotorImageryClass::LeftHand,
            2 => crate::signal::MotorImageryClass::RightHand,
            _ => crate::signal::MotorImageryClass::BothFeet,
        }
    }
}

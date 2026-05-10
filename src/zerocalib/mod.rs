//! Riemannian classifier calibration
//!
//! Placeholder for geometric mean shift on SPD manifold.

use crate::signal::EegFrame;

/// Riemannian zero-calibration
pub struct ZeroCalib;

impl ZeroCalib {
    /// Update calibration with new frame
    pub fn update(&mut self, _frame: EegFrame) {
        // TODO: implement covariance estimation on SPD manifold
    }

    /// Apply calibration inverse
    pub fn apply(&self, frame: EegFrame) -> EegFrame {
        // Identity pass-through until calibrated
        frame
    }
}

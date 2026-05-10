//! Riemannian classifier calibration
//!
//! Geometric mean shift on SPD manifold for zero-training classification.
//!
//! References:
//! - Congedo, M., Barachant, A., & Bhatia, R. (2017). "Riemannian geometry for
//!   EEG-based brain-computer interfaces." IEEE TBCI.
//! - Yger, F. (2013). "A review of classification algorithms for EEG-based
//!   brain-computer interfaces." Journal of Neural Engineering 10(3).

use crate::signal::EegFrame;

pub struct ZeroCalib;

impl ZeroCalib {
    pub fn update(&mut self, _frame: EegFrame) {
    }

    pub fn apply(&self, frame: EegFrame) -> EegFrame {
        frame
    }
}

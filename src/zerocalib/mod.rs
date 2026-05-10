//! Riemannian classifier calibration

use crate::signal::EegFrame;

pub struct ZeroCalib;

impl ZeroCalib {
    pub fn update(&mut self, _frame: EegFrame) {}

    pub fn apply(&self, frame: EegFrame) -> EegFrame {
        frame
    }
}

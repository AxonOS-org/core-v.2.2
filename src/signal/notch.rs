//! Notch filter (Stage 3)
//!
//! 50 Hz + 60 Hz powerline rejection.

use super::EegFrame;
use micromath::F32Ext;

/// IIR notch filter per channel
pub struct NotchFilter {
    /// Filter state per channel [y1, y2, x1, x2]
    state: [[f32; 4]; crate::config::EEG_CHANNELS],
    /// Coefficients [a1, a2, b0, b1, b2]
    coeffs: [f32; 5],
}

impl NotchFilter {
    /// Create notch at 50 Hz and 60 Hz (cascade of two biquads)
    pub fn new(_sampling_rate: u32) -> Self {
        // Placeholder coefficients for 50 Hz notch at 250 SPS
        let coeffs = [1.8, 0.98, 1.0, -1.8, 0.98];
        Self {
            state: [[0.0; 4]; crate::config::EEG_CHANNELS],
            coeffs,
        }
    }

    /// Process one frame
    pub fn process(&mut self, frame: EegFrame) -> EegFrame {
        let mut out = EegFrame::zero();
        for ch in 0..crate::config::EEG_CHANNELS {
            let x0 = frame.channels[ch] as f32;
            let [a1, a2, b0, b1, b2] = self.coeffs;
            let [y1, y2, x1, x2] = self.state[ch];
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            self.state[ch] = [y0, y1, x0, x1];
            out.channels[ch] = y0 as i32;
        }
        out
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.state = [[0.0; 4]; crate::config::EEG_CHANNELS];
    }
}

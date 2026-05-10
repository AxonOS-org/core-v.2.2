//! Common Spatial Patterns filter (Stage 5)
//!
//! 8×8 spatial filter for motor imagery.
//!
//! References:
//! - Blankertz et al. (2008). IEEE Signal Processing Magazine 25(1), 41–56.
//! - Ramoser et al. (2000). IEEE TBME 47(4), 583–584.

use super::EegFrame;

pub struct CspFilter {
    w: [[f32; crate::config::EEG_CHANNELS]; crate::config::EEG_CHANNELS],
}

impl CspFilter {
    pub fn new(_channels: usize) -> Self {
        let mut w = [[0.0f32; crate::config::EEG_CHANNELS]; crate::config::EEG_CHANNELS];
        for i in 0..crate::config::EEG_CHANNELS {
            w[i][i] = 1.0;
        }
        Self { w }
    }

    pub fn project(&mut self, frame: EegFrame) -> EegFrame {
        let mut out = EegFrame::zero();
        for i in 0..crate::config::EEG_CHANNELS {
            let mut acc = 0.0f32;
            for j in 0..crate::config::EEG_CHANNELS {
                acc += self.w[i][j] * frame.channels[j] as f32;
            }
            out.channels[i] = acc as i32;
        }
        out
    }

    pub fn reset(&mut self) {}
}

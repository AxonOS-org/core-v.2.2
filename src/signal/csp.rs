//! Common Spatial Patterns filter (Stage 5)
//!
//! 8×8 spatial filter for motor imagery.

use super::EegFrame;

/// CSP spatial filter
pub struct CspFilter {
    /// Filter matrix (8×8)
    w: [[f32; crate::config::EEG_CHANNELS]; crate::config::EEG_CHANNELS],
}

impl CspFilter {
    /// Create with identity matrix (placeholder — trained offline)
    pub fn new(_channels: usize) -> Self {
        let mut w = [[0.0f32; crate::config::EEG_CHANNELS]; crate::config::EEG_CHANNELS];
        for i in 0..crate::config::EEG_CHANNELS {
            w[i][i] = 1.0;
        }
        Self { w }
    }

    /// Project frame through CSP matrix
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

    /// Reset (no adaptive state)
    pub fn reset(&mut self) {}
}

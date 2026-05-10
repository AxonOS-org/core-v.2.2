//! Common Spatial Patterns filter (Stage 5)
use super::EegFrame;
pub struct CspFilter {
    w: [[f32; crate::config::EEG_CHANNELS]; crate::config::EEG_CHANNELS],
}
impl CspFilter {
    pub fn new(_: usize) -> Self {
        let mut w = [[0.0f32; 8]; 8];
        for i in 0..8 { w[i][i] = 1.0; }
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

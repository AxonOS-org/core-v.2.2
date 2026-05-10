//! Notch filter (Stage 3)
use super::EegFrame;
use micromath::F32Ext;
pub struct NotchFilter {
    state: [[f32; 4]; crate::config::EEG_CHANNELS],
    coeffs: [f32; 5],
}
impl NotchFilter {
    pub fn new(_: u32) -> Self {
        Self { state: [[0.0; 4]; 8], coeffs: [1.8, 0.98, 1.0, -1.8, 0.98] }
    }
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
    pub fn reset(&mut self) { self.state = [[0.0; 4]; 8]; }
}

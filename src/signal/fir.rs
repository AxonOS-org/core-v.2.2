//! FIR bandpass filter bank (Stage 2)
//!
//! Dominant stage: ~50% of pipeline WCET.
//!
//! Reference: Oppenheim, A. V., & Schafer, R. W. (2009).
//! *Discrete-Time Signal Processing* (3rd ed.). Prentice Hall.

use super::EegFrame;

pub struct FirFilter {
    taps: [f32; crate::config::FIR_ORDER],
    delay: [[f32; crate::config::FIR_ORDER]; crate::config::EEG_CHANNELS],
    tap_idx: usize,
}

impl FirFilter {
    pub fn new(order: usize, _channels: usize) -> Self {
        let mut taps = [0.0f32; crate::config::FIR_ORDER];
        for i in 0..order.min(crate::config::FIR_ORDER) {
            let x = (i as f32 - (order as f32 - 1.0) / 2.0) * core::f32::consts::PI / 10.0;
            taps[i] = if x == 0.0 { 1.0 } else { x.sin() / x };
        }
        Self {
            taps,
            delay: [[0.0; crate::config::FIR_ORDER]; crate::config::EEG_CHANNELS],
            tap_idx: 0,
        }
    }

    pub fn process(&mut self, frame: EegFrame) -> EegFrame {
        let mut out = EegFrame::zero();
        for ch in 0..crate::config::EEG_CHANNELS {
            self.delay[ch][self.tap_idx] = frame.channels[ch] as f32;
            let mut acc = 0.0f32;
            for t in 0..crate::config::FIR_ORDER {
                let idx = (self.tap_idx + crate::config::FIR_ORDER - t) % crate::config::FIR_ORDER;
                acc += self.delay[ch][idx] * self.taps[t];
            }
            out.channels[ch] = acc as i32;
        }
        self.tap_idx = (self.tap_idx + 1) % crate::config::FIR_ORDER;
        out
    }

    pub fn reset(&mut self) {
        self.delay = [[0.0; crate::config::FIR_ORDER]; crate::config::EEG_CHANNELS];
        self.tap_idx = 0;
    }
}

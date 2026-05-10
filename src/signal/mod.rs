//! Signal processing pipeline
//!
//! Zero-copy 6-stage pipeline from ADC DMA to classifier.

pub mod artifact;
pub mod csp;
pub mod fir;
pub mod kalman;
pub mod lda;
pub mod notch;
pub mod pipeline;

pub use pipeline::{SignalPipeline, PipelineConfig, PipelineStage};

#[derive(Debug, Clone, Copy)]
pub struct EegFrame {
    pub channels: [i32; crate::config::EEG_CHANNELS],
}

impl EegFrame {
    pub fn zero() -> Self {
        Self { channels: [0; crate::config::EEG_CHANNELS] }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Epoch {
    pub index: u64,
    pub start_us: u32,
    pub elapsed_us: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotorImageryClass {
    Rest,
    LeftHand,
    RightHand,
    Feet,
    Tongue,
}

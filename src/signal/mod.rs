//! Signal processing pipeline
//!
//! Zero-copy 6-stage pipeline from ADC DMA to classifier.
//!
//! References:
//! - Blankertz et al. (2008). "Optimizing spatial filters for robust EEG
//!   single-trial analysis." IEEE Signal Processing Magazine 25(1), 41–56. [CSP]
//! - Ramoser et al. (2000). "Optimal spatial filtering of single trial EEG
//!   during imagined hand movement." IEEE TBME 47(4), 583–584. [CSP for MI]

pub mod artifact;
pub mod csp;
pub mod fir;
pub mod kalman;
pub mod lda;
pub mod notch;
pub mod pipeline;

pub use pipeline::{SignalPipeline, PipelineConfig, PipelineStage};

/// Single EEG sample frame (8 channels × 24-bit → stored as i32)
#[derive(Debug, Clone, Copy)]
pub struct EegFrame {
    /// Channel samples [µV]
    pub channels: [i32; crate::config::EEG_CHANNELS],
}

impl EegFrame {
    pub fn zero() -> Self {
        Self { channels: [0; crate::config::EEG_CHANNELS] }
    }
}

/// Epoch metadata for timing measurement
#[derive(Debug, Clone, Copy)]
pub struct Epoch {
    pub index: u64,
    pub start_us: u32,
    pub elapsed_us: u32,
}

/// Motor imagery classification output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotorImageryClass {
    Rest,
    LeftHand,
    RightHand,
    Feet,
    Tongue,
}

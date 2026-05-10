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

/// Single EEG sample frame (8 channels × 24-bit → stored as i32)
#[derive(Debug, Clone, Copy)]
pub struct EegFrame {
    /// Channel samples [µV]
    pub channels: [i32; crate::config::EEG_CHANNELS],
}

impl EegFrame {
    /// Create zeroed frame
    pub fn zero() -> Self {
        Self { channels: [0; crate::config::EEG_CHANNELS] }
    }
}

/// Epoch metadata for timing measurement
#[derive(Debug, Clone, Copy)]
pub struct Epoch {
    /// Epoch index (monotonic)
    pub index: u64,
    /// Start timestamp [µs]
    pub start_us: u32,
    /// Elapsed time [µs] (filled by pipeline)
    pub elapsed_us: u32,
}

/// Motor imagery classification output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotorImageryClass {
    /// No detectable motor imagery
    Rest,
    /// Left hand imagery
    LeftHand,
    /// Right hand imagery
    RightHand,
    /// Feet imagery
    Feet,
    /// Tongue imagery
    Tongue,
}

//! Signal Pipeline Orchestrator
//!
//! Zero-copy signal path from ADC DMA to classifier.
//! No heap allocation on hot path.
//!
//! ## Pipeline Stages (Table 3)
//!
//! | Stage | C_i (µs) | Derivation |
//! |-------|----------|------------|
//! | Kalman state estimator (8-ch) | 80.0 | 13,440 cycles / 168 MHz |
//! | FIR bandpass (order 64, 8-ch) | 320.0 | ≈40 µs/ch (Remark 5.5) |
//! | Notch filter (50 Hz + 60 Hz) | 60.0 | 10,080 cycles / 168 MHz |
//! | Artifact rejection (±120 µV) | 40.0 | 6,720 cycles / 168 MHz |
//! | CSP spatial filter (8 × 8) | 100.0 | 16,800 cycles / 168 MHz |
//! | LDA classifier | 40.2 | 6,754 cycles / 168 MHz |
//! | **Pipeline subtotal** | **640.2** | incl. SPSC push overhead |

use super::{EegFrame, MotorImageryClass, Epoch, PipelineStage};
use crate::ringbuf::SpscRingBuffer;
use crate::config;
use crate::consent::Interlock;

/// Signal pipeline state
pub struct SignalPipeline {
    kalman: crate::signal::kalman::KalmanEstimator,
    fir: crate::signal::fir::FirFilter,
    notch: crate::signal::notch::NotchFilter,
    artifact: crate::signal::artifact::ArtifactRejection,
    csp: crate::signal::csp::CspFilter,
    lda: crate::signal::lda::LdaClassifier,
    output: SpscRingBuffer<MotorImageryClass>,
    current_epoch: Option<Epoch>,
    epoch_count: u64,
    wcet_observed: u32,
}

/// Pipeline configuration
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// FIR filter order
    pub fir_order: usize,
    /// Number of EEG channels
    pub channels: usize,
    /// ADC sampling rate [SPS]
    pub sampling_rate: u32,
    /// Artifact rejection threshold [µV]
    pub artifact_threshold_uv: i32,
    /// Number of LDA classes
    pub num_classes: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            fir_order: config::FIR_ORDER,
            channels: config::EEG_CHANNELS,
            sampling_rate: config::ADC_SPS,
            artifact_threshold_uv: 120,
            num_classes: 4,
        }
    }
}

impl SignalPipeline {
    /// Create new signal pipeline
    pub fn new(cfg: PipelineConfig) -> Self {
        Self {
            kalman: crate::signal::kalman::KalmanEstimator::new(cfg.channels),
            fir: crate::signal::fir::FirFilter::new(cfg.fir_order, cfg.channels),
            notch: crate::signal::notch::NotchFilter::new(cfg.sampling_rate),
            artifact: crate::signal::artifact::ArtifactRejection::new(cfg.artifact_threshold_uv),
            csp: crate::signal::csp::CspFilter::new(cfg.channels),
            lda: crate::signal::lda::LdaClassifier::new(cfg.num_classes),
            output: SpscRingBuffer::new(),
            current_epoch: None,
            epoch_count: 0,
            wcet_observed: 0,
        }
    }

    /// Process one epoch of EEG data
    ///
    /// Called by scheduler at each ADC DMA completion interrupt.
    /// Must complete within 4 ms deadline.
    pub fn process(&mut self, frame: EegFrame, epoch: Epoch) -> Option<MotorImageryClass> {
        self.current_epoch = Some(epoch);
        self.epoch_count += 1;

        let estimated = self.kalman.update(frame);
        let filtered = self.fir.process(estimated);
        let notched = self.notch.process(filtered);

        if self.artifact.reject(notched) {
            return None;
        }

        let spatial = self.csp.project(notched);
        let class = self.lda.classify(spatial);

        // Push to output ring buffer — never silently drop
        if let Err(_e) = self.output.try_push(class) {
            // DC1 violation: signal path overrun
            Interlock::activate_safe_idle();
            crate::attestation::Attestation::log_violation(
                crate::ipc::DcClause::Dc1,
                crate::scheduler::TaskId(1),
                epoch.start_us,
            );
            return None;
        }

        // Update WCET observation
        if let Some(ref mut e) = self.current_epoch {
            let elapsed = e.elapsed_us;
            if elapsed > self.wcet_observed {
                self.wcet_observed = elapsed;
            }
        }

        Some(class)
    }

    /// Get observed WCET [µs]
    pub fn observed_wcet(&self) -> u32 {
        self.wcet_observed
    }

    /// Get epoch count
    pub fn epoch_count(&self) -> u64 {
        self.epoch_count
    }

    /// Reset pipeline state
    pub fn reset(&mut self) {
        self.kalman.reset();
        self.fir.reset();
        self.notch.reset();
        self.artifact.reset();
        self.csp.reset();
        self.lda.reset();
        self.epoch_count = 0;
        self.wcet_observed = 0;
    }
}

/// Pipeline stage trait for modular composition
pub trait PipelineStage {
    type Input;
    type Output;
    /// Process one sample
    fn process(&mut self, input: Self::Input) -> Self::Output;
    /// Reset internal state
    fn reset(&mut self);
}

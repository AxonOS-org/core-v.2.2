//! Signal Pipeline Orchestrator
use super::{EegFrame, MotorImageryClass, Epoch, PipelineStage};
use crate::ringbuf::SpscRingBuffer;
use crate::config;
use crate::consent::Interlock;

pub struct SignalPipeline {
    kalman: crate::signal::kalman::KalmanEstimator,
    fir: crate::signal::fir::FirFilter,
    notch: crate::signal::notch::NotchFilter,
    artifact: crate::signal::artifact::ArtifactRejection,
    csp: crate::signal::csp::CspFilter,
    lda: crate::signal::lda::LdaClassifier,
    output: SpscRingBuffer<MotorImageryClass>,
    epoch_count: u64,
    wcet_observed: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    pub fir_order: usize,
    pub channels: usize,
    pub sampling_rate: u32,
    pub artifact_threshold_uv: i32,
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
    pub fn new(cfg: PipelineConfig) -> Self {
        Self {
            kalman: crate::signal::kalman::KalmanEstimator::new(cfg.channels),
            fir: crate::signal::fir::FirFilter::new(cfg.fir_order, cfg.channels),
            notch: crate::signal::notch::NotchFilter::new(cfg.sampling_rate),
            artifact: crate::signal::artifact::ArtifactRejection::new(cfg.artifact_threshold_uv),
            csp: crate::signal::csp::CspFilter::new(cfg.channels),
            lda: crate::signal::lda::LdaClassifier::new(cfg.num_classes),
            output: SpscRingBuffer::new(),
            epoch_count: 0,
            wcet_observed: 0,
        }
    }

    pub fn process(&mut self, frame: EegFrame, epoch: Epoch) -> Option<MotorImageryClass> {
        self.epoch_count += 1;
        let estimated = self.kalman.update(frame);
        let filtered = self.fir.process(estimated);
        let notched = self.notch.process(filtered);
        if self.artifact.reject(notched) {
            return None;
        }
        let spatial = self.csp.project(notched);
        let class = self.lda.classify(spatial);
        if let Err(_e) = self.output.try_push(class) {
            Interlock::activate_safe_idle();
            crate::attestation::Attestation::log_violation(
                crate::ipc::DcClause::Dc1,
                crate::scheduler::TaskId(1),
                epoch.start_us,
            );
            return None;
        }
        let elapsed = epoch.elapsed_us;
        if elapsed > self.wcet_observed {
            self.wcet_observed = elapsed;
        }
        Some(class)
    }

    pub fn observed_wcet(&self) -> u32 { self.wcet_observed }
    pub fn epoch_count(&self) -> u64 { self.epoch_count }

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

pub trait PipelineStage {
    type Input;
    type Output;
    fn process(&mut self, input: Self::Input) -> Self::Output;
    fn reset(&mut self);
}

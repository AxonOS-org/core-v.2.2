//! Integration tests for AxonOS Kernel v0.2.2

use axonos_kernel::scheduler::{EdfScheduler, Task, TaskId, AdmissionError};
use axonos_kernel::ringbuf::SpscRingBuffer;
use axonos_kernel::consent::{ConsentFsm, ConsentOp, Interlock};
use axonos_kernel::signal::{SignalPipeline, PipelineConfig, EegFrame};
use axonos_kernel::config;

#[test]
fn test_scheduler_admission() {
    let mut sched = EdfScheduler::new();
    let t0 = Task::new(TaskId(0), 4000, 800);
    let t1 = Task::new(TaskId(1), 4000, 600);
    assert!(sched.register_task(t0).is_ok());
    assert!(sched.register_task(t1).is_ok());

    // Exceed ceiling
    let t2 = Task::new(TaskId(2), 4000, 2000);
    assert!(matches!(sched.register_task(t2), Err(AdmissionError::CeilingExceeded { .. })));
}

#[test]
fn test_spsc_wraparound_simulation() {
    let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
    // Simulate many push/pop cycles
    for epoch in 0..10_000u32 {
        assert!(ring.try_push(epoch).is_ok());
        assert_eq!(ring.try_pop().unwrap(), epoch);
    }
    assert!(ring.is_empty());
}

#[test]
fn test_consent_interlock_integration() {
    let mut consent = ConsentFsm::new();
    let mut interlock = Interlock::new();
    interlock.init_gpio();

    consent.transition(ConsentOp::Grant, 0);
    interlock.update(&consent, true);
    assert!(interlock.is_stimulating());

    consent.transition(ConsentOp::Suspend, 1000);
    interlock.update(&consent, true);
    assert!(!interlock.is_stimulating());
}

#[test]
fn test_signal_pipeline_end_to_end() {
    let mut pipeline = SignalPipeline::new(PipelineConfig::default());
    let frame = EegFrame { channels: [100; config::EEG_CHANNELS] };
    let epoch = axonos_kernel::signal::Epoch { index: 0, start_us: 0, elapsed_us: 0 };
    let result = pipeline.process(frame, epoch);
    // Result may be None (artifact) or Some(class)
    assert!(pipeline.epoch_count() == 1);
}

#[test]
fn test_spsc_overrun_safety() {
    let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
    for i in 0..config::RING_BUFFER_CAPACITY {
        assert!(ring.try_push(i as u32).is_ok());
    }
    // Overrun must be detected
    assert!(ring.try_push(999).is_err());
    assert!(ring.is_full());
}

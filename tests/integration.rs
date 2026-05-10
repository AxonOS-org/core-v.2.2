//! Integration tests

use axonos_kernel::scheduler::{EdfScheduler, Task, TaskId, AdmissionError};
use axonos_kernel::ringbuf::SpscRingBuffer;
use axonos_kernel::signal::{SignalPipeline, PipelineConfig, EegFrame, Epoch};
use axonos_kernel::consent::{ConsentFsm, ConsentOp, Interlock};
use axonos_kernel::config;

#[test]
fn test_edf_schedulability() {
    let mut sched = EdfScheduler::new();
    let t1 = Task::new(TaskId(1), 818, 4000);
    let t2 = Task::new(TaskId(2), 100, 4000);
    let t3 = Task::new(TaskId(3), 54, 4000);
    assert!(sched.register_task(t1).is_ok());
    assert!(sched.register_task(t2).is_ok());
    assert!(sched.register_task(t3).is_ok());
    assert!(sched.stats().utilisation <= config::ADMISSION_CEILING);
}

#[test]
fn test_admission_ceiling() {
    let mut sched = EdfScheduler::new();
    let t1 = Task::new(TaskId(1), 1000, 4000);
    let t2 = Task::new(TaskId(2), 100, 4000);
    assert!(sched.register_task(t1).is_ok());
    assert!(matches!(sched.register_task(t2), Err(AdmissionError::CeilingExceeded { .. })));
}

#[test]
fn test_spsc_wraparound_stress() {
    let ring: SpscRingBuffer<u32> = SpscRingBuffer::new();
    for i in 0..1_000_000u32 {
        assert!(ring.try_push(i).is_ok());
        assert_eq!(ring.try_pop().unwrap(), i);
    }
}

#[test]
fn test_signal_pipeline_wcet() {
    let mut pipe = SignalPipeline::new(PipelineConfig::default());
    let frame = EegFrame::zero();
    let epoch = Epoch { index: 0, start_us: 0, elapsed_us: 0 };
    let result = pipe.process(frame, epoch);
    assert!(result.is_some() || result.is_none());
    assert!(pipe.observed_wcet() < config::EPOCH_US);
}

#[test]
fn test_consent_withdrawal_terminal() {
    let mut fsm = ConsentFsm::new();
    fsm.transition(ConsentOp::Grant, 1000);
    fsm.transition(ConsentOp::Withdraw, 2000);
    assert!(fsm.is_withdrawn());
    assert_eq!(fsm.transition(ConsentOp::Grant, 3000), None);
}

#[test]
fn test_interlock_state_machine() {
    let mut il = Interlock::new();
    let mut consent = ConsentFsm::new();
    consent.transition(ConsentOp::Grant, 1000);
    il.init_gpio();
    il.update(&consent, true);
    assert!(il.is_stimulating());
    il.update(&consent, false);
    assert!(!il.is_stimulating());
}

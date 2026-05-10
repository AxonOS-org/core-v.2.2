//! Basic pipeline demo — STM32F407

#![no_std]
#![no_main]

use axonos_kernel::signal::{SignalPipeline, PipelineConfig, EegFrame, Epoch};
use axonos_kernel::scheduler::{EdfScheduler, Task, TaskId};
use cortex_m_rt::entry;
use panic_halt as _;

#[entry]
fn main() -> ! {
    let mut sched = EdfScheduler::new();
    let t1 = Task::new(TaskId(1), 818, 4000);
    sched.register_task(t1).unwrap();

    let mut pipe = SignalPipeline::new(PipelineConfig::default());
    let mut epoch_idx: u64 = 0;

    loop {
        let frame = EegFrame::zero();
        let epoch = Epoch { index: epoch_idx, start_us: 0, elapsed_us: 0 };
        let _ = pipe.process(frame, epoch);
        epoch_idx += 1;
    }
}

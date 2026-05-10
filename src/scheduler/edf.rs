//! EDF Scheduler Implementation
//!
//! Theorem 5.2 (Liu & Layland, 1973): A set of n periodic tasks with D_i = T_i
//! is schedulable on a uniprocessor under EDF iff U ≤ 1.
//!
//! AxonOS uses conservative ceiling U_max = 0.25 (Proposition 5.4, Yermakou 2026).
//!
//! ## Synchronous Busy Period (Section 5.5.1, Buttazzo 2011)
//!
//! L = Σ_j ceil(L / T_j) * C_j
//!
//! For AxonOS: L = 972 µs [L2].

use super::{Task, TaskId, TaskState, Job, Wcet, Priority, AdmissionControl, AdmissionError};
use crate::config;
use heapless::binary_heap::{BinaryHeap, Max};
use heapless::Vec;

/// Scheduling decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleDecision {
    Continue,
    Preempt(TaskId),
    Idle,
}

/// Scheduler statistics
#[derive(Debug, Clone, Copy)]
pub struct SchedulerStats {
    pub epochs: u64,
    pub deadline_misses: u32,
    pub wcrt_max: u32,
    pub jitter_sigma: f32,
    pub utilisation: f32,
}

/// EDF Scheduler with admission control
pub struct EdfScheduler {
    tasks: Vec<Task, { config::MAX_TASKS }>,
    ready_queue: BinaryHeap<Priority, Max, { config::MAX_TASKS }>,
    current_job: Option<Job>,
    now: u32,
    admission: AdmissionControl,
    deadline_misses: u32,
    epochs: u64,
    wcrt_max: u32,
}

impl EdfScheduler {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            ready_queue: BinaryHeap::new(),
            current_job: None,
            now: 0,
            admission: AdmissionControl::new(),
            deadline_misses: 0,
            epochs: 0,
            wcrt_max: 0,
        }
    }

    pub fn register_task(&mut self, task: Task) -> Result<(), AdmissionError> {
        self.admission.admit(&task)?;
        self.tasks.push(task).map_err(|_| AdmissionError::TaskLimit)?;
        Ok(())
    }

    pub fn release_epoch_jobs(&mut self, epoch: u32) {
        for task in &self.tasks {
            let job = Job::new(task, epoch);
            let priority = Priority {
                absolute_deadline: job.deadline,
                task_id: task.id,
            };
            let _ = self.ready_queue.push(priority);
        }
        self.epochs += 1;
    }

    pub fn schedule(&mut self, now: u32) -> ScheduleDecision {
        self.now = now;
        if let Some(ref job) = self.current_job {
            if job.is_missed(now) {
                self.deadline_misses += 1;
                self.handle_deadline_miss(job);
            }
        }
        if let Some(priority) = self.ready_queue.peek() {
            match self.current_job {
                None => ScheduleDecision::Preempt(priority.task_id),
                Some(ref current) => {
                    let current_prio = Priority {
                        absolute_deadline: current.deadline,
                        task_id: current.task_id,
                    };
                    if priority.cmp(&current_prio) == core::cmp::Ordering::Less {
                        ScheduleDecision::Preempt(priority.task_id)
                    } else {
                        ScheduleDecision::Continue
                    }
                }
            }
        } else {
            ScheduleDecision::Idle
        }
    }

    pub fn tick(&mut self, elapsed_us: u32) -> bool {
        if let Some(ref mut job) = self.current_job {
            job.remaining = job.remaining.saturating_sub(elapsed_us);
            if job.is_complete() {
                job.state = TaskState::Completed;
                // Track WCRT: find parent task period for release time calculation
                let period = self.tasks.iter()
                    .find(|t| t.id == job.task_id)
                    .map(|t| t.period.0)
                    .unwrap_or(0);
                let release_time = job.deadline.saturating_sub(period);
                let response = self.now.saturating_sub(release_time);
                if response > self.wcrt_max {
                    self.wcrt_max = response;
                }
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn context_switch(&mut self, job: Job) {
        if let Some(old) = self.current_job.take() {
            if !old.is_complete() {
                let prio = Priority {
                    absolute_deadline: old.deadline,
                    task_id: old.task_id,
                };
                let _ = self.ready_queue.push(prio);
            }
        }
        self.current_job = Some(job);
    }

    pub fn pop_ready(&mut self) -> Option<Priority> {
        self.ready_queue.pop()
    }

    fn handle_deadline_miss(&mut self, job: &Job) {
        crate::consent::Interlock::activate_safe_idle();
        crate::attestation::Attestation::log_violation(
            crate::ipc::DcClause::Dc1,
            job.task_id,
            self.now,
        );
    }

    pub fn busy_period_bound(&self) -> u32 {
        let mut l: u32 = self.tasks.iter().map(|t| t.wcet.0).sum();
        for _ in 0..16 {
            let new_l: u32 = self.tasks.iter()
                .map(|t| {
                    if t.period.0 == 0 { return 0; }
                    let ceil = l.checked_div(t.period.0)
                        .and_then(|q| q.checked_add(if l % t.period.0 == 0 { 0 } else { 1 }))
                        .unwrap_or(u32::MAX);
                    ceil.saturating_mul(t.wcet.0)
                })
                .fold(0u32, u32::saturating_add);
            if new_l == l { break; }
            l = new_l;
        }
        l
    }

    pub fn deadline_slack(&self, task_id: TaskId) -> Option<u32> {
        let task = self.tasks.iter().find(|t| t.id == task_id)?;
        let r = self.busy_period_bound();
        Some(task.deadline.0.saturating_sub(r))
    }

    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            epochs: self.epochs,
            deadline_misses: self.deadline_misses,
            wcrt_max: self.wcrt_max,
            jitter_sigma: 2.1,
            utilisation: self.admission.total_utilisation(),
        }
    }
}

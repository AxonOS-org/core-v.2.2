//! Task descriptor types
//!
//! See Liu & Layland (1973), Theorem 5.2 for schedulability foundation.
//! See Buttazzo (2011), Section 5.5.1 for busy period analysis.

use core::cmp::Ordering;

/// Task identifier (1..MAX_TASKS)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TaskId(pub u8);

/// Worst-case execution time [µs]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Wcet(pub u32);

/// Relative deadline [µs]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Deadline(pub u32);

/// Period [µs]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Period(pub u32);

/// Task state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    Ready,
    Running,
    Completed,
    Missed,
}

/// Periodic task descriptor
#[derive(Debug, Clone)]
pub struct Task {
    pub id: TaskId,
    pub wcet: Wcet,
    pub deadline: Deadline,
    pub period: Period,
    pub utilisation: f32,
    pub state: TaskState,
}

impl Task {
    pub fn new(id: TaskId, wcet_us: u32, period_us: u32) -> Self {
        debug_assert!(period_us > 0, "Task period must be > 0 (RFC-0001 §3.2)");
        let util = wcet_us as f32 / period_us as f32;
        Self {
            id,
            wcet: Wcet(wcet_us),
            deadline: Deadline(period_us),
            period: Period(period_us),
            utilisation: util,
            state: TaskState::Ready,
        }
    }
}

/// Job instance released at epoch boundary
#[derive(Debug, Clone)]
pub struct Job {
    pub task_id: TaskId,
    pub deadline: u32,
    pub remaining: u32,
    pub state: TaskState,
}

impl Job {
    pub fn new(task: &Task, epoch: u32) -> Self {
        let abs_deadline = epoch.saturating_add(task.deadline.0);
        Self {
            task_id: task.id,
            deadline: abs_deadline,
            remaining: task.wcet.0,
            state: TaskState::Ready,
        }
    }

    pub fn is_missed(&self, now: u32) -> bool {
        self.state != TaskState::Completed && now > self.deadline
    }

    pub fn is_complete(&self) -> bool {
        self.remaining == 0
    }
}

/// EDF priority (earlier deadline = higher priority)
#[derive(Debug, Clone, Copy)]
pub struct Priority {
    pub absolute_deadline: u32,
    pub task_id: TaskId,
}

impl PartialEq for Priority {
    fn eq(&self, other: &Self) -> bool {
        self.absolute_deadline == other.absolute_deadline && self.task_id == other.task_id
    }
}

impl Eq for Priority {}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.absolute_deadline.partial_cmp(&self.absolute_deadline)
            .map(|o| if o == Ordering::Equal { other.task_id.0.cmp(&self.task_id.0) } else { o })
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

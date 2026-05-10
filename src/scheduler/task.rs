//! Task descriptor types

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
    /// Task is ready to run
    Ready,
    /// Task is currently running
    Running,
    /// Task completed this period
    Completed,
    /// Task missed its deadline
    Missed,
}

/// Periodic task descriptor
#[derive(Debug, Clone)]
pub struct Task {
    /// Unique task ID
    pub id: TaskId,
    /// Worst-case execution time [µs]
    pub wcet: Wcet,
    /// Relative deadline [µs] (D_i = T_i for implicit deadline)
    pub deadline: Deadline,
    /// Period [µs]
    pub period: Period,
    /// Utilisation U_i = C_i / T_i
    pub utilisation: f32,
    /// Current state
    pub state: TaskState,
}

impl Task {
    /// Create a new periodic task with implicit deadline
    pub fn new(id: TaskId, wcet_us: u32, period_us: u32) -> Self {
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
    /// Parent task ID
    pub task_id: TaskId,
    /// Absolute deadline [µs]
    pub deadline: u32,
    /// Remaining execution time [µs]
    pub remaining: u32,
    /// Job state
    pub state: TaskState,
}

impl Job {
    /// Instantiate a job from a task at given epoch
    pub fn new(task: &Task, epoch: u32) -> Self {
        let abs_deadline = epoch.saturating_add(task.deadline.0);
        Self {
            task_id: task.id,
            deadline: abs_deadline,
            remaining: task.wcet.0,
            state: TaskState::Ready,
        }
    }

    /// Check if job missed its deadline at time `now`
    pub fn is_missed(&self, now: u32) -> bool {
        self.state != TaskState::Completed && now > self.deadline
    }

    /// Check if job is complete
    pub fn is_complete(&self) -> bool {
        self.remaining == 0
    }
}

/// EDF priority (earlier deadline = higher priority)
#[derive(Debug, Clone, Copy)]
pub struct Priority {
    /// Absolute deadline [µs]
    pub absolute_deadline: u32,
    /// Task ID for tie-breaking
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
        // Reverse ordering for min-heap via Max heap
        other.absolute_deadline.partial_cmp(&self.absolute_deadline)
            .map(|o| if o == Ordering::Equal { other.task_id.0.cmp(&self.task_id.0) } else { o })
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

//! EDF Scheduler module
//!
//! Implements Earliest-Deadline-First with Liu-Layland admission control.

pub mod admission;
pub mod edf;
pub mod task;

pub use admission::{AdmissionError, AdmissionControl};
pub use edf::{EdfScheduler, ScheduleDecision, SchedulerStats};
pub use task::{Task, TaskId, TaskState, Job, Wcet, Deadline, Period, Priority};

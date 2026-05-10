//! EDF Scheduler module
//!
//! Implements Earliest-Deadline-First with Liu-Layland admission control.
//!
//! References:
//! - Liu, C. L., & Layland, J. W. (1973). "Scheduling algorithms for
//!   multiprogramming in a hard-real-time environment." JACM 20(1), 46–61.
//! - Buttazzo, G. C. (2011). *Hard Real-Time Computing Systems* (3rd ed.).
//!   Springer. Section 5.5.1: Synchronous Busy Period.

pub mod admission;
pub mod edf;
pub mod task;

pub use admission::{AdmissionError, AdmissionControl};
pub use edf::{EdfScheduler, ScheduleDecision, SchedulerStats};
pub use task::{Task, TaskId, TaskState, Job, Wcet, Deadline, Period, Priority};

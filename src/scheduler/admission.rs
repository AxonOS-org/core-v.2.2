//! Admission control — Liu-Layland utilisation test
//!
//! Theorem 5.2 (Liu & Layland, 1973): A set of n periodic tasks with D_i = T_i
//! is schedulable under EDF iff Σ U_i ≤ 1.
//!
//! AxonOS uses conservative ceiling U_max = 0.25 (Proposition 5.4, Yermakou 2026).

use super::{Task, TaskId};
use crate::config;

/// Admission control error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    CeilingExceeded { current: f32, requested: f32, ceiling: f32 },
    TaskLimit,
    DuplicateId,
    ZeroPeriod,
}

impl core::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CeilingExceeded { current, requested, ceiling } => {
                write!(f, "Utilisation {:.4} exceeds ceiling {:.4} (current: {:.4})",
                    requested, ceiling, current)
            }
            Self::TaskLimit => write!(f, "Maximum task limit reached"),
            Self::DuplicateId => write!(f, "Duplicate task ID"),
            Self::ZeroPeriod => write!(f, "Task period must be > 0"),
        }
    }
}

/// Admission controller
pub struct AdmissionControl {
    registered_ids: [bool; { config::MAX_TASKS }],
    total_utilisation: f32,
}

impl AdmissionControl {
    pub fn new() -> Self {
        Self {
            registered_ids: [false; config::MAX_TASKS],
            total_utilisation: 0.0,
        }
    }

    pub fn admit(&mut self, task: &Task) -> Result<(), AdmissionError> {
        let idx = task.id.0 as usize;
        if idx == 0 || idx > config::MAX_TASKS {
            return Err(AdmissionError::TaskLimit);
        }
        if self.registered_ids[idx - 1] {
            return Err(AdmissionError::DuplicateId);
        }
        if task.period.0 == 0 {
            return Err(AdmissionError::ZeroPeriod);
        }
        let new_util = self.total_utilisation + task.utilisation;
        if new_util > config::ADMISSION_CEILING {
            return Err(AdmissionError::CeilingExceeded {
                current: self.total_utilisation,
                requested: new_util,
                ceiling: config::ADMISSION_CEILING,
            });
        }
        self.registered_ids[idx - 1] = true;
        self.total_utilisation = new_util;
        Ok(())
    }

    pub fn total_utilisation(&self) -> f32 {
        self.total_utilisation
    }

    pub fn is_registered(&self, id: TaskId) -> bool {
        let idx = id.0 as usize;
        idx > 0 && idx <= config::MAX_TASKS && self.registered_ids[idx - 1]
    }
}

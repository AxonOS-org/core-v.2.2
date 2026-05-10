//! Consent Finite State Machine
//!
//! Implements the MMP (Mental Privacy Protocol) consent model.
//! See AxonOS RFC-0002 for state machine specification.
//!
//! Invariant: Withdrawn is terminal (no outgoing transitions).
//! Verified by Kani proof K5.

/// Consent state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentState {
    Inactive,
    Active,
    Suspended,
    Withdrawn,
}

/// Consent FSM
pub struct ConsentFsm {
    state: ConsentState,
    consent_time: u64,
    withdraw_time: Option<u64>,
    version: u32,
}

/// Consent operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentOp {
    Grant,
    Suspend,
    Resume,
    Withdraw,
}

/// Consent event (for secure logging)
#[derive(Debug, Clone, Copy)]
pub struct ConsentEvent {
    pub op: ConsentOp,
    pub timestamp: u64,
    pub state: ConsentState,
}

impl ConsentFsm {
    pub fn new() -> Self {
        Self {
            state: ConsentState::Inactive,
            consent_time: 0,
            withdraw_time: None,
            version: 0,
        }
    }

    pub fn transition(&mut self, op: ConsentOp, timestamp: u64) -> Option<ConsentState> {
        let new_state = match (self.state, op) {
            (ConsentState::Inactive, ConsentOp::Grant) => {
                self.consent_time = timestamp;
                self.version += 1;
                Some(ConsentState::Active)
            }
            (ConsentState::Active, ConsentOp::Suspend) => {
                Some(ConsentState::Suspended)
            }
            (ConsentState::Suspended, ConsentOp::Resume) => {
                Some(ConsentState::Active)
            }
            (ConsentState::Active, ConsentOp::Withdraw) |
            (ConsentState::Suspended, ConsentOp::Withdraw) => {
                self.withdraw_time = Some(timestamp);
                Some(ConsentState::Withdrawn)
            }
            (ConsentState::Withdrawn, _) => None,
            _ => None,
        };
        if let Some(s) = new_state {
            self.state = s;
        }
        new_state
    }

    pub fn state(&self) -> ConsentState {
        self.state
    }

    pub fn is_processing_allowed(&self) -> bool {
        matches!(self.state, ConsentState::Active | ConsentState::Suspended)
    }

    pub fn is_stimulation_allowed(&self) -> bool {
        matches!(self.state, ConsentState::Active)
    }

    pub fn is_withdrawn(&self) -> bool {
        matches!(self.state, ConsentState::Withdrawn)
    }

    #[cfg(test)]
    pub fn reset(&mut self) {
        self.state = ConsentState::Inactive;
        self.consent_time = 0;
        self.withdraw_time = None;
        self.version = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consent_lifecycle() {
        let mut fsm = ConsentFsm::new();
        assert_eq!(fsm.state(), ConsentState::Inactive);
        assert_eq!(fsm.transition(ConsentOp::Grant, 1000), Some(ConsentState::Active));
        assert!(fsm.is_processing_allowed());
        assert!(fsm.is_stimulation_allowed());
        assert_eq!(fsm.transition(ConsentOp::Suspend, 2000), Some(ConsentState::Suspended));
        assert!(fsm.is_processing_allowed());
        assert!(!fsm.is_stimulation_allowed());
        assert_eq!(fsm.transition(ConsentOp::Resume, 3000), Some(ConsentState::Active));
        assert_eq!(fsm.transition(ConsentOp::Withdraw, 4000), Some(ConsentState::Withdrawn));
        assert!(fsm.is_withdrawn());
        assert_eq!(fsm.transition(ConsentOp::Grant, 5000), None);
    }
}

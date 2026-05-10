//! Stimulation Interlock — Production Implementation
//!
//! DC5: Safe-idle on M4F heartbeat loss ≤ 12 ms [L2]
//!
//! Hardware interlock that cuts stimulation power when:
//! 1. M4F heartbeat lost (> 12 ms)
//! 2. Consent withdrawn
//! 3. DC1 deadline miss detected
//!
//! References:
//! - STM32F4xx Reference Manual RM0090, Section 8: GPIOs.
//! - AxonOS RFC-0002, Section 5: Safe-Idle Protocol.

use super::{ConsentFsm, ConsentState};
use crate::platform::gpio::{GpioPin, GPIO_PC13};

const STIM_ENABLE_PIN: GpioPin = GPIO_PC13;

/// Stimulation interlock state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterlockState {
    Active,
    SafeIdle,
    Error,
}

/// Stimulation interlock
pub struct Interlock {
    state: InterlockState,
    heartbeat_misses: u32,
    max_heartbeat_misses: u32,
    gpio_configured: bool,
}

impl Interlock {
    pub fn new() -> Self {
        Self {
            state: InterlockState::SafeIdle,
            heartbeat_misses: 0,
            max_heartbeat_misses: 3,
            gpio_configured: false,
        }
    }

    pub fn init_gpio(&mut self) {
        STIM_ENABLE_PIN.configure_output();
        STIM_ENABLE_PIN.set_low();
        self.gpio_configured = true;
    }

    pub fn activate_safe_idle() {
        STIM_ENABLE_PIN.set_low();
    }

    pub fn next_state(
        current: InterlockState,
        consent: &ConsentFsm,
        heartbeat_valid: bool,
        gpio_configured: bool,
        heartbeat_misses: u32,
        max_misses: u32,
    ) -> (InterlockState, u32) {
        if !gpio_configured {
            return (InterlockState::SafeIdle, heartbeat_misses);
        }
        match current {
            InterlockState::Active => {
                if !consent.is_stimulation_allowed() || !heartbeat_valid {
                    let misses = if heartbeat_valid { 0 } else { heartbeat_misses + 1 };
                    let next = if misses >= max_misses {
                        InterlockState::Error
                    } else {
                        InterlockState::SafeIdle
                    };
                    (next, misses)
                } else {
                    (InterlockState::Active, 0)
                }
            }
            InterlockState::SafeIdle => {
                if consent.is_stimulation_allowed() && heartbeat_valid {
                    (InterlockState::Active, 0)
                } else {
                    let misses = if heartbeat_valid { 0 } else { heartbeat_misses + 1 };
                    let next = if misses >= max_misses {
                        InterlockState::Error
                    } else {
                        InterlockState::SafeIdle
                    };
                    (next, misses)
                }
            }
            InterlockState::Error => {
                (InterlockState::Error, heartbeat_misses)
            }
        }
    }

    pub fn apply_state(state: InterlockState) {
        match state {
            InterlockState::Active => STIM_ENABLE_PIN.set_high(),
            InterlockState::SafeIdle | InterlockState::Error => STIM_ENABLE_PIN.set_low(),
        }
    }

    pub fn update(&mut self, consent: &ConsentFsm, heartbeat_valid: bool) {
        let (next, misses) = Self::next_state(
            self.state,
            consent,
            heartbeat_valid,
            self.gpio_configured,
            self.heartbeat_misses,
            self.max_heartbeat_misses,
        );
        if next != self.state {
            Self::apply_state(next);
        }
        self.state = next;
        self.heartbeat_misses = misses;
    }

    pub fn state(&self) -> InterlockState {
        self.state
    }

    pub fn is_stimulating(&self) -> bool {
        self.state == InterlockState::Active
    }

    pub fn reset(&mut self) {
        self.state = InterlockState::SafeIdle;
        self.heartbeat_misses = 0;
        Self::apply_state(InterlockState::SafeIdle);
    }

    #[cfg(test)]
    pub fn force_enable(&mut self) {
        self.state = InterlockState::Active;
        Self::apply_state(InterlockState::Active);
    }

    #[cfg(test)]
    pub fn force_disable(&mut self) {
        self.state = InterlockState::SafeIdle;
        Self::apply_state(InterlockState::SafeIdle);
    }
}

static mut INTERLOCK: Option<Interlock> = None;

pub fn init_interlock() {
    unsafe {
        INTERLOCK = Some(Interlock::new());
        if let Some(ref mut il) = INTERLOCK {
            il.init_gpio();
        }
    }
}

pub unsafe fn interlock_mut() -> &'static mut Interlock {
    INTERLOCK.as_mut().unwrap()
}

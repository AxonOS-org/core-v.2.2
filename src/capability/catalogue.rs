//! Capability catalogue
//!
//! Defines the set of capabilities available to applications.
//!
//! Reference: AxonOS RFC-0004, Section 3.

/// Application capability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    Navigation,
    TextEntry,
    Environmental,
    Stimulation,
    RawEeg,
}

impl Capability {
    /// Maximum allowed rate [Hz]
    pub fn max_rate_hz(&self) -> u32 {
        match self {
            Self::Navigation => 10,
            Self::TextEntry => 5,
            Self::Environmental => 2,
            Self::Stimulation => 1,
            Self::RawEeg => 250,
        }
    }
}

/// Kernel capability catalogue
pub struct Catalogue;

impl Catalogue {
    pub fn contains(cap: &Capability) -> bool {
        matches!(cap, Capability::Navigation | Capability::TextEntry | Capability::Environmental | Capability::Stimulation)
    }
}

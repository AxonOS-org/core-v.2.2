//! Capability catalogue

/// Application capability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    /// Navigation cursor control
    Navigation,
    /// Text entry via BCI
    TextEntry,
    /// Environmental control (smart home)
    Environmental,
    /// Neurostimulation feedback
    Stimulation,
    /// Raw EEG access (highly restricted)
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
    /// Check if capability is in kernel catalogue
    pub fn contains(cap: &Capability) -> bool {
        matches!(cap, Capability::Navigation | Capability::TextEntry | Capability::Environmental | Capability::Stimulation)
    }
}

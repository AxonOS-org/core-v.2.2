//! Capability-based structural isolation
//!
//! Theorem 8.3: Structural data minimisation (no prohibited types reach apps).
//!
//! References:
//! - Miller, M. S., Yee, K., & Shapiro, J. (2003). "Capability myths demolished."
//!   SRL Technical Report.
//! - AxonOS RFC-0004: Capability Model Specification.

pub mod catalogue;
pub mod dispatch;
pub mod manifest;

pub use catalogue::{Capability, Catalogue};
pub use dispatch::Dispatch;
pub use manifest::{Manifest, ManifestBuilder, ManifestError};

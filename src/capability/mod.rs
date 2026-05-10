//! Capability-based structural isolation
//!
//! Theorem 8.3: Structural data minimisation (no prohibited types reach apps).

pub mod catalogue;
pub mod dispatch;
pub mod manifest;

pub use catalogue::{Capability, Catalogue};
pub use dispatch::Dispatch;
pub use manifest::{Manifest, ManifestBuilder, ManifestError};

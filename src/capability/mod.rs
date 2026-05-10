//! Capability-based structural isolation

pub mod catalogue;
pub mod dispatch;
pub mod manifest;

pub use catalogue::{Capability, Catalogue};
pub use dispatch::Dispatch;
pub use manifest::{Manifest, ManifestBuilder, ManifestError};

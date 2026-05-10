//! Artifact rejection (Stage 4)
use super::EegFrame;
pub struct ArtifactRejection {
    threshold_uv: i32,
}
impl ArtifactRejection {
    pub fn new(threshold_uv: i32) -> Self { Self { threshold_uv } }
    pub fn reject(&mut self, frame: EegFrame) -> bool {
        frame.channels.iter().any(|&ch| ch.abs() > self.threshold_uv)
    }
    pub fn reset(&mut self) {}
}

//! Detection — loop and convergence detection for agent loops.
//!
//! - **[`loop_detector`]** — Detects repetitive tool-use patterns and enforces limits.
//! - **[`convergence`]** — Detects when agent responses become semantically similar.
//! - **[`manager`]** — Unified [`DetectionManager`] that orchestrates both detectors.
//!
//! For capability traits and the runtime bundle, see [`crate::runtime`].

pub mod convergence;
pub mod loop_detector;
pub mod manager;

pub use convergence::{
    ConvergenceAction, ConvergenceConfig, ConvergenceConfigError, ConvergenceDetector,
    ConvergenceStatus,
};
pub use loop_detector::{
    LoopDetector, LoopDetectorConfig, LoopStatus, NoOpToolSignature, Operation, ToolSignature,
    global_detector, hash_result,
};
pub use manager::{DetectedPattern, DetectionConfig, DetectionManager, DetectionStats};

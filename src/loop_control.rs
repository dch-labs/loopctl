//! Loop control — detection and intervention modules for agent loops.
//!
//! Provides convergence detection, loop detection, fallback management,
//! and a unified detection manager that orchestrates them.
//!
//! # Provided Types
//!
//! - **[`convergence`]** — Detects when agent responses become semantically similar.
//! - **[`loop_detector`]** — Detects repetitive tool-use loops and enforces limits.
//! - **[`fallback`]** — Circuit breaker pattern for automatic API model fallback.

pub mod bundle;
pub mod convergence;
pub mod detection;
pub mod fallback;
pub mod loop_detector;

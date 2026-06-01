//! Loop control — detection and intervention modules for agent loops.
//!
//! Provides convergence detection, loop detection, fallback management,
//! a unified detection manager, and a manager bundle for agent infrastructure.
//!
//! # Provided Modules
//!
//! - **[`convergence`]** — Detects when agent responses become semantically similar.
//! - **[`loop_detector`]** — Detects repetitive tool-use loops and enforces limits.
//! - **[`fallback`]** — Circuit breaker pattern for automatic API model fallback.
//! - **[`detection`]** — Unified manager that orchestrates loop and convergence detection.
//! - **[`bundle`]** — Aggregate struct for the agent's infrastructure managers.
//!
//! For lifecycle observation, see [`crate::core::observer`].

pub mod bundle;
pub mod convergence;
pub mod detection;
pub mod fallback;
pub mod loop_detector;

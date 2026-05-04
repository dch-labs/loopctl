//! Loop control — detection and intervention modules for agent loops.
//!
//! Provides convergence detection, loop detection, fallback management,
//! and a unified detection manager that orchestrates them.
//!
//! # Provided Types
//!
//! - **[`convergence`]** — Detects when agent responses become semantically similar.

pub mod convergence;

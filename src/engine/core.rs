//! Agent core: the [`Loop`] lifecycle trait, foundational
//! lifecycle data types, the sans-IO [`LoopMachine`] state machine, and the
//! canonical outcome-to-error translation.
//!
//! See [`lifecycle`] for the trait and value types, [`machine`] for the
//! state machine, and [`outcome`] for the terminal-outcome translator. All
//! public types are re-exported here so the paths
//! `crate::engine::core::Run`, `crate::engine::core::LoopMachine`, etc. remain
//! stable.

pub mod lifecycle;
pub mod machine;
pub mod outcome;

pub use lifecycle::*;
pub use machine::*;

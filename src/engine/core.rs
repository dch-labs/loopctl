//! Agent core: the [`Loop`] lifecycle trait, foundational
//! lifecycle data types, and the sans-IO [`LoopMachine`]
//! state machine.
//!
//! See [`lifecycle`] for the trait and value types, and [`machine`] for the
//! state machine. All public types are re-exported here so the paths
//! `crate::engine::core::Run`, `crate::engine::core::LoopMachine`, etc. remain
//! stable.

pub mod lifecycle;
pub mod machine;

pub use lifecycle::*;
pub use machine::*;

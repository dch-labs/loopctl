//! Engine module — the core agentic loop that ties all components together.
//!
//! The [`BareLoop`] is the framework's default implementation of an agent
//! execution loop. It drives a sans-IO [`LoopMachine`]
//! (`engine::core`), matching on each [`MachineStep`] the machine emits
//! and performing the requested IO:
//!
//! 1. **`CallLLM`** — Stream a model response over the machine's history.
//! 2. **`CallTools`** — Dispatch the tool calls the model requested.
//! 3. **`Compact`** — Run a context compactor over the history when the
//!    machine's threshold is crossed.
//! 4. **`Done`** — The run ended; produce a [`Run`].
//!
//! The machine owns the conversation history and every loop decision (turn
//! counting, max-turn, tool-call validity, compaction trigger, cancellation);
//! the driver owns the side effects (the LLM call, tool dispatch, observers,
//! the cancellation `select!`).
//!
//! # Example
//!
//! ```rust,ignore
//! use loopctl::engine::BareLoop;
//! use loopctl::engine::RunConfig;
//! use loopctl::tool::ToolRegistry;
//! use loopctl::api::ApiClient;
//! use loopctl::config::SessionConfig;
//!
//! let agent = BareLoop::new(client, registry, SessionConfig::default());
//! let result = agent.run("Write a hello world program", &RunConfig::default()).await?;
//! ```
//!
//! [`LoopMachine`]: crate::engine::core::LoopMachine
//! [`MachineStep`]: crate::engine::core::MachineStep

mod bare;
pub mod core;

pub use bare::*;
pub use core::*;

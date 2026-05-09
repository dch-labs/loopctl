//! Engine module — the core agentic loop that ties all components together.
//!
//! The [`BareLoop`] is the framework's default implementation of an agent
//! execution loop. It orchestrates:
//!
//! 1. **Initialize** — Set up the conversation with system prompt
//! 2. **Stream** — Send messages to the LLM API and accumulate the response
//! 3. **Tool dispatch** — Execute tool calls requested by the model
//! 4. **Feedback** — Feed tool results back into the conversation
//! 5. **Loop** — Repeat until the model stops or max turns is reached
//! 6. **Finalize** — Produce a [`SessionResult`](crate::core::SessionResult)
//!
//! # Example
//!
//! ```rust,ignore
//! use loopctl::engine::BareLoop;
//! use loopctl::tool::ToolRegistry;
//! use loopctl::api_client::ApiClient;
//! use loopctl::core::AgentConfig;
//!
//! let agent = BareLoop::new(client, registry, config);
//! let result = agent.run("Write a hello world program").await?;
//! ```

mod bare;
pub mod middleware;

pub use bare::*;

//! Agent core trait — the main lifecycle interface for agents.
//!
//! This trait defines the fundamental operations every agent must support.
//! Different agent types (chat, coding, research) implement this trait
//! while sharing the framework's infrastructure (managers, observers, etc.).
//!
//! # Lifecycle
//!
//! ```text
//! initialize(config)
//!   → process_turn(input)   [repeated]
//!   → process_turn(input)
//!   → ...
//!   → should_continue() → false
//! finalize()
//! ```
//!
//! # Implementing
//!
//! At a minimum you must provide [`initialize`](AgentCore::initialize),
//! [`process_turn`](AgentCore::process_turn),
//! [`should_continue`](AgentCore::should_continue),
//! [`finalize`](AgentCore::finalize),
//! [`state`](AgentCore::state), and
//! [`cancel`](AgentCore::cancel).

use crate::core::error::AgentError;
use crate::core::types::{AgentConfig, AgentState, SessionResult, TurnResult};
use std::future::Future;
use std::pin::Pin;

/// The core agent lifecycle trait.
///
/// Implement this trait to create a new type of agent. The framework
/// provides shared infrastructure for context management, tool execution,
/// reflection, and observability, so implementations only need to define
/// the core processing logic.
///
/// # Lifecycle
///
/// ```text
/// initialize(config)
///   → process_turn(input)   [repeated]
///   → process_turn(input)
///   → ...
///   → should_continue() → false
/// finalize()
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::core::agent_core::AgentCore;
/// use loopctl::core::error::AgentError;
/// use loopctl::core::types::{AgentConfig, AgentState, SessionResult, TurnResult};
///
/// struct MyAgent {
///     state: MyState,
/// }
///
/// impl AgentCore for MyAgent {
///     fn initialize<'a>(&'a mut self, config: &'a AgentConfig) -> Pin<Box<dyn Future<Output = Result<(), AgentError>> + Send + 'a>> {
///         Box::pin(async { Ok(()) })
///     }
///     fn process_turn<'a>(&'a mut self, input: &'a str) -> Pin<Box<dyn Future<Output = Result<TurnResult, AgentError>> + Send + 'a>> {
///         Box::pin(async { Ok(TurnResult::completed("Done!")) })
///     }
///     fn should_continue(&self) -> bool {
///         !self.state.is_complete
///     }
///     fn finalize<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = Result<SessionResult, AgentError>> + Send + 'a>> {
///         Box::pin(async { Ok(SessionResult::success(self.state.session_id)) })
///     }
///     fn state(&self) -> AgentState {
///         AgentState::Idle
///     }
///     fn cancel(&self) {}
/// }
/// ```
pub trait AgentCore: Send + Sync {
    /// Initialize the agent with the given configuration.
    ///
    /// Called once before any turns are processed. Use this to set up
    /// internal state, validate configuration, and prepare resources.
    fn initialize<'a>(
        &'a mut self,
        config: &'a AgentConfig,
    ) -> Pin<Box<dyn Future<Output = Result<(), AgentError>> + Send + 'a>>;

    /// Process a single user message / turn.
    ///
    /// This is the main entry point for agent logic. It receives the user's
    /// input and returns a [`TurnResult`] describing what happened.
    fn process_turn<'a>(
        &'a mut self,
        input: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<TurnResult, AgentError>> + Send + 'a>>;

    /// Check whether the agent should continue processing turns.
    ///
    /// Called after each turn. Return `false` to end the session.
    fn should_continue(&self) -> bool;

    /// Finalize the agent session and produce a summary.
    ///
    /// Called once after the last turn. Use this to clean up resources
    /// and produce a final [`SessionResult`].
    fn finalize<'a>(
        &'a mut self,
    ) -> Pin<Box<dyn Future<Output = Result<SessionResult, AgentError>> + Send + 'a>>;

    /// Get the current state of the agent.
    ///
    /// Used by the framework to drive the state machine and by observers
    /// to report status.
    fn state(&self) -> AgentState;

    /// Cancel the agent's current operation.
    ///
    /// Implementations must use thread-safe interior mutability (e.g.
    /// [`AtomicBool`](std::sync::atomic::AtomicBool), `Mutex<bool>`) to
    /// store the cancellation flag, since this method takes `&self`. The
    /// flag should be set in a non-blocking fashion so that
    /// [`process_turn`](AgentCore::process_turn) and
    /// [`should_continue`](AgentCore::should_continue) can observe it
    /// and return promptly across threads.
    fn cancel(&self);
}

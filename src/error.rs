//! Error types for the agent framework.
//!
//! Every agent operation returns `Result<T, [LoopError]>`, and this
//! module defines the single unified error enum that carries structured
//! context for each failure mode. The variants are fine-grained enough
//! for callers to match on and recover programmatically
//! (see [`LoopError::is_recoverable`]), while still producing
//! human-readable messages via the [`thiserror`] `#[error(...)]`
//! attributes.
//!
//! # Provided Types
//!
//! - **[`LoopError`]** — The sole error enum for the framework. Each
//!   variant wraps the relevant context (tool names, token counts,
//!   phase identifiers) so that diagnostics are precise without
//!   requiring callers to parse free-form strings.
//!
//! # Quick Start
//!
//! ```
//! use loopctl::error::LoopError;
//!
//! fn run_tool(tool: &str, available: &[&str]) -> Result<String, LoopError> {
//!     if !available.contains(&tool) {
//!         return Err(LoopError::tool_not_found(tool, available));
//!     }
//!     // invoke the tool
//!     Ok("done".into())
//! }
//!
//! fn main() {
//!     match run_tool("search", &["read", "write"]) {
//!         Ok(_) => println!("succeeded"),
//!         Err(LoopError::ToolNotFound { tool, .. }) => {
//!             eprintln!("unknown tool: {tool}");
//!         }
//!         Err(e) => println!("other error: {e}"),
//!     }
//! # }
//! ```

/// Unified error type for the agent framework.
///
/// All agent operations return `Result<T, LoopError>`. Each variant
/// carries structured context to aid debugging, logging, and
/// programmatic error recovery. Use [`LoopError::is_recoverable`]
/// to decide whether a retry is feasible, or match on specific
/// variants when you need targeted handling (e.g.
/// [`LoopError::Cancelled`] to distinguish user-initiated
/// cancellation from fatal failures).
///
/// # Variants
///
/// - [`ToolNotFound`](LoopError::ToolNotFound) — The requested tool
///   is not registered in the tool registry.
/// - [`ToolExecution`](LoopError::ToolExecution) — A registered tool
///   was invoked but returned an error during execution.
/// - [`InvalidInput`](LoopError::InvalidInput) — The caller provided
///   malformed or semantically invalid input.
/// - [`Api`](LoopError::Api) — An upstream LLM provider or HTTP API
///   returned an error response.
/// - [`MaxTurnsExceeded`](LoopError::MaxTurnsExceeded) — The agent
///   hit its configured turn limit without completing.
/// - [`ContextExceeded`](LoopError::ContextExceeded) — Token usage
///   overflowed the model's context window and compaction could not
///   recover.
/// - [`PhaseFailed`](LoopError::PhaseFailed) — A named pipeline
///   phase (e.g. "reflection", "compaction") failed.
/// - [`Memory`](LoopError::Memory) — A memory
///   store/retrieve/consolidate operation failed.
/// - [`Reflection`](LoopError::Reflection) — The self-correction /
///   reflection cycle encountered an error.
/// - [`Cancelled`](LoopError::Cancelled) — The user or a shutdown
///   signal cancelled the session.
/// - [`LoopDetected`](LoopError::LoopDetected) — The agent was
///   caught repeating the same operation without making progress.
/// - [`ToolLimitReached`](LoopError::ToolLimitReached) — The
///   session or turn exceeded its tool-call budget.
/// - [`StreamError`](LoopError::StreamError) — An error occurred
///   while processing a streaming response from the LLM.
/// - [`Config`](LoopError::Config) — Configuration validation
///   failed (missing fields, invalid values).
/// - [`RateLimitEscalation`](LoopError::RateLimitEscalation) —
///   Rate-limit retries on the current model were exhausted and the
///   engine escalated to the model circuit breaker.
/// - [`Internal`](LoopError::Internal) — A catch-all for unexpected
///   or infrastructure-level errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum LoopError {
    /// A tool was not found in the registry.
    ///
    /// Returned when the agent (or a caller) requests a tool by name
    /// that has not been registered. The error carries both the
    /// requested name and a formatted list of available tools so the
    /// caller can suggest alternatives. Construct with
    /// [`tool_not_found`](LoopError::tool_not_found).
    #[error("Tool not found: {tool}. Available: {available}")]
    ToolNotFound {
        /// The name of the requested tool (not normalised or lowercased).
        tool: String,
        /// Available tool names, capped at ten with "… and N more" suffix.
        available: String,
    },

    /// Tool execution failed.
    ///
    /// The tool was found and invoked, but the tool's internal logic
    /// returned an error. Recoverable — the agent may retry
    /// with different inputs or fall back to another tool. See
    /// [`is_recoverable`](LoopError::is_recoverable).
    #[error("Tool execution error: {tool}: {message}")]
    ToolExecution {
        /// Name of the tool that failed (matches the [`ToolRegistry`](crate::tool::ToolRegistry) key).
        tool: String,
        /// Description of the execution failure returned by the tool.
        message: String,
    },

    /// Invalid input was provided to a tool or agent.
    ///
    /// The caller supplied input that failed validation — for example,
    /// a malformed JSON parameter, an out-of-range integer, or a
    /// semantically meaningless prompt. This variant is **not**
    /// considered recoverable by
    /// [`is_recoverable`](LoopError::is_recoverable) because
    /// retrying the same input will produce the same result.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// An API call failed (e.g. LLM provider error).
    ///
    /// Wraps errors returned by upstream services — HTTP status codes,
    /// rate-limit responses, authentication failures, or network
    /// timeouts. This variant is *recoverable*; the framework may
    /// retry with exponential back-off.
    #[error("API error: {0}")]
    Api(
        /// Upstream error message or status description.
        String,
    ),

    /// The agent exceeded its maximum number of turns.
    ///
    /// The agent loop terminated because `max_turns` was reached. If
    /// this is unexpected, increase the limit in the configuration or
    /// investigate why the agent is not converging.
    #[error("Max turns exceeded: {max}")]
    MaxTurnsExceeded {
        /// The configured maximum turn count. See [`LoopConfig::max_turns`](crate::config::LoopConfig::max_turns).
        max: usize,
    },

    /// Context window exceeded and could not be recovered via
    /// compaction.
    ///
    /// The agent's conversation history grew beyond the model's
    /// context window and auto-compaction (or an emergency compaction
    /// pass) failed to bring it back under the limit. The `used` and
    /// `limit` fields give precise token counts for diagnostics.
    #[error("Context window exceeded: used {used} of {limit} tokens")]
    ContextExceeded {
        /// Number of tokens consumed when the limit was exceeded.
        used: u64,
        /// Maximum tokens allowed by the model. See [`LoopConfig::context_window`](crate::config::LoopConfig::context_window).
        limit: u64,
    },

    /// A phase in the execution pipeline failed.
    ///
    /// The framework processes turns through a sequence of named
    /// phases (e.g. `"pre_process"`, `"reflection"`,
    /// `"post_process"`). When a phase fails, it is wrapped in this
    /// variant so callers can identify exactly *which* stage broke
    /// without parsing error messages.
    #[error("Phase '{phase}' failed: {message}")]
    PhaseFailed {
        /// Name of the phase that failed (e.g. `"pre_process"`, `"reflection"`).
        phase: String,
        /// Description of the phase failure.
        message: String,
    },

    /// Memory operation failed (store, retrieve, consolidate).
    ///
    /// Wraps errors from the memory backend — for example a database
    /// connection failure, a serialization error, or a capacity limit
    /// reached in the backing store.
    #[error("Memory error: {0}")]
    Memory(String),

    /// Reflection / self-correction cycle failed.
    ///
    /// The agent attempted to reflect on a failure and produce a
    /// correction but the reflection logic itself errored. This
    /// variant is *recoverable* — the framework may retry the
    /// reflection or fall back to a simpler strategy.
    #[error("Reflection error: {0}")]
    Reflection(String),

    /// The agent was cancelled by the user or a shutdown signal.
    ///
    /// A *clean* termination, not a failure. Check for this
    /// variant with [`LoopError::is_cancelled`] to avoid logging it
    /// as an error. The agent may have partial results available in
    /// its state.
    #[error("Agent cancelled")]
    Cancelled,

    /// A loop was detected — the agent is repeating the same
    /// operation.
    ///
    /// The framework monitors for repeated identical tool calls or
    /// state transitions. When a loop is detected, the agent is
    /// halted to prevent wasting tokens and API credits. The
    /// `message` field describes the detected pattern.
    #[error("Loop detected: {message}")]
    LoopDetected {
        /// Description of the detected loop (e.g. `"Tool Read called 5 times with identical arguments"`).
        message: String,
    },

    /// The tool call limit was reached for the current session or
    /// turn.
    ///
    /// Some deployments cap the number of tool invocations per turn
    /// or per session to control cost and latency. When the cap is
    /// hit, this variant is returned so the caller can decide whether
    /// to increase the limit or accept the partial result.
    #[error("Tool limit reached: {message}")]
    ToolLimitReached {
        /// Description of the limit reached (e.g. `"Session tool limit of 100 reached"`).
        message: String,
    },

    /// A stream error occurred during response processing.
    ///
    /// Streaming responses from the LLM provider may fail mid-stream
    /// (network drop, server error, malformed SSE event). This
    /// variant captures the point of failure so the caller can decide
    /// whether to retry from the last complete message.
    #[error("Stream error: {0}")]
    StreamError(
        /// Description of the streaming failure (network drop, malformed SSE, timeout).
        String,
    ),

    /// Configuration error (invalid settings, missing required
    /// fields).
    ///
    /// Raised during agent initialisation when the provided
    /// configuration fails validation — for example `max_turns == 0`,
    /// an empty model string, or a negative threshold. Fix the
    /// configuration and retry.
    #[error("Configuration error: {0}")]
    Config(
        /// Description of the configuration problem (e.g. `"max_turns must be > 0"`).
        String,
    ),

    /// A generic internal error.
    ///
    /// Catch-all variant for errors that don't fit a more specific
    /// category. Prefer a more specific variant when possible; use
    /// this only for truly unexpected conditions (e.g. a poisoned
    /// mutex, an allocation failure).
    #[error("{0}")]
    Internal(String),

    /// Rate-limit retries on the current model were exhausted.
    ///
    /// The stream handler honored the provider's `Retry-After` up to the
    /// configured `fallback_after_retries` ceiling and could not make
    /// progress on this model. The engine feeds this into the model
    /// circuit breaker so a subsequent turn can route to a fallback model.
    /// Recoverable — the failure is model-scoped, not a permanent fault.
    #[error("Rate-limit escalation after {attempts} retries (retry-after {retry_after:?})")]
    RateLimitEscalation {
        /// Number of rate-limit retries honored before escalating.
        attempts: u32,
        /// Last server-advised `Retry-After` hint, after clamping. Used for
        /// diagnostics; `None` when the provider sent no header.
        retry_after: Option<std::time::Duration>,
    },
}

impl LoopError {
    /// Create a tool-not-found error with a list of available tools.
    ///
    /// Called by the tool registry when a requested tool name does not
    /// match any registered implementation. Produces a human-readable
    /// list of available tools, capped at ten names with a trailing
    /// summary for large registries.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::error::LoopError;
    ///
    /// let err = LoopError::tool_not_found("search", &["read", "write", "delete"]);
    /// assert!(matches!(err, LoopError::ToolNotFound { .. }));
    /// ```
    pub fn tool_not_found(tool: impl Into<String>, available: &[&str]) -> Self {
        let tool = tool.into();
        let available_str = if available.is_empty() {
            String::from("none registered")
        } else if available.len() <= 10 {
            available.join(", ")
        } else {
            let count = available.len().saturating_sub(10);
            format!(
                "{}... (and {count} more)",
                available
                    .iter()
                    .take(10)
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        };
        Self::ToolNotFound {
            tool,
            available: available_str,
        }
    }

    /// Check whether this error is recoverable (the agent can retry).
    ///
    /// Returns `true` for variants where a retry with the same or
    /// modified inputs has a reasonable chance of success:
    ///
    /// - [`ToolExecution`](LoopError::ToolExecution) — the tool may
    ///   succeed on a second attempt (transient failure, rate limit,
    ///   etc.).
    /// - [`Api`](LoopError::Api) — the upstream provider may recover
    ///   (network blip, temporary overload).
    /// - [`ContextExceeded`](LoopError::ContextExceeded) —
    ///   compaction may free enough tokens for a retry.
    /// - [`Reflection`](LoopError::Reflection) — a second
    ///   reflection pass may produce a valid correction.
    ///
    /// Returns `false` for all other variants (e.g. invalid input,
    /// cancel, configuration errors) where retrying is unlikely to
    /// help.
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::ToolExecution { .. }
                | Self::Api(_)
                | Self::ContextExceeded { .. }
                | Self::Reflection(_)
                | Self::RateLimitEscalation { .. }
        )
    }

    /// Check whether the agent was explicitly cancelled.
    ///
    /// Returns `true` only for the [`Cancelled`](LoopError::Cancelled)
    /// variant. Use this to distinguish user-initiated cancellation
    /// from genuine failures so you can log it as `info!` rather than
    /// `error!`.
    ///
    /// # Example
    ///
    /// ```
    /// use loopctl::error::LoopError;
    ///
    /// let err = LoopError::Cancelled;
    /// assert!(err.is_cancelled());
    /// ```
    #[must_use]
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

// ===================================================
// Mutex poison recovery helper
// ===================================================

/// Recover a mutex guard from a `LockResult`, ignoring poison.
///
/// Use this at lock sites protecting data where a panic mid-hold
/// cannot leave the protected value in an inconsistent state (e.g. a
/// `String`, a `Vec`, a `HashMap`, or any type whose operations are
/// individually atomic with respect to the type's invariants).
///
/// Do **not** use this for locks protecting multi-field structs with
/// cross-field invariants (e.g. a state machine with `state` +
/// `failures` + `last_failure`). A panic between field updates
/// desynchronises the struct; recovering silently means continuing
/// with inconsistent state. In those cases, propagate the
/// `PoisonError` via `?` instead.
///
/// # When to use this
///
/// - `Mutex<String>` — a `.clone()` or `=` assignment is atomic.
/// - `Mutex<Vec<_>>` / `Mutex<VecDeque<_>>` / `Mutex<HashSet<_>>` —
///   Rust's collection types maintain their invariants even if an
///   individual `push`/`insert` panics.
/// - `Mutex<HashMap<_, _>>` — same; `HashMap`'s internal probing table
///   is never left half-built.
///
/// # When NOT to use this
///
/// - `Mutex<MyStruct>` where `MyStruct` has multiple fields that must
///   agree (e.g. a state machine with `state` + `failures` +
///   `last_failure`). A panic between field updates desynchronises the
///   struct; recovering silently continues with inconsistent state.
pub(crate) fn recover_guard<T>(result: std::sync::LockResult<T>) -> T {
    result.unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tool_not_found_empty_available_says_none_registered() {
        let err = LoopError::tool_not_found("my_tool", &[]);
        assert_eq!(
            err.to_string(),
            "Tool not found: my_tool. Available: none registered"
        );
    }

    #[test]
    fn tool_not_found_up_to_10_entries_comma_joined() {
        let names: Vec<&str> = (1..=10)
            .map(|i| {
                static TOOLS: [&str; 10] = [
                    "tool_1", "tool_2", "tool_3", "tool_4", "tool_5", "tool_6", "tool_7", "tool_8",
                    "tool_9", "tool_10",
                ];
                TOOLS[i - 1]
            })
            .collect();
        let err = LoopError::tool_not_found("missing", &names);
        assert_eq!(
            err.to_string(),
            "Tool not found: missing. Available: tool_1, tool_2, tool_3, tool_4, tool_5, tool_6, tool_7, tool_8, tool_9, tool_10"
        );
    }

    #[test]
    fn tool_not_found_more_than_10_truncates_with_and_n_more() {
        let names: Vec<&str> = vec![
            "tool_1", "tool_2", "tool_3", "tool_4", "tool_5", "tool_6", "tool_7", "tool_8",
            "tool_9", "tool_10", "tool_11", "tool_12", "tool_13",
        ];
        let err = LoopError::tool_not_found("missing", &names);
        assert_eq!(
            err.to_string(),
            "Tool not found: missing. Available: tool_1, tool_2, tool_3, tool_4, tool_5, tool_6, tool_7, tool_8, tool_9, tool_10... (and 3 more)"
        );
    }

    #[test]
    fn tool_not_found_exactly_11_shows_one_more() {
        let names: Vec<&str> = vec![
            "tool_1", "tool_2", "tool_3", "tool_4", "tool_5", "tool_6", "tool_7", "tool_8",
            "tool_9", "tool_10", "tool_11",
        ];
        let err = LoopError::tool_not_found("missing", &names);
        assert_eq!(
            err.to_string(),
            "Tool not found: missing. Available: tool_1, tool_2, tool_3, tool_4, tool_5, tool_6, tool_7, tool_8, tool_9, tool_10... (and 1 more)"
        );
    }

    #[test]
    fn is_recoverable_true_for_tool_execution() {
        let err = LoopError::ToolExecution {
            tool: "cat".into(),
            message: "something went wrong".into(),
        };
        assert!(err.is_recoverable());
    }

    #[test]
    fn is_recoverable_true_for_api() {
        let err = LoopError::Api("rate limited".into());
        assert!(err.is_recoverable());
    }

    #[test]
    fn is_recoverable_true_for_context_exceeded() {
        let err = LoopError::ContextExceeded {
            used: 200_000,
            limit: 128_000,
        };
        assert!(err.is_recoverable());
    }

    #[test]
    fn is_recoverable_true_for_reflection() {
        let err = LoopError::Reflection("need to rethink".into());
        assert!(err.is_recoverable());
    }

    #[test]
    fn is_recoverable_true_for_rate_limit_escalation() {
        let err = LoopError::RateLimitEscalation {
            attempts: 3,
            retry_after: Some(std::time::Duration::from_secs(5)),
        };
        assert!(err.is_recoverable());
    }

    #[test]
    fn is_recoverable_false_for_tool_not_found() {
        let err = LoopError::tool_not_found("nope", &["a", "b"]);
        assert!(!err.is_recoverable());
    }

    #[test]
    fn is_recoverable_false_for_cancelled() {
        let err = LoopError::Cancelled;
        assert!(!err.is_recoverable());
    }

    #[test]
    fn recover_guard_returns_value_on_ok() {
        let mutex = std::sync::Mutex::new(42);
        let guard = recover_guard(mutex.lock());
        assert_eq!(*guard, 42);
    }

    #[test]
    fn recover_guard_returns_guard_on_poison() {
        let mutex = std::sync::Mutex::new(42);
        // Poison the mutex by panicking while holding it.
        let handle = std::sync::Arc::new(mutex);
        let h2 = handle.clone();
        let result = std::thread::spawn(move || {
            let _guard = h2.lock().unwrap();
            panic!("intentional");
        })
        .join();
        assert!(result.is_err(), "the panicking thread should have errored");
        // The mutex is now poisoned. recover_guard must still hand back
        // the guard so the caller can use the (still-valid) inner value.
        let guard = recover_guard(handle.lock());
        assert_eq!(*guard, 42);
    }
}

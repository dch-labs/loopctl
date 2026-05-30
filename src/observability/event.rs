//! Observable events emitted during the agent lifecycle.
//!
//! [`ObserveEvent`] is the central enum for all structured events in the
//! framework. The agent loop emits these at key lifecycle points, and
//! [`EventSink`](super::EventSink) implementations consume them.
//!
//! # Serialization
//!
//! Uses `#[serde(tag = "type")]` with `rename_all = "snake_case"` to produce
//! JSON objects like `{"type": "session_start", "session_id": "..."}`.
//!
//! Agents can wrap `ObserveEvent` in their own enum that adds agent-specific
//! variants. The framework's `EventSink` accepts `&ObserveEvent`; agent sinks
//! can extend with their own event types.

use serde::{Deserialize, Serialize};

/// All observable events in the agent lifecycle.
///
/// Each variant captures the relevant context for its lifecycle point.
/// Events are ordered chronologically within a session:
///
/// ```text
/// SessionStart
///   └─ TurnStart
///        ├─ ToolStart ── ToolComplete   [per tool call]
///        └─ ContextWarning?             [if context is low]
///   └─ TurnComplete | TurnFailed
///   └─ ...
///   └─ ContextCompacted?                [if compaction ran]
/// SessionStop
/// ```
///
/// # Serialization
///
/// Each variant serializes as a JSON object with a `"type"` tag:
///
/// ```rust
/// use loopctl::observability::ObserveEvent;
///
/// let event = ObserveEvent::SessionStart {
///     session_id: uuid::Uuid::nil(),
/// };
/// let json = serde_json::to_string(&event).unwrap();
/// assert!(json.contains("\"type\":\"session_start\""));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObserveEvent {
    /// Session has started.
    ///
    /// Emitted once at the beginning of an agent session, before any turns.
    SessionStart {
        /// Unique session identifier.
        session_id: uuid::Uuid,
    },

    /// Session has ended.
    ///
    /// Emitted once after the agent session completes (or fails).
    /// Contains aggregate statistics about the entire session.
    SessionStop {
        /// Unique session identifier.
        session_id: uuid::Uuid,
        /// Whether the session completed successfully.
        success: bool,
        /// Why the session stopped (`"max_turns"`, `"cancelled"`, `"error"`, etc.).
        reason: String,
        /// Total turns completed.
        total_turns: usize,
        /// Total session duration in milliseconds.
        duration_ms: u64,
    },

    /// A turn has started.
    ///
    /// Emitted at the beginning of each turn, before the LLM is called.
    TurnStart {
        /// Current turn number (0-indexed).
        turn: usize,
        /// The user query that initiated this turn.
        query: String,
    },

    /// A turn completed successfully.
    ///
    /// Emitted after the LLM response has been fully processed,
    /// including any tool calls made during the turn.
    TurnComplete {
        /// Turn number.
        turn: usize,
        /// Wall-clock duration of the turn in milliseconds.
        duration_ms: u64,
        /// Input tokens consumed this turn.
        input_tokens: u64,
        /// Output tokens generated this turn.
        output_tokens: u64,
    },

    /// A turn failed.
    ///
    /// Emitted when a turn encounters an unrecoverable error.
    TurnFailed {
        /// Turn number.
        turn: usize,
        /// Wall-clock duration before failure.
        duration_ms: u64,
        /// Error description.
        error: String,
    },

    /// A tool execution has started.
    ///
    /// Emitted just before a tool is invoked.
    ToolStart {
        /// Tool name.
        name: String,
        /// Tool input as a JSON string.
        input: String,
    },

    /// A tool execution has completed.
    ///
    /// Emitted after the tool returns, whether successfully or with an error.
    ToolComplete {
        /// Tool name.
        name: String,
        /// Tool output as a string.
        output: String,
        /// Whether the tool reported an error.
        is_error: bool,
        /// Wall-clock duration in milliseconds.
        duration_ms: u64,
    },

    /// Context window is running low on capacity.
    ///
    /// Emitted when token usage exceeds a warning threshold,
    /// before compaction is triggered.
    ContextWarning {
        /// Estimated tokens currently used.
        tokens_used: u64,
        /// Estimated tokens remaining.
        tokens_remaining: u64,
    },

    /// A context compaction occurred.
    ///
    /// Emitted after the conversation history has been compacted
    /// to free up context window space.
    ContextCompacted {
        /// Messages before compaction.
        messages_before: usize,
        /// Messages after compaction.
        messages_after: usize,
        /// Estimated tokens saved by compaction.
        tokens_saved: u64,
    },

    /// A generic error event.
    ///
    /// Emitted for errors that don't fit a more specific category.
    Error {
        /// Error description.
        message: String,
        /// Error source or category.
        source: String,
    },

    /// A loop was detected in tool operations.
    ///
    /// Emitted when the detection manager observes the same tool call
    /// pattern repeated beyond the configured loop threshold.
    LoopDetected {
        /// Tool name that was repeating.
        tool: String,
        /// Number of repetitions observed.
        repetitions: usize,
    },

    /// Convergence was detected in agent responses.
    ///
    /// Emitted when the detection manager observes that recent agent
    /// responses have become semantically similar beyond the configured
    /// threshold.
    ConvergenceDetected {
        /// Configured action to take (e.g. `"stop"`, `"warn"`, `"compact"`).
        action: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_start_round_trip() {
        let event = ObserveEvent::SessionStart {
            session_id: uuid::Uuid::nil(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"session_start\""));

        let de: ObserveEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(de, ObserveEvent::SessionStart { .. }));
    }

    #[test]
    fn session_stop_round_trip() {
        let event = ObserveEvent::SessionStop {
            session_id: uuid::Uuid::nil(),
            success: true,
            reason: "max_turns".to_string(),
            total_turns: 5,
            duration_ms: 10_000,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"session_stop\""));
        assert!(json.contains("\"success\":true"));

        let de: ObserveEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(de, ObserveEvent::SessionStop { .. }));
    }

    #[test]
    fn turn_complete_round_trip() {
        let event = ObserveEvent::TurnComplete {
            turn: 3,
            duration_ms: 1200,
            input_tokens: 450,
            output_tokens: 200,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"turn_complete\""));

        let de: ObserveEvent = serde_json::from_str(&json).unwrap();
        if let ObserveEvent::TurnComplete {
            turn,
            input_tokens,
            output_tokens,
            ..
        } = de
        {
            assert_eq!(turn, 3);
            assert_eq!(input_tokens, 450);
            assert_eq!(output_tokens, 200);
        } else {
            panic!("expected TurnComplete");
        }
    }

    #[test]
    fn tool_complete_round_trip() {
        let success = ObserveEvent::ToolComplete {
            name: "read_file".to_string(),
            output: "file contents".to_string(),
            is_error: false,
            duration_ms: 50,
        };
        let json = serde_json::to_string(&success).unwrap();
        assert!(json.contains("\"type\":\"tool_complete\""));
        assert!(json.contains("\"is_error\":false"));

        let failure = ObserveEvent::ToolComplete {
            name: "read_file".to_string(),
            output: "not found".to_string(),
            is_error: true,
            duration_ms: 10,
        };
        let json = serde_json::to_string(&failure).unwrap();
        assert!(json.contains("\"is_error\":true"));

        let de: ObserveEvent = serde_json::from_str(&json).unwrap();
        if let ObserveEvent::ToolComplete { is_error, .. } = de {
            assert!(is_error);
        } else {
            panic!("expected ToolComplete");
        }
    }

    #[test]
    fn context_compacted_round_trip() {
        let event = ObserveEvent::ContextCompacted {
            messages_before: 50,
            messages_after: 20,
            tokens_saved: 10_000,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"context_compacted\""));

        let de: ObserveEvent = serde_json::from_str(&json).unwrap();
        if let ObserveEvent::ContextCompacted {
            messages_before,
            messages_after,
            tokens_saved,
        } = de
        {
            assert_eq!(messages_before, 50);
            assert_eq!(messages_after, 20);
            assert_eq!(tokens_saved, 10_000);
        } else {
            panic!("expected ContextCompacted");
        }
    }

    #[test]
    fn error_round_trip() {
        let event = ObserveEvent::Error {
            message: "something went wrong".to_string(),
            source: "api".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"error\""));

        let de: ObserveEvent = serde_json::from_str(&json).unwrap();
        if let ObserveEvent::Error { message, source } = de {
            assert_eq!(message, "something went wrong");
            assert_eq!(source, "api");
        } else {
            panic!("expected Error");
        }
    }

    #[test]
    fn context_warning_round_trip() {
        let event = ObserveEvent::ContextWarning {
            tokens_used: 180_000,
            tokens_remaining: 20_000,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"context_warning\""));

        let de: ObserveEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(de, ObserveEvent::ContextWarning { .. }));
    }

    #[test]
    fn turn_start_round_trip() {
        let event = ObserveEvent::TurnStart {
            turn: 0,
            query: "hello".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"turn_start\""));

        let de: ObserveEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(de, ObserveEvent::TurnStart { .. }));
    }

    #[test]
    fn turn_failed_round_trip() {
        let event = ObserveEvent::TurnFailed {
            turn: 2,
            duration_ms: 500,
            error: "timeout".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"turn_failed\""));

        let de: ObserveEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(de, ObserveEvent::TurnFailed { .. }));
    }

    #[test]
    fn tool_start_round_trip() {
        let event = ObserveEvent::ToolStart {
            name: "read_file".to_string(),
            input: r#"{"path":"/tmp/x"}"#.to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"tool_start\""));

        let de: ObserveEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(de, ObserveEvent::ToolStart { .. }));
    }
}

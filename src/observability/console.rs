//! A console-based [`EventSink`] that prints human-readable event summaries.
//!
//! [`ConsoleSink`] provides lightweight, readable output suitable for
//! development and debugging. Not intended for production logging — use
//! a structured sink (JSONL, metrics) for that.

use super::event::ObserveEvent;
use super::sink::EventSink;

/// Prints a human-readable summary of each event to stdout.
///
/// Output is designed for developer ergonomics, not machine parsing.
/// Use this during development to see what the agent loop is doing.
/// Each event type gets a formatted one-line summary with the relevant
/// context (turn number, duration, token counts, etc.).
///
/// # Example
///
/// ```rust
/// use loopctl::observability::{ConsoleSink, EventSink, ObserveEvent};
///
/// let sink = ConsoleSink;
/// sink.on_event(&ObserveEvent::TurnComplete {
///     turn: 3,
///     duration_ms: 1200,
///     input_tokens: 450,
///     output_tokens: 200,
/// });
/// // Prints: [turn 3] complete in 1200ms (450 in / 200 out tokens)
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ConsoleSink;

impl EventSink for ConsoleSink {
    fn on_event(&self, event: &ObserveEvent) {
        match event {
            ObserveEvent::SessionStart { session_id } => {
                println!("[session] started {session_id}");
            }
            ObserveEvent::SessionStop {
                session_id,
                success,
                total_turns,
                duration_ms,
                reason,
            } => {
                let status = if *success { "ok" } else { "failed" };
                println!(
                    "[session] {status} {session_id} \u{2014} {total_turns} turns in {duration_ms}ms ({reason})"
                );
            }
            ObserveEvent::TurnStart { turn, query } => {
                let preview = truncate(query, 60);
                println!("[turn {turn}] start: {preview}");
            }
            ObserveEvent::TurnComplete {
                turn,
                duration_ms,
                input_tokens,
                output_tokens,
            } => {
                println!(
                    "[turn {turn}] complete in {duration_ms}ms ({input_tokens} in / {output_tokens} out tokens)"
                );
            }
            ObserveEvent::TurnFailed {
                turn,
                duration_ms,
                error,
            } => {
                println!("[turn {turn}] failed after {duration_ms}ms: {error}");
            }
            ObserveEvent::ToolStart { name, .. } => {
                println!("[tool] {name} started");
            }
            ObserveEvent::ToolComplete {
                name,
                is_error: false,
                duration_ms,
                ..
            } => {
                println!("[tool] {name} completed in {duration_ms}ms");
            }
            ObserveEvent::ToolComplete {
                name,
                is_error: true,
                duration_ms,
                ..
            } => {
                println!("[tool] {name} failed after {duration_ms}ms");
            }
            ObserveEvent::ContextWarning {
                tokens_used,
                tokens_remaining,
            } => {
                println!("[context] warning: {tokens_used} used, {tokens_remaining} remaining");
            }
            ObserveEvent::ContextCompacted {
                messages_before,
                messages_after,
                tokens_saved,
            } => {
                println!(
                    "[context] compacted {messages_before} \u{2192} {messages_after} messages ({tokens_saved} tokens saved)"
                );
            }
            ObserveEvent::Error { message, source } => {
                println!("[error] {source}: {message}");
            }
        }
    }
}

/// Truncate a string to approximately `max_len` characters, appending an
/// ellipsis (`\u{2026}`) if truncated.
///
/// Handles multi-byte characters correctly by operating on char boundaries.
fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max_len).collect();
    format!("{truncated}\u{2026}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_sink_handles_all_event_types() {
        let sink = ConsoleSink;

        sink.on_event(&ObserveEvent::SessionStart {
            session_id: uuid::Uuid::nil(),
        });
        sink.on_event(&ObserveEvent::TurnStart {
            turn: 0,
            query: "hello".to_string(),
        });
        sink.on_event(&ObserveEvent::ToolStart {
            name: "read_file".to_string(),
            input: "{}".to_string(),
        });
        sink.on_event(&ObserveEvent::ToolComplete {
            name: "read_file".to_string(),
            output: "contents".to_string(),
            is_error: false,
            duration_ms: 50,
        });
        sink.on_event(&ObserveEvent::ToolComplete {
            name: "read_file".to_string(),
            output: "not found".to_string(),
            is_error: true,
            duration_ms: 10,
        });
        sink.on_event(&ObserveEvent::TurnComplete {
            turn: 0,
            duration_ms: 100,
            input_tokens: 10,
            output_tokens: 5,
        });
        sink.on_event(&ObserveEvent::TurnFailed {
            turn: 1,
            duration_ms: 200,
            error: "timeout".to_string(),
        });
        sink.on_event(&ObserveEvent::ContextWarning {
            tokens_used: 180_000,
            tokens_remaining: 20_000,
        });
        sink.on_event(&ObserveEvent::ContextCompacted {
            messages_before: 50,
            messages_after: 20,
            tokens_saved: 10_000,
        });
        sink.on_event(&ObserveEvent::Error {
            message: "something broke".to_string(),
            source: "api".to_string(),
        });
        sink.on_event(&ObserveEvent::SessionStop {
            session_id: uuid::Uuid::nil(),
            success: true,
            reason: "done".to_string(),
            total_turns: 5,
            duration_ms: 10_000,
        });
    }

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_long_string() {
        let long = "a".repeat(100);
        let result = truncate(&long, 10);
        assert_eq!(result.chars().count(), 11); // 10 chars + ellipsis
        assert!(result.ends_with('\u{2026}'));
    }

    #[test]
    fn truncate_exact_length() {
        assert_eq!(truncate("exactly!", 8), "exactly!");
    }

    #[test]
    fn truncate_empty_string() {
        assert_eq!(truncate("", 10), "");
    }

    #[test]
    fn truncate_multibyte_chars() {
        let input = "héllo😊world";
        let result = truncate(input, 5);
        assert_eq!(result, "héllo…");
    }

    #[test]
    fn truncate_zero_max_len() {
        let result = truncate("hello", 0);
        assert_eq!(result, "\u{2026}");
    }
}

//! Structured event streaming for agent observability.
//!
//! This module provides the [`EventSink`] trait and [`ObserveEvent`] enum —
//! the primary observability abstractions in `loopctl`. Every consumer —
//! console logging, JSONL files, metrics, custom monitors — implements
//! [`EventSink`].
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────┐
//! │        Agent Loop         │
//! │                           │
//! │  emits ObserveEvent       ├──▶ dyn EventSink::on_event()
//! │                           │         │
//! └───────────────────────────┘         ├─▶ ConsoleSink
//!                                       ├─▶ NullSink
//!                                       ├─▶ CompositeSink ─┬─▶ Sink 1
//!                                       │                  ├─▶ Sink 2
//!                                       │                  └─▶ Sink 3
//!                                       └─▶ MetricsSink (feature-gated)
//! ```
//!
//! # Provided Sinks
//!
//! | Sink                | Purpose                                              |
//! |---------------------|------------------------------------------------------|
//! | [`NullSink`]        | Discards all events. Useful as a default.            |
//! | [`ConsoleSink`]     | Prints human-readable summaries to stdout.           |
//! | [`CompositeSink`]   | Fans out to multiple sinks with panic isolation.     |
//!
//! # Event Types
//!
//! [`ObserveEvent`] covers the full agent lifecycle:
//!
//! - Session lifecycle (`SessionStart`, `SessionStop`)
//! - Turn lifecycle (`TurnStart`, `TurnComplete`, `TurnFailed`)
//! - Tool execution (`ToolStart`, `ToolComplete`)
//! - Context management (`ContextWarning`, `ContextCompacted`)
//! - Errors (`Error`)
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::observability::{EventSink, NullSink, ObserveEvent};
//!
//! let sink = NullSink;
//! sink.on_event(&ObserveEvent::SessionStart {
//!     session_id: uuid::Uuid::nil(),
//! });
//! ```
//!
//! # Composing Sinks
//!
//! Use [`CompositeSink`] to fan out to multiple sinks:
//!
//! ```rust
//! use loopctl::observability::{CompositeSink, ConsoleSink, NullSink};
//!
//! let sink = CompositeSink::new(vec![
//!     Box::new(ConsoleSink),
//!     Box::new(NullSink),
//! ]);
//! ```

pub mod console;
pub mod event;
pub mod sink;

pub use console::ConsoleSink;
pub use event::ObserveEvent;
pub use sink::{CompositeSink, EventSink, NullSink};

//! The [`EventSink`] trait and provided implementations.
//!
//! [`EventSink`] is the primary observability abstraction in `loopctl`.
//! Every consumer — console logging, JSONL files, metrics, custom monitors —
//! implements this trait.
//!
//! # Implementations
//!
//! - [`NullSink`] — Discards all events. Useful as a default.
//! - [`CompositeSink`] — Fans out to multiple sinks with panic isolation.

use super::event::ObserveEvent;
use std::panic::catch_unwind;
use std::sync::Arc;

// ===================================================
// EventSink trait
// ===================================================

/// The primary observability abstraction in `loopctl`.
///
/// Every consumer — console logging, JSONL files, metrics, custom monitors —
/// implements `EventSink`. The agent loop calls [`on_event`](EventSink::on_event)
/// at key lifecycle points with an [`ObserveEvent`].
///
/// # Object Safety
///
/// The trait is object-safe: `Box<dyn EventSink>` and `Arc<dyn EventSink>` work.
///
/// # Example
///
/// ```rust
/// use loopctl::observability::{EventSink, ObserveEvent};
///
/// struct PrintSink;
///
/// impl EventSink for PrintSink {
///     fn on_event(&self, event: &ObserveEvent) {
///         match event {
///             ObserveEvent::ToolStart { name, .. } => {
///                 println!("tool started: {name}");
///             }
///             ObserveEvent::TurnComplete { turn, duration_ms, .. } => {
///                 println!("turn {turn} done in {duration_ms}ms");
///             }
///             _ => {}
///         }
///     }
/// }
/// ```
pub trait EventSink: Send + Sync {
    /// Handle an observed event.
    ///
    /// Called by the agent loop at each lifecycle point. Implementations
    /// should be fast and non-blocking — heavy work should be offloaded
    /// to a channel or background task.
    fn on_event(&self, event: &ObserveEvent);
}

// ===================================================
// NullSink
// ===================================================

/// A sink that discards all events.
///
/// Useful as a default when no observability is needed, or as a
/// placeholder during testing. Equivalent to `/dev/null` for events.
///
/// # Example
///
/// ```rust
/// use loopctl::observability::{EventSink, NullSink, ObserveEvent};
///
/// let sink = NullSink;
/// sink.on_event(&ObserveEvent::SessionStart {
///     session_id: uuid::Uuid::nil(),
/// });
/// // Event is silently discarded.
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct NullSink;

impl EventSink for NullSink {
    fn on_event(&self, _event: &ObserveEvent) {}
}

// ===================================================
// CompositeSink
// ===================================================

/// A composite sink that fans out every event to multiple inner sinks.
///
/// Holds an ordered list of [`EventSink`] trait objects behind [`Arc`]
/// and forwards each event to every inner sink in sequence. This
/// implements the classic **Composite** pattern, allowing you to combine
/// console logging, JSONL output, and custom sinks without writing a
/// new struct.
///
/// Sinks are called in insertion order. If any individual sink panics,
/// the remaining sinks in the list are still called — this is achieved
/// internally via [`std::panic::catch_unwind`]. This ensures one
/// misbehaving sink does not prevent others from receiving events.
///
/// # Architecture
///
/// ```text
///                CompositeSink
///              ┌──────────────┐
/// on_event() ─►│  sinks[]     │──▶ sink[0].on_event()
///              │              │──▶ sink[1].on_event()
///              │              │──▶ sink[2].on_event()
///              └──────────────┘
/// ```
///
/// # Thread Safety
///
/// Each inner sink is stored as `Arc<dyn EventSink>`, so the same sink
/// can be shared across multiple [`CompositeSink`] instances or other
/// parts of the system. The fan-out loop borrows `&self`, meaning all
/// callbacks must be `&self`-safe (no `&mut self`).
///
/// # Example
///
/// ```rust
/// use loopctl::observability::{CompositeSink, ConsoleSink, NullSink, EventSink};
/// use std::sync::Arc;
///
/// // Build with owned sinks:
/// let sink = CompositeSink::new(vec![
///     Box::new(ConsoleSink),
///     Box::new(NullSink),
/// ]);
/// assert_eq!(sink.len(), 2);
///
/// // Build with the builder pattern:
/// let sink = CompositeSink::new(vec![])
///     .with(ConsoleSink)
///     .with(NullSink);
/// assert_eq!(sink.len(), 2);
///
/// // Build with pre-Arc'd sinks:
/// let shared = Arc::new(NullSink);
/// let sink = CompositeSink::new(vec![])
///     .with_arc(shared.clone())
///     .with_arc(shared);  // same sink twice
/// assert_eq!(sink.len(), 2);
/// ```
pub struct CompositeSink {
    /// The ordered list of inner sinks to fan out to.
    ///
    /// Each sink is stored as `Arc<dyn EventSink>` so it can be shared
    /// across threads. Sinks are called in insertion order — first added
    /// is first notified.
    sinks: Vec<Arc<dyn EventSink>>,
}

impl std::fmt::Debug for CompositeSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeSink")
            .field("sink_count", &self.sinks.len())
            .finish()
    }
}

impl Default for CompositeSink {
    /// Returns an empty composite sink with no inner sinks.
    ///
    /// The default instance contains zero inner sinks, so all events
    /// are silently discarded until sinks are added.
    fn default() -> Self {
        Self { sinks: Vec::new() }
    }
}

impl CompositeSink {
    /// Create a new composite sink that fans out to the given sinks.
    ///
    /// Each sink is wrapped in `Arc<dyn EventSink>` and stored for
    /// ordered fan-out. Returns a ready-to-use composite.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::observability::{CompositeSink, ConsoleSink, NullSink};
    ///
    /// let sink = CompositeSink::new(vec![
    ///     Box::new(ConsoleSink),
    ///     Box::new(NullSink),
    /// ]);
    /// assert_eq!(sink.len(), 2);
    /// ```
    #[must_use]
    pub fn new(sinks: Vec<Box<dyn EventSink>>) -> Self {
        let sinks: Vec<Arc<dyn EventSink>> = sinks.into_iter().map(Arc::from).collect();
        Self { sinks }
    }

    /// Add an owned sink to the fan-out list.
    ///
    /// The sink is wrapped as `Arc<dyn EventSink>` and appended to the
    /// end of the list. Returns `self` for chaining.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::observability::{CompositeSink, ConsoleSink, NullSink};
    ///
    /// let sink = CompositeSink::new(vec![])
    ///     .with(ConsoleSink)
    ///     .with(NullSink);
    /// assert_eq!(sink.len(), 2);
    /// ```
    #[must_use]
    pub fn with<S: EventSink + 'static>(mut self, sink: S) -> Self {
        self.sinks.push(Arc::new(sink));
        self
    }

    /// Add a sink that is already behind an `Arc`.
    ///
    /// Useful when multiple [`CompositeSink`] instances need to share the
    /// same inner sink, or when the sink is constructed externally and
    /// already wrapped in an `Arc`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::observability::{CompositeSink, NullSink};
    /// use std::sync::Arc;
    ///
    /// let shared = Arc::new(NullSink);
    /// let sink = CompositeSink::new(vec![])
    ///     .with_arc(shared.clone())
    ///     .with_arc(shared);
    /// assert_eq!(sink.len(), 2);
    /// ```
    #[must_use]
    pub fn with_arc<S: EventSink + 'static>(mut self, sink: Arc<S>) -> Self {
        self.sinks.push(sink);
        self
    }

    /// Number of inner sinks in the fan-out list.
    ///
    /// Returns the count of sinks that will receive events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sinks.len()
    }

    /// Whether there are no inner sinks in the fan-out list.
    ///
    /// Returns `true` when [`len`](CompositeSink::len) is zero. When
    /// empty, [`on_event`](EventSink::on_event) is effectively a no-op
    /// (the internal loop body never executes).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sinks.is_empty()
    }

    /// Append a sink after construction.
    ///
    /// The sink is wrapped as `Arc<dyn EventSink>` and appended to the
    /// end of the fan-out list.
    pub fn add(&mut self, sink: Box<dyn EventSink>) {
        self.sinks.push(Arc::from(sink));
    }
}

impl EventSink for CompositeSink {
    fn on_event(&self, event: &ObserveEvent) {
        for sink in &self.sinks {
            // Panic isolation: a failing sink must not break other sinks.
            let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
                sink.on_event(event);
            }));
            if let Err(panic_payload) = result {
                let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                tracing::error!("EventSink panicked: {msg}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn null_sink_accepts_events() {
        let sink = NullSink;
        sink.on_event(&ObserveEvent::SessionStart {
            session_id: uuid::Uuid::nil(),
        });
    }

    #[test]
    fn composite_fans_out() {
        static COUNT_A: AtomicUsize = AtomicUsize::new(0);
        static COUNT_B: AtomicUsize = AtomicUsize::new(0);

        struct SinkA;
        struct SinkB;

        impl EventSink for SinkA {
            fn on_event(&self, _event: &ObserveEvent) {
                COUNT_A.fetch_add(1, Ordering::Relaxed);
            }
        }

        impl EventSink for SinkB {
            fn on_event(&self, _event: &ObserveEvent) {
                COUNT_B.fetch_add(1, Ordering::Relaxed);
            }
        }

        let composite = CompositeSink::new(vec![Box::new(SinkA), Box::new(SinkB)]);
        composite.on_event(&ObserveEvent::SessionStart {
            session_id: uuid::Uuid::nil(),
        });

        assert_eq!(COUNT_A.load(Ordering::Relaxed), 1);
        assert_eq!(COUNT_B.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn composite_isolates_panics() {
        static COUNT: AtomicUsize = AtomicUsize::new(0);

        struct PanicSink;
        struct CountSink;

        impl EventSink for PanicSink {
            fn on_event(&self, _event: &ObserveEvent) {
                panic!("boom");
            }
        }

        impl EventSink for CountSink {
            fn on_event(&self, _event: &ObserveEvent) {
                COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        // PanicSink is first; CountSink should still receive the event.
        let composite = CompositeSink::new(vec![Box::new(PanicSink), Box::new(CountSink)]);
        composite.on_event(&ObserveEvent::SessionStart {
            session_id: uuid::Uuid::nil(),
        });

        assert_eq!(COUNT.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn composite_with_builder() {
        let composite = CompositeSink::new(vec![])
            .with(NullSink)
            .with(NullSink)
            .with(NullSink);
        assert_eq!(composite.len(), 3);
    }

    #[test]
    fn composite_with_arc_builder() {
        let shared = Arc::new(NullSink);
        let composite = CompositeSink::new(vec![])
            .with_arc(shared.clone())
            .with_arc(shared);
        assert_eq!(composite.len(), 2);
    }

    #[test]
    fn composite_add_and_len() {
        let mut composite = CompositeSink::new(vec![Box::new(NullSink)]);
        assert_eq!(composite.len(), 1);
        assert!(!composite.is_empty());

        composite.add(Box::new(NullSink));
        assert_eq!(composite.len(), 2);
    }

    #[test]
    fn composite_empty() {
        let composite = CompositeSink::new(vec![]);
        assert!(composite.is_empty());
        composite.on_event(&ObserveEvent::SessionStart {
            session_id: uuid::Uuid::nil(),
        });
    }

    #[test]
    fn composite_default_is_empty() {
        let composite = CompositeSink::default();
        assert!(composite.is_empty());
        assert_eq!(composite.len(), 0);
    }

    #[test]
    fn event_sink_is_object_safe() {
        let _boxed: Box<dyn EventSink> = Box::new(NullSink);
        let _arc: Arc<dyn EventSink> = Arc::new(NullSink);
    }
}

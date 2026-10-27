//! Run metrics: per-run snapshot value types and the metric registry catalog.
//!
//! *Requires the `metrics` feature.*
//!
//! Two halves ride this feature:
//!
//! - [`session`] — the typed per-run snapshot [`AgentMetrics`]: turn
//!   counts, turn-latency [`Percentile`] readings, per-tool
//!   [`ToolStats`], and cumulative [`TokenTotals`] with the cache and
//!   reasoning detail. An observer fills it as a run proceeds; any host
//!   can also assemble it directly, serialize it, and ship it.
//! - [`registry`] — the static catalog of every metric name this crate
//!   emits under the telemetry contract: [`MetricDescriptor`] entries
//!   carrying kind and the closed label set, queried through
//!   [`lookup`] and [`validate`].
//!
//! Everything here is pure data — construction, aggregation math, and
//! serde. Nothing subscribes, locks, or spawns; collection and export
//! live outside this module.
//!
//! # Examples
//!
//! ```
//! use loopctl::metrics::{AgentMetrics, TokenTotals, registry};
//!
//! // Assemble one run's snapshot the way an observer would.
//! let mut m = AgentMetrics::default();
//! m.total_turns = 2;
//! m.turn_latency.record(300);
//! m.turn_latency.record(400);
//! m.tokens.add(TokenTotals {
//!     input: 800,
//!     output: 90,
//!     ..Default::default()
//! });
//! m.record_tool("read", 5, false);
//!
//! // Latency percentiles read off fixed millisecond buckets: both
//! // samples fall in the ≤500 ms bucket, so every rank reports 500.
//! let p = m.turn_latency_percentiles().expect("turns recorded");
//! assert_eq!(p.p50.value_ms, 500);
//! assert_eq!(m.total_tool_calls(), 1);
//!
//! // The catalog answers questions about emitted metric names.
//! let d = registry::lookup("loopctl.tools.calls").expect("declared name");
//! assert_eq!(d.labels, ["tool"]);
//! ```

pub mod registry;
pub mod session;

pub use registry::{MetricDescriptor, MetricKind, all, lookup, validate};
pub use session::{
    AgentMetrics, LatencyHistogram, LatencyPercentiles, Percentile, TokenTotals, ToolStats,
    ToolStatsMap,
};

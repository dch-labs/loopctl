//! Run metrics: the metric catalog, the live value store, and the
//! collectors that fill them.
//!
//! *Requires the `metrics` feature.*
//!
//! Four surfaces ride this feature:
//!
//! - [`registry`] — the static catalog of every metric name declared
//!   under the telemetry contract — the signals emitted today plus the
//!   rest of the contract's closed table: [`MetricDescriptor`]
//!   entries carrying kind and the closed label set, queried through
//!   [`lookup`] and [`validate`].
//! - [`session`] — the typed per-run snapshot [`AgentMetrics`]: turn
//!   counts, turn-latency [`Percentile`] readings, per-tool
//!   [`ToolStats`], and cumulative [`TokenTotals`] with the cache and
//!   reasoning detail. An observer fills it as a run proceeds; any host
//!   can also assemble it directly, serialize it, and ship it.
//! - [`store`] — the live value store [`MetricsRegistry`]: one
//!   [`MetricValue`] per (name, label-set) series, read as a
//!   [`RegistrySnapshot`] by exporters and tests.
//! - The two collectors: [`MetricsLayer`], a `tracing_subscriber`
//!   layer that turns recognized metric *events* into store series,
//!   and [`MetricsObserver`], a [`LoopObserver`](crate::observer::LoopObserver)
//!   that turns engine *callbacks* into session aggregates. Both are
//!   inert until wired — the layer until a host composes it into a
//!   subscriber, the observer until
//!   [`with_observer`](crate::engine::BareLoop::with_observer) attaches
//!   it — and both are read by the companion exporters
//!   (`loopctl-otel`, `loopctl-langfuse`).
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
//! assert_eq!(d.labels, ["tool", "outcome"]);
//! ```
//!
//! # Wiring a subscriber over the store
//!
//! Metric emissions are `DEBUG`-level on the layer's reserved target;
//! the subscriber's filter must admit them or the layer never sees an
//! event.
//!
//! ```
//! use std::sync::Arc;
//! use loopctl::metrics::{MetricsLayer, MetricsRegistry};
//! use tracing_subscriber::layer::SubscriberExt as _;
//!
//! let store = Arc::new(MetricsRegistry::default());
//! let subscriber = tracing_subscriber::registry()
//!     .with(MetricsLayer::new(store.clone()))
//!     .with(
//!         tracing_subscriber::filter::Targets::new()
//!             .with_target(MetricsLayer::TARGET, tracing::Level::DEBUG),
//!     );
//! let _guard = tracing::subscriber::set_default(subscriber);
//!
//! tracing::debug!(target: MetricsLayer::TARGET, metric = "loopctl.memoize.hits");
//!
//! let snap = store.snapshot();
//! assert_eq!(snap.series.len(), 1, "the emission stored one series");
//! ```
//!
//! # Collecting a run's session metrics
//!
//! ```
//! use std::sync::Arc;
//! use loopctl::config::SessionConfig;
//! use loopctl::engine::core::Loop as _;
//! use loopctl::engine::{BareLoop, RunConfig};
//! use loopctl::metrics::MetricsObserver;
//! use loopctl::testing::{MockApiClient, MockResponse};
//! use loopctl::tool::ToolRegistry;
//!
//! let observer = Arc::new(MetricsObserver::default());
//! let client = Arc::new(MockApiClient::new("test-model").with_responses(vec![MockResponse {
//!     text: "done".to_string(),
//!     tool_call: None,
//!     stop_reason: "end_turn".to_string(),
//! }]));
//! # async fn scenario(
//! #     client: std::sync::Arc<MockApiClient>,
//! #     observer: std::sync::Arc<MetricsObserver>,
//! # ) -> loopctl::engine::RunResult {
//! let mut agent = BareLoop::new(client, ToolRegistry::new(), SessionConfig::default())
//!     .with_observer(observer);
//! agent.run("measure this", &RunConfig::default()).await
//! # }
//! ```
//!
//! Every collector read returns `None` until its signal has data; a
//! metric nobody emitted never fabricates a number.

pub mod layer;
pub mod observer;
pub mod registry;
pub mod session;
pub mod store;

pub use layer::{METRIC_EXEMPT_FIELDS, MetricsLayer};
pub use observer::MetricsObserver;
pub use registry::{MetricDescriptor, MetricKind, REGISTRY, all, lookup, validate};
pub use session::{
    AgentMetrics, LatencyHistogram, LatencyPercentiles, Percentile, TokenTotals, ToolStats,
    ToolStatsMap,
};
pub use store::{
    HistogramSummary, MetricValue, MetricsRegistry, RegistrySnapshot, SeriesSnapshot, SeriesValue,
};

/// Recovered lock guard — the metrics store never panics while held,
/// and poison must not cascade to every later reader.
pub(crate) fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

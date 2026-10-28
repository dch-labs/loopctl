//! The Langfuse layer: classify, parent, and enqueue on span close.
//!
//! Every span carrying the loopctl vocabulary gets a `SpanMeta` extension
//! at open (trace id for `agent.run`; observation id + inherited trace for
//! turns/tools), every `record` merges into it, and at close the meta
//! becomes one ingestion envelope. Metric-target events are the metrics
//! path's signals and never render here.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::SystemTime;

use loopctl::metrics::AgentMetrics;
use tracing::span::{Attributes, Record};
use tracing::{Event, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::{LookupSpan, SpanRef};

use crate::ingest::{self, IngestSink};
use crate::mapping::{self, LangfuseField, LangfuseTarget};

/// The per-run rollup source for auto-scoring on run close.
///
/// Typically `move || Some(observer.snapshot())`: the layer polls it
/// when an `agent.run` span closes and derives that run's score set
/// from whatever the observer holds at that instant. A `None` result
/// skips scoring for that run rather than emitting partial scores.
pub type MetricsSource = Arc<dyn Fn() -> Option<AgentMetrics> + Send + Sync + 'static>;

/// The layer: loopctl spans in, Langfuse ingestion events out.
///
/// Closed runs become traces, their turns and tool calls become nested
/// observations, and — with a [`MetricsSource`] attached — each run's
/// score set rides onto its trace. Cloning is cheap: every copy enqueues
/// through the same [`crate::ingest`] sink.
#[derive(Clone)]
pub struct LangfuseLayer {
    /// The shared ingest queue every closed span's envelope rides in;
    /// cloned to the per-layer visitor so each close can enqueue.
    sink: IngestSink,

    /// Optional auto-scoring hook, run at each `agent.run` close to
    /// derive scores from a host-held snapshot; `None` ships spans
    /// without scores.
    metrics: Option<MetricsSource>,
}

impl std::fmt::Debug for LangfuseLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangfuseLayer")
            .field("metrics", &self.metrics.as_ref().map(|_| "MetricsSource"))
            .finish_non_exhaustive()
    }
}

impl LangfuseLayer {
    /// Bind the layer to its ingestion sink.
    ///
    /// The sink arrives alongside the layer from [`crate::install`];
    /// hosts never construct a layer without its flusher, so the
    /// constructor stays crate-internal.
    pub(crate) fn new(sink: IngestSink) -> Self {
        Self {
            sink,
            metrics: None,
        }
    }

    /// Auto-score runs: on each run close, derive scores from `source`
    /// (typically `move || Some(observer.snapshot())`) and enqueue them on
    /// that run's trace.
    #[must_use]
    pub fn with_metrics_source(self, source: MetricsSource) -> Self {
        Self {
            sink: self.sink,
            metrics: Some(source),
        }
    }
}

/// What a span accumulated between open and close.
///
/// Inserted as a span extension at open, merged into on every
/// `record`, and consumed at close to render exactly one ingestion
/// envelope: identity fields for the Langfuse object, first-class
/// slots routed by [`mapping::first_class`], and everything else as
/// free-form metadata in key order.
#[derive(Debug, Clone)]
struct SpanMeta {
    /// Which Langfuse object kind this span renders as.
    ///
    /// The two values of [`LangfuseTarget`]: a run span renders as the
    /// trace, every other span as a span-kind observation — the span's
    /// name at open decides which.
    target: LangfuseTarget,

    /// The Langfuse `traceId` this observation hangs under — the run
    /// id for every span the span tree produces.
    trace_id: String,

    /// This observation's id; the span id formatted the way the
    /// ingestion API accepts it.
    observation_id: Option<String>,

    /// The enclosing observation's id, `None` for a root — the flat
    /// tree's turns and tools both read the run as their parent.
    parent_observation_id: Option<String>,

    /// Display name for the rendered Langfuse object.
    ///
    /// The span's own name (`agent.turn`, `agent.tool`), carried
    /// through unchanged so a dashboard row matches the trace tree.
    name: String,

    /// Langfuse session key, filled when a `session_id` field rode the
    /// span; keys the session view on the dashboard.
    session_id: Option<String>,

    /// Model name the span recorded, from `gen_ai.request.model`.
    ///
    /// Observations render it in their dedicated `model` field; traces
    /// carry it inside their metadata object.
    model: Option<String>,

    /// `gen_ai.usage.*` fields collected at close, keyed by their span
    /// field names and rendered to JSON values for the usage slot.
    usage: BTreeMap<String, serde_json::Value>,

    /// Every other field the span recorded, JSON-rendered; key order
    /// because a `BTreeMap` holds it.
    metadata: BTreeMap<String, serde_json::Value>,

    /// Whether the span recorded a failure signal.
    ///
    /// Set by an `error.type` field or an `outcome = "error"` record;
    /// closes the envelope with level `ERROR`.
    errored: bool,

    /// When the span opened.
    ///
    /// Stamps the envelope's start time; the end time is read at close,
    /// when the envelope renders.
    start: SystemTime,
}

impl SpanMeta {
    /// File one recorded field into its first-class slot or metadata.
    ///
    /// Keys [`mapping::first_class`](crate::mapping::first_class)
    /// recognizes route to their dedicated [`SpanMeta`] slot — session,
    /// model, and the five usage figures — while `error.type` and
    /// `outcome = "error"` set the failure flag with the value kept in
    /// metadata for dashboards. Every other key lands in metadata under
    /// its emitted name.
    fn absorb(&mut self, key: &str, value: serde_json::Value) {
        match mapping::first_class(key) {
            Some(LangfuseField::SessionId) => {
                self.session_id = value
                    .as_str()
                    .map(str::to_string)
                    .or(Some(value.to_string()));
            }
            Some(LangfuseField::Model) => {
                self.model = value
                    .as_str()
                    .map(str::to_string)
                    .or(Some(value.to_string()));
            }
            Some(LangfuseField::UsageInput) => {
                self.usage.insert("input".to_string(), value);
            }
            Some(LangfuseField::UsageOutput) => {
                self.usage.insert("output".to_string(), value);
            }
            Some(LangfuseField::UsageCachedInput) => {
                self.usage.insert("cached_input".to_string(), value);
            }
            Some(LangfuseField::UsageCacheWrite) => {
                self.usage.insert("cache_write".to_string(), value);
            }
            Some(LangfuseField::UsageReasoning) => {
                self.usage.insert("reasoning".to_string(), value);
            }
            Some(LangfuseField::ErrorType) => {
                self.errored = true;
                self.metadata.insert(key.to_string(), value);
            }
            Some(LangfuseField::Outcome) => {
                if value.as_str() == Some("error") {
                    self.errored = true;
                }
                self.metadata.insert(key.to_string(), value);
            }
            None => {
                self.metadata.insert(key.to_string(), value);
            }
        }
    }
}

/// Field visitor collecting into JSON values; debug-rendered strings have
/// their wrapping quotes trimmed so metadata reads like the emitted text.
#[derive(Default)]
struct Collector {
    /// The visited fields in arrival order, as `(name, JSON value)` pairs.
    ///
    /// A plain `Vec` rather than a map because one visit sees each field
    /// at most once; the consumers sort into their own key-ordered maps.
    fields: Vec<(String, serde_json::Value)>,
}

impl tracing::field::Visit for Collector {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let text = format!("{value:?}");
        let trimmed = text.strip_prefix('"').and_then(|s| s.strip_suffix('"'));
        self.fields.push((
            field.name().to_string(),
            serde_json::Value::String(trimmed.unwrap_or(&text).to_string()),
        ));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.fields.push((
            field.name().to_string(),
            serde_json::Value::String(value.to_string()),
        ));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.fields
            .push((field.name().to_string(), serde_json::json!(value)));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.fields
            .push((field.name().to_string(), serde_json::json!(value)));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.fields
            .push((field.name().to_string(), serde_json::json!(value)));
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.fields
            .push((field.name().to_string(), serde_json::json!(value)));
    }
}

/// Record a span's opening fields into plain `(name, value)` pairs.
///
/// One [`Collector`] visit over `attrs`; `on_new_span` routes the
/// pairs through [`SpanMeta::absorb`] exactly as later `record`
/// visits are routed, so opening fields and recorded fields classify
/// identically.
fn collected(attrs: &Attributes<'_>) -> Vec<(String, serde_json::Value)> {
    let mut collector = Collector::default();
    attrs.record(&mut collector);
    collector.fields
}

/// Walk ancestors for the nearest Langfuse-instrumented span's meta.
///
/// Observations inherit trace id, session, and parent-observation ids
/// from the closest ancestor carrying [`SpanMeta`]. An unbroken walk
/// to the root means the span has no run ancestor — an orphaned
/// observation would reference a trace that never exists, so it is
/// dropped rather than exported.
fn ancestor_meta<'a, S: Subscriber + for<'l> LookupSpan<'l>>(
    span: &SpanRef<'a, S>,
) -> Option<SpanMeta> {
    let mut current = span.parent();
    while let Some(candidate) = current {
        let found = {
            let extensions = candidate.extensions();
            extensions.get::<SpanMeta>().cloned()
        };
        if let Some(meta) = found {
            return Some(meta);
        }
        current = candidate.parent();
    }
    None
}

impl<S> Layer<S> for LangfuseLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &tracing::span::Id, ctx: Context<'_, S>) {
        let name = attrs.metadata().name().to_string();
        let target = mapping::classify_span(&name);
        let Some(span) = ctx.span(id) else { return };
        let mut meta = SpanMeta {
            target,
            trace_id: uuid::Uuid::new_v4().to_string(),
            observation_id: match target {
                LangfuseTarget::Trace => None,
                LangfuseTarget::Observation => Some(uuid::Uuid::new_v4().to_string()),
            },
            parent_observation_id: None,
            name,
            session_id: None,
            model: None,
            usage: BTreeMap::new(),
            metadata: BTreeMap::new(),
            errored: false,
            start: SystemTime::now(),
        };
        match target {
            LangfuseTarget::Trace => {}
            LangfuseTarget::Observation => {
                let Some(parent_meta) = ancestor_meta(&span) else {
                    // No run ancestor: nothing to attach to — an orphan
                    // observation would reference a trace that never
                    // exists, so it is not exported at all.
                    return;
                };
                meta.trace_id = parent_meta.trace_id;
                meta.session_id = parent_meta.session_id;
                meta.parent_observation_id = parent_meta.observation_id;
            }
        }
        for (key, value) in collected(attrs) {
            meta.absorb(&key, value);
        }
        let mut extensions = span.extensions_mut();
        extensions.insert(meta);
    }

    fn on_record(&self, id: &tracing::span::Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let mut collector = Collector::default();
        values.record(&mut collector);
        let mut extensions = span.extensions_mut();
        if let Some(meta) = extensions.get_mut::<SpanMeta>() {
            for (key, value) in collector.fields {
                meta.absorb(&key, value);
            }
        }
    }

    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        // Langfuse ingests spans, not log events; metric-target events in
        // particular belong to the metrics path entirely.
        let _ = event;
    }

    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let meta = {
            let mut extensions = span.extensions_mut();
            extensions.remove::<SpanMeta>()
        };
        let Some(meta) = meta else { return };
        match meta.target {
            LangfuseTarget::Trace => {
                let mut metadata =
                    serde_json::Map::<String, serde_json::Value>::from_iter(meta.metadata);
                if let Some(model) = meta.model {
                    metadata.insert("model".to_string(), serde_json::Value::String(model));
                }
                let mut body = serde_json::json!({
                    "id": meta.trace_id,
                    "timestamp": ingest::rfc3339(meta.start),
                    "name": meta.name,
                    "tags": ["loopctl"],
                    "metadata": serde_json::Value::Object(metadata),
                });
                if let Some(session) = meta.session_id {
                    ingest::set(&mut body, "sessionId", serde_json::Value::String(session));
                }
                if meta.errored {
                    ingest::set(
                        &mut body,
                        "level",
                        serde_json::Value::String("ERROR".to_string()),
                    );
                }
                self.sink.push(ingest::envelope("trace-create", &body));
                if let Some(source) = &self.metrics
                    && let Some(agent) = source()
                {
                    // The one wire builder lives in `scores` — the same
                    // envelopes a host pushes manually through the guard.
                    for envelope in crate::scores::score_events(&meta.trace_id, &agent) {
                        self.sink.push(envelope);
                    }
                }
            }
            LangfuseTarget::Observation => {
                let mut body = serde_json::json!({
                    "id": meta.observation_id,
                    "traceId": meta.trace_id,
                    "name": meta.name,
                    "startTime": ingest::rfc3339(meta.start),
                    "endTime": ingest::rfc3339_now(),
                    "type": "span",
                    "metadata": serde_json::Value::Object(
                        serde_json::Map::<String, serde_json::Value>::from_iter(meta.metadata),
                    ),
                });
                if let Some(parent) = meta.parent_observation_id {
                    ingest::set(
                        &mut body,
                        "parentObservationId",
                        serde_json::Value::String(parent),
                    );
                }
                if let Some(model) = meta.model {
                    ingest::set(&mut body, "model", serde_json::Value::String(model));
                }
                if !meta.usage.is_empty() {
                    ingest::set(
                        &mut body,
                        "usageDetails",
                        serde_json::Value::Object(meta.usage.into_iter().collect()),
                    );
                }
                if meta.errored {
                    ingest::set(
                        &mut body,
                        "level",
                        serde_json::Value::String("ERROR".to_string()),
                    );
                }
                self.sink.push(ingest::envelope("span-create", &body));
            }
        }
    }
}

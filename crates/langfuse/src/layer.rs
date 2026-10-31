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
    /// The shared ingest queue every closed span's envelope rides in.
    ///
    /// Held by value and cloned onto the close path so an envelope can
    /// be enqueued while the visit is still in hand, without borrowing
    /// the layer back. Every copy addresses the same buffer, so layers
    /// composed from one install all drain through one flusher.
    sink: IngestSink,

    /// Optional auto-scoring hook, run at each `agent.run` close.
    ///
    /// The closure hands back the host's current [`AgentMetrics`] snapshot,
    /// which the layer turns into the score set riding onto that run's
    /// trace. `None` — the state [`LangfuseLayer::new`] starts from —
    /// ships spans and observations with no scores at all, the shape a
    /// host tracing structure only wants.
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

    /// Auto-score runs from `source`.
    ///
    /// The closure runs when a run span closes, and what it returns is
    /// turned into that trace's score set — so the figures are the run's
    /// own, read as its span closed rather than whenever the batch
    /// happens to flush. The usual binding captures an observer:
    /// `move || Some(observer.snapshot())`. A source answering `None`
    /// scores nothing and the run ships as spans only.
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

    /// The Langfuse `traceId` this observation hangs under.
    ///
    /// Minted for a root span and inherited from the parent otherwise, so
    /// the run and everything opened beneath it share one grouping key and
    /// no field recorded mid-span can move an observation to another
    /// trace. Every span the engine's tree produces carries the run's id
    /// here, which is what makes turns and tools one dashboard object.
    trace_id: String,

    /// This observation's own id, the span id formatted the way the
    /// ingestion API accepts it.
    ///
    /// `None` for the run span, which renders as the trace itself and so
    /// has no observation to name. Child spans read it as their parent's
    /// [`parent_observation_id`](Self::parent_observation_id), which is
    /// how the Langfuse tree reproduces the tracing tree.
    observation_id: Option<String>,

    /// The enclosing observation's id, `None` for a root.
    ///
    /// Taken from the parent span's metadata at open. The engine
    /// dispatches tool spans from the run scope rather than from inside
    /// their turn, so a tool's parent here is the run — which is why the
    /// dashboard shows tools beside turns under the run instead of nested
    /// in the turn that asked for them.
    parent_observation_id: Option<String>,

    /// Display name for the rendered Langfuse object.
    ///
    /// The span's own name (`agent.turn`, `agent.tool`), carried
    /// through unchanged so a dashboard row matches the trace tree.
    name: String,

    /// Langfuse session key, filled when a `session_id` field rode the
    /// span.
    ///
    /// Absent unless the host records one — tracing does not invent a
    /// grouping a caller never asked for — and it keys the dashboard's
    /// session view, which is where runs of one conversation line up
    /// beside each other.
    session_id: Option<String>,

    /// Model name the span recorded, from `gen_ai.request.model`.
    ///
    /// Observations render it in their dedicated `model` field; traces
    /// carry it inside their metadata object.
    model: Option<String>,

    /// The `gen_ai.usage.*` fields the span recorded, under their slot
    /// names.
    ///
    /// Five keys are possible — `input`, `output`, `cached_input`,
    /// `cache_write`, and `reasoning` — and only the ones a provider
    /// actually reported are present, so a reader can tell an unreported
    /// figure from a zero one instead of exporting the difference as a
    /// measurement.
    usage: BTreeMap<String, serde_json::Value>,

    /// Every other field the span recorded, JSON-rendered.
    ///
    /// Key order comes from the `BTreeMap`, so the same fields render in
    /// the same order every run and the metadata object can be compared
    /// as it stands. Failure vocabulary stays visible here even though
    /// it also sets [`SpanMeta::errored`]: the recorded `error.type` text
    /// is what a dashboard reads, not just the flag.
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
    /// metadata for dashboards. The turn span's `query` maps to the
    /// suppressed slot and is dropped: the assembled prompt never rides
    /// the export. Suppression is scoped to the turn span — a `query`
    /// recorded on any other span is that span's own field (a search
    /// tool's argument, say) and lands in metadata like any other key.
    /// Every other key lands in metadata under its emitted name.
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
            Some(LangfuseField::Suppressed) => {
                if self.name != mapping::TURN_SPAN {
                    self.metadata.insert(key.to_string(), value);
                }
            }
            None => {
                self.metadata.insert(key.to_string(), value);
            }
        }
    }
}

/// Field visitor collecting a span's fields as JSON values.
///
/// Strings recorded through `Debug` have their wrapping quotes trimmed,
/// so a field rendered from a `&str` lands in metadata as the text a
/// reader expects rather than a quoted literal. Numbers and booleans
/// keep their JSON types rather than being rendered, which is what lets
/// the collector answer questions about a numeric field without parsing
/// every value it reads.
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
                    for envelope in crate::scores::score_events(&meta.trace_id, &agent) {
                        self.sink.push(envelope);
                    }
                }
            }
            LangfuseTarget::Observation => {
                let generation = meta.name == mapping::TURN_SPAN;
                let mut body = serde_json::json!({
                    "id": meta.observation_id,
                    "traceId": meta.trace_id,
                    "name": meta.name,
                    "startTime": ingest::rfc3339(meta.start),
                    "endTime": ingest::rfc3339_now(),
                    "metadata": serde_json::Value::Object(
                        serde_json::Map::<String, serde_json::Value>::from_iter(meta.metadata),
                    ),
                });
                if !generation {
                    ingest::set(
                        &mut body,
                        "type",
                        serde_json::Value::String("span".to_string()),
                    );
                }
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
                self.sink.push(ingest::envelope(
                    if generation {
                        "generation-create"
                    } else {
                        "span-create"
                    },
                    &body,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::missing_panics_doc)]

    use std::time::SystemTime;

    use serde_json::json;

    use super::*;

    fn meta_for(name: &str) -> SpanMeta {
        SpanMeta {
            target: mapping::classify_span(name),
            trace_id: "trace-a".to_string(),
            observation_id: Some("obs-a".to_string()),
            parent_observation_id: None,
            name: name.to_string(),
            session_id: None,
            model: None,
            usage: BTreeMap::new(),
            metadata: BTreeMap::new(),
            errored: false,
            start: SystemTime::now(),
        }
    }

    #[test]
    fn a_query_recorded_on_the_turn_span_is_dropped() {
        let mut meta = meta_for(mapping::TURN_SPAN);
        meta.absorb("query", json!("the assembled prompt"));
        assert!(
            meta.metadata.is_empty(),
            "the assembled prompt and its PII must never ride the export"
        );
    }

    #[test]
    fn a_query_recorded_on_another_span_stays_in_metadata() {
        let mut meta = meta_for(mapping::TOOL_SPAN);
        meta.absorb("query", json!("rust mutex guide"));
        assert_eq!(
            meta.metadata.get("query"),
            Some(&json!("rust mutex guide")),
            "a query on a non-turn span is that span's own vocabulary, \
             not the prompt — dropping it would lose a host's metadata"
        );
    }
}

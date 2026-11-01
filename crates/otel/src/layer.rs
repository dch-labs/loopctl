//! The `OTel` bridge: stock `tracing-opentelemetry` layer + close transform.
//!
//! [`tracing_opentelemetry::layer`] does span lifecycle, parenting, and
//! field-to-attribute copying verbatim — exactly right for emitters
//! that already carry spec-shaped attribute names. This module adds the
//! two things the bridge must do around span close:
//!
//! * [`SpecExporter`] renames per the `GenAI` conventions (`invoke_agent`,
//!   `execute_tool <x>`), sets `OTel` status from the recorded
//!   `outcome`/`error.type` fields, and derives `gen_ai.system` on the
//!   run span;
//! * [`OtelTracingLayer`] — when it carries a [`Stamper`] — writes the
//!   run rollup (and, in `SpanAttributes` mode, the subsystem registry)
//!   onto the run span's `OTel` span at the run-end record, the moment
//!   the engine writes the run's outcome.
//!
//! Metric-target events never reach `OTel` span events:
//! [`OtelTracingLayer`] drops them at its own `on_event` so the metrics
//! path stays the single owner of those signals.

use std::borrow::Cow;
use std::sync::Arc;

use loopctl::metrics::{AgentMetrics, MetricsObserver, MetricsRegistry};
use opentelemetry::trace::{Status, TraceContextExt as _};
use opentelemetry::{KeyValue, Value};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::SpanData;
use tracing::Subscriber;
use tracing::span::Attributes;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

use crate::mapping;
use crate::metrics::{registry_attributes, run_attributes};
use loopctl::metrics::RegistrySnapshot;

/// The target the metrics layer's emissions use; never an `OTel` span event.
///
/// Metric events are the metrics path's own signals; the composed layer
/// filters this target at its `on_event` so they never mirror into span
/// events, which keeps one signal from being reported twice.
const METRICS_TARGET: &str = "loopctl::metrics";

/// Registry series attributes a single run span will carry.
///
/// The `OTel` SDK caps a span at 128 attributes and drops later
/// additions silently, so the stamp bounds its registry share: the run
/// rollup (≈21–30 attributes), the engine's own span fields, and the
/// five diagnostics that always ride stay inside the budget, and a
/// large registry cannot push them off the span. The bound takes the
/// snapshot's own sorted order, so two identical registries stamp
/// identically.
const MAX_STAMPED_SERIES_ATTRIBUTES: usize = 64;

/// Run-rollup attributes a single run span will carry.
///
/// The same 128-attribute SDK cap that bounds the registry share bounds
/// the rollup: per-tool figures grow with the tool count (up to four
/// attributes each), and an unbounded rollup would push the run span's
/// own fields and the always-appended `loopctl.telemetry.*` diagnostics
/// past what the SDK keeps — the SDK drops later additions silently.
/// The budget leaves room for the whole registry share
/// ([`MAX_STAMPED_SERIES_ATTRIBUTES`] plus its five diagnostics) and the
/// handful of fields an `agent.run` span carries, so no tool count can
/// cost the diagnostics their seats. Only the run-span copy is trimmed:
/// [`export_metrics`](crate::export_metrics) still ships every tool's
/// full detail on the metrics path.
const MAX_STAMPED_ROLLUP_ATTRIBUTES: usize = 48;

/// The registry share of a run-span stamp, bounded.
///
/// Series attributes ride in the snapshot's deterministic order up to
/// [`MAX_STAMPED_SERIES_ATTRIBUTES`]; the `loopctl.telemetry.*`
/// diagnostics are re-appended when the cut lands on them, because
/// drift visibility is the one part of the registry share a span
/// should never lose to size.
fn bounded_registry_attributes(snapshot: &RegistrySnapshot) -> Vec<KeyValue> {
    let flattened = registry_attributes(snapshot);
    let mut bounded: Vec<KeyValue> =
        Vec::with_capacity(MAX_STAMPED_SERIES_ATTRIBUTES.saturating_add(5));
    let mut diagnostics = Vec::new();
    for kv in flattened {
        if kv.key.as_str().starts_with("loopctl.telemetry.") {
            diagnostics.push(kv);
        } else if bounded.len() < MAX_STAMPED_SERIES_ATTRIBUTES {
            bounded.push(kv);
        }
    }
    bounded.extend(diagnostics);
    bounded
}

/// The run-rollup share of a run-span stamp, bounded by whole tool groups.
///
/// The base run figures — headline counters and derived rates — always
/// ride; tool groups ride in the rollup's own order until
/// [`MAX_STAMPED_ROLLUP_ATTRIBUTES`] is spent, and a group that would
/// not fit whole is skipped whole, because a half-stamped tool reads as
/// a tool that never errored. The cut is deterministic: the same rollup
/// always stamps the same attributes.
fn bounded_run_attributes(agent: &AgentMetrics) -> Vec<KeyValue> {
    let emitted = run_attributes(agent);
    let base_end = emitted
        .iter()
        .position(|kv| kv.key.as_str().starts_with("loopctl.tool."))
        .unwrap_or(emitted.len());
    let (base, tools) = emitted.split_at(base_end);
    let mut bounded: Vec<KeyValue> = base.to_vec();
    let mut rest = tools;
    while let Some(first) = rest.first() {
        let group_key = tool_group_key(first.key.as_str());
        let group_end = rest
            .iter()
            .skip(1)
            .position(|kv| tool_group_key(kv.key.as_str()) != group_key)
            .map_or(rest.len(), |found| found.saturating_add(1));
        let (group, tail) = rest.split_at(group_end);
        if bounded.len().saturating_add(group.len()) > MAX_STAMPED_ROLLUP_ATTRIBUTES {
            break;
        }
        bounded.extend_from_slice(group);
        rest = tail;
    }
    bounded
}

/// The group an attribute belongs to: its key without a trailing
/// overflow marker and without the final segment.
///
/// Tool rollup keys are `loopctl.tool.<tool>.<figure>`, so the head
/// identifies the tool; stripping only the last segment keeps a dotted
/// tool name inside one group. An overflowed percentile emits a boolean
/// `<figure>.overflow` marker beside the reading it qualifies, and that
/// marker must not peel off into a group of its own — `p99_ms.overflow`
/// drops the marker first, then the figure, so the honesty signal rides
/// or drops with the tool it belongs to.
fn tool_group_key(key: &str) -> &str {
    let stem = key.strip_suffix(".overflow").unwrap_or(key);
    stem.rsplit_once('.').map_or(stem, |head| head.0)
}

/// Sources the bridge polls when a run span closes.
///
/// Bundles everything the run-end stamp needs: the optional session
/// rollup source stamped onto the run span, and — in
/// `SpanAttributes` mode — the subsystem registry whose series
/// flatten alongside it. Built from
/// [`OtelConfig`](crate::OtelConfig) whenever an observer or a
/// registry is configured, and consumed by [`OtelTracingLayer`],
/// which polls it when the engine records the run's outcome — after
/// that run's observer callbacks, so the stamp reads the span's
/// **own** run's rollup.
#[derive(Clone)]
pub struct Stamper {
    /// Session rollup source (the metrics observer), when one exists.
    ///
    /// Snapshotted at the run-end record — the closing run is the last
    /// run included. A registry-only stamper leaves this `None` and
    /// stamps nothing but its series.
    pub observer: Option<Arc<MetricsObserver>>,

    /// Subsystem registry, for `SpanAttributes` mode.
    ///
    /// Carried always; consulted (and its snapshot flattened onto the
    /// run span) only when [`Stamper::span_attributes`] is set.
    pub registry: Option<Arc<MetricsRegistry>>,

    /// Whether registry series ride onto the run span.
    ///
    /// True under [`OtelMetricsMode::SpanAttributes`](crate::OtelMetricsMode::SpanAttributes);
    /// kept as a plain flag so the close path never re-derives the mode
    /// from config.
    pub span_attributes: bool,
}

impl std::fmt::Debug for Stamper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stamper")
            .field(
                "registry",
                &self.registry.as_ref().map(|_| "MetricsRegistry"),
            )
            .field("span_attributes", &self.span_attributes)
            .finish_non_exhaustive()
    }
}

/// Exporter wrapper applying the loopctl transform before transport.
///
/// Implements the `OTel` SDK `SpanExporter` contract by transforming every
/// [`SpanData`] in the batch, then delegating — so it composes with the
/// batch-configured pipelines the SDK builds. The transform is the
/// rename/re-kind/status pass; the run rollup is written onto the `OTel`
/// span at the run-end record by [`OtelTracingLayer`] carrying a
/// [`Stamper`].
pub struct SpecExporter<E> {
    /// The transport this exporter transforms for.
    ///
    /// Every transformed [`SpanData`] is handed to it unchanged, which is
    /// what lets one transform ride an in-memory recorder in tests and an
    /// `OTLP` exporter in production without either learning that the
    /// loopctl vocabulary exists. Flush, shutdown, and the `Resource` a
    /// provider hands down at build time are forwarded rather than
    /// reimplemented, so batching and the resource stay the inner
    /// exporter's.
    inner: E,
}

impl<E: std::fmt::Debug> std::fmt::Debug for SpecExporter<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpecExporter")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<E> SpecExporter<E> {
    /// Wrap `inner`, applying the rename, re-kind, and status transform
    /// to every exported span.
    ///
    /// Loopctl-named spans leave with their `GenAI` names and `OTel`
    /// statuses; spans outside the vocabulary pass through untouched.
    /// The run rollup is not this exporter's to add — see
    /// [`OtelTracingLayer::with_stamper`].
    pub fn new(inner: E) -> Self {
        Self { inner }
    }
}

impl<E: opentelemetry_sdk::trace::SpanExporter> opentelemetry_sdk::trace::SpanExporter
    for SpecExporter<E>
{
    fn export(
        &self,
        mut batch: Vec<SpanData>,
    ) -> impl std::future::Future<Output = OTelSdkResult> + Send {
        for data in &mut batch {
            transform(data);
        }
        self.inner.export(batch)
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn shutdown(&self) -> OTelSdkResult {
        self.inner.shutdown()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

/// Rename, re-kind, and set status on one exported span.
///
/// Loopctl-named spans get their `GenAI` name and kind, and their
/// `OTel` status derives from the recorded `outcome`/`error.type`
/// fields. The run span additionally gains the derived `gen_ai.system`.
/// Spans outside the loopctl vocabulary return untouched.
fn transform(data: &mut SpanData) {
    let raw_name = data.name.to_string();
    if !mapping::is_loopctl_span(&raw_name) {
        return;
    }
    let tool = data
        .attributes
        .iter()
        .find(|attr| attr.key.as_str() == mapping::TOOL_NAME_KEY)
        .and_then(|attr| match &attr.value {
            Value::String(s) => Some(s.as_ref().to_string()),
            _ => None,
        });
    let fields = mapping::SpanFields { tool };
    data.name = Cow::Owned(mapping::otel_span_name(&raw_name, &fields));
    data.span_kind = mapping::otel_span_kind(&raw_name);

    let outcome = string_attr(&data.attributes, mapping::OUTCOME_KEY);
    let error_type = string_attr(&data.attributes, mapping::ERROR_TYPE_KEY);
    data.status = match (outcome.as_deref(), error_type) {
        (Some("error"), Some(err)) => Status::error(err),
        (Some("error"), None) => Status::error("error"),
        (Some("ok"), _) => Status::Ok,
        _ => data.status.clone(),
    };

    if raw_name == mapping::AGENT_RUN
        && let Some(model) = string_attr(&data.attributes, mapping::REQUEST_MODEL_KEY)
        && string_attr(&data.attributes, mapping::GEN_AI_SYSTEM).is_none()
    {
        data.attributes.push(KeyValue::new(
            mapping::GEN_AI_SYSTEM,
            mapping::map_system(&model),
        ));
    }
}

/// The string value carried under `key`, if the attribute holds one.
///
/// Non-string values answer `None`, so a numeric or boolean
/// attribute can never masquerade as a name or an outcome.
fn string_attr(attrs: &[KeyValue], key: &str) -> Option<String> {
    attrs
        .iter()
        .find(|attr| attr.key.as_str() == key)
        .and_then(|attr| match &attr.value {
            Value::String(s) => Some(s.as_ref().to_string()),
            _ => None,
        })
}

/// Whether an event belongs to the metrics path rather than to tracing.
///
/// Events on `loopctl::metrics` are the signals the metric pipeline
/// consumes; mirroring them through the bridge would publish each one
/// twice — once as a metric, once as a span event on whatever span
/// happened to be current — so they are dropped here and everything else
/// delegates to the stock bridge.
fn metrics_event(event: &tracing::Event<'_>) -> bool {
    event.metadata().target() == METRICS_TARGET
}

/// The composed `OTel` layer, boxed for `Registry` subscribers.
///
/// The stock bridge's generic parameter is pinned to
/// [`tracing_subscriber::registry::Registry`] — the shape `fmt`-style
/// subscribers layer onto, and every stock host's subscriber. With a
/// [`Stamper`] attached it also writes the run rollup onto the run
/// span's `OTel` span at the run-end record (see
/// [`with_stamper`](Self::with_stamper)).
pub struct OtelTracingLayer {
    /// The stock `tracing-opentelemetry` bridge, boxed.
    ///
    /// Boxing keeps this layer's own type free of the bridge's generic
    /// parameters, so composing it onto a subscriber stays `Send + Sync`
    /// without a cascade of type arguments at every call site; the bridge
    /// is reached through the box.
    inner: Box<dyn Layer<Registry> + Send + Sync>,

    /// Run-end stamp sources for the run-span rollup.
    ///
    /// `None` stamps nothing: without sources there is no rollup to
    /// publish, and the layer stays a pure span bridge. When present,
    /// the stamp fires exactly once per run span, at the engine's
    /// run-end record.
    stamper: Option<Arc<Stamper>>,

    /// The provider whose tracer the bridge records through.
    ///
    /// Owned rather than borrowed so [`shutdown`](Self::shutdown) can
    /// flush and stop the exact provider this layer installed, with no
    /// second handle for the host to keep alive alongside it and no
    /// teardown ordering it could get wrong.
    tracer_provider: opentelemetry_sdk::trace::SdkTracerProvider,

    /// The pipeline's meter provider, present in
    /// [`Pipeline`](crate::OtelMetricsMode) mode.
    ///
    /// No span is recorded through it — metrics reach it through
    /// [`crate::export_metrics`] — but it is held beside the tracer so one
    /// [`shutdown`](Self::shutdown) stops both halves of a pipeline, and a
    /// host cannot be left with traces flushed and a metric reader still
    /// running.
    meter_provider: Option<opentelemetry_sdk::metrics::SdkMeterProvider>,

    /// The pipeline handle this layer's build created, when it built a
    /// meter.
    ///
    /// Shutdown marks **this** handle — the one its own install built,
    /// whether or not it won the process-wide slot — so a trace-only
    /// layer (which holds `None`) or a pipeline whose meter lost the
    /// set-once race cannot mark another pipeline's exports dead.
    installed_meter: Option<Arc<crate::metrics::MeterHandle>>,
}

use tracing_subscriber::Registry;

impl std::fmt::Debug for OtelTracingLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtelTracingLayer")
            .field(
                "meter_provider",
                &self.meter_provider.as_ref().map(|_| "SdkMeterProvider"),
            )
            .finish_non_exhaustive()
    }
}

impl OtelTracingLayer {
    /// Compose an already-built span layer into the loopctl shape.
    ///
    /// The inner layer is typically
    /// `tracing_opentelemetry::layer().with_tracer(..)`; it is boxed so
    /// the composed type stays `Send + Sync` without exposing the
    /// bridge's generics. `tracer_provider` names what
    /// [`shutdown`](Self::shutdown) will flush and stop, and
    /// `meter_provider` is retained only for
    /// [`crate::export_metrics`] — pass `None` wherever no metrics
    /// pipeline was built.
    #[must_use]
    pub fn new(
        inner: impl Layer<Registry> + Send + Sync + 'static,
        tracer_provider: opentelemetry_sdk::trace::SdkTracerProvider,
        meter_provider: Option<opentelemetry_sdk::metrics::SdkMeterProvider>,
    ) -> Self {
        Self {
            inner: Box::new(inner),
            stamper: None,
            tracer_provider,
            meter_provider,
            installed_meter: None,
        }
    }

    /// Record the pipeline handle this layer's build created.
    ///
    /// Called by the crate's constructors right after they install the
    /// meter: [`shutdown`](Self::shutdown) marks that handle — and only
    /// that handle — so the export pipeline ends exactly when the
    /// pipeline that owns it ends. A layer built without a meter keeps
    /// `None` and shuts no exports down.
    #[must_use]
    pub(crate) fn owning_meter(mut self, handle: Option<Arc<crate::metrics::MeterHandle>>) -> Self {
        self.installed_meter = handle;
        self
    }

    /// Stamp the run rollup onto every run span, from `stamper`'s sources.
    ///
    /// The stamp is written when the engine records the run's outcome on
    /// the `agent.run` span — the run-end boundary inside the span's
    /// scope, after that run's observer callbacks have landed — so each
    /// span carries **its own run's** rollup. Reading the observer at
    /// export time instead would snapshot a sustained observer at flush
    /// time and print every later run's numbers on this span.
    ///
    /// Mechanically, the layer resolves the span's already-built `OTel`
    /// span through the bridge's context (`context activation` is the
    /// stock default) and calls `set_attribute` on it directly — the
    /// one public seam that reaches the `OTel` span before export
    /// without reimplementing the bridge's lifecycle.
    #[must_use]
    pub fn with_stamper(self, stamper: Stamper) -> Self {
        Self {
            stamper: Some(Arc::new(stamper)),
            ..self
        }
    }

    /// Flush and shut the underlying providers down (graceful teardown for
    /// hosts that composed the layer themselves).
    ///
    /// # Errors
    ///
    /// [`OtelInstallError::Flush`](crate::OtelInstallError::Flush) when a
    /// provider's flush or shutdown fails; the providers are shut down
    /// either way.
    pub fn shutdown(self) -> Result<(), crate::OtelInstallError> {
        let mut first = Ok(());
        if let Some(handle) = &self.installed_meter {
            handle.mark_shut_down();
        }
        let flushed = self.tracer_provider.force_flush();
        let stopped = self.tracer_provider.shutdown();
        if let Err(err) = flushed.and(stopped) {
            first = Err(crate::OtelInstallError::Flush(err.to_string()));
        }
        if let Some(meter) = &self.meter_provider {
            let flush_result = meter
                .force_flush()
                .map_err(|err| crate::OtelInstallError::Flush(err.to_string()));
            let meter_result = meter
                .shutdown()
                .map_err(|err| crate::OtelInstallError::Flush(err.to_string()));
            if first.is_ok() {
                first = flush_result.and(meter_result);
            }
        }
        first
    }

    /// Write the run-end stamp if this record is a run span's outcome.
    ///
    /// The engine records `outcome` on the `agent.run` span exactly
    /// once, at the run-end boundary — inside the span's scope, after
    /// that run's observer callbacks — so the trigger is both precise
    /// and on time. The same boundary guarantees the read: the engine
    /// announces every run to its observers before it settles that run's
    /// span, so what the snapshot holds at this record is the run being
    /// stamped — and a run that fails before its first turn stamps its
    /// own empty rollup rather than the last completed run's. The stamp
    /// rides straight onto the span's `OTel` span through the bridge's
    /// activated context; nothing is stored for the exporter to guess
    /// later.
    fn stamp_run_end(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        ctx: &Context<'_, Registry>,
    ) {
        let Some(stamper) = &self.stamper else {
            return;
        };
        let is_run_span = ctx
            .span(id)
            .is_some_and(|span| span.metadata().name() == mapping::AGENT_RUN);
        if !is_run_span {
            return;
        }
        let mut probe = OutcomeProbe(false);
        values.record(&mut probe);
        if !probe.0 {
            return;
        }
        tracing::dispatcher::get_default(|dispatch| {
            let Some(otel_cx) = tracing_opentelemetry::get_otel_context(id, dispatch) else {
                return;
            };
            let otel_span = otel_cx.span();
            if let Some(observer) = &stamper.observer {
                for kv in bounded_run_attributes(&observer.snapshot()) {
                    otel_span.set_attribute(kv);
                }
            }
            if stamper.span_attributes
                && let Some(registry) = &stamper.registry
            {
                for kv in bounded_registry_attributes(&registry.snapshot()) {
                    otel_span.set_attribute(kv);
                }
            }
        });
    }
}

/// Marker that the composed layer requires `LookupSpan`-capable hosts.
///
/// A compile-time-only assertion: it instantiates a bound check proving
/// the pinned host type satisfies the stock bridge's registry
/// requirements, so a future generalization cannot compile away the
/// capability the layer depends on. Nothing reaches production code.
const _: fn() = || {
    fn assert_layer<S: Subscriber + for<'a> LookupSpan<'a>>() {}
    let _ = assert_layer::<Registry>;
};

/// Field probe answering whether a record carries the run's outcome.
///
/// `Record` exposes its fields only through a visitor; one boolean is
/// all the stamp trigger needs.
struct OutcomeProbe(bool);

impl tracing::field::Visit for OutcomeProbe {
    fn record_debug(&mut self, field: &tracing::field::Field, _: &dyn core::fmt::Debug) {
        if field.name() == mapping::OUTCOME_KEY {
            self.0 = true;
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, _: &str) {
        if field.name() == mapping::OUTCOME_KEY {
            self.0 = true;
        }
    }
}

impl Layer<Registry> for OtelTracingLayer {
    fn on_new_span(
        &self,
        attrs: &Attributes<'_>,
        id: &tracing::span::Id,
        ctx: Context<'_, Registry>,
    ) {
        self.inner.on_new_span(attrs, id, ctx);
    }

    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        ctx: Context<'_, Registry>,
    ) {
        self.stamp_run_end(id, values, &ctx);
        self.inner.on_record(id, values, ctx);
    }

    fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<'_, Registry>) {
        if metrics_event(event) {
            return;
        }
        self.inner.on_event(event, ctx);
    }

    fn on_enter(&self, id: &tracing::span::Id, ctx: Context<'_, Registry>) {
        self.inner.on_enter(id, ctx);
    }

    fn on_exit(&self, id: &tracing::span::Id, ctx: Context<'_, Registry>) {
        self.inner.on_exit(id, ctx);
    }

    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, Registry>) {
        self.inner.on_close(id, ctx);
    }

    unsafe fn downcast_raw(&self, id: std::any::TypeId) -> Option<*const ()> {
        if id == std::any::TypeId::of::<Self>() {
            return Some(std::ptr::from_ref(self).cast::<()>());
        }
        // SAFETY: the inner layer is alive for as long as `self` is, and
        // the pointer it returns borrows from it — the same lifetime the
        // caller already holds on this wrapper.
        unsafe { self.inner.downcast_raw(id) }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::missing_panics_doc, clippy::expect_used)]

    use super::*;

    /// The stamp's attributes as `(key, debug-value)` pairs, for comparing
    /// what a bounded stamp retained against what it dropped.
    fn stamped(attrs: &[KeyValue]) -> Vec<(&str, String)> {
        attrs
            .iter()
            .map(|kv| (kv.key.as_str(), format!("{:?}", kv.value)))
            .collect()
    }

    /// A snapshot past the stamp bound keeps its diagnostics and loses
    /// only series attributes — the SDK's 128-attribute span budget is
    /// shared with the run rollup and the engine's own fields, which an
    /// unbounded stamp would silently push off the span.
    #[test]
    fn registry_stamping_is_bounded_and_keeps_diagnostics() {
        let series = (0..200_u32).map(|index| {
            let mut labels = std::collections::BTreeMap::new();
            labels.insert("tool", format!("tool-{index}"));
            loopctl::metrics::SeriesSnapshot {
                name: "loopctl.tools.calls",
                kind: loopctl::metrics::MetricKind::Counter,
                labels,
                value: loopctl::metrics::SeriesValue::Counter { total: 1 },
            }
        });
        let snapshot = RegistrySnapshot {
            series: series.collect(),
            unknown_metric: 1,
            ..RegistrySnapshot::default()
        };
        let attrs = bounded_registry_attributes(&snapshot);
        let series_attrs = attrs
            .iter()
            .filter(|kv| kv.key.as_str().starts_with("loopctl.tools.calls"))
            .count();
        assert_eq!(
            series_attrs, MAX_STAMPED_SERIES_ATTRIBUTES,
            "the registry share of the stamp is bounded"
        );
        for diagnostic in [
            "loopctl.telemetry.unknown_metric",
            "loopctl.telemetry.unknown_labels",
            "loopctl.telemetry.missing_labels",
            "loopctl.telemetry.numeric_on_counter",
            "loopctl.telemetry.ambiguous_scalar",
        ] {
            assert!(
                attrs.iter().any(|kv| kv.key.as_str() == diagnostic),
                "drift visibility never yields its seat to size"
            );
        }
        let bounded_again = bounded_registry_attributes(&snapshot);
        assert_eq!(
            stamped(&attrs),
            stamped(&bounded_again),
            "the cut is deterministic in the snapshot's own order: the same \
             attributes ride, in order, with the same values — equal lengths \
             alone would let a different equal-size cut pass"
        );
    }

    #[test]
    fn a_rolled_up_run_stays_inside_its_attribute_budget() {
        let mut agent = AgentMetrics {
            total_turns: 3,
            ..Default::default()
        };
        for index in 0..40_u8 {
            let name = format!("tool{index:02}");
            let mut stats = loopctl::metrics::ToolStats::new(&name);
            stats.record(12, false);
            stats.record(30, true);
            agent.tools.insert(name, stats);
        }

        let attrs = bounded_run_attributes(&agent);
        assert!(
            attrs.len() <= MAX_STAMPED_ROLLUP_ATTRIBUTES,
            "the rollup share never exceeds its budget, whatever the tool count: {}",
            attrs.len()
        );
        assert!(
            attrs
                .iter()
                .any(|kv| kv.key.as_str() == "loopctl.run.turns"),
            "the base run figures always ride, whatever was cut"
        );
        let mut groups: Vec<&str> = Vec::new();
        for kv in &attrs {
            let key = kv.key.as_str();
            if !key.starts_with("loopctl.tool.") {
                continue;
            }
            let group = tool_group_key(key);
            if !groups.contains(&group) {
                groups.push(group);
            }
            if key.strip_suffix(".calls").is_some() {
                continue;
            }
            let calls = format!("{group}.calls");
            assert!(
                attrs.iter().any(|other| other.key.as_str() == calls),
                "a stamped tool carries its whole group — a tool stamped with \
                 errors but no calls reads as one that was never called: {calls}"
            );
        }
        assert!(
            groups.len() < 40,
            "with 40 tools the budget must bite, not silently exceed: {} groups rode",
            groups.len()
        );
    }

    #[test]
    fn a_tool_s_overflow_marker_stays_in_its_tool_s_group() {
        assert_eq!(
            tool_group_key("loopctl.tool.bash.p99_ms.overflow"),
            "loopctl.tool.bash",
            "the marker strips with its figure before the group split"
        );
        assert_eq!(
            tool_group_key("loopctl.tool.bash.calls"),
            "loopctl.tool.bash"
        );
        assert_eq!(
            tool_group_key("bare"),
            "bare",
            "a key with no dot is its own group"
        );

        // A tool whose every latency sample sat above the histogram's
        // last bound: its percentiles export as markers only, no values.
        let overflowing_tool = |name: &str| {
            let mut stats = loopctl::metrics::ToolStats::new(name);
            stats.record(4_000_000, false);
            stats
        };
        let mut lone = AgentMetrics {
            total_turns: 1,
            ..Default::default()
        };
        lone.tools
            .insert("bash".to_string(), overflowing_tool("bash"));
        let attrs = bounded_run_attributes(&lone);
        let has = |key: &str| attrs.iter().any(|kv| kv.key.as_str() == key);
        assert!(
            has("loopctl.tool.bash.p50_ms.overflow")
                && has("loopctl.tool.bash.p99_ms.overflow")
                && has("loopctl.tool.bash.calls"),
            "a lone overflowing tool rides whole: the markers are the honest \
             reading, and they group with the tool's own counters"
        );

        // Under budget pressure the invariant that matters: no marker
        // rides without the rest of its tool's group.
        let mut crowded = AgentMetrics {
            total_turns: 3,
            ..Default::default()
        };
        for index in 0..40_u8 {
            let name = format!("tool{index:02}");
            let stats = overflowing_tool(&name);
            crowded.tools.insert(name, stats);
        }
        let attrs = bounded_run_attributes(&crowded);
        for kv in &attrs {
            let key = kv.key.as_str();
            if !key.ends_with(".overflow") {
                continue;
            }
            let calls = format!("{}.calls", tool_group_key(key));
            assert!(
                attrs.iter().any(|other| other.key.as_str() == calls),
                "the marker for {key} rode without its tool's counters — a \
                 group split before the marker was stripped"
            );
        }
    }

    #[test]
    fn shutdown_marks_only_the_pipeline_that_owns_the_meter() {
        use opentelemetry::metrics::MeterProvider as _;
        use opentelemetry_sdk::metrics::{
            InMemoryMetricExporter, PeriodicReader, SdkMeterProvider,
        };

        // Private pipelines, none touching the process-wide slot: the
        // handles are what the crate's own constructors hand their
        // layers and guards.
        let build_handle = |probe: &'static str| {
            let exporter = InMemoryMetricExporter::default();
            let reader = PeriodicReader::builder(exporter).build();
            let provider = SdkMeterProvider::builder().with_reader(reader).build();
            std::sync::Arc::new(crate::metrics::MeterHandle::new(provider.meter(probe)))
        };
        let build_layer = |handle: Option<std::sync::Arc<crate::metrics::MeterHandle>>| {
            let tracer = opentelemetry_sdk::trace::SdkTracerProvider::builder().build();
            OtelTracingLayer::new(crate::compose_layer_inner(&tracer), tracer, None)
                .owning_meter(handle)
        };
        let snapshot = loopctl::metrics::RegistrySnapshot::default();
        let owner = build_handle("owner");
        let stranger = build_handle("stranger");

        let trace_only = build_layer(None);
        trace_only
            .shutdown()
            .expect("a trace-only layer shuts its own tracer down");
        assert!(
            crate::metrics::record_registry(&owner, 0xA1, &snapshot),
            "a trace-only layer never owned a meter and ends no exports"
        );

        let loser = build_layer(Some(stranger));
        loser.shutdown().expect("a losing pipeline shuts down");
        assert!(
            crate::metrics::record_registry(&owner, 0xA2, &snapshot),
            "a pipeline whose meter lost the set-once race marks only its \
             own orphaned handle"
        );

        let winner = build_layer(Some(std::sync::Arc::clone(&owner)));
        winner.shutdown().expect("the owning pipeline shuts down");
        assert!(
            !crate::metrics::record_registry(&owner, 0xA3, &snapshot),
            "the owning pipeline's shutdown ends the exports its handle \
             fronts"
        );
    }

    /// An exporter that remembers the `Resource` it was handed.
    ///
    /// A provider passes its resource to each exporter through
    /// `set_resource`; this is the observable for whether one arrived at
    /// the transport underneath a wrapper.
    #[derive(Debug, Default)]
    struct ResourceSpy {
        seen: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    }

    impl opentelemetry_sdk::trace::SpanExporter for ResourceSpy {
        fn export(
            &self,
            _batch: Vec<SpanData>,
        ) -> impl std::future::Future<Output = opentelemetry_sdk::error::OTelSdkResult> + Send
        {
            std::future::ready(Ok(()))
        }

        fn set_resource(&mut self, resource: &Resource) {
            let name = resource
                .get(&opentelemetry::Key::from("service.name"))
                .map(|value| value.as_str().to_string());
            *self.seen.lock().expect("an uncontested spy") = name;
        }
    }

    #[test]
    fn a_wrapped_exporter_still_receives_the_providers_resource() {
        use opentelemetry_sdk::trace::SpanExporter as _;

        let spy = ResourceSpy::default();
        let seen = std::sync::Arc::clone(&spy.seen);
        let mut exporter = SpecExporter::new(spy);
        exporter.set_resource(&Resource::builder().with_service_name("loopctl").build());
        assert_eq!(
            seen.lock().expect("an uncontested spy").as_deref(),
            Some("loopctl"),
            "a wrapper that leaves set_resource to the trait's no-op default \
             exports to an unnamed service: the batch still arrives, and the \
             backend files it under a placeholder no one looks in"
        );
    }

    /// A span exporter whose export always fails, recording that its
    /// shutdown ran.
    ///
    /// The shutdown pin needs a flush that fails while the provider is
    /// still required to stop: the flag is the observable for "shutdown
    /// ran anyway", which is the behavior a failed flush used to skip.
    #[derive(Debug)]
    struct FailingSpanExporter {
        stopped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl opentelemetry_sdk::trace::SpanExporter for FailingSpanExporter {
        fn export(
            &self,
            _batch: Vec<SpanData>,
        ) -> impl std::future::Future<Output = opentelemetry_sdk::error::OTelSdkResult> + Send
        {
            std::future::ready(Err(
                opentelemetry_sdk::error::OTelSdkError::InternalFailure(
                    "planted flush failure".to_string(),
                ),
            ))
        }

        fn shutdown(&self) -> opentelemetry_sdk::error::OTelSdkResult {
            self.stopped
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    /// A metric exporter in the same shape: export fails, shutdown is
    /// observable.
    struct FailingMetricExporter {
        stopped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl opentelemetry_sdk::metrics::exporter::PushMetricExporter for FailingMetricExporter {
        fn export(
            &self,
            _metrics: &opentelemetry_sdk::metrics::data::ResourceMetrics,
        ) -> impl std::future::Future<Output = opentelemetry_sdk::error::OTelSdkResult> + Send
        {
            std::future::ready(Err(
                opentelemetry_sdk::error::OTelSdkError::InternalFailure(
                    "planted flush failure".to_string(),
                ),
            ))
        }

        fn force_flush(&self) -> opentelemetry_sdk::error::OTelSdkResult {
            Ok(())
        }

        fn shutdown_with_timeout(
            &self,
            _timeout: std::time::Duration,
        ) -> opentelemetry_sdk::error::OTelSdkResult {
            self.stopped
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        fn temporality(&self) -> opentelemetry_sdk::metrics::Temporality {
            opentelemetry_sdk::metrics::Temporality::Cumulative
        }
    }

    #[test]
    fn a_failed_flush_still_stops_the_providers_and_reports_it() {
        use opentelemetry::metrics::MeterProvider as _;
        use opentelemetry::trace::Tracer as _;
        use opentelemetry::trace::TracerProvider as _;
        use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
        use opentelemetry_sdk::trace::SdkTracerProvider;

        let span_stopped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let meter_stopped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        let tracer_provider = SdkTracerProvider::builder()
            .with_batch_exporter(FailingSpanExporter {
                stopped: std::sync::Arc::clone(&span_stopped),
            })
            .build();
        let meter_provider = {
            let reader = PeriodicReader::builder(FailingMetricExporter {
                stopped: std::sync::Arc::clone(&meter_stopped),
            })
            .build();
            SdkMeterProvider::builder().with_reader(reader).build()
        };

        // One span and one counter, so each flush has work to carry and
        // meets the planted failure.
        let tracer = tracer_provider.tracer("probe");
        tracer.in_span("flush-me", |_cx| {});
        meter_provider
            .meter("probe")
            .u64_counter("probe.calls")
            .build()
            .add(1, &[]);

        let layer = OtelTracingLayer::new(
            crate::compose_layer_inner(&tracer_provider),
            tracer_provider,
            Some(meter_provider),
        );
        let outcome = layer.shutdown();
        assert!(
            matches!(&outcome, Err(crate::OtelInstallError::Flush(message)) if message.contains("planted")),
            "the flush failure is what the caller hears, got {outcome:?}"
        );
        assert!(
            span_stopped.load(std::sync::atomic::Ordering::SeqCst),
            "the tracer provider stops even when its flush failed"
        );
        assert!(
            meter_stopped.load(std::sync::atomic::Ordering::SeqCst),
            "the meter provider stops even when its flush failed"
        );
    }
}

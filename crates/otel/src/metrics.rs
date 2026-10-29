//! Metrics export — run rollups as span attributes, registry as `OTLP`.
//!
//! Two shapes, matching the two [`OtelMetricsMode`](crate::OtelMetricsMode)
//! settings:
//!
//! * **Span attributes** — [`run_attributes`] and [`registry_attributes`]
//!   flatten [`AgentMetrics`] and [`RegistrySnapshot`] into dotted-key
//!   attributes the bridge stamps onto the closing `invoke_agent` span.
//!   Dotted names are fine as span attributes; label-cardinality rules
//!   apply to metric labels only.
//! * **Pipeline** — [`export_metrics`](crate::export_metrics) replays a
//!   [`RegistrySnapshot`] into `OTLP` instruments: counters as monotonic sums
//!   (delta since last call), gauges as gauges, histograms reconstructed by
//!   recording each bucket's midpoint once per sample it holds.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use loopctl::metrics::{AgentMetrics, RegistrySnapshot, SeriesValue, session::LatencyPercentiles};
use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Gauge, Histogram, Meter};

/// Instruments cached by series name for the pipeline export.
///
/// One variant per registry series kind, mirroring
/// [`loopctl::metrics::MetricKind`]; the cache behind
/// [`MeterHandle::with_instrument`] keeps a series on a single
/// instrument across exports, which the SDK's accumulators require.
enum Instruments {
    /// A monotonic counter, exporting per-series deltas as increments.
    ///
    /// Built for counter-kind series; each export adds the growth since
    /// the previous one, so the instrument's running total tracks the
    /// registry's own accumulation.
    Counter(Counter<u64>),

    /// An instantaneous reading, exporting the last value recorded.
    ///
    /// Built for gauge-kind series; a gauge carries no delta state, so
    /// every export re-records the snapshot's current reading.
    Gauge(Gauge<f64>),

    /// A distribution, exporting replayed bucket-midpoint samples.
    ///
    /// Built for histogram-kind series; each export replays the
    /// snapshot's buckets through the instrument, bounded per bucket.
    Histogram(Histogram<f64>),
}

/// Handle installed process-wide while the metrics pipeline is active.
///
/// [`export_metrics`](crate::export_metrics) reaches the pipeline through
/// [`active_meter`] rather than through a handle it was handed, because a
/// metric export has none of its own: the host calls it from a ticker or
/// at run end and expects whatever pipeline is installed to serve. The
/// slot is set-once, so a second install in the same process exports
/// through the first one's meter.
pub(crate) struct MeterHandle {
    /// The SDK meter every cached instrument is built from.
    ///
    /// Held for the handle's lifetime so first-sight instrument
    /// creation and every later export go through one meter and scope.
    meter: Meter,

    /// The instrument cache and the per-series delta ledger.
    ///
    /// One lock covers both: an export resolves a series' instrument
    /// and its previous total together, so a concurrent export cannot
    /// interleave an instrument build with a delta read.
    state: Mutex<InstrumentState>,

    /// Whether the pipeline this handle fronts has been shut down.
    ///
    /// A shut-down meter provider still accepts measurements and
    /// exports none of them, so a caller that kept exporting would get
    /// `Ok` answers that deliver nothing.
    /// [`export_metrics`](crate::export_metrics) checks this and fails
    /// loudly instead; the shutdown paths set it through
    /// [`note_meter_shutdown`], whichever guard or layer owns the
    /// providers.
    shut_down: AtomicBool,
}

/// The reusable half of a pipeline installation: instruments and deltas.
///
/// Held apart from [`MeterHandle`] so the bookkeeping is constructible
/// on its own — the export tests seed a fresh ledger without installing
/// the process-wide meter handle.
#[derive(Default)]
struct InstrumentState {
    /// One instrument per series name, built on first sight.
    ///
    /// `OTLP` instruments carry label sets per reading, so one counter
    /// per name serves every label series of that name.
    instruments: Vec<(String, Instruments)>,

    /// Last exported total per canonical **series** identity.
    ///
    /// The key is [`series_key`] — name plus sorted `label=value` pairs —
    /// and never the name alone: two label series under one name must
    /// track their growth separately, or one series' increase silently
    /// swallows the other's delta. Registry snapshots are cumulative, so
    /// what an export records is this ledger's difference, and
    /// [`tracked_registry`](Self::tracked_registry) names the registry
    /// those differences belong to.
    counter_totals: Vec<(String, u64)>,

    /// Last replayed per-bucket counts per canonical series identity.
    ///
    /// Registry snapshots are cumulative; the replay hands the
    /// instrument only the samples added since the previous export, so
    /// an unchanged snapshot adds zero new samples instead of
    /// re-recording its whole distribution on every scrape — under
    /// cumulative temporality the re-records would inflate the
    /// exported counts and sums on each pass.
    histogram_counts: Vec<(String, Vec<u64>)>,

    /// Which registry the two ledgers were built from, by generation.
    ///
    /// A ledger only means "growth since the last export" against the
    /// registry that accumulated it. A replacement registry restarts
    /// counting; differencing its counts against the old ledger
    /// saturates every delta to zero until the newcomer outgrew its
    /// predecessor, dropping the samples it did record — so a changed
    /// identity clears both ledgers and the newcomer's first export
    /// replays its own counts whole. The identity is the registry's
    /// own
    /// [`generation`](loopctl::metrics::MetricsRegistry::generation),
    /// never its address: the allocator can hand a replacement the exact
    /// address a freed predecessor held, and a generation never repeats.
    tracked_registry: Option<u64>,
}

impl MeterHandle {
    /// Mark this pipeline shut down.
    ///
    /// Idempotent and sticky: installation is once-per-process, so no
    /// later handle replaces a marked one.
    pub(crate) fn mark_shut_down(&self) {
        self.shut_down.store(true, Ordering::SeqCst);
    }

    /// Whether this pipeline was marked shut down.
    ///
    /// Read by [`export_metrics`](crate::export_metrics) before it
    /// records anything: a `true` there is the difference between an
    /// export that fails loudly and one that answers `Ok` for
    /// measurements a shut-down provider will never deliver.
    pub(crate) fn is_shut_down(&self) -> bool {
        self.shut_down.load(Ordering::SeqCst)
    }

    /// Lock the instrument state, recovering from poison.
    ///
    /// The guarded data is cached instruments and a delta ledger —
    /// plain numbers, safe to keep using after any panic — so a
    /// poisoned lock never stalls every later export.
    fn lock(&self) -> MutexGuard<'_, InstrumentState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Ensure an instrument exists for `name`, then record through it.
    ///
    /// The first use of a name builds its instrument via `make` under
    /// the handle's lock; later uses reuse the cached one. `record`
    /// receives the instrument and the label set to attach to the
    /// reading.
    fn with_instrument(
        &self,
        name: &str,
        make: impl FnOnce(&Meter, String) -> Instruments,
        record: impl FnOnce(&mut Instruments, Vec<KeyValue>),
    ) {
        let mut state = self.lock();
        if !state.instruments.iter().any(|(key, _)| key == name) {
            state
                .instruments
                .push((name.to_string(), make(&self.meter, name.to_string())));
        }
        if let Some(entry) = state.instruments.iter_mut().find(|(key, _)| key == name) {
            record(&mut entry.1, Vec::new());
        }
    }

    /// Record one value on a cached f64 gauge named `name`.
    ///
    /// The run-rollup export path: every derived `loopctl.run.*`
    /// attribute becomes its own gauge reading through this call, so a
    /// host's dashboard sees the rollup as first-class series.
    pub(crate) fn with_gauge(&self, name: &str, value: f64) {
        self.with_instrument(
            name,
            |meter, key| Instruments::Gauge(meter.f64_gauge(key).build()),
            |instrument, labels| {
                if let Instruments::Gauge(gauge) = instrument {
                    gauge.record(value, &labels);
                }
            },
        );
    }
}

/// The process-wide pipeline handle; the first install wins.
///
/// [`OnceLock::set`] ignores later attempts, so a second pipeline
/// build leaves the first meter's instruments and delta ledger in
/// place — the set-once rule [`install_meter`] documents.
static ACTIVE_METER: OnceLock<Arc<MeterHandle>> = OnceLock::new();

/// Install the process-wide metrics pipeline handle.
///
/// `install()`/`otel_layer()` call this when a pipeline is configured;
/// hosts running their own `SdkMeterProvider` can hand a `Meter` over
/// directly. First installation wins; later ones are ignored.
pub fn install_meter(meter: Meter) {
    let handle = Arc::new(MeterHandle {
        meter,
        state: Mutex::new(InstrumentState::default()),
        shut_down: AtomicBool::new(false),
    });
    ACTIVE_METER.set(handle).ok();
}

/// The installed pipeline handle, if this process has one.
///
/// `None` before the first [`install_meter`] — or forever, in a
/// traces-only process; [`export_metrics`](crate::export_metrics)
/// reads this to answer its `NoMetricsPipeline` error.
pub(crate) fn active_meter() -> Option<Arc<MeterHandle>> {
    ACTIVE_METER.get().cloned()
}

/// Mark the installed pipeline, if any, as shut down.
///
/// Called by the shutdown paths — the guard's drop and
/// [`OtelTracingLayer::shutdown`](crate::OtelTracingLayer::shutdown) —
/// so a later [`export_metrics`](crate::export_metrics) fails loudly
/// instead of recording into a provider that exports nothing. The
/// tracer-only paths call it too: a pipeline whose tracer half was torn
/// down is no longer a live export target, whether or not a meter was
/// ever built.
pub(crate) fn note_meter_shutdown() {
    if let Some(handle) = ACTIVE_METER.get() {
        handle.mark_shut_down();
    }
}

/// Upper bound on the samples one bucket contributes to a replay.
///
/// Bucket counts arrive as cumulative totals, and replaying them means
/// recording that many individual observations. A pathological count — a
/// long-lived series, or an overflow bucket that accumulated most of the
/// distribution — would otherwise spend the export's whole budget
/// re-recording one bucket, so the excess is dropped: the export reports
/// less growth than happened, never more.
const BUCKET_REPLAY_CAP: u64 = 10_000;

/// Converts a wide unsigned integer to `f64` — exact below 2^53,
/// correctly rounded beyond.
///
/// The limb-composed construction (one fused `mul_add` rounding) keeps
/// exported magnitudes honest past `u32::MAX` — where a saturating
/// conversion would flatten distinct totals onto one ceiling — without
/// a lossy-cast lint to suppress. The core crate's metrics module uses
/// the same construction for its ratios.
fn wide_to_f64(value: u64) -> f64 {
    let hi = value.div_euclid(4_294_967_296);
    let lo = value.rem_euclid(4_294_967_296);
    f64::from(u32::try_from(hi).unwrap_or(u32::MAX)).mul_add(
        4_294_967_296.0,
        f64::from(u32::try_from(lo).unwrap_or(u32::MAX)),
    )
}

/// Converts a wide signed integer to `f64`, sign and magnitude both
/// preserved.
///
/// The `Value::I64` readings the run rollup carries are billing-scale
/// integers; the run-metric gauges re-record them, and a cap at
/// `i32::MAX` would make distinct large totals indistinguishable.
pub(crate) fn wide_i64_to_f64(value: i64) -> f64 {
    let magnitude = wide_to_f64(value.unsigned_abs());
    if value < 0 { -magnitude } else { magnitude }
}

/// Appends one length-prefixed part to a series key.
///
/// A part is its byte length in digits, a colon, then the bytes — a
/// prefix code, so a chain of parts parses back to exactly the parts
/// that built it and no label value (dot-laden, `=`-laden, or not) can
/// alias another series' key.
fn push_key_part(key: &mut String, part: &str) {
    use std::fmt::Write as _;
    write!(key, "{}:", part.len()).ok();
    key.push_str(part);
}

/// The canonical identity of one stored series: its name and label
/// parts, each length-prefixed.
///
/// Distinct series must key apart however their label values spell:
/// a dot-and-equals join lets a value like `read.outcome=ok` collide
/// with a genuinely richer label set, and one series' counter delta
/// would silently swallow the other's. The length prefixes make the
/// encoding collision-free by construction.
fn series_key(series: &loopctl::metrics::SeriesSnapshot) -> String {
    let mut key = String::new();
    push_key_part(&mut key, series.name);
    for (label, value) in &series.labels {
        push_key_part(&mut key, label);
        push_key_part(&mut key, value);
    }
    key
}

/// Replay a registry snapshot into the installed `OTLP` pipeline.
///
/// `registry_id` identifies which registry `snapshot` came from (the
/// caller passes its
/// [`generation`](loopctl::metrics::MetricsRegistry::generation)). The
/// delta ledgers belong to that one
/// registry, so a changing identity clears them: a replacement's lower
/// counts replay whole rather than saturating against a predecessor
/// they have no relation to — see
/// [`InstrumentState::tracked_registry`]. A caller exporting one
/// registry repeatedly passes the same id and sees pure deltas.
///
/// Counters export the delta since the previous call (registry totals are
/// cumulative, `OTLP` counters take increments); gauges export their current
/// value; histograms are reconstructed bucket by bucket; the registry's
/// diagnostics counters export as `loopctl.telemetry.*` gauges so store
/// drift is visible on dashboards.
pub(crate) fn record_registry(meter: &MeterHandle, registry_id: u64, snapshot: &RegistrySnapshot) {
    let mut guard = meter.lock();
    let state: &mut InstrumentState = &mut guard;
    if state.tracked_registry != Some(registry_id) {
        state.counter_totals.clear();
        state.histogram_counts.clear();
        state.tracked_registry = Some(registry_id);
    }
    let mut totals_updates: Vec<(String, u64)> = Vec::new();
    let mut histogram_updates: Vec<(String, Vec<u64>)> = Vec::new();
    for series in &snapshot.series {
        let labels: Vec<KeyValue> = series
            .labels
            .iter()
            .map(|(key, value)| KeyValue::new(*key, value.clone()))
            .collect();
        let name = series.name;
        if !state.instruments.iter().any(|(key, _)| key == name) {
            let instrument = match &series.value {
                SeriesValue::Counter { .. } => {
                    Instruments::Counter(meter.meter.u64_counter(name.to_string()).build())
                }
                SeriesValue::Gauge { .. } => {
                    Instruments::Gauge(meter.meter.f64_gauge(name.to_string()).build())
                }
                SeriesValue::Histogram { .. } => {
                    Instruments::Histogram(meter.meter.f64_histogram(name.to_string()).build())
                }
            };
            state.instruments.push((name.to_string(), instrument));
        }
        let Some(entry) = state.instruments.iter_mut().find(|(key, _)| key == name) else {
            continue;
        };
        match (&mut entry.1, &series.value) {
            (Instruments::Counter(counter), SeriesValue::Counter { total }) => {
                let key = series_key(series);
                let previous = state
                    .counter_totals
                    .iter()
                    .find(|(seen, _)| *seen == key)
                    .map_or(0, |(_, seen)| *seen);
                let delta = total.saturating_sub(previous);
                if delta > 0 {
                    counter.add(delta, &labels);
                }
                totals_updates.push((key, *total));
            }
            (Instruments::Gauge(gauge), SeriesValue::Gauge { value }) => {
                gauge.record(*value, &labels);
            }
            (Instruments::Histogram(histogram), SeriesValue::Histogram { summary }) => {
                let key = series_key(series);
                let previous = state
                    .histogram_counts
                    .iter()
                    .find(|(seen, _)| *seen == key)
                    .map_or_else(Vec::new, |(_, seen)| seen.clone());
                let deltas: Vec<u64> = summary
                    .counts
                    .iter()
                    .enumerate()
                    .map(|(slot, count)| {
                        count.saturating_sub(previous.get(slot).copied().unwrap_or(0))
                    })
                    .collect();
                replay_buckets(histogram, &labels, &summary.bounds, &deltas);
                histogram_updates.push((key, summary.counts.clone()));
            }
            _ => {}
        }
    }
    for (key, total) in totals_updates {
        match state
            .counter_totals
            .iter_mut()
            .find(|(seen, _)| *seen == key)
        {
            Some(seen) => seen.1 = total,
            None => state.counter_totals.push((key, total)),
        }
    }
    for (key, counts) in histogram_updates {
        match state
            .histogram_counts
            .iter_mut()
            .find(|(seen, _)| *seen == key)
        {
            Some(seen) => seen.1 = counts,
            None => state.histogram_counts.push((key, counts)),
        }
    }
    drop(guard);
    record_diagnostics(meter, snapshot);
}

/// Reconstruct a distribution's growth from explicit bucket counts by
/// recording the bucket midpoint once per **new** sample the bucket
/// gained.
///
/// The counts arriving here are per-export deltas (cumulative snapshot
/// counts minus what the previous export already replayed), so an
/// unchanged distribution records nothing. The final (overflow) bucket
/// has no upper bound, so its samples replay at the last known bound —
/// an honest underestimate, capped at [`BUCKET_REPLAY_CAP`] per bucket
/// so a pathological count cannot stall an export. Midpoint math rides
/// the magnitude-honest wide conversion, saturating adds, and euclid
/// division, so no step can overflow or panic.
fn replay_buckets(
    recorder: &mut Histogram<f64>,
    labels: &[KeyValue],
    bounds: &[u64],
    counts: &[u64],
) {
    for slot in 0..counts.len() {
        let Some(count) = counts.get(slot).copied() else {
            continue;
        };
        let midpoint = match (
            bounds.get(slot),
            slot.checked_sub(1).and_then(|i| bounds.get(i)),
        ) {
            (Some(hi), Some(lo)) => wide_to_f64(hi.saturating_add(*lo).div_euclid(2)),
            (Some(hi), None) => wide_to_f64(hi.div_euclid(2)),
            (None, Some(lo)) => wide_to_f64(*lo),
            (None, None) => 0.0,
        };
        let samples = count.min(BUCKET_REPLAY_CAP);
        for _ in 0..samples {
            recorder.record(midpoint, labels);
        }
    }
}

/// Registry diagnostics exported as `loopctl.telemetry.*` gauges.
///
/// Store drift — rejected names, undeclared or missing labels, counted
/// numerics, ambiguous scalars — becomes a dashboard-visible signal, so
/// the telemetry pipeline reports on its own health the same way it
/// reports on the agent's.
fn record_diagnostics(meter: &MeterHandle, snapshot: &RegistrySnapshot) {
    for (name, value) in [
        ("loopctl.telemetry.unknown_metric", snapshot.unknown_metric),
        ("loopctl.telemetry.unknown_labels", snapshot.unknown_labels),
        ("loopctl.telemetry.missing_labels", snapshot.missing_labels),
        (
            "loopctl.telemetry.numeric_on_counter",
            snapshot.numeric_on_counter,
        ),
        (
            "loopctl.telemetry.ambiguous_scalar",
            snapshot.ambiguous_scalar,
        ),
    ] {
        let value = wide_to_f64(value);
        meter.with_instrument(
            name,
            |m, key| Instruments::Gauge(m.f64_gauge(key).build()),
            |instrument, labels| {
                if let Instruments::Gauge(gauge) = instrument {
                    gauge.record(value, &labels);
                }
            },
        );
    }
}

/// Run rollup flattened to span attributes.
///
/// Flattens [`AgentMetrics`] into dotted `loopctl.run.*` and
/// `gen_ai.usage.*` keys: counters and latencies as integers, derived
/// rates only when their denominators were nonzero, latency percentiles
/// only when turns were recorded, and per-tool stats under
/// `loopctl.tool.<name>.*`. Stamped onto the closing `invoke_agent`
/// span at the run-end record by
/// [`OtelTracingLayer::with_stamper`](crate::layer::OtelTracingLayer::with_stamper)
/// (and, from a pipeline, re-recorded as gauges by
/// [`export_metrics`](crate::export_metrics)).
#[must_use]
pub fn run_attributes(agent: &AgentMetrics) -> Vec<KeyValue> {
    fn ms(value: u64) -> i64 {
        i64::try_from(value).unwrap_or(i64::MAX)
    }
    let mut attrs = vec![
        KeyValue::new("loopctl.run.success", agent.run_success),
        KeyValue::new("loopctl.run.duration_ms", ms(agent.run_duration_ms)),
        KeyValue::new("loopctl.run.turns", ms(agent.total_turns)),
        KeyValue::new("loopctl.run.failed_turns", ms(agent.failed_turns)),
        KeyValue::new("gen_ai.usage.input_tokens", ms(agent.tokens.input)),
        KeyValue::new("gen_ai.usage.output_tokens", ms(agent.tokens.output)),
        KeyValue::new(
            "gen_ai.usage.cached_input_tokens",
            ms(agent.tokens.cached_input),
        ),
        KeyValue::new(
            "gen_ai.usage.cache_write_tokens",
            ms(agent.tokens.cache_write),
        ),
        KeyValue::new("gen_ai.usage.reasoning_tokens", ms(agent.tokens.reasoning)),
        KeyValue::new("loopctl.run.stream_successes", ms(agent.stream_successes)),
        KeyValue::new("loopctl.run.stream_failures", ms(agent.stream_failures)),
        KeyValue::new("loopctl.run.compactions", ms(agent.compactions)),
        KeyValue::new("loopctl.run.fallbacks", ms(agent.fallbacks)),
        KeyValue::new(
            "loopctl.run.transport_fallbacks",
            ms(agent.transport_fallbacks),
        ),
        KeyValue::new("loopctl.run.model_switches", ms(agent.model_switches)),
        KeyValue::new("loopctl.run.budget_warnings", ms(agent.budget_warnings)),
        KeyValue::new("loopctl.run.loops_detected", ms(agent.loops_detected)),
        KeyValue::new(
            "loopctl.run.convergences_detected",
            ms(agent.convergences_detected),
        ),
    ];
    if let Some(rate) = agent.tokens.cache_hit_rate() {
        attrs.push(KeyValue::new("loopctl.run.cache_hit_rate", rate));
    }
    if let Some(share) = agent.tokens.reasoning_share() {
        attrs.push(KeyValue::new("loopctl.run.reasoning_share", share));
    }
    if let Some(LatencyPercentiles { p50, p90, p99 }) = agent.turn_latency_percentiles() {
        for (label, reading) in [("p50", p50), ("p90", p90), ("p99", p99)] {
            let key = format!("loopctl.run.turn_{label}_ms");
            push_percentile(&mut attrs, key, reading, ms);
        }
    }
    if let Some(error) = &agent.run_error {
        attrs.push(KeyValue::new("loopctl.run.error", error.clone()));
    }
    for stats in agent.tools.values() {
        let prefix = format!("loopctl.tool.{}", stats.tool);
        attrs.push(KeyValue::new(format!("{prefix}.calls"), ms(stats.calls)));
        if stats.errors > 0 {
            attrs.push(KeyValue::new(format!("{prefix}.errors"), ms(stats.errors)));
        }
        if let Some(percentiles) = stats.latency.percentiles() {
            for (label, reading) in [("p50", percentiles.p50), ("p99", percentiles.p99)] {
                let key = format!("{prefix}.{label}_ms");
                push_percentile(&mut attrs, key, reading, ms);
            }
        }
    }
    attrs
}

/// Files one percentile reading under `key`, omitting overflowed values.
///
/// A reading whose sample sat above the histogram's last bound is the
/// bound standing in for an unbounded sample — "at least this much",
/// not a measurement — so it never publishes as an ordinary finite
/// value; a boolean `<key>.overflow` marker rides instead, and a
/// dashboard can never read an exceeded objective as met.
fn push_percentile(
    attrs: &mut Vec<KeyValue>,
    key: String,
    reading: loopctl::metrics::Percentile,
    ms: impl Fn(u64) -> i64,
) {
    if reading.overflow {
        attrs.push(KeyValue::new(format!("{key}.overflow"), true));
    } else {
        attrs.push(KeyValue::new(key, ms(reading.value_ms)));
    }
}

/// Registry snapshot flattened to span attributes (`SpanAttributes` mode).
///
/// Keys are the series name plus sorted `label=value` pairs joined by dots;
/// histograms contribute a count and p50/p99 (where samples exist). The
/// registry's diagnostics counters ride along as `loopctl.telemetry.*` so
/// drift is visible on the trace itself.
#[must_use]
pub fn registry_attributes(snapshot: &RegistrySnapshot) -> Vec<KeyValue> {
    fn ms(value: u64) -> i64 {
        i64::try_from(value).unwrap_or(i64::MAX)
    }
    let mut attrs = Vec::with_capacity(snapshot.series.len().saturating_mul(2));
    for series in &snapshot.series {
        let mut key = String::from(series.name);
        for (label, value) in &series.labels {
            key.push('.');
            key.push_str(label);
            key.push('=');
            key.push_str(value);
        }
        match &series.value {
            SeriesValue::Counter { total } => attrs.push(KeyValue::new(key, ms(*total))),
            SeriesValue::Gauge { value } => attrs.push(KeyValue::new(key, *value)),
            SeriesValue::Histogram { summary } => {
                let count: u64 = summary.counts.iter().copied().sum();
                attrs.push(KeyValue::new(format!("{key}.count"), ms(count)));
                if let Some([p50, _p90, p99]) = summary.percentiles {
                    for (label, reading) in [("p50", p50), ("p99", p99)] {
                        let reading_key = format!("{key}.{label}");
                        push_percentile(&mut attrs, reading_key, reading, ms);
                    }
                }
            }
        }
    }
    attrs.push(KeyValue::new(
        "loopctl.telemetry.unknown_metric",
        ms(snapshot.unknown_metric),
    ));
    attrs.push(KeyValue::new(
        "loopctl.telemetry.unknown_labels",
        ms(snapshot.unknown_labels),
    ));
    attrs.push(KeyValue::new(
        "loopctl.telemetry.missing_labels",
        ms(snapshot.missing_labels),
    ));
    attrs.push(KeyValue::new(
        "loopctl.telemetry.numeric_on_counter",
        ms(snapshot.numeric_on_counter),
    ));
    attrs.push(KeyValue::new(
        "loopctl.telemetry.ambiguous_scalar",
        ms(snapshot.ambiguous_scalar),
    ));
    attrs
}

#[cfg(test)]
mod tests {
    #![allow(clippy::missing_panics_doc)]

    use loopctl::metrics::AgentMetrics;

    use super::*;

    const REGISTRY: u64 = 0xA1;
    const REPLACEMENT: u64 = 0xB2;

    #[test]
    fn exports_after_shutdown_report_it() {
        use opentelemetry::metrics::MeterProvider as _;
        use opentelemetry_sdk::metrics::{
            InMemoryMetricExporter, PeriodicReader, SdkMeterProvider,
        };

        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        install_meter(provider.meter("shutdown-probe"));
        let registry = loopctl::metrics::MetricsRegistry::default();
        assert!(
            crate::export_metrics(None, &registry).is_ok(),
            "a live pipeline accepts the export"
        );
        note_meter_shutdown();
        assert!(
            matches!(
                crate::export_metrics(None, &registry),
                Err(crate::OtelInstallError::PipelineShutDown)
            ),
            "an export after shutdown reports the dead pipeline instead \
             of returning Ok for measurements that cannot deliver"
        );
    }

    fn find<'a>(attrs: &'a [KeyValue], key: &str) -> Option<&'a opentelemetry::Value> {
        attrs
            .iter()
            .find(|attr| attr.key.as_str() == key)
            .map(|attr| &attr.value)
    }

    #[test]
    fn run_rollup_flattens_tokens_and_derived_rates() {
        let mut agent = AgentMetrics::default();
        agent.tokens.input = 100;
        agent.tokens.cached_input = 25;
        agent.total_turns = 3;
        agent.turn_latency.record(120);
        let attrs = run_attributes(&agent);
        assert_eq!(
            find(&attrs, "gen_ai.usage.input_tokens"),
            Some(&opentelemetry::Value::I64(100))
        );
        assert_eq!(
            find(&attrs, "loopctl.run.cache_hit_rate"),
            Some(&opentelemetry::Value::F64(0.25))
        );
        assert!(find(&attrs, "loopctl.run.turn_p50_ms").is_some());
    }

    #[test]
    fn overflowed_percentiles_omit_the_value_and_mark_the_overflow() {
        let mut exceeded = AgentMetrics {
            total_turns: 1,
            ..AgentMetrics::default()
        };
        exceeded.turn_latency.record(120_000);
        let attrs = run_attributes(&exceeded);
        assert!(
            find(&attrs, "loopctl.run.turn_p99_ms").is_none(),
            "an overflowed reading never publishes as an ordinary latency"
        );
        assert_eq!(
            find(&attrs, "loopctl.run.turn_p99_ms.overflow"),
            Some(&opentelemetry::Value::Bool(true)),
            "the marker says why the value is absent"
        );

        let mut bounded = AgentMetrics::default();
        bounded.turn_latency.record(120);
        let attrs = run_attributes(&bounded);
        assert!(find(&attrs, "loopctl.run.turn_p50_ms").is_some());
        assert!(
            find(&attrs, "loopctl.run.turn_p50_ms.overflow").is_none(),
            "a bounded reading carries no marker"
        );
    }

    #[test]
    fn wide_integers_keep_their_magnitude() {
        let cases = [
            (
                wide_to_f64(u64::from(u32::MAX).saturating_add(5)),
                f64::from(u32::MAX) + 5.0,
            ),
            (
                wide_i64_to_f64(i64::from(i32::MAX).saturating_add(5)),
                f64::from(i32::MAX) + 5.0,
            ),
            (wide_i64_to_f64(-5_000_000_000), -5_000_000_000.0),
            (wide_to_f64(1_000_000_000), 1_000_000_000.0),
        ];
        for (converted, wanted) in cases {
            assert!(
                (converted - wanted).abs() < 1e-6,
                "the conversion keeps the magnitude: {converted} vs {wanted}"
            );
        }
    }

    #[test]
    fn colliding_label_spellings_keep_their_deltas_apart() {
        use opentelemetry::metrics::MeterProvider as _;
        use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
        use opentelemetry_sdk::metrics::{
            InMemoryMetricExporter, PeriodicReader, SdkMeterProvider,
        };

        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let handle = MeterHandle {
            meter: provider.meter("collision-probe"),
            state: Mutex::new(InstrumentState::default()),
            shut_down: AtomicBool::new(false),
        };

        let rich = |tool: &str, outcome: &str, total: u64| loopctl::metrics::SeriesSnapshot {
            name: "loopctl.tools.calls",
            kind: loopctl::metrics::MetricKind::Counter,
            labels: std::collections::BTreeMap::from([
                ("tool", tool.to_string()),
                ("outcome", outcome.to_string()),
            ]),
            value: SeriesValue::Counter { total },
        };
        let sneaky = |tool: &str, total: u64| loopctl::metrics::SeriesSnapshot {
            name: "loopctl.tools.calls",
            kind: loopctl::metrics::MetricKind::Counter,
            labels: std::collections::BTreeMap::from([("tool", tool.to_string())]),
            value: SeriesValue::Counter { total },
        };
        let snapshot = |series: Vec<loopctl::metrics::SeriesSnapshot>| RegistrySnapshot {
            series,
            ..RegistrySnapshot::default()
        };
        let exported_totals = |probe: &InMemoryMetricExporter| -> Vec<(String, String, u64)> {
            let mut points = Vec::new();
            for resource in probe.get_finished_metrics().unwrap_or_default() {
                for scope in resource.scope_metrics() {
                    for metric in scope.metrics() {
                        if metric.name() != "loopctl.tools.calls" {
                            continue;
                        }
                        if let AggregatedMetrics::U64(MetricData::Sum(sum)) = metric.data() {
                            for point in sum.data_points() {
                                let tool = point
                                    .attributes()
                                    .find(|kv| kv.key.as_str() == "tool")
                                    .map(|kv| kv.value.as_str().into_owned())
                                    .unwrap_or_default();
                                let outcome = point
                                    .attributes()
                                    .find(|kv| kv.key.as_str() == "outcome")
                                    .map(|kv| kv.value.as_str().into_owned())
                                    .unwrap_or_default();
                                points.push((tool, outcome, point.value()));
                            }
                        }
                    }
                }
            }
            points
        };

        record_registry(
            &handle,
            REGISTRY,
            &snapshot(vec![rich("read", "ok", 3), sneaky("read.outcome=ok", 5)]),
        );
        assert!(provider.force_flush().is_ok(), "flush 1 lands");
        record_registry(
            &handle,
            REGISTRY,
            &snapshot(vec![rich("read", "ok", 4), sneaky("read.outcome=ok", 5)]),
        );
        assert!(provider.force_flush().is_ok(), "flush 2 lands");
        let points = exported_totals(&exporter);
        let rich_growth: Vec<u64> = points
            .iter()
            .filter(|(tool, outcome, _)| tool == "read" && outcome == "ok")
            .map(|(_, _, value)| *value)
            .collect();
        assert!(
            rich_growth.last() == Some(&4) || rich_growth.iter().sum::<u64>() == 4,
            "the richer label set's growth 3 → 4 exports (delta: the increment; \
             cumulative: the new total) — a colliding key swallows it: {points:?}"
        );
        let sneaky_last = points
            .iter()
            .filter(|(tool, outcome, _)| tool == "read.outcome=ok" && outcome.is_empty())
            .map(|(_, _, value)| *value)
            .next_back()
            .unwrap_or_default();
        assert_eq!(sneaky_last, 5, "the dot-laden series never drifts");
    }

    #[test]
    fn unchanged_histograms_replay_only_new_samples() {
        use opentelemetry::metrics::MeterProvider as _;
        use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
        use opentelemetry_sdk::metrics::{
            InMemoryMetricExporter, PeriodicReader, SdkMeterProvider,
        };

        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let handle = MeterHandle {
            meter: provider.meter("replay-probe"),
            state: Mutex::new(InstrumentState::default()),
            shut_down: AtomicBool::new(false),
        };

        let histogram_series = |first: u64, second: u64| {
            let mut histogram = loopctl::metrics::LatencyHistogram::with_buckets(vec![10, 100]);
            for _ in 0..first {
                histogram.record(5);
            }
            for _ in 0..second {
                histogram.record(50);
            }
            let counts = histogram.counts().to_vec();
            loopctl::metrics::SeriesSnapshot {
                name: "loopctl.vector.embed",
                kind: loopctl::metrics::MetricKind::Histogram,
                labels: std::collections::BTreeMap::new(),
                value: SeriesValue::Histogram {
                    summary: loopctl::metrics::HistogramSummary {
                        bounds: vec![10, 100],
                        counts,
                        percentiles: histogram.percentiles().map(|p| [p.p50, p.p90, p.p99]),
                    },
                },
            }
        };
        let snapshot = |series: loopctl::metrics::SeriesSnapshot| RegistrySnapshot {
            series: vec![series],
            ..RegistrySnapshot::default()
        };
        let exported_counts = |probe: &InMemoryMetricExporter| -> Vec<u64> {
            let mut counts = Vec::new();
            for resource in probe.get_finished_metrics().unwrap_or_default() {
                for scope in resource.scope_metrics() {
                    for metric in scope.metrics() {
                        if metric.name() != "loopctl.vector.embed" {
                            continue;
                        }
                        if let AggregatedMetrics::F64(MetricData::Histogram(histogram)) =
                            metric.data()
                        {
                            for point in histogram.data_points() {
                                counts.push(point.count());
                            }
                        }
                    }
                }
            }
            counts
        };

        record_registry(&handle, REGISTRY, &snapshot(histogram_series(1, 1)));
        assert!(provider.force_flush().is_ok(), "flush 1 lands");
        record_registry(&handle, REGISTRY, &snapshot(histogram_series(1, 1)));
        assert!(provider.force_flush().is_ok(), "flush 2 lands");
        record_registry(&handle, REGISTRY, &snapshot(histogram_series(2, 3)));
        assert!(provider.force_flush().is_ok(), "flush 3 lands");
        let counts = exported_counts(&exporter);
        assert!(
            counts.iter().all(|count| *count <= 5),
            "two samples then zero new then three new — no point may exceed the \
             five true samples, got {counts:?}"
        );
        assert!(
            counts.last() == Some(&5) || counts.iter().sum::<u64>() == 5,
            "the exported aggregate ends at the true sample count (delta sums to \
             it, cumulative ends at it): {counts:?}"
        );
    }

    #[test]
    fn a_replacement_registry_replays_its_own_counts() {
        use opentelemetry::metrics::MeterProvider as _;
        use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
        use opentelemetry_sdk::metrics::{
            InMemoryMetricExporter, PeriodicReader, SdkMeterProvider,
        };

        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let handle = MeterHandle {
            meter: provider.meter("replacement-probe"),
            state: Mutex::new(InstrumentState::default()),
            shut_down: AtomicBool::new(false),
        };

        let snapshot = |samples: u64, total: u64| {
            let mut histogram = loopctl::metrics::LatencyHistogram::with_buckets(vec![10, 100]);
            for _ in 0..samples {
                histogram.record(5);
            }
            let counts = histogram.counts().to_vec();
            RegistrySnapshot {
                series: vec![
                    loopctl::metrics::SeriesSnapshot {
                        name: "loopctl.vector.embed",
                        kind: loopctl::metrics::MetricKind::Histogram,
                        labels: std::collections::BTreeMap::new(),
                        value: SeriesValue::Histogram {
                            summary: loopctl::metrics::HistogramSummary {
                                bounds: vec![10, 100],
                                counts,
                                percentiles: histogram
                                    .percentiles()
                                    .map(|readings| [readings.p50, readings.p90, readings.p99]),
                            },
                        },
                    },
                    loopctl::metrics::SeriesSnapshot {
                        name: "loopctl.tools.calls",
                        kind: loopctl::metrics::MetricKind::Counter,
                        labels: std::collections::BTreeMap::from([("tool", "bash".to_string())]),
                        value: SeriesValue::Counter { total },
                    },
                ],
                ..RegistrySnapshot::default()
            }
        };
        let batches = |probe: &InMemoryMetricExporter| -> Vec<(u64, u64)> {
            let mut batches = Vec::new();
            for resource in probe.get_finished_metrics().unwrap_or_default() {
                let mut replayed = 0_u64;
                let mut counted = 0_u64;
                for scope in resource.scope_metrics() {
                    for metric in scope.metrics() {
                        match metric.name() {
                            "loopctl.vector.embed" => {
                                if let AggregatedMetrics::F64(MetricData::Histogram(histogram)) =
                                    metric.data()
                                {
                                    replayed = replayed.saturating_add(
                                        histogram.data_points().fold(0_u64, |acc, point| {
                                            acc.saturating_add(point.count())
                                        }),
                                    );
                                }
                            }
                            "loopctl.tools.calls" => {
                                if let AggregatedMetrics::U64(MetricData::Sum(sum)) = metric.data()
                                {
                                    counted = counted.saturating_add(
                                        sum.data_points().fold(0_u64, |acc, point| {
                                            acc.saturating_add(point.value())
                                        }),
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                }
                batches.push((replayed, counted));
            }
            batches
        };

        record_registry(&handle, REGISTRY, &snapshot(6, 9));
        assert!(provider.force_flush().is_ok(), "the first export lands");
        record_registry(&handle, REPLACEMENT, &snapshot(2, 4));
        assert!(
            provider.force_flush().is_ok(),
            "the replacement's export lands"
        );

        let emitted = batches(&exporter);
        let last = emitted.last().copied().unwrap_or_default();
        assert!(
            (last.0 >= 8 || last.0 == 2) && (last.1 >= 13 || last.1 == 4),
            "both ledgers reset when the registry does: the replacement's own \
             2 samples and 4 calls must reach the export (delta temporality shows \
             them as the increment, cumulative as the new total), got {emitted:?}"
        );
    }

    #[test]
    fn registry_series_flatten_with_labels_and_diagnostics() {
        let snapshot = RegistrySnapshot {
            series: vec![loopctl::metrics::SeriesSnapshot {
                name: "loopctl.tools.calls",
                kind: loopctl::metrics::MetricKind::Counter,
                labels: std::collections::BTreeMap::from([("tool", "bash".to_string())]),
                value: SeriesValue::Counter { total: 7 },
            }],
            unknown_metric: 1,
            unknown_labels: 0,
            missing_labels: 0,
            numeric_on_counter: 0,
            ambiguous_scalar: 0,
        };
        let attrs = registry_attributes(&snapshot);
        assert_eq!(
            find(&attrs, "loopctl.tools.calls.tool=bash"),
            Some(&opentelemetry::Value::I64(7))
        );
        assert_eq!(
            find(&attrs, "loopctl.telemetry.unknown_metric"),
            Some(&opentelemetry::Value::I64(1))
        );
        assert!(
            find(&attrs, "loopctl.telemetry.missing_labels").is_some(),
            "the refusal diagnostic rides the span-attribute stamp"
        );
    }

    #[test]
    fn counter_deltas_track_each_label_series_apart() {
        use opentelemetry::metrics::MeterProvider as _;
        use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
        use opentelemetry_sdk::metrics::{
            InMemoryMetricExporter, PeriodicReader, SdkMeterProvider,
        };

        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let handle = MeterHandle {
            meter: provider.meter("delta-probe"),
            state: Mutex::new(InstrumentState::default()),
            shut_down: AtomicBool::new(false),
        };

        let counter_series = |tool: &str, total: u64| loopctl::metrics::SeriesSnapshot {
            name: "loopctl.tools.calls",
            kind: loopctl::metrics::MetricKind::Counter,
            labels: std::collections::BTreeMap::from([("tool", tool.to_string())]),
            value: SeriesValue::Counter { total },
        };
        let snapshot = |series: Vec<loopctl::metrics::SeriesSnapshot>| RegistrySnapshot {
            series,
            ..RegistrySnapshot::default()
        };
        let exported_points = |probe: &InMemoryMetricExporter| -> Vec<(String, u64)> {
            let mut points = Vec::new();
            for resource in probe.get_finished_metrics().unwrap_or_default() {
                for scope in resource.scope_metrics() {
                    for metric in scope.metrics() {
                        if metric.name() != "loopctl.tools.calls" {
                            continue;
                        }
                        if let AggregatedMetrics::U64(MetricData::Sum(sum)) = metric.data() {
                            for point in sum.data_points() {
                                let tool = point
                                    .attributes()
                                    .find(|kv| kv.key.as_str() == "tool")
                                    .map(|kv| kv.value.as_str().into_owned())
                                    .unwrap_or_default();
                                points.push((tool, point.value()));
                            }
                        }
                    }
                }
            }
            points
        };

        record_registry(
            &handle,
            REGISTRY,
            &snapshot(vec![counter_series("read", 3), counter_series("exec", 5)]),
        );
        let flushed_one = provider.force_flush().is_ok();
        assert!(flushed_one, "flush 1 lands");
        let first = exported_points(&exporter);
        assert!(
            first.iter().any(|(t, v)| t == "read" && *v == 3)
                && first.iter().any(|(t, v)| t == "exec" && *v == 5),
            "first export carries each series from zero: {first:?}"
        );

        record_registry(
            &handle,
            REGISTRY,
            &snapshot(vec![counter_series("read", 4), counter_series("exec", 5)]),
        );
        let flushed_two = provider.force_flush().is_ok();
        assert!(flushed_two, "flush 2 lands");
        let second = exported_points(&exporter);
        assert!(
            second.iter().any(|(t, _)| t == "read"),
            "the read series grew 3 → 4 and must appear in batch 2 \
             (delta: the increment; cumulative: the new total) — a name-keyed \
             previous swallows it: {second:?}"
        );

        record_registry(
            &handle,
            REGISTRY,
            &snapshot(vec![counter_series("read", 10), counter_series("exec", 5)]),
        );
        let flushed_three = provider.force_flush().is_ok();
        assert!(flushed_three, "flush 3 lands");
        let third = exported_points(&exporter);

        let read_points: Vec<u64> = first
            .iter()
            .chain(second.iter())
            .chain(third.iter())
            .filter(|(t, _)| t == "read")
            .map(|(_, v)| *v)
            .collect();
        let read_sum: u64 = read_points.iter().copied().sum();
        let read_last = read_points.last().copied().unwrap_or_default();
        assert!(
            read_sum == 10 || read_last == 10,
            "read grew 3 + 1 + 6 — the exported increments must end at the true \
             growth of 10 (delta mode sums to it, cumulative ends at it): \
             {read_points:?}"
        );
        let exec_last = first
            .iter()
            .chain(second.iter())
            .chain(third.iter())
            .filter(|(t, _)| t == "exec")
            .map(|(_, v)| *v)
            .next_back()
            .unwrap_or_default();
        assert_eq!(exec_last, 5, "the untouched series never drifts");
    }
}

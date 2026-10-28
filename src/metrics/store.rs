//! The live value store behind the metric catalog.
//!
//! [`MetricsRegistry`] holds one [`MetricValue`] per (name, label-set)
//! series — the totals a Prometheus-style scrape would read. The
//! [`MetricsLayer`](super::layer::MetricsLayer) writes it as emissions
//! arrive; exporters and tests read it through
//! [`snapshot`](MetricsRegistry::snapshot). Each write keys its series
//! by the [`MetricDescriptor`] handed to it — the catalog's descriptors
//! on the layer path — and [`snapshot`](MetricsRegistry::snapshot)
//! reports every stored series with the shape its data holds, so a
//! write the store accepted can never vanish from a scrape.
//!
//! The store accumulates for the subscriber's lifetime — process-total
//! counters, gauges at last write, histograms over fixed buckets. It
//! never evicts and never windows; retention policy is an exporter-side
//! concern.
//!
//! # Examples
//!
//! ```
//! use std::collections::BTreeMap;
//! use loopctl::metrics::{MetricsRegistry, registry};
//!
//! let registry = MetricsRegistry::default();
//! let d = registry::lookup("loopctl.tools.calls").expect("declared");
//! let mut labels = BTreeMap::new();
//! labels.insert("tool", "read".to_string());
//! registry.counter_add(d, labels.clone(), 1);
//! registry.counter_add(d, labels, 2);
//!
//! let snap = registry.snapshot();
//! assert_eq!(snap.series.len(), 1);
//! ```

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

use super::lock;
use super::registry::MetricDescriptor;
use super::{MetricKind, Percentile, session::LatencyHistogram};

/// One stored series value, per the catalog's declared kind.
///
/// The variant mirrors the [`MetricKind`] declared for the series name;
/// dispatch through [`MetricsRegistry`] keeps one series to one kind.
/// A raw-API write that names a series under a foreign kind is refused
/// quietly by [`counter_add`](MetricsRegistry::counter_add) and
/// [`histogram_record`](MetricsRegistry::histogram_record) (the stored
/// series survives untouched) and replaced wholesale by
/// [`gauge_set`](MetricsRegistry::gauge_set) — the layer cannot produce
/// either shape because the descriptor fixes the kind, so these are
/// raw-API misuse paths, pinned by tests rather than diagnosed.
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Process-total monotonic count.
    ///
    /// Advanced by [`counter_add`](MetricsRegistry::counter_add),
    /// saturating at [`u64::MAX`].
    Counter(u64),

    /// Instantaneous value at the latest write.
    ///
    /// Replaced wholesale by [`gauge_set`](MetricsRegistry::gauge_set);
    /// last writer wins.
    Gauge(f64),

    /// Histogram samples, bucketed to whole units.
    ///
    /// Accumulates via [`histogram_record`](MetricsRegistry::histogram_record)
    /// on the default millisecond bucket scheme; non-time samples use
    /// the same bounds as ordinary sample units (fixed-bucket
    /// coarseness, not an error).
    Histogram(LatencyHistogram),
}

/// Series identity: catalog name plus its canonical label pairs.
///
/// The `BTreeMap` gives the map its own canonical iteration order, so
/// two emissions carrying the same label pairs in different arrival
/// orders key to the same series.
type SeriesKey = (&'static str, BTreeMap<&'static str, String>);

/// A distribution summary inside a [`SeriesValue`].
///
/// Bounds, counts, and the three headline percentiles travel together:
/// consumers that want the raw distribution read `bounds`/`counts`,
/// consumers that want the SLO read `percentiles`.
#[derive(Debug, Clone, Serialize)]
pub struct HistogramSummary {
    /// Bucket upper bounds, ascending.
    ///
    /// Exactly [`LatencyHistogram::buckets`] at snapshot time.
    pub bounds: Vec<u64>,

    /// Per-slot counts; one more entry than [`bounds`](Self::bounds),
    /// the last for overflow samples.
    pub counts: Vec<u64>,

    /// p50/p90/p99 readings, present only when samples exist.
    ///
    /// Computed from the [`LatencyHistogram`] interpolation at
    /// snapshot time. `None` means the series holds no
    /// samples yet — an empty histogram exports its buckets, but
    /// fabricating percentiles from them would lie.
    pub percentiles: Option<[Percentile; 3]>,
}

/// A stored value in its export-facing shape.
///
/// The serializable mirror of [`MetricValue`]: counters as a total,
/// gauges as a value, histograms as the full bucket-and-percentile
/// summary an exporter renders.
#[derive(Debug, Clone, Serialize)]
pub enum SeriesValue {
    /// A counter series and its process total.
    ///
    /// Monotonic for the life of the process — session-scoped counting
    /// has its own surface on [`MetricsObserver`](super::MetricsObserver),
    /// so this total is deliberately never reset by hand.
    Counter {
        /// The accumulated total.
        ///
        /// The sum of every increment the store has recorded for this
        /// series — `+1` per event for the catalog's counter-kind
        /// signals, which is what a counter counts (passes, not units).
        total: u64,
    },

    /// A gauge series and its last-written value.
    ///
    /// Gauges answer "how much is there now"; each write replaces what
    /// came before, so the store keeps no history for them — a scrape
    /// sees one instant, honestly labeled.
    Gauge {
        /// The value at the most recent write.
        ///
        /// Stored as `f64` because gauge emissions arrive in float and
        /// integer shapes alike; integral gauges lose nothing in the
        /// cast at the magnitudes telemetry reaches.
        value: f64,
    },

    /// A histogram series and its distribution summary.
    ///
    /// The only shape that preserves a distribution, so the only one
    /// whose percentiles survive the trip to an exporter — which is why
    /// durations are declared histograms in the catalog.
    Histogram {
        /// Bounds, counts, and percentiles together.
        ///
        /// Flattened into the variant's JSON so an exporter's payload
        /// sees the flat shape the OTLP mapping expects, without an
        /// nesting layer that carries no meaning.
        #[serde(flatten)]
        summary: HistogramSummary,
    },
}

/// One exported series: what it is and what it currently holds.
///
/// A self-describing row — name, declared kind, canonical labels, and
/// value — which is exactly what an OTLP metrics payload or a dashboard
/// ingestion needs per series; nothing further must be resolved
/// client-side.
#[derive(Debug, Clone, Serialize)]
pub struct SeriesSnapshot {
    /// Catalog name of this series.
    ///
    /// The declared `loopctl.*` name, identical to the descriptor's.
    pub name: &'static str,

    /// The kind this series' stored value holds.
    ///
    /// Equal to the descriptor's declared kind for every layer-written
    /// series (the layer routes writes by the descriptor); for raw-API
    /// writes it is the shape the data actually holds. Derived from the
    /// stored value itself at snapshot time, never resolved against the
    /// catalog — so nothing that was stored can be dropped here.
    pub kind: MetricKind,

    /// Canonical label pairs of this series.
    ///
    /// Closed-enum values under declared keys, sorted by key.
    pub labels: BTreeMap<&'static str, String>,

    /// The current value in export shape.
    ///
    /// Mirrors the declared kind — a counter total, a gauge value, or
    /// the full histogram summary — so an exporter renders the series
    /// without re-resolving anything against the catalog.
    pub value: SeriesValue,
}

/// A consistent read of the whole store for exporters and tests.
///
/// Produced by [`MetricsRegistry::snapshot`] under one lock hold, so
/// every series reflects a single moment; the four diagnostic counters
/// ride along so a consumer sees drift exactly once per scrape, next
/// to the data it explains.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RegistrySnapshot {
    /// Every stored series, sorted by name then labels.
    ///
    /// The order is structural ([`BTreeMap`] key order, not insertion
    /// order), so two scrapes of an unchanged store produce byte-equal
    /// payloads — a property diff-based exporters and tests rely on.
    pub series: Vec<SeriesSnapshot>,

    /// Count of emissions whose name is not in the catalog.
    ///
    /// Drift telemetry: an emitter speaking a name nobody declared.
    pub unknown_metric: u64,

    /// Count of label keys observed beyond a descriptor's set.
    ///
    /// The emission was still stored — under its declared labels only;
    /// the surplus keys are what this counts.
    pub unknown_labels: u64,

    /// Count of declared label keys that emissions omitted.
    ///
    /// Each missing key counts once and the emission itself is refused
    /// — no series is ever stored under a partial label set, so the
    /// stored shape cannot fork from the declared one.
    pub missing_labels: u64,

    /// Count of counter-kind events that carried numeric values.
    ///
    /// Counters count passes (`+1` per event); a numeric field on one
    /// means the emission site and the declaration disagree about what
    /// the signal measures.
    pub numeric_on_counter: u64,

    /// Count of gauge/histogram events with no resolvable scalar.
    ///
    /// Either no numeric field at all, or several unrelated ones with
    /// none matching the metric's unit — the layer refuses to guess and
    /// says so here instead.
    pub ambiguous_scalar: u64,
}

/// The live value store behind the [`REGISTRY`](super::registry::REGISTRY)
/// catalog.
///
/// One instance, shared via [`Arc`](std::sync::Arc), is written by the
/// [`MetricsLayer`](super::layer::MetricsLayer) and read by any number
/// of exporters and tests. All storage lives behind one [`Mutex`] with
/// short holds; poison is recovered (the guarded data is plain numbers,
/// safe to read after any panic), never propagated — a metrics store
/// must not take a process down with it.
#[derive(Default)]
pub struct MetricsRegistry {
    /// Series map, keyed canonically; sorted for stable snapshots.
    ///
    /// One [`Mutex`] hold covers every read and write, so a scrape
    /// never observes a half-applied increment; the map is plain data,
    /// safe to read even after a lock-holder panic.
    series: Mutex<BTreeMap<SeriesKey, MetricValue>>,

    /// Emissions whose name is not in the catalog.
    ///
    /// Bumped once per dropped event and carried into every
    /// [`RegistrySnapshot`], so an undeclared emitter is visible from
    /// the store alone — no log access required.
    unknown_metric: AtomicU64,

    /// Label keys beyond a descriptor's declared set.
    ///
    /// The emission itself is stored under its declared labels; this
    /// counts only the surplus keys the closed contract refused.
    unknown_labels: AtomicU64,

    /// Declared label keys that emissions omitted.
    ///
    /// Refused emissions bump here once per key they lacked; the count
    /// is the repair schedule's signal that a site and the catalog
    /// disagree on a signal's label shape.
    missing_labels: AtomicU64,

    /// Counter-kind events carrying numeric values.
    ///
    /// The event still counts as a pass; the number records that the
    /// emission site and the declaration disagree about what the
    /// signal measures.
    numeric_on_counter: AtomicU64,

    /// Gauge/histogram events with no resolvable scalar.
    ///
    /// These events are dropped rather than guessed at; the count is
    /// the evidence that the emission site needs a unit-suffixed field.
    ambiguous_scalar: AtomicU64,
}

/// Adds one to an atomic counter without wrap arithmetic.
///
/// `checked_add` keeps the bump overflow-honest at [`u64::MAX`] where a
/// `fetch_add` would wrap; the compare-exchange loop is the standard
/// lock-free way to apply the check.
fn bump(counter: &AtomicU64) {
    let mut current = counter.load(Ordering::Relaxed);
    while let Some(next) = current.checked_add(1) {
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(seen) => current = seen,
        }
    }
}

/// Adds `count` to an atomic counter without wrap arithmetic.
///
/// The multi-key twin of [`bump`]: one emission can refuse on several
/// omitted keys at once, and the accounting stays overflow-honest in a
/// single pass.
fn bump_by(counter: &AtomicU64, count: u64) {
    let mut current = counter.load(Ordering::Relaxed);
    while let Some(next) = current.checked_add(count) {
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(seen) => current = seen,
        }
    }
}

/// Snap one sample onto the histogram's whole-unit grid.
///
/// The crate's float→integer boundary (the `pricing::micro_usd`
/// doctrine): a rendering through `{:.0}` rounds to nearest, ties to
/// even, with no cast — so 120.7 stores as 121, not 120, and a tie
/// like 2.5 lands on 2. The guards read the sample the histogram can
/// hold: negative, zero, and NaN record as 0, and anything at or past
/// the `u32` ceiling (including +∞) saturates there.
fn bucket_units(sample: f64) -> u64 {
    if sample >= f64::from(u32::MAX) {
        u64::from(u32::MAX)
    } else if !sample.is_finite() || sample <= 0.0 {
        0
    } else {
        format!("{sample:.0}").parse().unwrap_or(u64::MAX)
    }
}

/// Sums every counter series stored under `name`, optionally narrowed
/// by label pairs (`label` must equal `want` on matching series).
///
/// `None` when no series for `name` exists at all; zero counters still
/// sum to a present `Some(0)` — a recorded zero is a fact, absence is
/// not. The `label`/`want` pair is applied only to label-narrowed
/// derived signals; `None` for the pair means "every series of name".
fn counter_sum(
    series: &BTreeMap<SeriesKey, MetricValue>,
    name: &str,
    label: Option<&str>,
    want: Option<&str>,
) -> Option<u64> {
    let mut found = false;
    let mut total = 0_u64;
    for (key, value) in series {
        if key.0 != name {
            continue;
        }
        if matches!(label.zip(want), Some((l, w))
            if key.1.get(l).map(String::as_str) != Some(w))
        {
            continue;
        }
        if let MetricValue::Counter(c) = value {
            found = true;
            total = total.saturating_add(*c);
        }
    }
    found.then_some(total)
}

impl MetricsRegistry {
    /// Add `delta` to a counter series, creating it on first sight.
    ///
    /// The descriptor's catalog name keys the series (so aliases a
    /// caller might carry never fork storage); the accumulation
    /// saturates at [`u64::MAX`]. A write onto a series the store
    /// already holds under a foreign kind (a gauge or histogram name
    /// driven through this counter API) is a silent no-op: the stored
    /// series survives untouched — see [`MetricValue`] for the full
    /// foreign-kind table.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use loopctl::metrics::{MetricsRegistry, registry};
    ///
    /// let store = MetricsRegistry::default();
    /// let d = registry::lookup("loopctl.memoize.hits").expect("declared");
    /// store.counter_add(d, BTreeMap::new(), 3);
    /// store.counter_add(d, BTreeMap::new(), 1);
    /// let s = store.snapshot();
    /// let total = match s.series[0].value {
    ///     loopctl::metrics::SeriesValue::Counter { total } => total,
    ///     _ => panic!("counter series"),
    /// };
    /// assert_eq!(total, 4);
    /// ```
    pub fn counter_add(
        &self,
        descriptor: &'static MetricDescriptor,
        labels: BTreeMap<&'static str, String>,
        delta: u64,
    ) {
        let mut store = lock(&self.series);
        let value = store
            .entry((descriptor.name, labels))
            .or_insert(MetricValue::Counter(0));
        if let MetricValue::Counter(total) = value {
            *total = total.saturating_add(delta);
        }
    }

    /// Set a gauge series to `value`, creating it on first sight.
    ///
    /// Last write wins outright — a gauge is a reading, not an
    /// accumulation. That rule also governs foreign kinds: a write
    /// onto a series the store holds as a counter or histogram
    /// **replaces** it (the accumulated total or distribution is
    /// gone), because a gauge write always means "this is the value
    /// now". See [`MetricValue`] for the full foreign-kind table.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use loopctl::metrics::{MetricsRegistry, registry};
    ///
    /// let store = MetricsRegistry::default();
    /// let d = registry::lookup("loopctl.tools.registered").expect("declared");
    /// store.gauge_set(d, BTreeMap::new(), 12.0);
    /// store.gauge_set(d, BTreeMap::new(), 9.0);
    /// let s = store.snapshot();
    /// let value = match s.series[0].value {
    ///     loopctl::metrics::SeriesValue::Gauge { value } => value,
    ///     _ => panic!("gauge series"),
    /// };
    /// assert_eq!(value, 9.0);
    /// ```
    pub fn gauge_set(
        &self,
        descriptor: &'static MetricDescriptor,
        labels: BTreeMap<&'static str, String>,
        value: f64,
    ) {
        let mut store = lock(&self.series);
        store.insert((descriptor.name, labels), MetricValue::Gauge(value));
    }

    /// Record one sample on a histogram series, creating it on first sight.
    ///
    /// Negative samples clamp to zero; fractional samples are rounded
    /// to the nearest whole unit (ties to even, the rendering's own
    /// rule) before bucketing — the buckets are whole-millisecond and
    /// the store keeps that invariant rather than inventing a
    /// fractional scheme. A write onto a foreign-kind series is a
    /// silent no-op; see [`MetricValue`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use loopctl::metrics::{MetricsRegistry, registry};
    ///
    /// let store = MetricsRegistry::default();
    /// let d = registry::lookup("loopctl.vector.embed").expect("declared");
    /// store.histogram_record(d, BTreeMap::new(), 120.0);
    /// let s = store.snapshot();
    /// let counts = match &s.series[0].value {
    ///     loopctl::metrics::SeriesValue::Histogram { summary } => &summary.counts,
    ///     _ => panic!("histogram series"),
    /// };
    /// assert_eq!(counts.iter().sum::<u64>(), 1);
    /// ```
    pub fn histogram_record(
        &self,
        descriptor: &'static MetricDescriptor,
        labels: BTreeMap<&'static str, String>,
        sample: f64,
    ) {
        let clamped = bucket_units(sample);
        let mut store = lock(&self.series);
        let value = store
            .entry((descriptor.name, labels))
            .or_insert_with(|| MetricValue::Histogram(LatencyHistogram::new()));
        if let MetricValue::Histogram(h) = value {
            h.record(clamped);
        }
    }

    /// Consistent read of every series plus the drift diagnostics.
    ///
    /// One lock hold produces the whole view: callers never see
    /// interleaved writes, and the diagnostics ride the same moment
    /// they describe. Series order is deterministic (name, then
    /// labels), which keeps export diffs stable.
    #[must_use]
    pub fn snapshot(&self) -> RegistrySnapshot {
        let store = lock(&self.series);
        let series = store
            .iter()
            .map(|((name, labels), value)| {
                let kind = match value {
                    MetricValue::Counter(_) => MetricKind::Counter,
                    MetricValue::Gauge(_) => MetricKind::Gauge,
                    MetricValue::Histogram(_) => MetricKind::Histogram,
                };
                let value = match value {
                    MetricValue::Counter(total) => SeriesValue::Counter { total: *total },
                    MetricValue::Gauge(v) => SeriesValue::Gauge { value: *v },
                    MetricValue::Histogram(h) => SeriesValue::Histogram {
                        summary: HistogramSummary {
                            bounds: h.buckets().to_vec(),
                            counts: h.counts().to_vec(),
                            percentiles: h.percentiles().map(|p| [p.p50, p.p90, p.p99]),
                        },
                    },
                };
                SeriesSnapshot {
                    name,
                    kind,
                    labels: labels.clone(),
                    value,
                }
            })
            .collect();
        RegistrySnapshot {
            series,
            unknown_metric: self.unknown_metric.load(Ordering::Relaxed),
            unknown_labels: self.unknown_labels.load(Ordering::Relaxed),
            missing_labels: self.missing_labels.load(Ordering::Relaxed),
            numeric_on_counter: self.numeric_on_counter.load(Ordering::Relaxed),
            ambiguous_scalar: self.ambiguous_scalar.load(Ordering::Relaxed),
        }
    }

    /// Count an emission whose name is not in the catalog.
    ///
    /// Called from the layer's drop path on every unrecognized event,
    /// so the bump itself is infallible and lock-free — drift
    /// accounting must never become a new failure mode.
    pub(crate) fn bump_unknown_metric(&self) {
        bump(&self.unknown_metric);
    }

    /// Count a label key beyond a descriptor's declared set.
    ///
    /// Raised once per undeclared key an emission carries — two surplus
    /// keys in one event bump twice. The snapshot answers "how many
    /// label keys disagree with the catalog", which is the question a
    /// contract repair schedules by; per-emission totals read back off
    /// the store's own series.
    pub(crate) fn bump_unknown_labels(&self) {
        bump(&self.unknown_labels);
    }

    /// Count declared label keys an emission omitted, refusing it.
    ///
    /// `missing` is how many declared keys one emission lacked; the
    /// event stores nothing — a partial label set would fork a series
    /// beside the conforming shape — so this counter is the only trace
    /// the emission left.
    pub(crate) fn bump_missing_labels(&self, missing: usize) {
        bump_by(
            &self.missing_labels,
            u64::try_from(missing).unwrap_or(u64::MAX),
        );
    }

    /// Count a numeric value carried on a counter-kind event.
    ///
    /// The counter still increments by its declared rule; this records
    /// that the site thought the event carried a magnitude, which is a
    /// design disagreement worth reading off the next scrape.
    pub(crate) fn bump_numeric_on_counter(&self) {
        bump(&self.numeric_on_counter);
    }

    /// Count a gauge/histogram event with no resolvable scalar.
    ///
    /// The event stores nothing — a refused sample beats a fabricated
    /// one — so this counter is the only trace the emission left.
    pub(crate) fn bump_ambiguous_scalar(&self) {
        bump(&self.ambiguous_scalar);
    }

    /// The contract's derived ratio signals, computed from stored series.
    ///
    /// Four named ratios over the counters the store already holds;
    /// `None` whenever a denominator is absent or zero — a missing
    /// denominator never fabricates a number. The token ratios
    /// (`usage.cache_hit_rate`, `usage.reasoning_share`) live on
    /// [`TokenTotals`](super::TokenTotals) instead, computed per run
    /// where the counts are.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use loopctl::metrics::{MetricsRegistry, registry};
    ///
    /// let store = MetricsRegistry::default();
    /// let passes = registry::lookup("loopctl.compaction.passes").expect("declared");
    /// let mut stage = BTreeMap::new();
    /// stage.insert("stage", "threshold".to_string());
    /// store.counter_add(passes, stage, 10);
    /// assert_eq!(store.derived("loopctl.compaction.never_failed"), Some(1.0));
    /// assert_eq!(store.derived("loopctl.unrelated.signal"), None);
    /// ```
    #[must_use]
    pub fn derived(&self, name: &str) -> Option<f64> {
        let store = lock(&self.series);
        let ratio = |numerator: (&str, Option<&str>, &str),
                     denominator: (&str, Option<&str>, &str)| {
            let num = counter_sum(&store, numerator.0, numerator.1, Some(numerator.2))?;
            let den = counter_sum(&store, denominator.0, denominator.1, Some(denominator.2))?;
            if den == 0 {
                return None;
            }
            let n = f64::from(u32::try_from(num).unwrap_or(u32::MAX));
            let d = f64::from(u32::try_from(den).unwrap_or(u32::MAX));
            Some(n / d)
        };
        match name {
            "loopctl.memory.demotion_roi" => ratio(
                ("loopctl.memory.stored", Some("origin"), "demoted"),
                ("loopctl.compaction.evicted.messages", None, ""),
            ),
            "loopctl.memory.passive_vs_active" => ratio(
                ("loopctl.memory.retrieve.results", Some("trigger"), "turn"),
                ("loopctl.memory.retrieve.results", Some("trigger"), "tool"),
            ),
            "loopctl.compaction.never_failed" => {
                let passes = counter_sum(&store, "loopctl.compaction.passes", None, None)?;
                if passes == 0 {
                    return None;
                }
                let degraded =
                    counter_sum(&store, "loopctl.compaction.degradations", None, None).unwrap_or(0);
                let d = f64::from(u32::try_from(degraded).unwrap_or(u32::MAX));
                let p = f64::from(u32::try_from(passes).unwrap_or(u32::MAX));
                Some(1.0 - d / p)
            }
            "loopctl.compaction.summarize.ratio" => ratio(
                (
                    "loopctl.compaction.summarizer.tokens",
                    Some("direction"),
                    "out",
                ),
                (
                    "loopctl.compaction.summarizer.tokens",
                    Some("direction"),
                    "in",
                ),
            ),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::registry::lookup;
    use super::*;

    fn labels(pairs: &[(&'static str, &str)]) -> BTreeMap<&'static str, String> {
        pairs.iter().map(|(k, v)| (*k, (*v).to_string())).collect()
    }

    #[test]
    fn all_three_kinds_store_and_snapshot() {
        let store = MetricsRegistry::default();
        store.counter_add(
            lookup("loopctl.tools.calls").unwrap(),
            labels(&[("tool", "read")]),
            2,
        );
        store.gauge_set(
            lookup("loopctl.tools.registered").unwrap(),
            BTreeMap::new(),
            7.0,
        );
        store.histogram_record(
            lookup("loopctl.vector.embed").unwrap(),
            BTreeMap::new(),
            50.0,
        );

        let snap = store.snapshot();
        assert_eq!(snap.series.len(), 3);
        assert!(snap.series.iter().any(|s| s.kind == MetricKind::Counter));
        assert!(snap.series.iter().any(|s| s.kind == MetricKind::Gauge));
        assert!(snap.series.iter().any(|s| s.kind == MetricKind::Histogram));
    }

    #[test]
    fn derived_ratios_and_honest_none() {
        let store = MetricsRegistry::default();
        let passes = lookup("loopctl.compaction.passes").unwrap();
        store.counter_add(passes, labels(&[("stage", "threshold")]), 8);
        let degrades = lookup("loopctl.compaction.degradations").unwrap();
        store.counter_add(
            degrades,
            labels(&[("from", "threshold"), ("to", "truncate")]),
            2,
        );
        assert_eq!(store.derived("loopctl.compaction.never_failed"), Some(0.75));

        assert_eq!(
            store.derived("loopctl.memory.demotion_roi"),
            None,
            "no data at all"
        );

        let summarizer = lookup("loopctl.compaction.summarizer.tokens").unwrap();
        store.counter_add(summarizer, labels(&[("direction", "in")]), 0);
        assert_eq!(
            store.derived("loopctl.compaction.summarize.ratio"),
            None,
            "zero denominator is not zero ratio"
        );
        store.counter_add(summarizer, labels(&[("direction", "out")]), 30);
        store.counter_add(summarizer, labels(&[("direction", "in")]), 60);
        assert_eq!(
            store.derived("loopctl.compaction.summarize.ratio"),
            Some(0.5)
        );
    }

    #[test]
    fn demotion_roi_reads_stored_demotions() {
        let store = MetricsRegistry::default();
        let stored = lookup("loopctl.memory.stored").unwrap();
        store.counter_add(
            stored,
            labels(&[("origin", "demoted"), ("category", "fact")]),
            3,
        );
        store.counter_add(
            stored,
            labels(&[("origin", "turn"), ("category", "fact")]),
            7,
        );
        let evicted = lookup("loopctl.compaction.evicted.messages").unwrap();
        store.counter_add(evicted, BTreeMap::new(), 6);
        assert_eq!(
            store.derived("loopctl.memory.demotion_roi"),
            Some(0.5),
            "three demotions per six evicted messages"
        );
    }

    #[test]
    fn foreign_kind_writes_follow_the_documented_rule() {
        let store = MetricsRegistry::default();
        let calls = lookup("loopctl.tools.calls").unwrap();
        let window = lookup("loopctl.context.window").unwrap();
        let value_of = |store: &MetricsRegistry, name: &str| {
            store
                .snapshot()
                .series
                .iter()
                .find(|s| s.name == name)
                .map(|s| s.value.clone())
        };

        store.counter_add(calls, BTreeMap::new(), 5);
        assert!(matches!(
            value_of(&store, "loopctl.tools.calls"),
            Some(SeriesValue::Counter { total: 5 })
        ));

        store.gauge_set(calls, BTreeMap::new(), 9.0);
        assert!(
            matches!(
                value_of(&store, "loopctl.tools.calls"),
                Some(SeriesValue::Gauge { value }) if (value - 9.0).abs() < f64::EPSILON
            ),
            "gauge_set replaces a foreign-kind series outright"
        );
        store.counter_add(calls, BTreeMap::new(), 4);
        assert!(
            matches!(
                value_of(&store, "loopctl.tools.calls"),
                Some(SeriesValue::Gauge { value }) if (value - 9.0).abs() < f64::EPSILON
            ),
            "a counter write onto a gauge-shaped series is a silent no-op"
        );

        store.gauge_set(window, BTreeMap::new(), 2.0);
        store.counter_add(window, BTreeMap::new(), 5);
        store.histogram_record(window, BTreeMap::new(), 7.0);
        assert!(
            matches!(
                value_of(&store, "loopctl.context.window"),
                Some(SeriesValue::Gauge { value }) if (value - 2.0).abs() < f64::EPSILON
            ),
            "counter and histogram writes onto a gauge series both no-op"
        );
    }

    #[test]
    fn bucket_units_rounds_to_nearest_ties_to_even() {
        assert_eq!(bucket_units(120.7), 121, "nearest, not truncated");
        assert_eq!(bucket_units(120.4), 120, "nearest rounds down too");
        assert_eq!(bucket_units(2.5), 2, "ties follow the even rule");
        assert_eq!(bucket_units(3.5), 4, "…from either side");
        assert_eq!(bucket_units(-3.0), 0, "negatives record as zero");
        assert_eq!(bucket_units(f64::NAN), 0, "NaN records as zero");
        assert_eq!(
            bucket_units(5_000_000_000.0),
            u64::from(u32::MAX),
            "past the u32 ceiling the sample saturates"
        );
    }

    #[test]
    fn label_arrival_order_does_not_fork_series() {
        let store = MetricsRegistry::default();
        let d = lookup("loopctl.memory.retrieve.results").unwrap();
        store.counter_add(d, labels(&[("trigger", "tool"), ("outcome", "hit")]), 1);
        store.counter_add(d, labels(&[("outcome", "hit"), ("trigger", "tool")]), 1);
        let snap = store.snapshot();
        assert_eq!(snap.series.len(), 1, "same pairs, same series");
        let SeriesValue::Counter { total } = snap.series[0].value else {
            panic!("counter series")
        };
        assert_eq!(total, 2);
    }

    #[test]
    fn hand_rolled_descriptors_snapshot_with_their_written_shape() {
        let descriptor: &'static super::super::registry::MetricDescriptor =
            Box::leak(Box::new(super::super::registry::MetricDescriptor {
                name: "loopctl.custom.hand",
                kind: MetricKind::Counter,
                labels: &[],
            }));
        let store = MetricsRegistry::default();
        store.counter_add(descriptor, BTreeMap::new(), 3);
        let snap = store.snapshot();
        assert_eq!(
            snap.series.len(),
            1,
            "a write the store accepted cannot vanish from the snapshot"
        );
        assert_eq!(snap.series[0].name, "loopctl.custom.hand");
        assert_eq!(snap.series[0].kind, MetricKind::Counter);
        let SeriesValue::Counter { total } = snap.series[0].value else {
            panic!("counter shape");
        };
        assert_eq!(total, 3);
    }

    #[test]
    fn diagnostics_count_without_storing_offenders() {
        let store = MetricsRegistry::default();
        store.bump_unknown_metric();
        store.bump_unknown_metric();
        store.bump_unknown_labels();
        store.bump_numeric_on_counter();
        store.bump_ambiguous_scalar();
        let snap = store.snapshot();
        assert_eq!(snap.unknown_metric, 2);
        assert_eq!(snap.unknown_labels, 1);
        assert_eq!(snap.numeric_on_counter, 1);
        assert_eq!(snap.ambiguous_scalar, 1);
        assert!(snap.series.is_empty(), "diagnostics alone store no series");
    }
}

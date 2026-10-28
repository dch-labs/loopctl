//! The `tracing` layer that turns metric events into stored series.
//!
//! [`MetricsLayer`] is a [`tracing_subscriber::Layer`] composed into the
//! host's subscriber. It watches for events on the reserved metrics
//! [`target`](MetricsLayer::TARGET), checks each against the
//! [catalog](super::registry), and converts it into writes on a shared
//! [`MetricsRegistry`]. An event on the target that carries no `metric`
//! field is not a metric event — the engine's own non-metric warnings
//! ride the same target — and is ignored before any catalog check. The
//! check runs in both directions of the closed
//! label set: surplus keys are counted
//! ([`unknown_labels`](super::RegistrySnapshot::unknown_labels)) and
//! dropped, while an emission missing a declared key is refused
//! outright and counted
//! ([`missing_labels`](super::RegistrySnapshot::missing_labels)) — a
//! partial label set would fork a series beside the declared shape.
//! Emissions the catalog does not recognize are counted
//! ([`unknown_metric`](super::RegistrySnapshot::unknown_metric)), never
//! stored under an invented name — drift becomes visible instead of
//! silent.
//!
//! The layer is inert until composed: with no subscriber built on it,
//! nothing it owns ever runs. Emissions are `DEBUG`-level, so the
//! host's filter must admit [`RECOMMENDED_FILTER`](MetricsLayer::RECOMMENDED_FILTER)
//! or the layer sees nothing.
//!
//! # Examples
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

use std::collections::BTreeMap;
use std::sync::Arc;

use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

use super::registry::{self, MetricDescriptor};
use super::store::MetricsRegistry;

/// Fields that may ride a metric event without being labels or drift.
///
/// The telemetry contract exempts model names and session/request
/// identity from the closed label set: they are correlation metadata
/// the emitting subsystem carries for joins, not signal dimensions. A
/// signal that ever *declares* one of these as a label still gets the
/// label (the descriptor check runs first); the exemption only spares
/// the drift counter. The emission-side contract test reads this same
/// list, so the layer's classifier and the contract cannot diverge.
pub const METRIC_EXEMPT_FIELDS: &[&str] = &["model", "session_id", "request_id"];

/// One event's worth of extracted fields, collected before dispatch.
///
/// The `metric` field names the signal; string fields are label
/// candidates; numeric fields are value candidates. Display-formatted
/// numbers (`duration_ms = %elapsed.as_millis()`) arrive through the
/// debug formatter and are re-parsed here — the contract permits that
/// emission shape, so the layer reads it as the number it is.
#[derive(Default)]
struct Collected {
    /// The `metric` field's value, the catalog name under test.
    ///
    /// `None` when the emission carried no `metric` field at all —
    /// that is not a metric, and it is counted as drift before any
    /// lookup is attempted.
    metric: Option<String>,

    /// String fields other than `metric` — label candidates.
    ///
    /// Includes numbers rendered by `%expr` that did not re-parse as
    /// digits (uuids, prose); which of these become labels is the
    /// descriptor's decision, not this struct's.
    strings: Vec<(String, String)>,

    /// Numeric fields, whichever shape they arrived in — value
    /// candidates.
    ///
    /// Genuine `record_*` numerics and re-parsed `%expr` renderings
    /// land in the same list; the declared kind decides which one, if
    /// any, becomes the stored scalar.
    numbers: Vec<(String, f64)>,
}

/// Parses a rendered field into an exact numeric value, if it is one.
///
/// Digit spellings count as numbers — integers and simple decimals
/// (with an optional leading sign), which is everything the engine's
/// `%expr` renderings of counts, durations, and `%f64` scores produce.
/// Anything else — an enum variant name, a uuid, prose — is not a
/// number wearing a disguise, it is a string.
fn digits_only(rendered: &str) -> Option<f64> {
    let body = body_after_sign(rendered);
    let digits = body.split_once('.').unwrap_or((body, ""));
    let integral = !digits.0.is_empty() && digits.0.bytes().all(|b| b.is_ascii_digit());
    let fraction = digits.1.is_empty() || digits.1.bytes().all(|b| b.is_ascii_digit());
    if !integral || !fraction || (digits.0.is_empty() && digits.1.is_empty()) {
        return None;
    }
    rendered.parse::<f64>().ok()
}

/// The digit body of a possibly signed rendering.
///
/// Strips a leading `-` or `+` so [`digits_only`] can check the body
/// uniformly. Both signs are accepted because a hand-written number in
/// a `Debug` rendering may carry one, even though the engine's
/// `%Duration::as_millis()` and `%f64` renderings never do.
fn body_after_sign(text: &str) -> &str {
    text.strip_prefix(['-', '+']).unwrap_or(text)
}

/// [`Visit`] implementation sorting event fields into [`Collected`].
///
/// `tracing` hands fields over one at a time in whatever shape the
/// emission site used — including the macro's prose message, recorded
/// as a field named `message`; prose is neither label nor value, so it
/// is lifted out beside `metric`. Every other field is classified by
/// shape — digits become value candidates, everything else strings —
/// and defers all meaning to the layer's declared-kind dispatch.
struct FieldVisitor<'a> {
    /// The destination every recorded field lands in.
    ///
    /// Borrowed rather than owned: one visit per event, and the
    /// [`Collected`] is consumed by dispatch the moment the visit
    /// ends, so a temporary is all it ever needs.
    out: &'a mut Collected,
}

impl FieldVisitor<'_> {
    /// Files one rendered field, deciding number vs string by shape.
    ///
    /// The `metric` field is lifted out of the label lane entirely:
    /// it names the series, it is never one of the series' labels.
    /// The event's `message` is lifted out too: it is the emission's
    /// prose, not a candidate for either lane.
    fn push_rendered(&mut self, name: &str, rendered: &str) {
        if name == "metric" {
            self.out.metric = Some(rendered.to_string());
            return;
        }
        if name == "message" {
            return;
        }
        match digits_only(rendered) {
            Some(number) => self.out.numbers.push((name.to_string(), number)),
            None => self
                .out
                .strings
                .push((name.to_string(), rendered.to_string())),
        }
    }
}

impl Visit for FieldVisitor<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "metric" {
            self.out.metric = Some(value.to_string());
            return;
        }
        if field.name() == "message" {
            return;
        }
        self.out
            .strings
            .push((field.name().to_string(), value.to_string()));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn core::fmt::Debug) {
        let rendered = format!("{value:?}");
        self.push_rendered(field.name(), &rendered);
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.out.numbers.push((field.name().to_string(), value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        let magnitude = wide_u64_to_f64(value.unsigned_abs());
        let rendered = if value < 0 { -magnitude } else { magnitude };
        self.out.numbers.push((field.name().to_string(), rendered));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.out
            .numbers
            .push((field.name().to_string(), wide_u64_to_f64(value)));
    }
}

/// Converts a wide unsigned integer to `f64` the way a rendered digit
/// string parses — exact below 2^53, correctly rounded beyond.
///
/// The f64 mantissa holds every integer up to 2^53 exactly, and above
/// it the nearest-representable value is what `parse::<f64>` of the
/// rendered digits answers too — the shared construction lives in
/// [`super::wide_to_f64`], so the direct `record_u64` shape and the
/// `%expr`-rendered shape of the same number land on one stored value
/// everywhere, including past `u32::MAX` where a saturating conversion
/// would fork them.
fn wide_u64_to_f64(value: u64) -> f64 {
    super::wide_to_f64(value)
}

/// Resolves the single scalar a gauge or histogram event is reporting.
///
/// Precedence: an explicit `value` field; then a numeric field whose
/// name is the metric's own last segment (the unit-in-name grammar:
/// `…jsonl_bytes` carries `jsonl_bytes`); then, unambiguously, the
/// event's only numeric field. Multiple unrelated numerics with no
/// name match is ambiguity — the caller counts a diagnostic rather
/// than the layer guessing which one the ticket meant.
fn resolve_scalar(metric_name: &str, numbers: &[(String, f64)]) -> Option<f64> {
    if let Some((_, value)) = numbers.iter().find(|(name, _)| name == "value") {
        return Some(*value);
    }
    let tail = metric_name.rsplit('.').next().unwrap_or(metric_name);
    if let Some((_, value)) = numbers.iter().find(|(name, _)| name == tail) {
        return Some(*value);
    }
    if numbers.len() == 1 {
        return numbers.first().map(|(_, value)| *value);
    }
    None
}

/// The unit-named numeric of a counter event, when it carries one.
///
/// A counter whose own unit name rides the event (`…summarizer.tokens`
/// with `tokens = 60`) counts that magnitude as its delta rather than
/// one pass — the same unit-in-name grammar [`resolve_scalar`] applies
/// to gauges and histograms. Returns `None` when no such field rides
/// the event, leaving the counter at its default `+1` pass count.
fn unit_named_delta(metric_name: &str, numbers: &[(String, f64)]) -> Option<u64> {
    let tail = metric_name.rsplit('.').next().unwrap_or(metric_name);
    let (_, value) = numbers.iter().find(|(name, _)| name == tail)?;
    if value.is_finite() && *value > 0.0 {
        Some(format!("{value:.0}").parse().unwrap_or(u64::MAX))
    } else {
        Some(0)
    }
}

/// Canonicalizes label pairs against a descriptor's declared keys.
///
/// String fields whose key the descriptor declares become labels,
/// keyed by the descriptor's own `'static` key string (arrival order
/// and borrowed spellings can never fork a series key). Undeclared
/// keys are counted on the registry as label drift and dropped —
/// the series stays stored under its declared shape.
fn canonical_labels(
    descriptor: &'static MetricDescriptor,
    strings: &[(String, String)],
    store: &MetricsRegistry,
) -> BTreeMap<&'static str, String> {
    let mut labels = BTreeMap::new();
    for (key, value) in strings {
        match descriptor.labels.iter().find(|declared| *declared == key) {
            Some(declared) => {
                labels.insert(*declared, value.clone());
            }
            None => {
                if !METRIC_EXEMPT_FIELDS.contains(&key.as_str()) {
                    store.bump_unknown_labels();
                }
            }
        }
    }
    labels
}

/// The descriptor's declared keys an emission did not carry.
///
/// The closed-set contract's missing direction: a declared key nobody
/// provided means the emission's label shape disagrees with the
/// declaration, and storing it would fork a partial-label series beside
/// the conforming one. The caller refuses the emission and counts one
/// per key named here.
fn missing_declared_keys(
    descriptor: &'static MetricDescriptor,
    strings: &[(String, String)],
) -> Vec<&'static str> {
    descriptor
        .labels
        .iter()
        .filter(|declared| !strings.iter().any(|(key, _)| key.as_str() == **declared))
        .copied()
        .collect()
}

/// The [`tracing_subscriber::Layer`] converting metric events to series.
///
/// Compose it into the host's subscriber with an [`Arc`]ged
/// [`MetricsRegistry`] shared with whatever reads it later. Events on
/// any other target pass through untouched; the only work done for
/// them is one string compare.
pub struct MetricsLayer {
    /// The store every recognized emission writes into.
    ///
    /// Shared through [`Arc`](std::sync::Arc) with exporters and tests;
    /// the layer never owns or swaps it, so the store an exporter
    /// scrapes is exactly the one a running engine writes.
    registry: Arc<MetricsRegistry>,
}

impl MetricsLayer {
    /// The event target reserved for cataloged metrics.
    ///
    /// Emission sites write `target: MetricsLayer::TARGET` (today a
    /// literal at each site); the layer recognizes exactly this target
    /// and nothing else.
    pub const TARGET: &'static str = "loopctl::metrics";

    /// The filter directive this layer needs, for copy-paste wiring.
    ///
    /// Metric emissions are `DEBUG`-level; a subscriber filtering
    /// tighter never shows the layer an event. As an `EnvFilter`
    /// directive: `loopctl::metrics=debug`.
    pub const RECOMMENDED_FILTER: &'static str = "loopctl::metrics=debug";

    /// Builds the layer over a shared registry.
    ///
    /// The `Arc` is the sharing handle: the layer writes, exporters
    /// read, neither owns the store alone.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use loopctl::metrics::{MetricsLayer, MetricsRegistry};
    ///
    /// let store = Arc::new(MetricsRegistry::default());
    /// let layer = MetricsLayer::new(store);
    /// assert_eq!(MetricsLayer::TARGET, "loopctl::metrics");
    /// ```
    #[must_use]
    pub fn new(registry: Arc<MetricsRegistry>) -> Self {
        Self { registry }
    }

    /// Dispatches one collected event per the catalog's declared kind.
    ///
    /// The label check runs first and in both directions: surplus keys
    /// count as drift while the event still stores, but an emission
    /// missing a declared key is refused — counted per key, stored
    /// nowhere. Counters count passes — `+1` per event — except where
    /// the event carries a numeric named as the metric's own unit
    /// (`tokens` on `…summarizer.tokens`): that magnitude is the delta,
    /// and every other numeric on a counter is counted as a
    /// `numeric_on_counter` diagnostic (the site and the declaration
    /// disagree about the unit of the signal; guessing one is not the
    /// layer's job). Gauges and histograms need a scalar; where
    /// [`resolve_scalar`] finds none unambiguously, the
    /// `ambiguous_scalar` diagnostic records the event instead of a
    /// fabricated value.
    fn dispatch(&self, collected: Collected) {
        let Some(metric_name) = collected.metric else {
            return;
        };
        let Some(descriptor) = registry::lookup(&metric_name) else {
            self.registry.bump_unknown_metric();
            return;
        };
        let missing = missing_declared_keys(descriptor, &collected.strings);
        if !missing.is_empty() {
            self.registry.bump_missing_labels(missing.len());
            return;
        }
        let labels = canonical_labels(descriptor, &collected.strings, &self.registry);
        match descriptor.kind {
            super::MetricKind::Counter => {
                let unit = unit_named_delta(&metric_name, &collected.numbers);
                for (name, _) in &collected.numbers {
                    let is_unit =
                        unit.is_some() && Some(name.as_str()) == metric_name.rsplit('.').next();
                    if !is_unit {
                        self.registry.bump_numeric_on_counter();
                    }
                }
                self.registry
                    .counter_add(descriptor, labels, unit.unwrap_or(1));
            }
            super::MetricKind::Gauge | super::MetricKind::Histogram => {
                match resolve_scalar(&metric_name, &collected.numbers) {
                    Some(scalar) => match descriptor.kind {
                        super::MetricKind::Histogram => {
                            self.registry.histogram_record(descriptor, labels, scalar);
                        }
                        _ => self.registry.gauge_set(descriptor, labels, scalar),
                    },
                    None => self.registry.bump_ambiguous_scalar(),
                }
            }
        }
    }
}

impl<S: Subscriber> Layer<S> for MetricsLayer {
    fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
        if event.metadata().target() != Self::TARGET {
            return;
        }
        let mut collected = Collected::default();
        event.record(&mut FieldVisitor {
            out: &mut collected,
        });
        if collected.metric.is_none() {
            return;
        }
        self.dispatch(collected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Installs a subscriber built over `store` and runs `emit` under it.
    ///
    /// `set_default` is thread-local and every test here runs to
    /// completion on its own thread, so scenarios cannot cross-talk
    /// through a global dispatcher.
    fn capture(store: &Arc<MetricsRegistry>, emit: impl FnOnce()) {
        use tracing_subscriber::layer::SubscriberExt as _;
        let subscriber = tracing_subscriber::registry().with(MetricsLayer::new(store.clone()));
        let guard = tracing::subscriber::set_default(subscriber);
        emit();
        drop(guard);
    }

    fn counter_total(store: &Arc<MetricsRegistry>, name: &str) -> Option<u64> {
        store.snapshot().series.iter().find_map(|series| {
            (series.name == name).then(|| match series.value {
                super::super::store::SeriesValue::Counter { total } => total,
                _ => panic!("{name} declared counter"),
            })
        })
    }

    #[test]
    fn counter_labels_and_undeclared_keys_and_display_duration() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.tools.calls",
                tool = "read",
                outcome = "ok",
                region = "eu",
                duration_ms = %123_u64,
            );
        });
        let snap = store.snapshot();
        assert_eq!(snap.series.len(), 1);
        assert_eq!(
            snap.series[0].labels.keys().copied().collect::<Vec<_>>(),
            vec!["outcome", "tool"],
            "the declared pair becomes labels; an undeclared key never joins them"
        );
        assert_eq!(counter_total(&store, "loopctl.tools.calls"), Some(1));
        assert_eq!(snap.unknown_labels, 1, "region is drift, counted once");
        assert_eq!(
            snap.numeric_on_counter, 1,
            "a duration on a counter is drift"
        );
    }

    #[test]
    fn rendered_decimals_are_value_candidates_not_label_drift() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.trajectory.jsonl_bytes",
                jsonl_bytes = 512_u64,
                top_score = %0.85,
            );
        });
        let snap = store.snapshot();
        assert_eq!(
            snap.unknown_labels, 0,
            "a rendered decimal is a number, not a label"
        );
    }

    #[test]
    fn histogram_scalar_by_unit_name_and_single_fallback() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.trajectory.jsonl_bytes",
                jsonl_bytes = 4096_u64,
            );
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.vector.embed",
                provider = "hashing",
                duration = %7_u64,
            );
        });
        let snap = store.snapshot();
        let bytes = snap
            .series
            .iter()
            .find(|s| s.name == "loopctl.trajectory.jsonl_bytes")
            .expect("gauge series");
        match bytes.value {
            super::super::store::SeriesValue::Gauge { value } => {
                assert!((value - 4096.0).abs() < f64::EPSILON);
            }
            _ => panic!("declared gauge"),
        }
        let embed = snap
            .series
            .iter()
            .find(|s| s.name == "loopctl.vector.embed")
            .expect("histogram series");
        match &embed.value {
            super::super::store::SeriesValue::Histogram { summary } => {
                assert_eq!(summary.counts.iter().sum::<u64>(), 1);
            }
            _ => panic!("declared histogram"),
        }
    }

    #[test]
    fn bogus_name_counts_unknown_and_stores_nothing() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.nope.something",
                value = 1_u64,
            );
        });
        let snap = store.snapshot();
        assert_eq!(snap.unknown_metric, 1);
        assert!(snap.series.is_empty());
    }

    #[test]
    fn events_on_other_targets_pass_through() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::debug!(target: "something.else", metric = "loopctl.memoize.hits");
        });
        let snap = store.snapshot();
        assert!(snap.series.is_empty());
        assert_eq!(snap.unknown_metric, 0, "not even counted as drift");
    }

    /// A counter whose own unit name rides the event counts that
    /// magnitude, not one: `loopctl.compaction.summarizer.tokens`
    /// carrying `tokens = 60` adds sixty, and the derived
    /// summarize ratio becomes the token ratio it was declared to be.
    /// A foreign numeric on a counter stays drift, exactly as before.
    #[test]
    fn a_unit_named_counter_field_is_its_delta() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.compaction.summarizer.tokens",
                direction = "in",
                estimated = true,
                tokens = 60_u64,
                "summarizer input tokens for one compaction pass"
            );
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.compaction.summarizer.tokens",
                direction = "out",
                estimated = true,
                tokens = 30_u64,
                "summarizer output tokens for one compaction pass"
            );
        });
        let snap = store.snapshot();
        let input = snap
            .series
            .iter()
            .find(|s| s.labels.get("direction").map(String::as_str) == Some("in"))
            .expect("the input series stored");
        match input.value {
            super::super::store::SeriesValue::Counter { total } => assert_eq!(
                total, 60,
                "the unit-named field is the counter's delta, not a pass count"
            ),
            _ => panic!("counter series"),
        }
        assert_eq!(
            snap.numeric_on_counter, 0,
            "the metric's own unit-named numeric is the delta, never drift"
        );
        assert_eq!(
            store.derived("loopctl.compaction.summarize.ratio"),
            Some(0.5),
            "the derived ratio reads tokens, not event passes"
        );

        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.tools.calls",
                tool = "read",
                outcome = "ok",
                duration_ms = %5_u64,
            );
        });
        assert_eq!(
            store.snapshot().numeric_on_counter,
            1,
            "a numeric the metric does not name stays drift"
        );
    }

    /// The engine itself emits non-metric events on the reserved target
    /// (the embedding dimension-mismatch warning); those are not metric
    /// events and must not read as catalog drift.
    #[test]
    fn events_without_a_metric_field_are_ignored() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::warn!(
                target: MetricsLayer::TARGET,
                "embedding dimension mismatch: configured 256, served 128"
            );
        });
        let snap = store.snapshot();
        assert!(
            snap.series.is_empty(),
            "nothing without a metric name stores a series"
        );
        assert_eq!(
            snap.unknown_metric, 0,
            "a metric-less event on the target is not catalog drift"
        );
    }

    #[test]
    fn two_undeclared_keys_count_once_each() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.tools.calls",
                tool = "read",
                outcome = "ok",
                region = "eu",
                shard = "s2",
            );
        });
        assert_eq!(
            store.snapshot().unknown_labels,
            2,
            "one count per surplus key — the snapshot answers which emissions disagree, \
             measured per key"
        );
    }

    #[test]
    fn contract_exempt_fields_are_never_drift() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.tools.calls",
                tool = "read",
                outcome = "ok",
                model = "some-model",
                session_id = "s-42",
            );
        });
        let snap = store.snapshot();
        assert_eq!(
            snap.unknown_labels, 0,
            "contract-exempt correlation metadata is conformant, not drift"
        );
        assert_eq!(
            snap.series[0].labels.keys().copied().collect::<Vec<_>>(),
            vec!["outcome", "tool"],
            "exempt fields stay out of the label set too"
        );
    }

    /// A wide integer must land in the store at the same value whether
    /// the site passed it directly or `%`-rendered it: the extraction
    /// rules bless both shapes, and a saturation on only the direct
    /// path would fork the series by emission style above
    /// [`u32::MAX`] (a GB-scale `jsonl_bytes`, a 49-day `duration_ms`).
    #[test]
    fn direct_and_rendered_ints_store_one_value_past_u32_max() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.trajectory.jsonl_bytes",
                jsonl_bytes = 5_000_000_000_u64,
            );
        });
        let super::super::store::SeriesValue::Gauge { value: direct } =
            store.snapshot().series[0].value
        else {
            panic!("declared gauge")
        };
        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.trajectory.jsonl_bytes",
                jsonl_bytes = %5_000_000_000_u64,
            );
        });
        let super::super::store::SeriesValue::Gauge { value: rendered } =
            store.snapshot().series[0].value
        else {
            panic!("declared gauge")
        };
        // Bit equality is the claim: the two emission shapes must land
        // on the one f64 that is exactly 5 000 000 000 — any tolerance
        // would hide the fork this pin exists to catch.
        assert_eq!(
            direct.to_bits(),
            (5_000_000_000.0_f64).to_bits(),
            "the direct shape stores exactly"
        );
        assert_eq!(
            direct.to_bits(),
            rendered.to_bits(),
            "both emission shapes agree past u32::MAX"
        );

        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.context.window",
                source = "live",
                depth = -4_294_967_300_i64,
            );
        });
        let snap = store.snapshot();
        let negative = snap
            .series
            .iter()
            .find(|s| s.name == "loopctl.context.window")
            .expect("window gauge");
        match negative.value {
            super::super::store::SeriesValue::Gauge { value } => {
                assert_eq!(
                    value.to_bits(),
                    (-4_294_967_300.0_f64).to_bits(),
                    "the signed path is exact too"
                );
            }
            _ => panic!("declared gauge"),
        }
    }

    /// An emission missing a declared label key is refused outright and
    /// counted per key — the closed-set contract's other half. Storing
    /// it would fork a partial-label series beside the conforming shape
    /// (`{tool: read}` next to `{tool: read, outcome: ok}`), invisible
    /// to every drift counter.
    #[test]
    fn missing_declared_labels_are_refused_and_counted() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.tools.calls",
                tool = "read",
            );
        });
        let snap = store.snapshot();
        assert!(
            snap.series.is_empty(),
            "a partial label set must not fork a series"
        );
        assert_eq!(
            snap.missing_labels, 1,
            "the omitted `outcome` key is counted once"
        );

        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.memory.retrieve.results",
            );
        });
        let snap = store.snapshot();
        assert!(
            snap.series.is_empty(),
            "omitting both declared keys still stores nothing"
        );
        assert_eq!(
            snap.missing_labels, 3,
            "each of the two omitted keys counts on its own"
        );

        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.tools.calls",
                tool = "read",
                outcome = "ok",
            );
        });
        let snap = store.snapshot();
        assert_eq!(
            snap.series.len(),
            1,
            "the conforming emission still stores after the refusals"
        );
        assert_eq!(
            snap.missing_labels, 3,
            "a conformant emission counts no missing keys"
        );
    }

    #[test]
    fn ambiguous_multi_numeric_gauge_is_diagnosed_not_guessed() {
        let store = Arc::new(MetricsRegistry::default());
        capture(&store, || {
            tracing::debug!(
                target: MetricsLayer::TARGET,
                metric = "loopctl.context.window",
                source = "live",
                ratio_bp = 80_u64,
                sample_bp = 90_u64,
            );
        });
        let snap = store.snapshot();
        assert_eq!(snap.ambiguous_scalar, 1);
        assert!(snap.series.is_empty());
    }
}

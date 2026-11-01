//! The metric registry catalog: every signal the crate declares.
//!
//! The catalog follows the telemetry naming contract: names are
//! `loopctl.<subsystem>.<name>[.<unit>]`, every label set is **closed**
//! (label values are closed enums chosen at the emission site; session
//! ids and model names ride span fields, never metric labels), and one
//! [`MetricDescriptor`] entry exists per declared signal.
//!
//! This half of the feature is the catalog — *descriptors*, not values.
//! A signal observed in the wild with no entry here is a gap between
//! emitter and catalog: the validation path in [`validate`] refuses it
//! so the mismatch is counted and surfaced rather than papered over
//! with an invented name.
//!
//! The other direction holds too, and is enforced rather than asserted:
//! every row here has at least one emission site in the crate, so a
//! declared name is a promise the crate keeps, not a wish.
//! The repository's own gate, `tests/registry_emission_audit.rs`, reads
//! this table and fails the build on a row no `metric = "…"` site
//! names — which is what makes it safe for a consumer to build a
//! dashboard on a name declared here and see nothing: silence then means
//! the signal never fired, never that nobody wired it.
//!
//! # Examples
//!
//! ```
//! use loopctl::metrics::registry;
//!
//! let d = registry::lookup("loopctl.tools.calls").expect("declared");
//! assert_eq!(d.kind, loopctl::metrics::MetricKind::Counter);
//! assert!(registry::validate("loopctl.tools.calls", &["tool", "outcome"]).is_some());
//! ```

use core::cmp::Ordering;

/// The kind of a registered metric.
///
/// Only these three signal types carry stored values in the telemetry
/// contract; spans, events, and observer callbacks are lifecycle
/// surfaces and never registry entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum MetricKind {
    /// Monotonic count.
    ///
    /// A value that only ever increases within a process; decreasing is
    /// reserved for the deliberate zero-reset at process start.
    Counter,

    /// Instantaneous value.
    ///
    /// A reading of current state — a size, a percentage — that may
    /// move in either direction between observations.
    Gauge,

    /// Sample distribution.
    ///
    /// Samples collected over fixed buckets, from which percentile
    /// readings are taken.
    Histogram,
}

/// One declared signal: its name, kind, and closed label set.
///
/// Descriptors are the contract an emitter and an exporter agree on.
/// The [`name`](Self::name) grammar and the closed [`labels`](Self::labels)
/// set are both pinned here; anything the catalog does not state is not
/// part of the metric surface.
#[derive(Debug, Clone, Copy)]
pub struct MetricDescriptor {
    /// Full metric name.
    ///
    /// Dotted path under the `loopctl.` prefix following
    /// `loopctl.<subsystem>.<name>[.<unit>]`; this exact string is what
    /// emissions carry and exporters key on.
    pub name: &'static str,

    /// Signal kind.
    ///
    /// Determines how the value store aggregates samples for this name
    /// and how exporters render it.
    pub kind: MetricKind,

    /// Closed label keys.
    ///
    /// The complete, exact set of label keys this signal carries. An
    /// emission with a different set — extra or missing keys — is a
    /// contract violation, rejected by [`validate`].
    pub labels: &'static [&'static str],
}

/// Every declared signal, as one auditable table.
///
/// Lookup is a linear scan over these entries; it is called once per
/// distinct (name, labels) pair at first sight by the validation path,
/// never in a hot loop, so the slice form is deliberate: one visible
/// table, no hidden map construction.
pub const REGISTRY: &[MetricDescriptor] = &[
    c("loopctl.trajectory.records", &["outcome"]),
    g("loopctl.trajectory.jsonl_bytes", &[]),
    c("loopctl.structured.prompted", &["outcome"]),
    c("loopctl.structured.prompted.tokens", &["direction"]),
    h("loopctl.structured.prompted.attempts_count", &[]),
    c("loopctl.think.first_in_turn", &[]),
    c("loopctl.think.calls", &[]),
    h("loopctl.estimate.calibration", &[]),
    c("loopctl.window.token_counter", &["preset"]),
    c("loopctl.memory.extracted", &["category"]),
    c("loopctl.memory.extract.attempts", &["outcome"]),
    h("loopctl.memory.extract.duration_ms", &[]),
    c("loopctl.memory.consolidated", &[]),
    c("loopctl.memory.persist.writes", &["backend"]),
    h("loopctl.memory.persist.flush.duration_ms", &["backend"]),
    g("loopctl.memory.persist.file_bytes", &["backend"]),
    g("loopctl.memory.entries", &["store"]),
    c("loopctl.memory.stored", &["origin", "category"]),
    c("loopctl.memory.retrieve.results", &["trigger", "outcome"]),
    h("loopctl.memory.retrieve.duration_ms", &["store"]),
    c("loopctl.embed.inputs", &["provider"]),
    c("loopctl.embed.requests", &["provider", "outcome"]),
    h("loopctl.embed.batch.duration_ms", &["provider", "outcome"]),
    c("loopctl.vector.provision.errors", &["backend"]),
    c("loopctl.vector.provision", &["backend"]),
    h("loopctl.vector.embed", &["provider"]),
    h("loopctl.vector.index.search", &[]),
    h("loopctl.memory.store.retrieve", &[]),
    h("loopctl.memory.store.consolidate", &[]),
    c("loopctl.memoize.hits", &[]),
    c("loopctl.memoize.misses", &[]),
    c("loopctl.compaction.passes", &["stage"]),
    c("loopctl.compaction.degradations", &["from", "to"]),
    c(
        "loopctl.compaction.summarizer.tokens",
        &["direction", "estimated"],
    ),
    c(
        "loopctl.compaction.structured.sections",
        &["section", "state"],
    ),
    c("loopctl.compaction.evicted.messages", &[]),
    c("loopctl.compaction.no_action", &[]),
    g("loopctl.compaction.headroom_pct", &[]),
    h("loopctl.compaction.tokens_saved", &[]),
    h("loopctl.compaction.summarize.duration_ms", &[]),
    c("loopctl.tools.calls", &["tool", "outcome"]),
    c("loopctl.tools.first_calls", &[]),
    c("loopctl.verify.passes", &["outcome"]),
    c("loopctl.demotion.delivered", &["sink"]),
    c("loopctl.demotion.errors", &["sink"]),
    g("loopctl.profile.active", &["profile"]),
    c("loopctl.prune.cleared", &[]),
    g("loopctl.attachments.count", &["kind"]),
    h("loopctl.attachments.tokens_estimate", &["kind"]),
    c("loopctl.compaction.attachments.carried_count", &[]),
    c("loopctl.compaction.attachments.carried_bytes", &[]),
    c("loopctl.mcp.structured", &["outcome", "tool"]),
    c("loopctl.mcp.structured.emitted", &["tool"]),
    c("loopctl.tools.registry.mutations", &["op"]),
    c("loopctl.tools.registry.inflight_drains", &[]),
    g("loopctl.tools.registered", &[]),
    c("loopctl.tools.selected", &["selector"]),
    c("loopctl.tools.schema_tokens", &["direction"]),
    h("loopctl.tools.advertised_count", &["selector"]),
    c("loopctl.provider.wire_failures", &["class"]),
    c("loopctl.context.probe.failures", &["kind"]),
    g("loopctl.context.window", &["source"]),
    c("loopctl.reasoning.blocks", &["action"]),
    c("loopctl.reasoning.dropped_at_turn_end", &[]),
    c("loopctl.reasoning.redaction_skips", &[]),
];

/// Counter descriptor shorthand.
///
/// Builds a [`MetricDescriptor`] with [`MetricKind::Counter`], keeping
/// the [`REGISTRY`] table one entry per line wherever the entry fits.
const fn c(name: &'static str, labels: &'static [&'static str]) -> MetricDescriptor {
    MetricDescriptor {
        name,
        kind: MetricKind::Counter,
        labels,
    }
}

/// Gauge descriptor shorthand.
///
/// Builds a [`MetricDescriptor`] with [`MetricKind::Gauge`]; same role
/// as [`c`] for the gauge rows of the [`REGISTRY`] table.
const fn g(name: &'static str, labels: &'static [&'static str]) -> MetricDescriptor {
    MetricDescriptor {
        name,
        kind: MetricKind::Gauge,
        labels,
    }
}

/// Histogram descriptor shorthand.
///
/// Builds a [`MetricDescriptor`] with [`MetricKind::Histogram`]; same
/// role as [`c`] for the histogram rows of the [`REGISTRY`] table.
const fn h(name: &'static str, labels: &'static [&'static str]) -> MetricDescriptor {
    MetricDescriptor {
        name,
        kind: MetricKind::Histogram,
        labels,
    }
}

/// Every declared descriptor.
///
/// Same slice as [`REGISTRY`], offered as a function so callers treat
/// the catalog as an iterable surface rather than a static to name.
///
/// # Examples
///
/// ```
/// use loopctl::metrics::registry;
///
/// assert!(registry::all().iter().any(|d| d.name == "loopctl.tools.calls"));
/// ```
#[must_use]
pub const fn all() -> &'static [MetricDescriptor] {
    REGISTRY
}

/// Find a descriptor by exact metric name.
///
/// Returns `None` for any name the catalog does not declare, including
/// near-misses — matching is exact string equality, never prefix or
/// case-insensitive, so an unregistered name surfaces as a gap rather
/// than silently borrowing a neighboring entry.
///
/// # Examples
///
/// ```
/// use loopctl::metrics::registry;
///
/// assert!(registry::lookup("loopctl.embed.inputs").is_some());
/// assert!(registry::lookup("loopctl.embed.nope").is_none());
/// ```
#[must_use]
pub fn lookup(name: &str) -> Option<&'static MetricDescriptor> {
    REGISTRY.iter().find(|d| d.name == name)
}

/// Validate an observed emission against its declaration.
///
/// The name must be registered and the observed label keys must be
/// **exactly** the declared set (order insensitive, no duplicates — a
/// key carried twice is a doubled identity, not the declared one).
/// Returns the descriptor on success and `None` for an unregistered
/// name or any label-set mismatch, so callers count the refusal instead
/// of storing data under a shape nobody declared.
///
/// # Examples
///
/// ```
/// use loopctl::metrics::registry;
///
/// assert!(registry::validate("loopctl.embed.requests", &["provider", "outcome"]).is_some());
/// assert!(registry::validate("loopctl.embed.requests", &["provider"]).is_none());
/// assert!(registry::validate("loopctl.embed.requests", &["provider", "provider", "outcome"]).is_none());
/// ```
#[must_use]
pub fn validate(name: &str, label_keys: &[&str]) -> Option<&'static MetricDescriptor> {
    let d = lookup(name)?;
    let exact = label_keys.len() == d.labels.len()
        && matches!(cmp_label_sets(d.labels, label_keys), Ordering::Equal);
    exact.then_some(d)
}

/// Order-insensitive set comparison of declared against observed label keys.
///
/// [`Ordering::Less`] when `observed` is missing declared keys,
/// [`Ordering::Greater`] when it carries undeclared ones (or both at
/// once), [`Ordering::Equal`] when the two sets are identical. Only the
/// `Equal` leg is a valid emission; every other outcome is a refusal
/// the caller is expected to count.
fn cmp_label_sets(declared: &[&str], observed: &[&str]) -> Ordering {
    let missing = declared.iter().filter(|k| !observed.contains(k)).count();
    let extra = observed.iter().filter(|k| !declared.contains(k)).count();
    match (missing, extra) {
        (0, 0) => Ordering::Equal,
        (0, _) => Ordering::Greater,
        (_, 0) => Ordering::Less,
        _ => Ordering::Greater,
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    #[test]
    fn registry_names_follow_grammar_and_are_unique() {
        let mut names: Vec<&str> = REGISTRY.iter().map(|d| d.name).collect();
        assert_eq!(names.len(), REGISTRY.len(), "descriptor count");
        for d in REGISTRY {
            let segs: Vec<&str> = d.name.split('.').collect();
            assert!(segs.len() >= 3, "too short: {}", d.name);
            assert_eq!(segs[0], "loopctl", "prefix: {}", d.name);
            assert!(
                segs.iter().skip(1).all(|s| !s.is_empty()),
                "empty seg: {}",
                d.name
            );
            assert!(
                d.labels.iter().all(|l| !l.is_empty() && !l.contains('.')),
                "bad label key on {}",
                d.name
            );
        }
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate metric names in REGISTRY");
    }

    #[test]
    fn histogram_names_carry_unit_suffix_modulo_legacy() {
        const LEGACY: &[&str] = &[
            "loopctl.vector.embed",
            "loopctl.vector.index.search",
            "loopctl.memory.store.retrieve",
            "loopctl.memory.store.consolidate",
            "loopctl.compaction.tokens_saved",
            "loopctl.attachments.tokens_estimate",
            "loopctl.estimate.calibration",
        ];
        const SUFFIXES: &[&str] = &[
            "_ms",
            "_bytes",
            "_count",
            "_tokens",
            "_pct",
            "_attempts_count",
        ];
        for d in REGISTRY {
            if d.kind == MetricKind::Histogram {
                if LEGACY.contains(&d.name) {
                    continue;
                }
                assert!(
                    SUFFIXES.iter().any(|s| d.name.ends_with(s)),
                    "histogram without unit suffix: {}",
                    d.name
                );
            }
        }
    }

    #[test]
    fn lookup_answers_representative_names_and_misses_bogus() {
        for name in [
            "loopctl.trajectory.records",
            "loopctl.memory.extracted",
            "loopctl.embed.batch.duration_ms",
            "loopctl.compaction.passes",
            "loopctl.tools.calls",
            "loopctl.attachments.count",
            "loopctl.context.window",
            "loopctl.reasoning.blocks",
        ] {
            assert!(lookup(name).is_some(), "expected registered: {name}");
        }
        assert!(lookup("loopctl.nonexistent.thing").is_none());
        assert!(lookup("prometheus.up").is_none());
    }

    #[test]
    fn validate_requires_exact_closed_label_set() {
        let d = validate("loopctl.memory.retrieve.results", &["outcome", "trigger"]);
        assert!(d.is_some(), "declared set must validate (any order)");
        assert!(
            validate(
                "loopctl.memory.retrieve.results",
                &["trigger", "outcome", "store"],
            )
            .is_none(),
            "superset (extra `store` label) must be rejected"
        );
        assert!(
            validate("loopctl.memory.retrieve.results", &["trigger"]).is_none(),
            "subset must be rejected"
        );
        assert!(validate("loopctl.tools.calls", &["tool", "outcome"]).is_some());
        assert!(
            validate("loopctl.tools.calls", &["tool"]).is_none(),
            "the released emission carries the outcome split; a partial set is not it"
        );
        assert!(validate("loopctl.prune.cleared", &[]).is_some());
        assert!(validate("loopctl.prune.cleared", &["tool"]).is_none());
        assert!(validate("loopctl.unregistered.name", &[]).is_none());
    }

    #[test]
    fn duplicate_observed_keys_never_validate() {
        assert!(
            validate("loopctl.tools.calls", &["tool", "outcome", "outcome"]).is_none(),
            "a key carried twice is a doubled identity, not the declared set"
        );
        assert!(
            validate("loopctl.compaction.passes", &["stage", "stage"]).is_none(),
            "duplicating the only declared key must not pass the set test"
        );
    }

    #[test]
    fn every_emitted_metric_name_is_registered() {
        const EMITTED_TODAY: &[&str] = &[
            "loopctl.compaction.degradations",
            "loopctl.compaction.passes",
            "loopctl.compaction.structured.sections",
            "loopctl.compaction.summarizer.tokens",
            "loopctl.context.probe.failures",
            "loopctl.context.window",
            "loopctl.embed.batch.duration_ms",
            "loopctl.estimate.calibration",
            "loopctl.mcp.structured",
            "loopctl.mcp.structured.emitted",
            "loopctl.memory.consolidated",
            "loopctl.memory.extract.attempts",
            "loopctl.memory.extracted",
            "loopctl.memory.retrieve.results",
            "loopctl.memory.store.consolidate",
            "loopctl.memory.store.retrieve",
            "loopctl.prune.cleared",
            "loopctl.structured.prompted",
            "loopctl.structured.prompted.attempts_count",
            "loopctl.structured.prompted.tokens",
            "loopctl.think.calls",
            "loopctl.tools.calls",
            "loopctl.trajectory.jsonl_bytes",
            "loopctl.trajectory.records",
            "loopctl.vector.embed",
            "loopctl.vector.index.search",
            "loopctl.window.token_counter",
        ];
        for name in EMITTED_TODAY {
            assert!(
                lookup(name).is_some(),
                "emitted name not in REGISTRY: {name}"
            );
        }
    }

    #[test]
    fn all_is_registry() {
        assert_eq!(all().len(), REGISTRY.len());
    }
}

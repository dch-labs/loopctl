//! Per-run metrics value types: the typed snapshot a run produces and
//! an exporter reads.
//!
//! Pure data — construction, aggregation math, and serde. Nothing here
//! subscribes to anything; [`crate::metrics::registry`] says what the
//! metric names mean.
//!
//! The centerpiece is [`AgentMetrics`]: turn counts, a
//! [`LatencyHistogram`] per latency surface, per-tool [`ToolStats`], and
//! cumulative [`TokenTotals`]. A fresh [`Default`] snapshot belongs to
//! one run; identity and outcome land from the run boundaries and every
//! other field accumulates from callbacks along the way.
//!
//! # Examples
//!
//! ```
//! use loopctl::metrics::AgentMetrics;
//!
//! // A snapshot is a value: assemble, serialize, ship, re-read.
//! let mut m = AgentMetrics::default();
//! m.run_success = true;
//! m.turn_latency.record(250);
//! let json = serde_json::to_string(&m).expect("plain data");
//! let back: AgentMetrics = serde_json::from_str(&json).expect("round-trip");
//! assert_eq!(back, m);
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Cumulative token counts, including the cache and reasoning detail.
///
/// Mirrors [`crate::stream::Usage`] at run granularity: populated per
/// turn from the engine's authoritative per-turn totals and summed
/// across the run. Every field saturates on [`TokenTotals::add`], so a
/// long-running accumulation is a clamp, never a wrap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenTotals {
    /// Total input (prompt) tokens across the run.
    ///
    /// Every token the engine sent to the model, summed over all turns;
    /// the denominator of [`TokenTotals::cache_hit_rate`].
    #[serde(default)]
    pub input: u64,

    /// Total output (completion) tokens across the run.
    ///
    /// Every token the model generated, summed over all turns; the
    /// denominator of [`TokenTotals::reasoning_share`].
    #[serde(default)]
    pub output: u64,

    /// Input tokens served from provider prompt cache.
    ///
    /// Zero where the provider does not report the figure; a partial
    /// report is indistinguishable from silence, so treat it with the
    /// cache-hit ratio's [`Option::None`] discipline in mind.
    #[serde(default)]
    pub cached_input: u64,

    /// Input tokens written to prompt cache.
    ///
    /// Anthropic-style write accounting: tokens the provider charged to
    /// populate the cache this run, zero where unreported.
    #[serde(default)]
    pub cache_write: u64,

    /// Reasoning tokens inside `output`.
    ///
    /// Thinking tokens the model charged as output (o-series, Claude
    /// thinking); zero where the provider does not report them.
    #[serde(default)]
    pub reasoning: u64,
}

impl TokenTotals {
    /// Total tokens: input plus output, saturating.
    ///
    /// The headline size of a run. Cache fields are overlap, not
    /// additional tokens, so they never enter this sum.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::TokenTotals;
    ///
    /// let t = TokenTotals { input: 100, output: 25, ..Default::default() };
    /// assert_eq!(t.total(), 125);
    /// ```
    #[must_use]
    pub fn total(self) -> u64 {
        self.input.saturating_add(self.output)
    }

    /// Add another token total into this one, saturating per field.
    ///
    /// The accumulation primitive for turn-by-turn accounting: each
    /// settled turn contributes its usage and the running snapshot
    /// never wraps, whatever the run length.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::TokenTotals;
    ///
    /// let mut t = TokenTotals { input: 100, output: 25, ..Default::default() };
    /// t.add(TokenTotals { input: 50, output: 10, ..Default::default() });
    /// assert_eq!(t.total(), 185);
    /// ```
    pub fn add(&mut self, other: TokenTotals) {
        self.input = self.input.saturating_add(other.input);
        self.output = self.output.saturating_add(other.output);
        self.cached_input = self.cached_input.saturating_add(other.cached_input);
        self.cache_write = self.cache_write.saturating_add(other.cache_write);
        self.reasoning = self.reasoning.saturating_add(other.reasoning);
    }

    /// `cached_input / input` — the share of input served from cache.
    ///
    /// `None` when no input tokens were reported, distinguishing
    /// "nothing ran" from "nothing was cached". Conversions saturate
    /// through `u32` so the `f64` arithmetic stays exact for any real
    /// token count.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::TokenTotals;
    ///
    /// let t = TokenTotals { input: 100, cached_input: 25, ..Default::default() };
    /// assert_eq!(t.cache_hit_rate(), Some(0.25));
    /// assert_eq!(TokenTotals::default().cache_hit_rate(), None);
    /// ```
    #[must_use]
    pub fn cache_hit_rate(self) -> Option<f64> {
        if self.input == 0 {
            return None;
        }
        let cached = f64::from(u32::try_from(self.cached_input).unwrap_or(u32::MAX));
        let input = f64::from(u32::try_from(self.input).unwrap_or(u32::MAX));
        Some(cached / input)
    }

    /// `reasoning / output` — the share of output spent reasoning.
    ///
    /// `None` when no output tokens were reported. Same exact-for-reals
    /// `u32`-saturating conversion as [`TokenTotals::cache_hit_rate`].
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::TokenTotals;
    ///
    /// let t = TokenTotals { output: 40, reasoning: 10, ..Default::default() };
    /// assert_eq!(t.reasoning_share(), Some(0.25));
    /// assert_eq!(TokenTotals::default().reasoning_share(), None);
    /// ```
    #[must_use]
    pub fn reasoning_share(self) -> Option<f64> {
        if self.output == 0 {
            return None;
        }
        let reasoning = f64::from(u32::try_from(self.reasoning).unwrap_or(u32::MAX));
        let output = f64::from(u32::try_from(self.output).unwrap_or(u32::MAX));
        Some(reasoning / output)
    }
}

/// One percentile reading off a [`LatencyHistogram`].
///
/// The value is the upper bound of the bucket holding the rank's
/// sample — a conservative over-estimate, the right claim for an SLO
/// ("the p99 turn was in the ≤5000 ms bucket").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Percentile {
    /// Percentile rank this reading answers.
    ///
    /// One of the requested ranks (e.g. `50`, `90`, `99`), echoed so a
    /// lone [`Percentile`] says which question it answers.
    pub rank: u8,

    /// Latency in milliseconds at this rank.
    ///
    /// The upper bound of the bucket containing the rank's sample; for
    /// samples above the last bucket bound, that last bound itself.
    pub value_ms: u64,
}

/// The headline p50/p90/p99 triple for one latency surface.
///
/// Produced only when samples exist ([`LatencyHistogram::percentiles`]
/// answers `None` on an empty histogram), so its presence means at
/// least one sample was recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    /// Median sample reading.
    ///
    /// The 50th-percentile [`Percentile`] of the underlying histogram.
    pub p50: Percentile,

    /// 90th-percentile reading.
    ///
    /// The ninth-of-ten boundary; the usual alarm line for turn and
    /// tool latency.
    pub p90: Percentile,

    /// 99th-percentile reading.
    ///
    /// The tail reading; on small sample counts it is usually the same
    /// bucket as `p90`, which is honest for a bucketed estimate.
    pub p99: Percentile,
}

/// Fixed-bucket latency histogram in milliseconds.
///
/// Bucket bounds are ascending upper bounds in ms; the trailing count
/// slot collects samples above the last bound ("overflow"), and
/// percentile reads report that last bound for overflow samples.
/// Agent-loop latencies span roughly 5 ms to 60 s, which the default
/// scheme covers. Fixed buckets keep the dependency footprint at zero
/// and per-sample [`record`](Self::record) branch-light: one binary
/// search plus one saturating add.
///
/// [`merge`](Self::merge) combines same-scheme histograms only — a
/// foreign bucket scheme is refused, so cross-run aggregation is a
/// deliberate decision, never an accident of two schemes meeting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "LatencyHistogramWire")]
pub struct LatencyHistogram {
    /// Ascending, de-duplicated bucket upper bounds in ms.
    ///
    /// The scheme identity: two histograms are mergeable exactly when
    /// this list is equal.
    buckets: Vec<u64>,

    /// Per-slot sample counts, one per bucket plus an overflow slot.
    ///
    /// Always `buckets.len() + 1` entries; the last is the count of
    /// samples above the last bound.
    counts: Vec<u64>,
}

/// Wire shape for deserialization of a [`LatencyHistogram`].
///
/// Tolerates hand-edited or partial JSON by re-normalizing on the way
/// in: bounds are sorted and de-duplicated (an empty set falls back to
/// [`default_buckets`]) and counts are resized to the slot count, so a
/// degraded document yields a valid, if conservative, histogram rather
/// than an error or a panic.
#[derive(Deserialize)]
struct LatencyHistogramWire {
    /// Bucket upper bounds as written.
    ///
    /// Any order, any repetition; normalized by the [`From`] conversion
    /// before a [`LatencyHistogram`] ever sees them.
    #[serde(default)]
    buckets: Vec<u64>,

    /// Slot counts as written.
    ///
    /// Short lists read zero-padded to the slot count; surplus entries
    /// are dropped by the same conversion.
    #[serde(default)]
    counts: Vec<u64>,
}

impl From<LatencyHistogramWire> for LatencyHistogram {
    fn from(wire: LatencyHistogramWire) -> Self {
        let mut buckets = wire.buckets;
        buckets.sort_unstable();
        buckets.dedup();
        if buckets.is_empty() {
            buckets = default_buckets();
        }
        let mut counts = wire.counts;
        counts.resize(buckets.len().saturating_add(1), 0);
        Self { buckets, counts }
    }
}

/// The default bucket bounds in ms, covering the agent-loop latency band.
///
/// Bounds are 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000,
/// 30000, 60000, with an implicit overflow slot above 60 s. Shared by
/// [`LatencyHistogram::new`] and the deserialization fallback.
///
/// # Examples
///
/// ```
/// use loopctl::metrics::session::default_buckets;
///
/// let buckets = default_buckets();
/// assert_eq!(buckets.first(), Some(&5));
/// assert_eq!(buckets.last(), Some(&60_000));
/// ```
#[must_use]
pub fn default_buckets() -> Vec<u64> {
    vec![
        5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000,
    ]
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyHistogram {
    /// Empty histogram on the [`default_buckets`] scheme.
    ///
    /// The constructor for the ordinary case; use
    /// [`with_buckets`](Self::with_buckets) only when a coarser or
    /// finer scheme is genuinely wanted.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::LatencyHistogram;
    ///
    /// let h = LatencyHistogram::new();
    /// assert_eq!(h.count(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::with_buckets(default_buckets())
    }

    /// Empty histogram on a custom bucket scheme.
    ///
    /// Bounds are sorted and de-duplicated; a list that empties out
    /// falls back to [`default_buckets`], so no input produces an
    /// unusable histogram.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::LatencyHistogram;
    ///
    /// let h = LatencyHistogram::with_buckets(vec![10, 10, 5]);
    /// assert_eq!(h.buckets(), &[5, 10]);
    /// ```
    #[must_use]
    pub fn with_buckets(mut buckets: Vec<u64>) -> Self {
        buckets.sort_unstable();
        buckets.dedup();
        if buckets.is_empty() {
            buckets = default_buckets();
        }
        let counts = vec![0_u64; buckets.len().saturating_add(1)];
        Self { buckets, counts }
    }

    /// The bucket upper bounds of this histogram's scheme.
    ///
    /// Ascending and de-duplicated as constructed; equality of two
    /// slices is exactly the [`merge`](Self::merge) precondition.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::LatencyHistogram;
    ///
    /// let h = LatencyHistogram::new();
    /// assert_eq!(h.buckets().first(), Some(&5));
    /// ```
    #[must_use]
    pub fn buckets(&self) -> &[u64] {
        &self.buckets
    }

    /// Record one latency sample in milliseconds.
    ///
    /// The sample lands in the first bucket whose bound is not below
    /// it; above the last bound it lands in the overflow slot. The
    /// operation is branch-light by design — it runs per turn and per
    /// tool call.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::LatencyHistogram;
    ///
    /// let mut h = LatencyHistogram::new();
    /// h.record(120);
    /// h.record(40);
    /// assert_eq!(h.count(), 2);
    /// ```
    pub fn record(&mut self, ms: u64) {
        let slot = self.buckets.partition_point(|bound| *bound < ms);
        if let Some(count) = self.counts.get_mut(slot) {
            *count = count.saturating_add(1);
        }
    }

    /// Total recorded samples, overflow included.
    ///
    /// Saturating sum over the slots; the denominator behind every
    /// "of how many samples?" question an exporter must answer.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::LatencyHistogram;
    ///
    /// let mut h = LatencyHistogram::new();
    /// h.record(50);
    /// assert_eq!(h.count(), 1);
    /// ```
    #[must_use]
    pub fn count(&self) -> u64 {
        self.counts
            .iter()
            .fold(0_u64, |acc, c| acc.saturating_add(*c))
    }

    /// The `pct`-th percentile: the bucket bound holding the rank's sample.
    ///
    /// The sampled rank is `floor(N * pct / 100) + 1` — strict
    /// majority, "more than pct% of samples are at or below this
    /// value". The reading names the upper bound of the bucket holding
    /// that sample; a sample in the overflow slot reads the last bound
    /// (an index the bound lookup answers with the last element rather
    /// than a slot that has no bound of its own). `None` for an empty
    /// histogram or a rank outside 1..=100.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::LatencyHistogram;
    ///
    /// let mut h = LatencyHistogram::new();
    /// h.record(120);
    /// h.record(40);
    /// let p50 = h.percentile(50).expect("two samples");
    /// assert_eq!(p50.rank, 50);
    /// assert_eq!(p50.value_ms, 250);
    /// ```
    #[must_use]
    pub fn percentile(&self, pct: u8) -> Option<Percentile> {
        if pct == 0 || pct > 100 {
            return None;
        }
        let total = self.count();
        if total == 0 {
            return None;
        }
        let target = total
            .saturating_mul(u64::from(pct))
            .div_euclid(100)
            .saturating_add(1);
        let mut cumulative = 0_u64;
        for (slot, count) in self.counts.iter().enumerate() {
            cumulative = cumulative.saturating_add(*count);
            if cumulative >= target {
                let value_ms = self
                    .buckets
                    .get(slot)
                    .or_else(|| self.buckets.last())
                    .copied()?;
                return Some(Percentile {
                    rank: pct,
                    value_ms,
                });
            }
        }
        None
    }

    /// The p50/p90/p99 triple, if any samples exist.
    ///
    /// Convenience over three [`percentile`](Self::percentile) calls;
    /// `None` whenever the histogram is empty, so the triple's presence
    /// always guarantees at least one sample behind it.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::LatencyHistogram;
    ///
    /// let mut h = LatencyHistogram::new();
    /// h.record(50);
    /// let p = h.percentiles().expect("one sample");
    /// assert_eq!(p.p50.value_ms, p.p99.value_ms);
    /// assert_eq!(p.p99.rank, 99);
    /// ```
    #[must_use]
    pub fn percentiles(&self) -> Option<LatencyPercentiles> {
        let p50 = self.percentile(50)?;
        let p90 = self.percentile(90)?;
        let p99 = self.percentile(99)?;
        Some(LatencyPercentiles { p50, p90, p99 })
    }

    /// Add another histogram's counts into this one.
    ///
    /// Only same-scheme merges are meaningful: returns `false` and
    /// changes nothing when the bucket schemes differ, so a refused
    /// merge is never a half-counted sample set.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::LatencyHistogram;
    ///
    /// let mut a = LatencyHistogram::new();
    /// a.record(50);
    /// let mut b = LatencyHistogram::new();
    /// b.record(500);
    /// assert!(a.merge(&b));
    /// assert_eq!(a.count(), 2);
    /// ```
    pub fn merge(&mut self, other: &Self) -> bool {
        if self.buckets != other.buckets {
            return false;
        }
        for (mine, theirs) in self.counts.iter_mut().zip(other.counts.iter()) {
            *mine = mine.saturating_add(*theirs);
        }
        true
    }

    /// Zero all counts, keeping the bucket scheme.
    ///
    /// The per-run reset path: the scheme survives so histograms from
    /// consecutive runs stay [`merge`](Self::merge)-able.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::LatencyHistogram;
    ///
    /// let mut h = LatencyHistogram::new();
    /// h.record(50);
    /// h.reset();
    /// assert_eq!(h.count(), 0);
    /// assert_eq!(h.buckets().first(), Some(&5));
    /// ```
    pub fn reset(&mut self) {
        self.counts.fill(0);
    }
}

/// Per-tool call statistics for one run.
///
/// One entry per tool the run actually called, holding attempt counts,
/// error counts, and the call-latency distribution — enough for a
/// per-tool error rate and a latency read without any raw logs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolStats {
    /// Tool name, the [`ToolStatsMap`] key.
    ///
    /// Stored as well as keyed so a borrowed [`ToolStats`] carries its
    /// own identity; the key and this field always agree.
    #[serde(default)]
    pub tool: String,

    /// Call attempts.
    ///
    /// One per settled call, retries included: a failure followed by a
    /// successful retry counts twice, which is what makes
    /// [`error_rate`](Self::error_rate) an attempt-weighted rate.
    #[serde(default)]
    pub calls: u64,

    /// Attempts that ended as errors.
    ///
    /// The numerator of [`error_rate`](Self::error_rate); an attempt
    /// the caller observed as failed, by the caller's own error signal.
    #[serde(default)]
    pub errors: u64,

    /// Call-latency distribution in ms.
    ///
    /// One sample per attempt on the default bucket scheme; percentiles
    /// read off it follow the same bucket-bound semantics as turn
    /// latency.
    #[serde(default)]
    pub latency: LatencyHistogram,
}

impl ToolStats {
    /// Empty stats for a named tool.
    ///
    /// The constructor behind [`AgentMetrics::record_tool`]'s
    /// first-sight insert; latency starts empty until the first call
    /// lands.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::ToolStats;
    ///
    /// let mut s = ToolStats::new("read");
    /// s.record(10, false);
    /// s.record(20, true);
    /// assert_eq!(s.calls, 2);
    /// assert_eq!(s.errors, 1);
    /// assert!((s.error_rate() - 0.5).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn new(tool: &str) -> Self {
        Self {
            tool: tool.to_string(),
            calls: 0,
            errors: 0,
            latency: LatencyHistogram::new(),
        }
    }

    /// Record one settled call attempt: duration in ms, error flag.
    ///
    /// Increments [`calls`](Self::calls) (and [`errors`](Self::errors)
    /// when flagged) and samples [`latency`](Self::latency). The shape
    /// matches what a tool-completion callback knows at settle time.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::ToolStats;
    ///
    /// let mut s = ToolStats::new("exec");
    /// s.record(1_000, true);
    /// assert_eq!(s.calls, 1);
    /// assert_eq!(s.errors, 1);
    /// assert_eq!(s.latency.count(), 1);
    /// ```
    pub fn record(&mut self, duration_ms: u64, is_error: bool) {
        self.calls = self.calls.saturating_add(1);
        if is_error {
            self.errors = self.errors.saturating_add(1);
        }
        self.latency.record(duration_ms);
    }

    /// `errors / calls` as a rate; `0.0` when nothing ran.
    ///
    /// An empty tool record reads as "no errors observed", not
    /// "unknown" — an absent tool is absent from the map, so this
    /// function only ever sees tools that ran. Saturates through `u32`
    /// like the token ratios: exact for any realistic per-run count.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::ToolStats;
    ///
    /// let mut s = ToolStats::new("web");
    /// assert_eq!(s.error_rate(), 0.0);
    /// s.record(10, true);
    /// s.record(10, false);
    /// assert_eq!(s.error_rate(), 0.5);
    /// ```
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        if self.calls == 0 {
            return 0.0;
        }
        let errors = f64::from(u32::try_from(self.errors).unwrap_or(u32::MAX));
        let calls = f64::from(u32::try_from(self.calls).unwrap_or(u32::MAX));
        errors / calls
    }
}

/// Per-tool stats keyed by tool name.
///
/// A `BTreeMap` for deterministic serialization order — export diffs
/// stay stable between runs that made the same calls.
pub type ToolStatsMap = BTreeMap<String, ToolStats>;

/// A per-run metrics snapshot: the typed artifact of one run.
///
/// A fresh [`Default`] snapshot belongs to one run. Run identity and
/// outcome land from the run boundaries (the
/// [`RunStartContext`](crate::observer::RunStartContext) /
/// [`RunEndContext`](crate::observer::RunEndContext) pair); turn, tool,
/// and event fields accumulate from the matching
/// [`LoopObserver`](crate::observer::LoopObserver) callbacks as the run
/// proceeds, so a filled snapshot serializes to everything a dashboard
/// or export needs and nothing the run did not do.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Run identity: the session/run uuid.
    ///
    /// From [`RunStartContext::session_id`](crate::observer::RunStartContext::session_id);
    /// nil on a snapshot built without it (deserialization default).
    #[serde(default)]
    pub session_id: uuid::Uuid,

    /// Whether the run completed successfully.
    ///
    /// From [`RunEndContext::success`](crate::observer::RunEndContext::success):
    /// `true` only on a normal exit, `false` on error or cancellation.
    #[serde(default)]
    pub run_success: bool,

    /// Terminal error class, if the run ended in error.
    ///
    /// From [`RunEndContext::error`](crate::observer::RunEndContext::error);
    /// `None` exactly when [`run_success`](Self::run_success) is `true`.
    #[serde(default)]
    pub run_error: Option<String>,

    /// Whole-run wall clock in ms.
    ///
    /// From [`RunEndContext::duration_ms`](crate::observer::RunEndContext::duration_ms);
    /// zero where the run boundary did not report one.
    #[serde(default)]
    pub run_duration_ms: u64,

    /// Turns completed during the run.
    ///
    /// One per settled turn; an in-flight turn at fatal error is not
    /// counted, matching the run-end context's own accounting.
    #[serde(default)]
    pub total_turns: u64,

    /// Turns that ended unsuccessfully.
    ///
    /// The subset of [`total_turns`](Self::total_turns) whose
    /// turn-end reported failure; the turn-level error numerator.
    #[serde(default)]
    pub failed_turns: u64,

    /// Turn-latency distribution in ms.
    ///
    /// One sample per settled turn;
    /// [`turn_latency_percentiles`](Self::turn_latency_percentiles)
    /// is the usual read.
    #[serde(default)]
    pub turn_latency: LatencyHistogram,

    /// Cumulative tokens, cache and reasoning detail included.
    ///
    /// Summed from the engine's per-turn totals; the source for
    /// cache-hit and reasoning-share reads.
    #[serde(default)]
    pub tokens: TokenTotals,

    /// Successful model streams.
    ///
    /// One per stream that completed without error; with
    /// [`stream_failures`](Self::stream_failures) this is the
    /// stream-level success/failure pair.
    #[serde(default)]
    pub stream_successes: u64,

    /// Failed model streams.
    ///
    /// One per stream that ended in an error path, retries at the
    /// provider level already resolved.
    #[serde(default)]
    pub stream_failures: u64,

    /// Per-tool stats, keyed by tool name.
    ///
    /// Populated on first call per tool; see [`ToolStats`] for the
    /// attempt-counting rule.
    #[serde(default)]
    pub tools: ToolStatsMap,

    /// Completed compactions.
    ///
    /// One per compaction that changed the context; a no-action pass
    /// does not count.
    #[serde(default)]
    pub compactions: u64,

    /// Failed compactions.
    ///
    /// One per compaction attempt that errored; failures do not
    /// decrement [`compactions`](Self::compactions).
    #[serde(default)]
    pub compaction_failures: u64,

    /// Model fallbacks.
    ///
    /// One per switch to a fallback model after a provider refusal or
    /// outage signaled mid-run.
    #[serde(default)]
    pub fallbacks: u64,

    /// Transport downgrades.
    ///
    /// One per mid-run change of transport (e.g. streaming to
    /// non-streaming) taken to survive a wire-level failure.
    #[serde(default)]
    pub transport_fallbacks: u64,

    /// Mid-run model switches.
    ///
    /// One per deliberate model change through the run's model-switch
    /// surface, fallbacks aside.
    #[serde(default)]
    pub model_switches: u64,

    /// Budget warnings.
    ///
    /// One per warning the engine emitted on approaching a budget
    /// bound (tokens, cost, or turns, per its configuration).
    #[serde(default)]
    pub budget_warnings: u64,

    /// Loop-detection firings.
    ///
    /// One per loop-detector trip during the run.
    #[serde(default)]
    pub loops_detected: u64,

    /// Convergence-detection firings.
    ///
    /// One per convergence-detector trip; pairs with
    /// [`loops_detected`](Self::loops_detected) as the two guardrails.
    #[serde(default)]
    pub convergences_detected: u64,
}

impl AgentMetrics {
    /// Headline turn-latency p50/p90/p99, if any turns ran.
    ///
    /// Reads [`turn_latency`](Self::turn_latency); `None` on a run with
    /// zero settled turns, distinguishing "no turns" from "fast turns".
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::AgentMetrics;
    ///
    /// let mut m = AgentMetrics::default();
    /// m.turn_latency.record(300);
    /// let p = m.turn_latency_percentiles().expect("one turn");
    /// assert_eq!(p.p99.value_ms, 500);
    /// ```
    #[must_use]
    pub fn turn_latency_percentiles(&self) -> Option<LatencyPercentiles> {
        self.turn_latency.percentiles()
    }

    /// Aggregate tool-call attempts across all tools.
    ///
    /// Sums [`ToolStats::calls`] over the map; retries included, since
    /// attempts, not logical calls, are the counted unit.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::AgentMetrics;
    ///
    /// let mut m = AgentMetrics::default();
    /// m.record_tool("read", 5, false);
    /// m.record_tool("read", 7, false);
    /// assert_eq!(m.total_tool_calls(), 2);
    /// ```
    #[must_use]
    pub fn total_tool_calls(&self) -> u64 {
        self.tools
            .values()
            .fold(0_u64, |acc, s| acc.saturating_add(s.calls))
    }

    /// Aggregate tool errors across all tools.
    ///
    /// Sums [`ToolStats::errors`] over the map; pair with
    /// [`total_tool_calls`](Self::total_tool_calls) for the run-level
    /// tool error rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::AgentMetrics;
    ///
    /// let mut m = AgentMetrics::default();
    /// m.record_tool("exec", 50, true);
    /// m.record_tool("read", 5, false);
    /// assert_eq!(m.total_tool_errors(), 1);
    /// ```
    #[must_use]
    pub fn total_tool_errors(&self) -> u64 {
        self.tools
            .values()
            .fold(0_u64, |acc, s| acc.saturating_add(s.errors))
    }

    /// Record one settled tool-call attempt.
    ///
    /// Creates the tool's [`ToolStats`] on first sight and records the
    /// sample. One pair per attempt — a retry is another attempt, so a
    /// failure followed by a successful retry yields `calls = 2,
    /// errors = 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::AgentMetrics;
    ///
    /// let mut m = AgentMetrics::default();
    /// m.record_tool("web", 100, true);
    /// m.record_tool("web", 50, false);
    /// let s = m.tools.get("web").expect("recorded");
    /// assert_eq!(s.calls, 2);
    /// assert_eq!(s.errors, 1);
    /// ```
    pub fn record_tool(&mut self, tool: &str, duration_ms: u64, is_error: bool) {
        self.tools
            .entry(tool.to_string())
            .or_insert_with(|| ToolStats::new(tool))
            .record(duration_ms, is_error);
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    fn sample_uuid() -> uuid::Uuid {
        uuid::Uuid::parse_str("3f2504e0-4f89-11d3-9a0c-0305e82c3301").unwrap()
    }

    #[test]
    fn record_places_samples_at_bucket_edges() {
        let mut h = LatencyHistogram::new();
        for ms in [0, 5, 10, 100, 5_000, 70_000] {
            h.record(ms);
        }
        assert_eq!(h.count(), 6);
        // Rank 3 (p50 of six under strict-majority) is the 100 sample in
        // the 100 bucket; the 99th percentile is the 70000 overflow,
        // reported at the last bound.
        assert_eq!(h.percentile(50).map(|p| p.value_ms), Some(100));
        assert_eq!(h.percentile(99).map(|p| p.value_ms), Some(60_000));
    }

    #[test]
    fn percentiles_of_known_distributions() {
        let mut all_250 = LatencyHistogram::new();
        for _ in 0..100 {
            all_250.record(250);
        }
        assert_eq!(all_250.percentile(50).map(|p| p.value_ms), Some(250));
        assert_eq!(all_250.percentile(90).map(|p| p.value_ms), Some(250));
        assert_eq!(all_250.percentile(99).map(|p| p.value_ms), Some(250));

        let mut mixed = LatencyHistogram::new();
        for _ in 0..10 {
            mixed.record(5);
        }
        for _ in 0..80 {
            mixed.record(100);
        }
        for _ in 0..10 {
            mixed.record(5_000);
        }
        assert_eq!(mixed.percentile(50).map(|p| p.value_ms), Some(100));
        assert_eq!(mixed.percentile(90).map(|p| p.value_ms), Some(5_000));
        assert_eq!(mixed.percentile(99).map(|p| p.value_ms), Some(5_000));
    }

    #[test]
    fn empty_percentiles_are_none() {
        let h = LatencyHistogram::new();
        assert_eq!(h.count(), 0);
        assert!(h.percentile(50).is_none());
        assert!(h.percentiles().is_none());
        assert!(h.percentile(0).is_none());
        assert!(h.percentile(101).is_none());
    }

    #[test]
    fn percentiles_triple_is_consistent() {
        let mut h = LatencyHistogram::new();
        for ms in [7_u64, 7, 400, 400, 400, 61_000] {
            h.record(ms);
        }
        let p = h.percentiles().unwrap();
        assert_eq!(p.p50, h.percentile(50).unwrap());
        assert_eq!(p.p90, h.percentile(90).unwrap());
        assert_eq!(p.p99, h.percentile(99).unwrap());
    }

    #[test]
    fn merge_requires_same_scheme() {
        let mut a = LatencyHistogram::new();
        a.record(50);
        let mut b = LatencyHistogram::new();
        b.record(500);
        assert!(a.merge(&b), "same scheme merges");
        assert_eq!(a.count(), 2);

        let mut custom = LatencyHistogram::with_buckets(vec![100]);
        assert!(!a.merge(&custom), "foreign scheme must be rejected");
        assert_eq!(a.count(), 2, "rejected merge changes nothing");
        custom.record(1);
        assert_eq!(custom.count(), 1);
    }

    #[test]
    fn reset_keeps_scheme() {
        let mut h = LatencyHistogram::with_buckets(vec![1, 2, 4, 8]);
        h.record(3);
        assert_eq!(h.buckets(), &[1, 2, 4, 8]);
        h.reset();
        assert_eq!(h.count(), 0);
        assert_eq!(h.buckets(), &[1, 2, 4, 8]);
        assert!(h.percentile(99).is_none());
    }

    #[test]
    fn wire_normalizes() {
        let empty: LatencyHistogram = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.buckets(), default_buckets().as_slice());
        assert_eq!(empty.count(), 0);
        let messy: LatencyHistogram =
            serde_json::from_str(r#"{"buckets":[10,5,5],"counts":[1]}"#).unwrap();
        assert_eq!(messy.buckets(), &[5, 10]);
        assert_eq!(messy.count(), 1);
    }

    #[test]
    fn token_totals_math() {
        let mut t = TokenTotals {
            input: 200,
            output: 100,
            cached_input: 50,
            cache_write: 10,
            reasoning: 40,
        };
        t.add(TokenTotals {
            input: 1,
            output: 2,
            cached_input: 3,
            cache_write: 4,
            reasoning: 5,
        });
        assert_eq!(t.total(), 303);
        assert_eq!(t.cached_input, 53);
        assert_eq!(t.reasoning, 45);
        let rate = t.cache_hit_rate().unwrap();
        assert!((rate - 53.0 / 201.0).abs() < 1e-12);
        let share = t.reasoning_share().unwrap();
        assert!((share - 45.0 / 102.0).abs() < 1e-12);

        assert!(TokenTotals::default().cache_hit_rate().is_none());
        assert!(TokenTotals::default().reasoning_share().is_none());

        let mut sat = TokenTotals {
            input: u64::MAX,
            ..TokenTotals::default()
        };
        sat.add(TokenTotals {
            input: 5,
            output: u64::MAX,
            ..TokenTotals::default()
        });
        assert_eq!(sat.input, u64::MAX);
        assert_eq!(sat.output, u64::MAX);
    }

    #[test]
    fn tool_stats_records() {
        let mut s = ToolStats::new("read");
        s.record(10, false);
        s.record(20, true);
        s.record(20, false);
        assert_eq!(s.tool, "read");
        assert_eq!(s.calls, 3);
        assert_eq!(s.errors, 1);
        assert!((s.error_rate() - 1.0 / 3.0).abs() < 1e-12);
        assert!((ToolStats::new("x").error_rate() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn agent_metrics_serde_round_trips_and_fills_defaults() {
        let mut m = AgentMetrics::default();
        m.session_id = sample_uuid();
        m.run_success = true;
        m.total_turns = 3;
        m.failed_turns = 1;
        m.turn_latency.record(400);
        m.tokens = TokenTotals {
            input: 900,
            output: 100,
            cached_input: 300,
            cache_write: 0,
            reasoning: 20,
        };
        m.record_tool("read", 10, false);
        m.record_tool("exec", 50, true);
        m.run_error = Some("tool_timeout".to_string());
        let json = serde_json::to_string(&m).unwrap();
        let back: AgentMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);

        // Forward-compat: partial JSON fills the rest with defaults.
        let partial = format!(r#"{{"session_id":"{id}"}}"#, id = sample_uuid());
        let partial_m: AgentMetrics = serde_json::from_str(&partial).unwrap();
        assert_eq!(partial_m.session_id, sample_uuid());
        assert!(!partial_m.run_success);
        assert_eq!(partial_m.total_turns, 0);
        assert!(partial_m.tools.is_empty());
    }

    #[test]
    fn tool_map_order_is_deterministic() {
        let mut m = AgentMetrics::default();
        m.record_tool("zeta", 1, false);
        m.record_tool("alpha", 1, false);
        let json = serde_json::to_string(&m.tools).unwrap();
        let a = json.find("alpha").unwrap();
        let z = json.find("zeta").unwrap();
        assert!(a < z, "BTreeMap key order in JSON: {json}");
    }

    /// A run-shaped drive: the whole callback sequence expressed
    /// directly on the value types.
    #[test]
    fn run_shaped_aggregation() {
        // reset(): fresh snapshot per run.
        let mut m = AgentMetrics::default();
        // on_run_start: identity.
        m.session_id = sample_uuid();
        // 3x on_turn_end (one failed; five-field tokens).
        let turns = [
            (500_u64, true, (300_u64, 40_u64, 150_u64, 0_u64, 10_u64)),
            (800_u64, true, (500_u64, 60_u64, 250_u64, 0_u64, 20_u64)),
            (900_u64, false, (0_u64, 0_u64, 0_u64, 0_u64, 0_u64)),
        ];
        for (ms, ok, tok) in turns {
            m.total_turns = m.total_turns.saturating_add(1);
            if !ok {
                m.failed_turns = m.failed_turns.saturating_add(1);
            }
            m.turn_latency.record(ms);
            m.tokens.add(TokenTotals {
                input: tok.0,
                output: tok.1,
                cached_input: tok.2,
                cache_write: tok.3,
                reasoning: tok.4,
            });
        }
        // 4x on_tool_post across 2 tools.
        m.record_tool("read", 5, false);
        m.record_tool("read", 7, false);
        m.record_tool("exec", 1_000, true);
        m.record_tool("exec", 900, false);
        // stream + event counters.
        m.stream_successes = 2;
        m.stream_failures = 1;
        m.compactions = 1;
        m.fallbacks = 1;
        m.budget_warnings = 2;
        // on_run_end.
        m.run_success = true;
        m.run_duration_ms = 4_000;

        assert_eq!(m.total_turns, 3);
        assert_eq!(m.failed_turns, 1);
        assert_eq!(m.turn_latency.count(), 3);
        // 800 ms median rounds up to the 1000 ms bucket bound.
        assert_eq!(
            m.turn_latency_percentiles().map(|p| p.p50.value_ms),
            Some(1_000)
        );
        assert_eq!(m.tokens.input, 800);
        assert_eq!(m.tokens.cached_input, 400);
        assert_eq!(m.tokens.reasoning, 30);
        assert!((m.tokens.cache_hit_rate().unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(m.total_tool_calls(), 4);
        assert_eq!(m.total_tool_errors(), 1);
        assert_eq!(m.tools.get("read").map(|s| s.calls), Some(2));
        assert_eq!(m.tools.get("exec").map(|s| s.errors), Some(1));
        assert!(m.run_success);
        assert_eq!(m.run_duration_ms, 4_000);
    }

    #[test]
    fn record_tool_splits_attempt_counts() {
        let mut m = AgentMetrics::default();
        m.record_tool("web", 100, true);
        m.record_tool("web", 50, false);
        let s = m.tools.get("web").unwrap();
        assert_eq!(s.calls, 2, "one failed attempt plus one retry = 2");
        assert_eq!(s.errors, 1);
    }
}

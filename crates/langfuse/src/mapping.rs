//! loopctl → Langfuse mapping: classification, first-class fields, scores.
//!
//! The classification keys on the engine's span-name contract (`agent.run`,
//! `agent.turn`, `agent.tool`), so a renamed span in core renames here at
//! compile time via [`loopctl::tracing_spans`] constants. Field mapping is
//! deliberately small: Langfuse renders a handful of keys natively
//! (`session_id`, `model`, `usage_details.*`, `level`); everything else
//! rides as free-form metadata under its emitted name — Langfuse metadata
//! is open-schema and the fields are already spec-shaped at emission.

use loopctl::metrics::AgentMetrics;
// Re-exported so consumers can key off the same span-name contract.
pub use loopctl::tracing_spans::{AGENT_RUN_SPAN, TOOL_SPAN, TURN_SPAN};

/// What a closed span becomes in Langfuse's model.
///
/// The classification is decided once, from the span name at open, and
/// travels with the span's meta; the close path only renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LangfuseTarget {
    /// One run → one trace (with session grouping).
    ///
    /// Carries the `sessionId` when the run recorded a conversation
    /// id; nested observations reference this trace's id.
    Trace,

    /// A turn or tool execution → an observation under the trace.
    ///
    /// Inherits trace, session, and parent-observation ids from the
    /// nearest instrumented ancestor; without one it is orphaned and
    /// never exported.
    Observation,
}

/// Route a span to trace or observation by its tracing name.
///
/// Keyed on the core span-name contract, so renaming `agent.run`
/// there renames the constant this compares against and lands here at
/// compile time. Everything that is not the run span — turns, tools,
/// and any foreign span — classifies as an observation.
#[must_use]
pub fn classify_span(span_name: &str) -> LangfuseTarget {
    if span_name == AGENT_RUN_SPAN {
        LangfuseTarget::Trace
    } else {
        LangfuseTarget::Observation
    }
}

/// A Langfuse first-class destination for an emitted field.
///
/// Langfuse renders a small set of keys natively — session, model,
/// usage details, level; [`first_class`] routes emitted names onto
/// these slots, and anything it does not recognize rides as
/// free-form metadata under its emitted name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LangfuseField {
    /// Trace `sessionId`.
    ///
    /// Fed from `gen_ai.conversation.id`; groups related runs in the
    /// Langfuse sessions view.
    SessionId,

    /// Observation `model` (and the usage block's model echo).
    ///
    /// Fed from `gen_ai.request.model`; traces keep it in metadata,
    /// observations render it in the dedicated field.
    Model,

    /// `usage_details.input`.
    ///
    /// Non-cached input tokens; accepted under both the loopctl
    /// `gen_ai.usage.input_tokens` name and the spec's
    /// `gen_ai.usage.input`.
    UsageInput,

    /// `usage_details.output`.
    ///
    /// Generated tokens, emitted as `gen_ai.usage.output_tokens`.
    UsageOutput,

    /// `usage_details.cached_input`.
    ///
    /// Cache-read tokens — the numerator of the `cache_hit_rate`
    /// score.
    UsageCachedInput,

    /// `usage_details.cache_write`.
    ///
    /// Cache-creation tokens; part of that rate's denominator.
    UsageCacheWrite,

    /// `usage_details.reasoning`.
    ///
    /// Thinking tokens billed outside the visible output, rendered as
    /// their own usage line.
    UsageReasoning,

    /// Observation `level = "ERROR"` (the value also rides as metadata).
    ///
    /// Any presence of `error.type` marks the object errored; the
    /// classification itself stays in metadata for dashboards.
    ErrorType,

    /// Observation end-time status: `outcome = "error"` marks the level.
    ///
    /// `"ok"` changes nothing; together with [`LangfuseField::ErrorType`]
    /// it means either vocabulary can flag a failed span.
    Outcome,

    /// Never exported — the turn span's assembled-prompt field.
    ///
    /// The engine records `query` on DEBUG turn spans for hosts that
    /// opt into content capture; the exporter never uploads that text
    /// (or the PII inside it), so the field maps here instead of
    /// falling through to free-form metadata.
    Suppressed,
}

/// Map an emitted attribute name to its Langfuse-native slot, if any.
///
/// `None` means "free-form metadata under the emitted name" — the pass-
/// through path, correct for every `gen_ai.*` key Langfuse has no column
/// for (tool names, call ids, durations) and every loopctl extra. The
/// one exception is the turn span's `query`, which maps to
/// [`LangfuseField::Suppressed`] so the assembled prompt never reaches
/// that pass-through path.
#[must_use]
pub fn first_class(field: &str) -> Option<LangfuseField> {
    match field {
        "gen_ai.conversation.id" => Some(LangfuseField::SessionId),
        "gen_ai.request.model" => Some(LangfuseField::Model),
        "gen_ai.usage.input_tokens" | "gen_ai.usage.input" => Some(LangfuseField::UsageInput),
        "gen_ai.usage.output_tokens" => Some(LangfuseField::UsageOutput),
        "gen_ai.usage.cache_read.input_tokens" => Some(LangfuseField::UsageCachedInput),
        "gen_ai.usage.cache_creation.input_tokens" => Some(LangfuseField::UsageCacheWrite),
        "gen_ai.response.reasoning_tokens" => Some(LangfuseField::UsageReasoning),
        "error.type" => Some(LangfuseField::ErrorType),
        "outcome" => Some(LangfuseField::Outcome),
        "query" => Some(LangfuseField::Suppressed),
        _ => None,
    }
}

/// One score on a trace, numeric or categorical.
///
/// Built for the standard set by [`metrics_to_scores`]; exactly one of
/// [`LangfuseScore::value`] or [`LangfuseScore::string_value`] is
/// populated, and the wire builder follows Langfuse's `dataType`
/// convention for the categorical case.
#[derive(Debug, Clone, PartialEq)]
pub struct LangfuseScore {
    /// Score name (dashboard key).
    ///
    /// Dashboards pin on these names, so a change here is a breaking
    /// change for hosts with saved views.
    pub name: String,

    /// Numeric value; mutually exclusive with `string_value`.
    ///
    /// Set for token totals, turns, latency, and rates. On the wire a
    /// categorical score still ships a numeric `value` — a zero
    /// anchor beside `stringValue` — but the struct keeps exactly one
    /// field populated.
    pub value: Option<f64>,

    /// Categorical value (e.g. `"success"`).
    ///
    /// The `outcome` score is today's only categorical entry; Langfuse
    /// renders it with `dataType: "CATEGORICAL"`.
    pub string_value: Option<String>,
}

impl LangfuseScore {
    /// Build a numeric score — the dashboard's plotted form.
    ///
    /// The shape behind the token, turn, latency, and rate entries;
    /// `string_value` stays `None` so the wire builder emits a plain
    /// `value` and nothing else.
    fn numeric(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value: Some(value),
            string_value: None,
        }
    }

    /// Build a categorical score — Langfuse's named-outcome form.
    ///
    /// `value` stays `None` on the struct; the wire builder anchors
    /// the numeric field at zero and carries the category in
    /// `stringValue` under `dataType: "CATEGORICAL"`, the shape the
    /// cloud API requires.
    fn categorical(name: &str, value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: None,
            string_value: Some(value.to_string()),
        }
    }
}

/// Derive the standard score set for one closed run.
///
/// Names and definitions: `input_tokens`, `output_tokens`, `token_total`
/// (`input + output`), `total_turns`, `p90_turn_latency_ms` (omitted
/// when the reading sits in the histogram's overflow slot — a bound
/// standing in for an unbounded sample is not a latency),
/// `tool_error_rate` (errors over calls across tools), `cache_hit_rate`
/// (`cached_input / input` when the denominator is positive), and
/// categorical `outcome` (`"success"`/`"error"`).
#[must_use]
pub fn metrics_to_scores(m: &AgentMetrics) -> Vec<LangfuseScore> {
    let mut scores = vec![
        LangfuseScore::numeric("input_tokens", m.tokens.input.to_f64()),
        LangfuseScore::numeric("output_tokens", m.tokens.output.to_f64()),
        LangfuseScore::numeric("token_total", token_total(m).to_f64()),
        LangfuseScore::numeric("total_turns", m.total_turns.to_f64()),
        LangfuseScore::categorical("outcome", if m.run_success { "success" } else { "error" }),
    ];
    if let Some(percentiles) = m.turn_latency_percentiles()
        && !percentiles.p90.overflow
    {
        scores.push(LangfuseScore::numeric(
            "p90_turn_latency_ms",
            percentiles.p90.value_ms.to_f64(),
        ));
    }
    if let Some(rate) = tool_error_rate(m) {
        scores.push(LangfuseScore::numeric("tool_error_rate", rate));
    }
    if let Some(rate) = cache_hit_rate(m) {
        scores.push(LangfuseScore::numeric("cache_hit_rate", rate));
    }
    scores
}

/// The non-overlapping total: input plus output.
///
/// The cache and reasoning detail fields are overlap — subsets of the
/// input and output totals per the crate-wide usage convention, never
/// additive classes — so the headline total is
/// [`TokenTotals::total`](loopctl::metrics::TokenTotals::total).
/// Saturating through the run so an absurd token table cannot wrap.
#[must_use]
pub fn token_total(m: &AgentMetrics) -> u64 {
    m.tokens.total()
}

/// Σ errors / Σ calls across the run's tool stats; `None` with no calls.
///
/// The `tool_error_rate` score's definition is deliberately run-wide
/// rather than per-tool; per-tool calls, errors, and percentiles ride
/// as span metadata.
#[must_use]
pub fn tool_error_rate(m: &AgentMetrics) -> Option<f64> {
    let mut calls = 0_u64;
    let mut errors = 0_u64;
    for stats in m.tools.values() {
        calls = calls.saturating_add(stats.calls);
        errors = errors.saturating_add(stats.errors);
    }
    if calls == 0 {
        return None;
    }
    Some(errors.to_f64() / calls.to_f64())
}

/// `cached_input / input`; `None` when nothing ran through the cache
/// window's denominator.
///
/// The denominator is the input total as reported — it already
/// contains the cache-write tokens (overlap, not addition), so adding
/// `cache_write` again would shrink a real hit rate.
#[must_use]
pub fn cache_hit_rate(m: &AgentMetrics) -> Option<f64> {
    if m.tokens.input == 0 {
        return None;
    }
    Some(m.tokens.cached_input.to_f64() / m.tokens.input.to_f64())
}

/// Lossless-for-telemetry-magnitudes `u64 → f64` (exact below 2^53,
/// correctly rounded beyond).
///
/// Composing the value from its two 32-bit limbs (one fused `mul_add`
/// rounding) keeps score magnitudes honest — distinct large totals stay
/// distinct — without a lossy-cast lint to suppress.
trait ToF64 {
    /// Convert to `f64`, exact below 2^53 and correctly rounded beyond.
    ///
    /// Score values are magnitudes first — distinct large totals must
    /// stay distinct on a dashboard — so the conversion rounds rather
    /// than saturating at a ceiling.
    fn to_f64(self) -> f64;
}

impl ToF64 for u64 {
    fn to_f64(self) -> f64 {
        let hi = self.div_euclid(4_294_967_296);
        let lo = self.rem_euclid(4_294_967_296);
        f64::from(u32::try_from(hi).unwrap_or(u32::MAX)).mul_add(
            4_294_967_296.0,
            f64::from(u32::try_from(lo).unwrap_or(u32::MAX)),
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::missing_panics_doc)]

    use loopctl::metrics::{AgentMetrics, session::ToolStats};

    use super::*;

    #[test]
    fn run_spans_become_traces_and_everything_else_observations() {
        assert_eq!(classify_span(AGENT_RUN_SPAN), LangfuseTarget::Trace);
        assert_eq!(classify_span(TURN_SPAN), LangfuseTarget::Observation);
        assert_eq!(classify_span(TOOL_SPAN), LangfuseTarget::Observation);
        assert_eq!(classify_span("whatever"), LangfuseTarget::Observation);
    }

    #[test]
    fn first_class_slots_and_pass_through() {
        assert_eq!(
            first_class("gen_ai.conversation.id"),
            Some(LangfuseField::SessionId)
        );
        assert_eq!(
            first_class("gen_ai.usage.cache_read.input_tokens"),
            Some(LangfuseField::UsageCachedInput)
        );
        assert_eq!(first_class("gen_ai.tool.name"), None);
        assert_eq!(first_class("duration_ms"), None);
        assert_eq!(first_class("query"), Some(LangfuseField::Suppressed));
    }

    #[test]
    fn token_total_and_cache_hit_rate_do_not_double_count() {
        let mut m = AgentMetrics::default();
        m.tokens.input = 100;
        m.tokens.output = 50;
        m.tokens.cached_input = 40;
        m.tokens.cache_write = 20;
        m.tokens.reasoning = 10;
        assert_eq!(
            token_total(&m),
            150,
            "the cache and reasoning detail ride inside input/output — \
             overlap, never additive classes"
        );
        let rate = cache_hit_rate(&m).expect("positive denominator");
        assert!(
            (rate - 0.4).abs() < 1e-12,
            "the denominator is input, which already contains the writes: {rate}"
        );
    }

    #[test]
    fn scores_cover_tokens_turns_latency_tools_and_outcome() {
        let mut m = AgentMetrics {
            run_success: true,
            total_turns: 4,
            ..AgentMetrics::default()
        };
        m.tokens.input = 100;
        m.tokens.cached_input = 40;
        m.turn_latency.record(500);
        let mut stats = ToolStats::new("bash");
        stats.record(10, true);
        stats.record(10, false);
        m.tools.insert("bash".to_string(), stats);
        let scores = metrics_to_scores(&m);
        let names: Vec<&str> = scores.iter().map(|s| s.name.as_str()).collect();
        for want in [
            "input_tokens",
            "output_tokens",
            "token_total",
            "total_turns",
            "outcome",
            "p90_turn_latency_ms",
            "tool_error_rate",
            "cache_hit_rate",
        ] {
            assert!(names.contains(&want), "missing {want} in {names:?}");
        }
        let outcome = scores
            .iter()
            .find(|s| s.name == "outcome")
            .expect("outcome");
        assert_eq!(outcome.string_value.as_deref(), Some("success"));
        let rate = scores
            .iter()
            .find(|s| s.name == "tool_error_rate")
            .expect("rate");
        assert_eq!(rate.value, Some(0.5));
        let cache = scores
            .iter()
            .find(|s| s.name == "cache_hit_rate")
            .expect("cache");
        assert_eq!(cache.value, Some(0.4));
    }
}

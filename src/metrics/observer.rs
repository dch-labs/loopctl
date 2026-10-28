//! The lifecycle collector: engine callbacks in, [`AgentMetrics`] out.
//!
//! [`MetricsObserver`] implements [`LoopObserver`] and fills one
//! per-run [`AgentMetrics`] snapshot from the engine's callback stream.
//! It is inert until attached — a run without it collects nothing,
//! and an attached instance costs one lock per callback.
//!
//! The accounting rules mirror the snapshot's own documentation: token
//! totals and turn counts land from the engine's per-turn callbacks
//! ([`on_turn_end`](LoopObserver::on_turn_end)) and never from a
//! run-boundary handover, so streaming detail counts events without
//! touching the ledger; every counter saturates rather than wraps; and
//! each [`on_run_start`](LoopObserver::on_run_start) opens a fresh
//! snapshot — a reused observer reports one run at a time, whatever
//! [`RunConfig::reset_managers`](crate::engine::RunConfig::reset_managers)
//! says — while [`reset`](LoopObserver::reset) remains the manual
//! equivalent.
//!
//! Concurrent runs are the one shape this surface cannot separate: the
//! observer callbacks carry no run identity (the session id is stable
//! across a loop's runs by contract, and no per-turn callback carries
//! one at all), so two engines sharing one observer interleave
//! undistinguishably. Hosts running concurrent engines give each its
//! own observer.
//!
//! Read the snapshot at any time with [`snapshot`](MetricsObserver::snapshot) —
//! mid-run for live dashboards, after [`on_run_end`](LoopObserver::on_run_end)
//! for the completed record; a sustained observer's boundary read is
//! the run that just ended, never the runs before it.
//!
//! # Examples
//!
//! ```
//! use loopctl::metrics::MetricsObserver;
//! use loopctl::observer::LoopObserver as _;
//! use loopctl::observer::RunEndContext;
//!
//! let observer = MetricsObserver::default();
//! observer.on_run_end(&RunEndContext::new(true, None, 2, 4500));
//!
//! let m = observer.snapshot();
//! assert!(m.run_success);
//! assert_eq!(m.run_duration_ms, 4500);
//! ```

use std::sync::Mutex;

use super::lock;
use super::session::{AgentMetrics, TokenTotals};
use crate::observer::LoopObserver;
use crate::observer::context::{
    BudgetWarnContext, CompactedContext, CompactionFailedContext, ConvergenceDetectedContext,
    FallbackContext, LoopDetectedContext, ModelSwitchedContext, RunEndContext, RunStartContext,
    StreamContext, StreamFailureContext, ToolPostContext, TransportFallbackContext, TurnEndContext,
};

/// The per-run [`AgentMetrics`] collector on the observer surface.
///
/// One instance collects one run's snapshot at a time: each
/// [`on_run_start`](LoopObserver::on_run_start) opens a fresh
/// [`AgentMetrics`], so a reused observer reports per-run records
/// without anyone resetting it between runs. The internal
/// [`Mutex`] is held only for field bumps — callbacks never run engine
/// code under the lock — and poison is recovered, because the guarded
/// data is plain counters. Concurrent runs cannot share one: the
/// callback surface carries no run identity, so hosts running
/// concurrent engines attach one observer each (see the module docs).
///
/// # Examples
///
/// ```
/// use loopctl::metrics::MetricsObserver;
///
/// let one_shot = MetricsObserver::new();
/// let sustained = MetricsObserver::default();
/// assert_eq!(one_shot.snapshot().total_turns, 0, "nothing observed yet");
/// ```
#[derive(Default)]
pub struct MetricsObserver {
    /// The snapshot under collection; fresh after every `reset`.
    ///
    /// Every callback mutates it under one short [`Mutex`] hold, and
    /// [`snapshot`](Self::snapshot) clones out of the same lock, so an
    /// exporter scrape never sees a half-updated aggregate.
    metrics: Mutex<AgentMetrics>,
}

/// Advances a run counter without wrap arithmetic.
///
/// Metrics are evidence: a counter that wrapped would read as
/// instrument failure, so the arithmetic saturates instead — a stuck
/// maximum is at least an honest one.
fn bump(counter: &mut u64) {
    *counter = counter.saturating_add(1);
}

impl MetricsObserver {
    /// A fresh collector over an empty snapshot.
    ///
    /// Equivalent to [`Default::default`]; named for call-site clarity
    /// when the collector is constructed inline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A consistent read of the snapshot as it stands.
    ///
    /// Cheap enough to call between turns: one lock, one clone of
    /// plain data. Mid-run reads show work completed so far; the
    /// canonical read is after [`on_run_end`](LoopObserver::on_run_end)
    /// (or after the next `reset` boundary lands the next run).
    ///
    /// # Examples
    ///
    /// ```
    /// use loopctl::metrics::MetricsObserver;
    ///
    /// let observer = MetricsObserver::new();
    /// let m = observer.snapshot();
    /// assert_eq!(m.total_turns, 0);
    /// assert!(m.turn_latency_percentiles().is_none(), "no turns yet");
    /// ```
    #[must_use]
    pub fn snapshot(&self) -> AgentMetrics {
        lock(&self.metrics).clone()
    }
}

impl LoopObserver for MetricsObserver {
    fn name(&self) -> &'static str {
        "metrics"
    }

    fn on_run_start(&self, ctx: &RunStartContext) {
        let mut metrics = lock(&self.metrics);
        *metrics = AgentMetrics::default();
        metrics.session_id = ctx.session_id;
    }

    fn on_run_end(&self, ctx: &RunEndContext) {
        let mut m = lock(&self.metrics);
        m.run_success = ctx.success;
        m.run_error.clone_from(&ctx.error);
        m.run_duration_ms = ctx.duration_ms;
    }

    fn on_turn_end(&self, ctx: &TurnEndContext) {
        let mut m = lock(&self.metrics);
        bump(&mut m.total_turns);
        if !ctx.success {
            bump(&mut m.failed_turns);
        }
        m.turn_latency.record(ctx.duration_ms);
        m.tokens.add(TokenTotals {
            input: ctx.input_tokens,
            output: ctx.output_tokens,
            cached_input: ctx.cached_input_tokens,
            cache_write: ctx.cache_write_tokens,
            reasoning: ctx.reasoning_tokens,
        });
    }

    fn on_stream_success(&self, _ctx: &StreamContext) {
        bump(&mut lock(&self.metrics).stream_successes);
    }

    fn on_stream_failure(&self, _ctx: &StreamFailureContext) {
        bump(&mut lock(&self.metrics).stream_failures);
    }

    fn on_tool_post(&self, ctx: &ToolPostContext) {
        let duration_ms = u64::try_from(ctx.duration.as_millis()).unwrap_or(u64::MAX);
        lock(&self.metrics).record_tool(&ctx.tool, duration_ms, ctx.is_error);
    }

    fn on_compaction(&self, _ctx: &CompactedContext) {
        bump(&mut lock(&self.metrics).compactions);
    }

    fn on_compaction_failed(&self, _ctx: &CompactionFailedContext) {
        bump(&mut lock(&self.metrics).compaction_failures);
    }

    fn on_fallback(&self, _ctx: &FallbackContext) {
        bump(&mut lock(&self.metrics).fallbacks);
    }

    fn on_transport_fallback(&self, _ctx: &TransportFallbackContext) {
        bump(&mut lock(&self.metrics).transport_fallbacks);
    }

    fn on_model_switched(&self, _ctx: &ModelSwitchedContext) {
        bump(&mut lock(&self.metrics).model_switches);
    }

    fn on_budget_warn(&self, _ctx: &BudgetWarnContext) {
        bump(&mut lock(&self.metrics).budget_warnings);
    }

    fn on_loop_detected(&self, _ctx: &LoopDetectedContext) {
        bump(&mut lock(&self.metrics).loops_detected);
    }

    fn on_convergence_detected(&self, _ctx: &ConvergenceDetectedContext) {
        bump(&mut lock(&self.metrics).convergences_detected);
    }

    fn reset(&self) {
        *lock(&self.metrics) = AgentMetrics::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compact::{CompactReason, CompactTelemetry, PostCompactStats, PreCompactStats};
    use crate::stream::StreamStopReason;
    use std::sync::Arc;
    use std::time::Duration;

    fn turn(
        turn: usize,
        success: bool,
        duration_ms: u64,
        input: u64,
        output: u64,
    ) -> TurnEndContext {
        TurnEndContext {
            turn,
            success,
            error: if success {
                None
            } else {
                Some("boom".to_string())
            },
            duration_ms,
            input_tokens: input,
            output_tokens: output,
            stop_reason: StreamStopReason::EndTurn,
            context_tokens: 0,
            context_window: None,
            history: Arc::new(Vec::new()),
            cached_input_tokens: 0,
            cache_write_tokens: 0,
            reasoning_tokens: 0,
        }
    }

    fn telemetry() -> CompactTelemetry {
        CompactTelemetry {
            trigger: CompactReason::ThresholdExceeded,
            pre_compact: PreCompactStats {
                total_messages: 10,
                estimated_tokens: 5_000,
                user_messages: 4,
                assistant_messages: 4,
                tool_messages: 2,
                user_tokens: 2_000,
                assistant_tokens: 3_000,
                density: 1.0,
            },
            post_compact: PostCompactStats {
                total_messages: 4,
                estimated_tokens: 2_000,
                tokens_saved: 3_000,
                percent_saved: 60,
                user_messages: 1,
                assistant_messages: 1,
                tool_messages: 2,
                user_tokens: 500,
                assistant_tokens: 1_500,
                density: 1.0,
            },
            duration: Duration::from_millis(40),
            compression_ratio: 2.5,
            headroom_pct: 30.0,
            compactor_name: None,
        }
    }

    #[test]
    fn run_boundaries_land_identity_and_outcome() {
        let observer = MetricsObserver::new();
        let session_id = uuid::Uuid::new_v4();
        observer.on_run_start(&RunStartContext { session_id });
        for turn_index in 0..3 {
            observer.on_turn_end(&turn(turn_index, true, 100, 10, 1));
        }
        observer.on_run_end(&RunEndContext::new(
            false,
            Some("exploded".into()),
            3,
            9_100,
        ));

        let m = observer.snapshot();
        assert_eq!(m.session_id, session_id);
        assert!(!m.run_success);
        assert_eq!(m.run_error.as_deref(), Some("exploded"));
        assert_eq!(
            m.total_turns, 3,
            "the incremental count is authoritative; the boundary confirms, never overwrites"
        );
        assert_eq!(m.run_duration_ms, 9_100);
    }

    #[test]
    fn turns_accumulate_tokens_latency_and_failures() {
        let observer = MetricsObserver::new();
        observer.on_turn_end(&turn(0, true, 300, 1_000, 100));
        observer.on_turn_end(&turn(1, false, 2_000, 500, 50));

        let m = observer.snapshot();
        assert_eq!(m.total_turns, 2);
        assert_eq!(m.failed_turns, 1);
        assert_eq!(m.tokens.input, 1_500);
        assert_eq!(m.tokens.output, 150);
        assert!(
            m.turn_latency_percentiles().is_some(),
            "two samples recorded"
        );
    }

    #[test]
    fn tool_posts_split_per_tool_and_count_attempts() {
        let observer = MetricsObserver::new();
        for (tool, is_error) in [("read", false), ("exec", true), ("read", false)] {
            observer.on_tool_post(&ToolPostContext {
                tool_call_id: "call-1".into(),
                turn: 0,
                tool: tool.into(),
                result_hash: None,
                is_error,
                duration: Duration::from_millis(40),
                display_hint: None,
            });
        }
        let m = observer.snapshot();
        assert_eq!(m.total_tool_calls(), 3);
        assert_eq!(m.total_tool_errors(), 1);
        assert_eq!(m.tools.get("read").map(|s| s.calls), Some(2));
        assert_eq!(m.tools.get("exec").map(|s| s.errors), Some(1));
    }

    #[test]
    fn guardrail_counters_all_land() {
        let observer = MetricsObserver::new();
        observer.on_stream_success(&StreamContext {
            turn: 0,
            model: "m".into(),
            input_tokens: 10,
            output_tokens: 1,
            cached_input_tokens: 0,
            cache_write_tokens: 0,
            reasoning_tokens: 0,
        });
        observer.on_fallback(&FallbackContext {
            from: "a".into(),
            to: "b".into(),
        });
        observer.on_transport_fallback(&TransportFallbackContext {
            turn: 0,
            stop_reason: StreamStopReason::MaxTokens,
        });
        observer.on_model_switched(&ModelSwitchedContext {
            from: "a".into(),
            to: "b".into(),
        });
        observer.on_budget_warn(&BudgetWarnContext {
            dimension: crate::budget::BudgetDimension::Tokens,
            spent: 90_000,
            limit: 100_000,
        });
        observer.on_loop_detected(&LoopDetectedContext {
            pattern: "p".into(),
            repetitions: 3,
        });
        observer.on_convergence_detected(&ConvergenceDetectedContext {
            action: "stop".into(),
        });
        observer.on_compaction(&CompactedContext {
            tokens_before: 5_000,
            tokens_after: 2_000,
            tokens_saved: 3_000,
            reason: CompactReason::ThresholdExceeded,
            evicted_messages: 6,
            telemetry: telemetry(),
            stage: None,
        });
        observer.on_compaction_failed(&CompactionFailedContext {
            reason: CompactReason::ThresholdExceeded,
            turn: 1,
            tokens_before: 5_000,
            context_window: 8_000,
            error: Some("provider down".into()),
        });

        let m = observer.snapshot();
        assert_eq!(m.stream_successes, 1);
        assert_eq!(m.fallbacks, 1);
        assert_eq!(m.transport_fallbacks, 1);
        assert_eq!(m.model_switches, 1);
        assert_eq!(m.budget_warnings, 1);
        assert_eq!(m.loops_detected, 1);
        assert_eq!(m.convergences_detected, 1);
        assert_eq!(m.compactions, 1);
        assert_eq!(m.compaction_failures, 1);
    }

    #[test]
    fn a_reused_observer_resets_at_each_run_start() {
        let observer = MetricsObserver::new();
        observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        observer.on_turn_end(&turn(0, true, 100, 10, 1));
        observer.on_run_end(&RunEndContext::new(true, None, 1, 100));

        observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        let opened = observer.snapshot();
        assert_eq!(
            opened.total_turns, 0,
            "the second run opens on a clean snapshot, not the first run's ledger"
        );
        assert_eq!(opened.tokens.input, 0);

        observer.on_turn_end(&turn(0, true, 200, 20, 2));
        observer.on_run_end(&RunEndContext::new(true, None, 1, 200));
        let settled = observer.snapshot();
        assert_eq!(
            settled.total_turns, 1,
            "the boundary read carries only this run's turns"
        );
        assert_eq!(settled.tokens.input, 20);
    }

    #[test]
    fn reset_hands_the_next_run_a_clean_snapshot() {
        let observer = MetricsObserver::new();
        observer.on_run_start(&RunStartContext {
            session_id: uuid::Uuid::new_v4(),
        });
        observer.on_turn_end(&turn(0, true, 10, 100, 5));
        observer.reset();

        let m = observer.snapshot();
        assert_eq!(m.total_turns, 0);
        assert_eq!(m.tokens.input, 0);
        assert_eq!(
            m.session_id,
            uuid::Uuid::default(),
            "identity is per-run too"
        );
    }
}

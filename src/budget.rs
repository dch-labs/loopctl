//! The budget gate — spend accumulation with soft and hard lines.
//!
//! One engine-side gate behind the "budgets are gates" decision: it
//! accumulates what a run spends (provider-reported tokens, completed
//! turns, wall-clock measured by the caller's clock seam) and enforces two
//! lines per dimension — a **soft** line that emits a warn event once
//! per crossing, and a **hard** line that refuses the *next turn's*
//! model request before it is sent. Decisions flow through
//! [`GateDecision`](crate::tool::permission::GateDecision) records
//! like any gate, so audit and replay see budget enforcement through
//! the same trail. Both lines scope to the run's turn requests:
//! compaction-pass model calls are outside the accounting and the
//! enforcement alike.

use std::sync::Mutex;
use std::time::Duration;

use serde::Deserialize;
use serde::Serialize;

use crate::error::LoopError;

/// One budget line a run can spend against.
///
/// Names the dimension in [`LoopError::BudgetExhausted`], warn
/// contexts, and gate-decision reasons — the one vocabulary for
/// "which budget ran out" across errors, events, and audit records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetDimension {
    /// Provider-reported tokens, input plus output.
    ///
    /// Counted from the turn-serving usage records the engine feeds,
    /// never estimates — a provider that reports no usage spends
    /// nothing against this line, and compaction-pass model calls are
    /// outside the accounting.
    Tokens,

    /// Completed assistant turns.
    ///
    /// Incremented once per successful turn end, so an in-flight turn
    /// never counts against itself and a turn that fails mid-run
    /// counts nothing.
    Turns,

    /// Wall-clock time since the run began.
    ///
    /// Read exclusively from the engine's clock seam at check time,
    /// so a replayed run crosses the same lines at the same logical
    /// instants.
    WallClock,
}

impl std::fmt::Display for BudgetDimension {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Tokens => "tokens",
            Self::Turns => "turns",
            Self::WallClock => "wall-clock",
        };
        formatter.write_str(name)
    }
}

/// The enforceable budget lines for one run.
///
/// The gate's own input shape, independent of any manifest type: a
/// host supplies limits directly, or maps the manifest's advisory
/// [`Budgets`](crate::manifest::Budgets) stanza through the
/// `From<&Budgets>` impl behind the `manifest` feature. `None`
/// disables a line — no threshold, no warn, no refusal. No price
/// table is consulted, so cost has no line; a configured `cost_usd`
/// maps to nothing (the gate refuses to guess a cost).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BudgetLimits {
    /// The hard ceiling on provider-reported tokens per run.
    ///
    /// Input plus output, counted from usage records; the soft line
    /// is a fraction of this value.
    pub tokens: Option<u64>,

    /// The hard ceiling on completed turns per run.
    ///
    /// The soft line is a fraction of this value.
    pub turns: Option<u64>,

    /// The hard ceiling on run wall-clock time.
    ///
    /// Measured from the run's `begin` instant through the clock
    /// seam; the soft line is a fraction of this value.
    pub duration: Option<Duration>,
}

impl BudgetLimits {
    /// Build limits with every line disabled.
    ///
    /// The empty budget: a gate over these limits never warns and
    /// never refuses — the additive identity a host composes from
    /// when only some dimensions are configured.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(feature = "manifest")]
impl From<&crate::manifest::Budgets> for BudgetLimits {
    fn from(budgets: &crate::manifest::Budgets) -> Self {
        let duration = budgets.duration.as_deref().and_then(|text| {
            parse_duration(text)
                .map_err(|error| {
                    tracing::warn!(
                        target: "loopctl::metrics",
                        text,
                        error = %error,
                        "ignoring an unparseable duration budget"
                    );
                })
                .ok()
        });
        Self {
            tokens: budgets.tokens,
            turns: budgets.turns,
            duration,
        }
    }
}

/// Parse a duration budget string — whole numbers with `s`/`m`/`h` units.
///
/// The grammar the manifest's duration strings use: one or more
/// `number + unit` pairs concatenated, units `s` (seconds), `m`
/// (minutes), `h` (hours) — `90s`, `5m`, `4h`, `1h30m`, `2h45m10s`.
/// Numbers are plain ASCII digits without signs or fractions; the
/// manifest keeps the string unparsed precisely so the consumer can
/// define the grammar, and this is the budget gate's. Unparseable
/// text is a loud [`LoopError::InvalidInput`] naming the string and
/// the expected shape, never a silent zero.
///
/// # Errors
///
/// Returns [`LoopError::InvalidInput`] when the string is empty,
/// carries anything but digit-then-unit pairs, uses an unknown
/// unit, or overflows a `u64` second count.
pub fn parse_duration(text: &str) -> Result<Duration, LoopError> {
    let original = text;
    let mut rest = text;
    let mut seconds: u64 = 0;
    let mut saw_any = false;
    while let Some(digit_end) = rest.find(|c: char| !c.is_ascii_digit()) {
        let (digits, tail) = rest.split_at(digit_end);
        if digits.is_empty() {
            return Err(unparseable_duration(original, "a number first"));
        }
        let unit_end = tail
            .find(|c: char| c.is_ascii_digit())
            .unwrap_or(tail.len());
        let (unit, next) = tail.split_at(unit_end);
        let multiplier = match unit {
            "s" => 1,
            "m" => 60,
            "h" => 3600,
            "" => return Err(unparseable_duration(original, "a unit after the number")),
            _ => return Err(unparseable_duration(original, "unit s, m, or h")),
        };
        let count: u64 = digits
            .parse()
            .map_err(|_| unparseable_duration(original, "a number u64 can hold"))?;
        let addend = count
            .checked_mul(multiplier)
            .and_then(|value| seconds.checked_add(value))
            .ok_or_else(|| unparseable_duration(original, "a duration u64 seconds can hold"))?;
        seconds = addend;
        saw_any = true;
        rest = next;
    }
    if !rest.is_empty() || !saw_any {
        return Err(unparseable_duration(
            original,
            "number-unit pairs like 4h or 1h30m",
        ));
    }
    Ok(Duration::from_secs(seconds))
}

/// The soft line for a hard limit: `limit * fraction`, rounded down.
///
/// Exact integer arithmetic on the fraction's rational form
/// (denominator `10_000`; the builder's range check keeps the value
/// in `(0.0, 1.0]`), computed through a `u128` product so no float
/// rounding can push the soft line above the exact product and no
/// intermediate saturation can push it below — at fraction `1.0` the
/// soft line is the hard limit itself, for every limit a `u64` can
/// hold. The warn can therefore never trail the refusal.
fn soft_line(limit: u64, fraction: f64) -> Option<u64> {
    let (numerator, denominator) = rationalize(fraction)?;
    let product = u128::from(limit).saturating_mul(u128::from(numerator));
    let scaled = product
        .checked_div(u128::from(denominator))
        .unwrap_or(u128::from(limit));
    Some(u64::try_from(scaled).unwrap_or(limit))
}

/// The fraction as `numerator / 10_000`, when representable.
///
/// `None` covers the non-finite and out-of-range shapes the builder
/// already rejects; the denominator is fixed so the division stays
/// exact for every fraction the gate accepts.
fn rationalize(fraction: f64) -> Option<(u64, u64)> {
    if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
        return None;
    }
    let text = format!("{fraction:.4}");
    let (whole_text, decimal_text) = text.split_once('.')?;
    let whole: u64 = whole_text.parse().ok()?;
    let decimal: u64 = decimal_text.parse().ok()?;
    let numerator = whole.saturating_mul(10_000).saturating_add(decimal);
    if numerator == 0 {
        return None;
    }
    Some((numerator, 10_000))
}

/// The invalid-input error for a bad duration string.
///
/// One wording for every parse failure, naming the original text and
/// the expected shape so the correction prompt is self-contained.
fn unparseable_duration(original: &str, expected: &str) -> LoopError {
    LoopError::InvalidInput(format!(
        "budget duration '{original}' is not parseable: expected {expected}"
    ))
}

/// A soft-line crossing the gate is reporting.
///
/// Produced by [`BudgetGate::pre_request`] the first time a
/// dimension reaches its soft line; carries the numbers an observer
/// (or a hub consumer) needs to report the crossing without
/// re-reading the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetWarn {
    /// The dimension whose soft line was crossed.
    ///
    /// Tokens, turns, or wall-clock — the vocabulary every budget
    /// surface shares.
    pub dimension: BudgetDimension,

    /// The spend at the crossing.
    ///
    /// Tokens or turns counted, or elapsed milliseconds for
    /// [`BudgetDimension::WallClock`].
    pub spent: u64,

    /// The hard limit the soft line derives from.
    ///
    /// Reporting the ceiling (not the soft line itself) lets a
    /// consumer phrase the warning as "N of M" — the number the
    /// user set, not the fraction the gate derived.
    pub limit: u64,
}

/// The accumulator and threshold evaluator for one budget set.
///
/// The engine begins a run (`begin` with the clock's now), feeds it
/// usage and turn completions as they happen, and consults
/// `pre_request` before sending each turn's model request: soft
/// crossings come back as warns — each dimension warns exactly once
/// per run — and a hard crossing refuses, identifying the dimension
/// and the numbers for the caller's [`LoopError::BudgetExhausted`]
/// and deny record. The in-flight request that crosses a line is
/// never the one refused: accumulation is post-response, so the
/// crossing call completes and the *next* check refuses.
#[derive(Debug)]
pub struct BudgetGate {
    /// The configured lines and derived soft thresholds.
    ///
    /// Immutable after construction; only the spend state mutates.
    config: BudgetConfig,

    /// The mutable spend state, latched per run.
    ///
    /// Interior-mutex so the loop's request, response, and turn-end
    /// paths can update and check the same gate.
    state: Mutex<GateState>,
}

/// The immutable gate configuration: limits plus the soft fraction.
///
/// Captured at construction so every check reads the same lines for
/// the gate's lifetime; `with_soft_fraction` is the only mutator and
/// it consumes the builder before any run begins.
#[derive(Debug, Clone)]
struct BudgetConfig {
    /// The hard ceilings; `None` disables a line.
    ///
    /// The manifest stanza mapped into the gate's own shape.
    limits: BudgetLimits,

    /// The fraction of each hard limit the soft line sits at.
    ///
    /// In `[0.0001, 1.0]` — a fraction at or above 1 would never warn
    /// before the refusal, one at or below 0 would warn before
    /// anything was spent, and one below the rationalization floor
    /// could not warn at all.
    soft_fraction: f64,
}

/// The per-run spend state under the gate's mutex.
///
/// Counters and latches only — every line and threshold lives in the
/// immutable config, and the critical sections are flag writes and
/// saturating adds that cannot panic, so the lock cannot be poisoned.
/// Were it poisoned regardless, every gate method would go silent —
/// no warns, no refusals, never a refusal on stale data — a fail-open
/// shape the panic-free sections keep unreachable by construction.
#[derive(Debug, Default)]
struct GateState {
    /// Provider-reported tokens spent so far this run.
    ///
    /// Input plus output, summed saturating.
    spent_tokens: u64,

    /// Completed turns so far this run.
    ///
    /// Incremented at turn end only.
    turns: u64,

    /// Which dimensions have already warned.
    ///
    /// One latch per dimension: the soft line warns exactly once per
    /// run, however many requests land between the soft line and
    /// the hard one.
    warned_tokens: bool,

    /// The turns dimension's warn latch.
    ///
    /// Set exactly when the dimension's first warn is pushed.
    warned_turns: bool,

    /// The wall-clock dimension's warn latch.
    ///
    /// Set exactly when the dimension's first warn is pushed.
    warned_wall_clock: bool,
}

impl BudgetGate {
    /// Build a gate over `limits` with the default soft fraction (80%).
    ///
    /// The hard lines come from the limits; each soft line sits at
    /// 80% of its hard line unless [`with_soft_fraction`](Self::with_soft_fraction)
    /// moves it.
    #[must_use]
    pub fn new(limits: BudgetLimits) -> Self {
        Self {
            config: BudgetConfig {
                limits,
                soft_fraction: 0.80,
            },
            state: Mutex::new(GateState::default()),
        }
    }

    /// Move every soft line to `fraction` of its hard line.
    ///
    /// The fraction must sit in `[0.0001, 1.0]`: at or above one the
    /// soft line would coincide with or trail the refusal (no warn
    /// before the block), at or below zero it would warn before any
    /// spend, and below the four-decimal rationalization floor
    /// (`0.0001`) no soft line could fire at all — a warless gate
    /// wearing a fraction. Out-of-range fractions are rejected loudly
    /// rather than clamped — a mistyped 8.0 for 0.80 should surface,
    /// not silently become a different gate.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::InvalidInput`] when `fraction` is below
    /// the rationalization floor, zero, negative, NaN, or greater
    /// than one.
    pub fn with_soft_fraction(mut self, fraction: f64) -> Result<Self, LoopError> {
        if !fraction.is_finite() || fraction < 0.0001 || fraction > 1.0 {
            return Err(LoopError::InvalidInput(format!(
                "budget soft fraction must be in [0.0001, 1.0] — 0.0001 is the \
                 four-decimal rationalization floor below which no soft line \
                 could fire — got {fraction}"
            )));
        }
        self.config.soft_fraction = fraction;
        Ok(self)
    }

    /// Begin (or restart) the run's accounting.
    ///
    /// Resets every spend counter and warn latch. The engine calls
    /// this at run start; a host re-running over one gate calls it
    /// between runs so each run starts from zero. Wall-clock origin
    /// is the caller's business: the engine holds its clock-seam
    /// start instant and passes the elapsed [`Duration`] to
    /// [`pre_request`](Self::pre_request), so the gate itself never
    /// reads a clock.
    pub fn begin(&self) {
        if let Ok(mut state) = self.state.lock() {
            *state = GateState::default();
        }
    }

    /// Record provider-reported token usage for the run.
    ///
    /// Input plus output tokens from one response's usage record;
    /// the spend lands after the response arrives, which is what
    /// makes the crossing request the one that completes.
    pub fn observe_usage(&self, input_tokens: u64, output_tokens: u64) {
        if let Ok(mut state) = self.state.lock() {
            state.spent_tokens = state
                .spent_tokens
                .saturating_add(input_tokens)
                .saturating_add(output_tokens);
        }
    }

    /// Record one completed turn.
    ///
    /// Called at a successful turn's end so an in-flight turn never
    /// counts against itself — the turns line refuses the *next*
    /// request, after the current turn has fully completed. The
    /// engine does not call it for a failed turn end: the dimension
    /// counts completed turns, and every failed turn end is terminal
    /// for its run.
    pub fn observe_turn_end(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.turns = state.turns.saturating_add(1);
        }
    }

    /// Evaluate every enabled dimension's line before a model request.
    ///
    /// Returns the soft crossings to report — each dimension at most
    /// once per run — and the hard crossing that refuses the request:
    /// the dimension, the spend, and the limit, ready for
    /// [`LoopError::BudgetExhausted`] and the deny record. Every
    /// dimension's first crossing latches in the same check,
    /// whichever dimension's hard line refuses the request — the
    /// tokens → turns → wall-clock priority picks the refusal, never
    /// which warns a consumer sees. The elapsed [`Duration`] is
    /// the caller's clock-seam reading since the run began — the
    /// gate itself never reads a clock, so a replayed run crosses
    /// the same lines at the same logical instants.
    ///
    /// Returns the warns and the hard crossing together: the `Vec`
    /// carries every first soft crossing this check latched (a
    /// spend rocketing past both lines in one response still warns —
    /// the record of the approach belongs on the trail even when the
    /// refusal lands with it), and the `Option` names the crossed
    /// hard line for the caller's error and decision record.
    pub fn pre_request(&self, elapsed: Duration) -> (Vec<BudgetWarn>, Option<HardCrossing>) {
        let Ok(mut state) = self.state.lock() else {
            return (Vec::new(), None);
        };
        let mut warns = Vec::new();
        let mut hard = None;

        if let Some(limit) = self.config.limits.tokens {
            if state.spent_tokens >= limit {
                let spent = state.spent_tokens;
                latch_warn(
                    &mut state.warned_tokens,
                    &mut warns,
                    BudgetDimension::Tokens,
                    spent,
                    limit,
                );
                hard = Some(HardCrossing {
                    dimension: BudgetDimension::Tokens,
                    spent,
                    limit,
                });
            } else if !state.warned_tokens
                && soft_line(limit, self.config.soft_fraction)
                    .is_some_and(|soft| state.spent_tokens >= soft)
            {
                state.warned_tokens = true;
                warns.push(BudgetWarn {
                    dimension: BudgetDimension::Tokens,
                    spent: state.spent_tokens,
                    limit,
                });
            }
        }

        if let Some(limit) = self.config.limits.turns {
            if state.turns >= limit {
                let spent = state.turns;
                latch_warn(
                    &mut state.warned_turns,
                    &mut warns,
                    BudgetDimension::Turns,
                    spent,
                    limit,
                );
                if hard.is_none() {
                    hard = Some(HardCrossing {
                        dimension: BudgetDimension::Turns,
                        spent,
                        limit,
                    });
                }
            } else if !state.warned_turns
                && soft_line(limit, self.config.soft_fraction)
                    .is_some_and(|soft| state.turns >= soft)
            {
                state.warned_turns = true;
                warns.push(BudgetWarn {
                    dimension: BudgetDimension::Turns,
                    spent: state.turns,
                    limit,
                });
            }
        }

        if let Some(limit) = self.config.limits.duration {
            let elapsed_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
            let limit_ms = limit.as_millis().try_into().unwrap_or(u64::MAX);
            if elapsed_ms >= limit_ms {
                latch_warn(
                    &mut state.warned_wall_clock,
                    &mut warns,
                    BudgetDimension::WallClock,
                    elapsed_ms,
                    limit_ms,
                );
                if hard.is_none() {
                    hard = Some(HardCrossing {
                        dimension: BudgetDimension::WallClock,
                        spent: elapsed_ms,
                        limit: limit_ms,
                    });
                }
            } else if !state.warned_wall_clock
                && soft_line(limit_ms, self.config.soft_fraction)
                    .is_some_and(|soft| elapsed_ms >= soft)
            {
                state.warned_wall_clock = true;
                warns.push(BudgetWarn {
                    dimension: BudgetDimension::WallClock,
                    spent: elapsed_ms,
                    limit: limit_ms,
                });
            }
        }

        (warns, hard)
    }
}

/// A hard line the check refused the request over.
///
/// The dimension, the spend, and the crossed ceiling — everything
/// the caller needs for [`LoopError::BudgetExhausted`] and the deny
/// record, in one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardCrossing {
    /// The budget line that was crossed.
    ///
    /// The dimension the caller reports in its error and record.
    pub dimension: BudgetDimension,

    /// The spend at the refusal.
    ///
    /// Tokens or turns counted, or elapsed milliseconds for the
    /// wall-clock line.
    pub spent: u64,

    /// The hard ceiling that was crossed.
    ///
    /// The configured limit — the number the operator set, not the
    /// derived soft line.
    pub limit: u64,
}

/// Latch one soft crossing if it has not fired yet.
///
/// The once-per-run semantics live here: the latch is set only when
/// the warn is pushed, so however many checks pass between the soft
/// line and the hard one, the dimension reports exactly once.
fn latch_warn(
    latched: &mut bool,
    warns: &mut Vec<BudgetWarn>,
    dimension: BudgetDimension,
    spent: u64,
    limit: u64,
) {
    if !*latched {
        *latched = true;
        warns.push(BudgetWarn {
            dimension,
            spent,
            limit,
        });
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_lines
)]
mod tests {
    use super::*;

    fn gate(limits: BudgetLimits) -> BudgetGate {
        let gate = BudgetGate::new(limits);
        gate.begin();
        gate
    }

    #[test]
    fn duration_grammar_parses_plain_units() {
        assert_eq!(parse_duration("90s"), Ok(Duration::from_secs(90)));
        assert_eq!(parse_duration("5m"), Ok(Duration::from_mins(5)));
        assert_eq!(parse_duration("4h"), Ok(Duration::from_hours(4)));
    }

    #[test]
    fn duration_grammar_parses_sequences() {
        assert_eq!(parse_duration("1h30m"), Ok(Duration::from_mins(90)));
        assert_eq!(
            parse_duration("2h45m10s"),
            Ok(Duration::from_hours(2) + Duration::from_mins(45) + Duration::from_secs(10))
        );
    }

    #[test]
    fn duration_grammar_rejects_garbage_loudly() {
        for bad in ["", "4", "h", "4x", "1.5h", "-5m", "4 h", "4H"] {
            let error = parse_duration(bad)
                .expect_err("unparseable durations must be rejected, not defaulted");
            match error {
                LoopError::InvalidInput(message) => assert!(
                    message.contains("budget duration"),
                    "the error names the field: {message}"
                ),
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
    }

    #[test]
    fn soft_fraction_out_of_range_is_rejected() {
        for bad in [0.0, -0.5, 1.5, 8.0, f64::NAN] {
            let error = BudgetGate::new(BudgetLimits::new())
                .with_soft_fraction(bad)
                .expect_err("an out-of-range fraction is a correction prompt");
            match error {
                LoopError::InvalidInput(message) => assert!(
                    message.contains("soft fraction"),
                    "the error names the knob: {message}"
                ),
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
        assert!(
            BudgetGate::new(BudgetLimits::new())
                .with_soft_fraction(0.5)
                .is_ok()
        );
    }

    #[test]
    fn a_soft_crossing_warns_exactly_once_per_run() {
        let gate = gate(BudgetLimits {
            tokens: Some(200),
            ..BudgetLimits::new()
        });
        let (warns, crossing) = gate.pre_request(Duration::ZERO);
        assert!(warns.is_empty() && crossing.is_none());
        gate.observe_usage(90, 90);
        let (warns, crossing) = gate.pre_request(Duration::ZERO);
        assert_eq!(warns.len(), 1, "180 of 200 crosses the 80% soft line");
        assert_eq!(warns[0].spent, 180);
        assert_eq!(warns[0].limit, 200);
        assert!(crossing.is_none(), "the hard line has not been reached");
        let (warns, _) = gate.pre_request(Duration::ZERO);
        assert!(
            warns.is_empty(),
            "the latch holds: one warn per dimension per run"
        );
    }

    #[test]
    fn a_hard_crossing_carries_the_numbers_and_still_latches_the_warn() {
        let gate = gate(BudgetLimits {
            tokens: Some(100),
            ..BudgetLimits::new()
        });
        gate.observe_usage(60, 60);
        let (warns, crossing) = gate.pre_request(Duration::ZERO);
        let crossing = crossing.expect("120 of 100 crosses the hard line");
        assert_eq!(crossing.dimension, BudgetDimension::Tokens);
        assert_eq!(crossing.spent, 120);
        assert_eq!(crossing.limit, 100);
        assert_eq!(
            warns.len(),
            1,
            "a spend rocketing past both lines still records the approach"
        );
    }

    #[test]
    fn turns_dimension_counts_completed_turns() {
        let gate = gate(BudgetLimits {
            turns: Some(3),
            ..BudgetLimits::new()
        });
        gate.observe_turn_end();
        gate.observe_turn_end();
        let (warns, crossing) = gate.pre_request(Duration::ZERO);
        assert_eq!(warns.len(), 1, "2 of 3 crosses the soft line");
        assert!(crossing.is_none());
        gate.observe_turn_end();
        let (_, crossing) = gate.pre_request(Duration::ZERO);
        let crossing = crossing.expect("3 of 3 crosses the hard line");
        assert_eq!(crossing.dimension, BudgetDimension::Turns);
        assert_eq!(crossing.spent, 3);
    }

    #[test]
    fn wall_clock_dimension_reads_the_elapsed_argument() {
        let gate = gate(BudgetLimits {
            duration: Some(Duration::from_secs(100)),
            ..BudgetLimits::new()
        });
        let (warns, crossing) = gate.pre_request(Duration::from_secs(80));
        assert_eq!(warns.len(), 1, "80s of 100s crosses the soft line");
        assert_eq!(warns[0].dimension, BudgetDimension::WallClock);
        assert!(crossing.is_none());
        let (_, crossing) = gate.pre_request(Duration::from_secs(101));
        let crossing = crossing.expect("101s of 100s crosses the hard line");
        assert_eq!(crossing.spent, 101_000);
    }

    #[test]
    fn begin_resets_counters_and_latches_for_the_next_run() {
        let gate = gate(BudgetLimits {
            tokens: Some(100),
            ..BudgetLimits::new()
        });
        gate.observe_usage(50, 50);
        let (_, crossing) = gate.pre_request(Duration::ZERO);
        assert!(crossing.is_some(), "the first run exhausts its line");
        gate.begin();
        let (warns, crossing) = gate.pre_request(Duration::ZERO);
        assert!(
            warns.is_empty() && crossing.is_none(),
            "the next run starts from zero"
        );
    }

    #[test]
    fn disabled_lines_never_warn_nor_refuse() {
        let gate = gate(BudgetLimits::new());
        gate.observe_usage(u64::MAX / 2, u64::MAX / 2);
        for _ in 0..100 {
            gate.observe_turn_end();
        }
        let (warns, crossing) = gate.pre_request(Duration::MAX);
        assert!(warns.is_empty() && crossing.is_none());
    }

    #[test]
    fn budget_exhausted_is_not_recoverable() {
        let error = LoopError::BudgetExhausted {
            dimension: BudgetDimension::Tokens,
            spent: 120,
            limit: 100,
        };
        assert!(
            !error.is_recoverable(),
            "more spend is what the line exists to prevent"
        );
        assert!(error.to_string().contains("tokens"));
        assert!(error.to_string().contains("120"));
    }

    #[test]
    fn the_soft_line_never_exceeds_the_hard_one() {
        for limit in [1u64, 2, 7, 100, 9_999, 100_000] {
            for fraction in [0.1, 0.5, 0.8, 0.99, 1.0] {
                let soft = soft_line(limit, fraction).unwrap_or(u64::MAX);
                assert!(
                    soft <= limit,
                    "soft {soft} must sit at or below hard {limit} (fraction {fraction})"
                );
            }
        }
        // The 80% default lands where the docs say.
        assert_eq!(soft_line(200, 0.80), Some(160));
        assert_eq!(soft_line(1, 0.80), Some(0));
    }

    #[test]
    fn a_fraction_below_the_rationalization_floor_is_rejected_loudly() {
        // `rationalize` keeps four decimals; a fraction below 0.0001
        // rounds to a zero numerator, which `soft_line` would read as
        // "no warn, ever" — an accepted builder value silently
        // disabling every soft line. The builder rejects it instead.
        let limits = BudgetLimits {
            tokens: Some(100),
            ..BudgetLimits::new()
        };
        let sub_floor = BudgetGate::new(limits.clone()).with_soft_fraction(0.00001);
        assert!(
            sub_floor.is_err(),
            "a sub-floor fraction must be a loud InvalidInput, not a \
             warless gate: {sub_floor:?}"
        );
        let at_floor = BudgetGate::new(limits).with_soft_fraction(0.0001);
        assert!(
            at_floor.is_ok(),
            "the floor itself rationalizes (1/10_000) and is accepted: \
             {at_floor:?}"
        );
    }

    #[test]
    fn the_soft_line_at_full_fraction_is_the_hard_limit_where_products_saturate() {
        // `limit * 10_000` overflows a `u64` product for limits past
        // `u64::MAX / 10_000`; the soft line must still coincide with
        // the hard limit at fraction 1.0, or a warn could fire ahead of
        // the refusal the builder doc rules out.
        for limit in [
            (u64::MAX / 10_000).saturating_add(1),
            u64::MAX / 3,
            u64::MAX,
        ] {
            assert_eq!(
                soft_line(limit, 1.0),
                Some(limit),
                "fraction 1.0 derives the hard limit itself, even where \
                 limit times 10_000 saturates a u64 product (limit {limit})"
            );
        }
    }

    #[cfg(feature = "manifest")]
    mod manifest_mapping {
        use super::*;
        use crate::manifest::Budgets;

        #[test]
        fn manifest_budgets_map_to_limits() {
            let budgets = Budgets {
                turns: Some(10),
                tokens: Some(5_000),
                cost_usd: Some(1.5),
                duration: Some("4h".to_string()),
            };
            let limits = BudgetLimits::from(&budgets);
            assert_eq!(limits.turns, Some(10));
            assert_eq!(limits.tokens, Some(5_000));
            assert_eq!(limits.duration, Some(Duration::from_hours(4)));
        }

        #[test]
        fn an_unparseable_duration_maps_to_no_wall_clock_line() {
            let budgets = Budgets {
                duration: Some("forever".to_string()),
                ..Budgets::default()
            };
            let limits = BudgetLimits::from(&budgets);
            assert_eq!(
                limits.duration, None,
                "an unparseable duration disables the line rather than guessing"
            );
        }
    }
}

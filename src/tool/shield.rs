//! Tool Safety Shield — multi-turn adversarial defense.
//!
//! [`ToolSafetyShield`] trait — a generic,
//! platform-agnostic boundary for evaluating tool call safety — and a
//! reference [`UnixShield`] implementation that matches dangerous Unix
//! shell patterns.
//!
//! The trait itself places no constraints on tool names, input formats,
//! or operating systems. Consumers targeting non-Unix environments
//! (Windows `PowerShell`, cloud APIs, database tools) should implement
//! [`ToolSafetyShield`] with their own pattern database.
//!
//! # Evaluation Dimensions
//!
//! Each invocation is scored across three independent dimensions:
//!
//! | Dimension         | What it detects                                        |
//! |-------------------|--------------------------------------------------------|
//! | Single-turn risk  | Dangerous argument patterns in the current call        |
//! | Multi-turn risk   | Accumulation of calls that individually look benign    |
//! | Combination risk  | Dangerous tool sequences (e.g. write → execute)        |
//!
//! # Architecture
//!
//! The shield sits in the agent-loop middleware path and is consulted twice
//! per tool call:
//!
//! 1. **Before dispatch** —
//!    [`ToolSafetyShield::evaluate()`] inspects the `tool_call` and returns
//!    a [`SafetyDecision`] of `Allow`, `Warn`, or `Block`, which the loop
//!    honors before running the tool.
//! 2. **After dispatch** — `record_invocation()` is fed the `tool_result`
//!    so the shield can update its multi-turn state (call history,
//!    combination tracking) for future `evaluate()` calls.
//!
//! # Provided Implementations
//!
//! | Type           | Feature       | Purpose                                    |
//! |----------------|---------------|--------------------------------------------|
//! | [`UnixShield`] | `tool_shield` | Pattern matching for Unix shell tools      |
//! | [`NullShield`] | —             | Zero-cost no-op (used when feature is off) |
//!
//! # Usage
//!
//! ```rust,ignore
//! use loopctl::tool::shield::{UnixShield, ToolSafetyShield, ShieldContext};
//!
//! let shield = UnixShield::new();
//!
//! let ctx = ShieldContext {
//!     tool_name: "Bash".into(),
//!     input: serde_json::json!({ "command": "ls -la" }),
//!     turn: 3,
//!     recent_calls: vec![],
//! };
//!
//! match shield.evaluate(&ctx) {
//!     d if d.is_allowed() => { /* proceed */ },
//!     d if d.is_blocked() => { /* reject */ },
//!     _ => { /* warn */ },
//! }
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use serde_json::Value;

/// Risk classification produced by shield evaluation.
///
/// Each dimension of the shield (single-turn, multi-turn, combination)
/// scores into a float in `[0.0, 1.0]`. The aggregate score is then
/// mapped to a [`RiskLevel`] using configurable thresholds
/// (`warn_threshold`, `block_threshold`) on [`UnixShield`].
///
/// # Ordering
///
/// Variants are written in increasing severity. There is intentionally no
/// `Ord` derive — risk levels are labels produced by threshold
/// comparisons, not a totally-ordered set; `RiskLevel::Medium` is not
/// "less than" `RiskLevel::High` in any numeric sense callers should rely
/// on. Compare the underlying aggregate score if you need ordering.
///
/// [`UnixShield`]: crate::tool::shield::UnixShield
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiskLevel {
    /// No risk detected.
    ///
    /// Aggregate score below `0.2`. The default decision mapping for
    /// `Safe` is [`SafetyAction::Allow`] with no warning.
    Safe,

    /// Low risk — below the warn threshold but not negligible.
    ///
    /// Aggregate score in `[0.2, warn_threshold)`. Like [`Safe`], this
    /// maps to [`SafetyAction::Allow`]; the level exists so callers
    /// inspecting a [`SafetyDecision`] can distinguish "nothing matched"
    /// from "minor patterns matched, but below the warn bar."
    ///
    /// [`Safe`]: Self::Safe
    /// [`SafetyAction::Allow`]: crate::tool::shield::SafetyAction::Allow
    Low,

    /// Moderate risk — at or above the warn threshold.
    ///
    /// Aggregate score in `[warn_threshold, block_threshold)`. Maps to
    /// [`SafetyAction::Warn`]: the call proceeds, but the middleware
    /// surfaces a reason and category to observers / logs.
    ///
    /// [`SafetyAction::Warn`]: crate::tool::shield::SafetyAction::Warn
    Medium,

    /// High risk — at or above the block threshold.
    ///
    /// Aggregate score in `[block_threshold, 0.9)`. Maps to
    /// [`SafetyAction::Block`]: the call does not proceed.
    ///
    /// [`SafetyAction::Block`]: crate::tool::shield::SafetyAction::Block
    High,

    /// Critical risk — the most dangerous category.
    ///
    /// Aggregate score at or above `0.9`. Reserved for patterns the
    /// shield treats as maximally dangerous (e.g. `rm -rf /`, `curl … |
    /// sh`). Maps to [`SafetyAction::Block`].
    ///
    /// [`SafetyAction::Block`]: crate::tool::shield::SafetyAction::Block
    Critical,
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Safe => write!(f, "safe"),
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// The action the shield recommends for a tool call.
///
/// Produced as the [`SafetyDecision::action`] field. The middleware is
/// expected to honor it: proceed on [`Allow`], proceed-and-log on
/// [`Warn`], refuse to dispatch on [`Block`].
///
/// There is intentionally no `Ord` derive — these are categorical
/// recommendations, not an ordered severity scale. Use [`RiskLevel`] for
/// severity comparisons.
///
/// [`Allow`]: Self::Allow
/// [`Warn`]: Self::Warn
/// [`Block`]: Self::Block
/// [`SafetyDecision::action`]: crate::tool::shield::SafetyDecision::action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SafetyAction {
    /// Allow the tool call to proceed.
    ///
    /// The shield found no concerning patterns, or every pattern it found
    /// scored below the warn threshold. No metadata is required on the
    /// accompanying [`SafetyDecision`].
    ///
    /// [`SafetyDecision`]: crate::tool::shield::SafetyDecision
    Allow,

    /// Allow the tool call to proceed, but emit a warning.
    ///
    /// The shield detected moderate risk (score in the warn band). The
    /// call is permitted to run — warnings are advisory, not blocking —
    /// but the [`SafetyDecision`] carries a human-readable `reason` and
    /// a machine-readable `category` for observers, logs, and TUIs.
    ///
    /// [`SafetyDecision`]: crate::tool::shield::SafetyDecision
    Warn,

    /// Block the tool call entirely.
    ///
    /// The shield detected high or critical risk (score at or above the
    /// block threshold). The middleware must refuse to dispatch the call;
    /// the [`SafetyDecision`] carries a `reason` and `category`
    /// describing what was matched, for surfacing to the user.
    ///
    /// [`SafetyDecision`]: crate::tool::shield::SafetyDecision
    Block,
}

/// The shield's decision for a single tool invocation.
///
/// Produced by [`ToolSafetyShield::evaluate`]. Carries the action the
/// middleware should take, an optional human-readable reason explaining
/// the decision, and an optional machine-readable category tag for
/// programmatic handling / filtering in logs.
///
/// # Construction
///
/// Use the [`allow`](Self::allow), [`warn`](Self::warn), and
/// [`block`](Self::block) constructors rather than building the struct
/// directly — they keep the reason/category fields consistent with the
/// chosen action (e.g. `allow` always has `None` for both).
#[derive(Debug, Clone)]
pub struct SafetyDecision {
    /// The recommended action for this tool call.
    ///
    /// Always set. See [`SafetyAction`] for the three variants and the
    /// middleware's responsibility for each.
    pub action: SafetyAction,

    /// Human-readable explanation of the decision.
    ///
    /// Set for [`SafetyAction::Warn`] and [`SafetyAction::Block`]
    /// decisions (the constructors populate it from the caller-supplied
    /// string); `None` for [`SafetyAction::Allow`]. Suitable for
    /// surfacing to a user or writing to a log line.
    pub reason: Option<String>,

    /// Machine-readable category tag for the decision.
    ///
    /// Set for `Warn` and `Block`; `None` for `Allow`. Free-form but
    /// conventionally a stable identifier like `"safety_evaluation"` or
    /// `"pattern_match"` so downstream consumers can route on it without
    /// parsing the `reason` text.
    pub category: Option<String>,
}

impl SafetyDecision {
    /// Create an [`SafetyAction::Allow`] decision with no attached
    /// metadata.
    ///
    /// Both `reason` and `category` are `None` — an allowed call has
    /// nothing to warn or block about.
    #[must_use]
    pub fn allow() -> Self {
        Self {
            action: SafetyAction::Allow,
            reason: None,
            category: None,
        }
    }

    /// Create a [`SafetyAction::Warn`] decision with a reason and
    /// category.
    ///
    /// The call proceeds, but the middleware should surface `reason` to
    /// observers / logs. `category` should be a stable machine-readable
    /// identifier consumers can filter on.
    #[must_use]
    pub fn warn(reason: String, category: &str) -> Self {
        Self {
            action: SafetyAction::Warn,
            reason: Some(reason),
            category: Some(category.to_string()),
        }
    }

    /// Create a [`SafetyAction::Block`] decision with a reason and
    /// category.
    ///
    /// The middleware must refuse to dispatch the call. `reason` should
    /// explain what was matched so the user can understand why; `category`
    /// should be a stable identifier for programmatic routing.
    #[must_use]
    pub fn block(reason: String, category: &str) -> Self {
        Self {
            action: SafetyAction::Block,
            reason: Some(reason),
            category: Some(category.to_string()),
        }
    }

    /// Returns `true` if the decision's action is
    /// [`SafetyAction::Allow`].
    ///
    /// Convenience predicate for middleware that wants to short-circuit
    /// "this call is fine, dispatch it" without a full `match`.
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        self.action == SafetyAction::Allow
    }

    /// Returns `true` if the decision's action is
    /// [`SafetyAction::Block`].
    ///
    /// Convenience predicate for middleware that wants to short-circuit
    /// "refuse this call" without a full `match`.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.action == SafetyAction::Block
    }

    /// Returns `true` if the decision's action is
    /// [`SafetyAction::Warn`].
    ///
    /// Convenience predicate for middleware that wants to branch on
    /// "proceed but log" without a full `match`.
    #[must_use]
    pub fn is_warn(&self) -> bool {
        self.action == SafetyAction::Warn
    }
}

/// Context provided to the shield for evaluation.
///
/// Constructed by the middleware before each tool call. Carries what the
/// shield needs to evaluate both single-turn risk (the current call in
/// isolation) and multi-turn / combination risk (the current call in the
/// context of the recent call history).
///
/// # Lifecycle
///
/// A fresh `ShieldContext` is built per tool invocation and passed by
/// reference to [`ToolSafetyShield::evaluate`]. It is not stored across
/// calls — shields that need persistent history keep their own
/// (typically `Mutex<Vec<…>>`-protected) state, updated in
/// [`ToolSafetyShield::record_invocation`].
#[derive(Debug, Clone)]
pub struct ShieldContext {
    /// Name of the tool being invoked.
    ///
    /// Matches the registry key the engine used to dispatch the call.
    /// Single-turn risk patterns are looked up under this name, so a
    /// tool's risk configuration is keyed off the same identifier the
    /// agent uses to call it.
    pub tool_name: String,

    /// The JSON input passed to the tool.
    ///
    /// Shield patterns operate on the stringified form of this value
    /// (substring matching), so any JSON-shaped input is admissible —
    /// objects, arrays, primitives. The same value is later handed to
    /// [`ToolSafetyShield::record_invocation`] so the shield can store
    /// it for combination-rule matching.
    pub input: Value,

    /// Current turn number in the agent session, 0-indexed.
    ///
    /// Lets the shield factor recency into its scoring (e.g. weight
    /// recent calls more heavily) and lets logs correlate shield
    /// decisions with the turn that produced them.
    pub turn: usize,

    /// Recent tool invocations in this session, as `(tool_name, turn)`
    /// pairs.
    ///
    /// A read-only snapshot provided by the middleware for shields that
    /// prefer not to maintain their own history. Shields that *do*
    /// maintain their own history (like [`UnixShield`]) can ignore this
    /// field and consult their internal state instead.
    ///
    /// [`UnixShield`]: crate::tool::shield::UnixShield
    pub recent_calls: Vec<(String, usize)>,
}

/// A named risk pattern matched against a tool's input.
///
/// Patterns are stored per tool name. When a tool call is evaluated,
/// its stringified JSON input is checked against every pattern
/// registered for that tool name; any pattern whose `pattern` substring
/// appears contributes its `score` to the single-turn risk dimension.
///
/// # Matching
///
/// Matching is plain substring search on `input.to_string()` — there is
/// no regex, word-boundary, or JSON-aware path matching. Patterns must
/// be chosen so that a substring match implies the intended risk (e.g.
/// `"rm -rf"` is distinctive enough; a bare `"rm"` would over-match).
#[derive(Clone)]
pub struct RiskPattern {
    /// Human-readable name for diagnostics and log messages.
    ///
    /// Appears in shield output and logs when this pattern is the
    /// matched one. Should be unique within a tool's pattern list and
    /// descriptive enough to identify the risk at a glance (e.g.
    /// `"recursive_delete"`, `"write_ssh"`, `"curl_pipe_sh"`).
    pub name: &'static str,

    /// Score contributed to the single-turn dimension if this pattern
    /// matches, in `[0.0, 1.0]`.
    ///
    /// When multiple patterns match a single input, the highest score
    /// wins ([`UnixShield::assess_single_tool_risk`] takes the `max`).
    /// Convention: `0.9` for critical patterns (`rm -rf`, `curl | sh`),
    /// `0.5`–`0.8` for serious patterns, `< 0.5` for minor ones.
    ///
    /// [`UnixShield::assess_single_tool_risk`]: crate::tool::shield::UnixShield::assess_single_tool_risk
    pub score: f32,

    /// Substring to search for in the stringified input.
    ///
    /// Matched verbatim (no regex, no case folding). Picked so that a
    /// substring hit reliably indicates the intended risk — see the
    /// type-level docs on matching.
    pub pattern: &'static str,
}

/// A rule that scores a dangerous *sequence* of tool calls.
///
/// A combination rule triggers when **all** of its [`triggers`](Self::triggers)
/// appear in recent history or the current call, in the order specified.
/// Each trigger is a `(tool_name, optional_substring)` pair: the
/// `tool_name` must match exactly, and when the substring is present the
/// trigger only fires if that substring is in the call's stringified
/// input.
///
/// # Example
///
/// A rule with triggers `&[("Bash", Some("curl")), ("Bash", Some("| sh"))]`
/// fires only when a `curl` call is followed by a `| sh` call in the
/// session — catching the classic "download and execute" pattern that
/// neither call would flag in isolation.
#[derive(Clone)]
pub struct CombinationRule {
    /// Human-readable description for diagnostics and log messages.
    ///
    /// Surfaces in shield output when the rule fires. Should name the
    /// dangerous sequence (e.g. `"download then execute"`,
    /// `"write then chmod +x"`) so a user reading the warning
    /// understands what the shield saw.
    pub description: &'static str,

    /// Score contributed to the combination dimension if this rule
    /// triggers, in `[0.0, 1.0]`.
    ///
    /// When multiple rules fire on the same call, the highest score
    /// wins ([`UnixShield::assess_combination`] takes the `max`).
    /// Convention: `0.8+` for sequences that are almost always
    /// adversarial; `0.5`–`0.7` for suspicious-but-defensible ones.
    ///
    /// [`UnixShield::assess_combination`]: crate::tool::shield::UnixShield::assess_combination
    pub score: f32,

    /// Pairs of `(tool_name, optional_substring)` that must all appear,
    /// in order, for the rule to trigger.
    ///
    /// The match is chronological against the candidate sequence
    /// (recorded history followed by the current call). Each trigger
    /// advances only when both the tool name matches and, if a
    /// substring is supplied, that substring appears in the call's
    /// stringified input. A `None` substring means "any call to this
    /// tool".
    pub triggers: &'static [(&'static str, Option<&'static str>)],
}

/// Trait for evaluating tool call safety.
///
/// Implementations assess whether a tool invocation is safe to execute,
/// considering the current input, the session's call history, and any
/// known dangerous patterns. The trait is fully generic — it places no
/// constraints on tool names, input formats, or operating systems.
///
/// # Provided Implementations
///
/// - [`UnixShield`] — behind `tool_shield` feature; static pattern
///   matching for Unix shell tools (`Bash`, `Write`, `Edit`).
/// - [`NullShield`] — always allows; used when the feature is disabled.
///
/// # Integration Points
///
/// - **Middleware (engine):** `SafetyShieldMiddleware` wraps the tool
///   dispatch, calling [`evaluate`](ToolSafetyShield::evaluate) before
///   each invocation and [`record_invocation`](ToolSafetyShield::record_invocation)
///   after.
/// - **Hooks:** The middleware can be bypassed by a hook that returns
///   `Allow` unconditionally for specific tools.
pub trait ToolSafetyShield: Send + Sync {
    /// Evaluate whether the tool call described by `ctx` should be
    /// allowed, warned about, or blocked.
    ///
    /// Called by the middleware before dispatch. The shield inspects the
    /// call (and, if it maintains its own history, prior calls) and
    /// returns a [`SafetyDecision`] naming the action to take, with a
    /// reason/category populated for `Warn` and `Block`.
    ///
    /// Implementations must be deterministic for a given `(self, ctx)`
    /// pair so that replaying the same call produces the same decision;
    /// non-determinism breaks the VCR / cassette test path.
    fn evaluate(&self, ctx: &ShieldContext) -> SafetyDecision;

    /// Called after the tool executes so the shield can update internal
    /// state for multi-turn / combination analysis.
    ///
    /// The middleware calls this with the same `input` [`Value`] passed
    /// to [`evaluate`](Self::evaluate), plus a `success` flag indicating
    /// whether the tool returned an error. Shields that maintain call
    /// history (e.g. [`UnixShield`]) append to it here; shields with no
    /// multi-turn state (e.g. [`NullShield`]) no-op.
    ///
    /// [`UnixShield`]: crate::tool::shield::UnixShield
    /// [`NullShield`]: crate::tool::shield::NullShield
    fn record_invocation(&self, tool_name: &str, input: &Value, success: bool);

    /// Return the set of tool names this shield wants to inspect.
    ///
    /// Optimization: the middleware consults this before calling
    /// [`evaluate`](Self::evaluate) and skips evaluation for tools not
    /// in the set. Shields should return the keys of their per-tool
    /// pattern database. An empty set means "no tools watched" and
    /// causes the middleware to skip evaluation entirely (used by
    /// [`NullShield`] to be truly zero-cost).
    ///
    /// [`NullShield`]: crate::tool::shield::NullShield
    fn watched_tools(&self) -> HashSet<String>;
}

/// Reference shield implementation with Unix shell pattern matching.
///
/// Detects dangerous patterns in tools commonly found in Unix-based
/// agent environments: shell execution (`Bash`), file writes (`Write`),
/// and file edits (`Edit`). **Not suitable for Windows or non-shell
/// tool environments** — implement [`ToolSafetyShield`] directly for
/// those cases.
///
/// # Scoring
///
/// The aggregate score is computed as:
///
/// ```text
/// total = (single + multi × 0.5 + combo × 0.3).min(1.0)
/// ```
///
/// where each dimension contributes:
///
/// | Dimension         | Method                       | Weight |
/// |-------------------|------------------------------|--------|
/// | Single-turn risk  | [`assess_single_tool_risk`]  | 1.0    |
/// | Multi-turn risk   | [`assess_multi_turn`]        | 0.5    |
/// | Combination risk  | [`assess_combination`]       | 0.3    |
///
/// The aggregate is then mapped to a [`RiskLevel`] via configurable
/// thresholds (defaults: warn ≥ 0.4, block ≥ 0.7).
///
/// [`assess_single_tool_risk`]: UnixShield::assess_single_tool_risk
/// [`assess_multi_turn`]: UnixShield::assess_multi_turn
/// [`assess_combination`]: UnixShield::assess_combination
pub struct UnixShield {
    /// Aggregate score at or above which [`SafetyAction::Warn`] is
    /// returned.
    ///
    /// Configurable via [`with_thresholds`](Self::with_thresholds) or the
    /// builder; defaults to `0.4`. Clamped to `[0.0, 1.0]` on
    /// construction.
    warn_threshold: f32,

    /// Aggregate score at or above which [`SafetyAction::Block`] is
    /// returned.
    ///
    /// Must be `>= warn_threshold` for sensible behavior; the shield does
    /// not enforce this at construction time, so a misconfigured builder
    /// can produce surprising results. Defaults to `0.7`.
    block_threshold: f32,

    /// Per-invocation call history: `(tool_name, stringified_input)`.
    ///
    /// Mutex-protected because [`ToolSafetyShield`] requires `Send +
    /// Sync` and `evaluate`/`record_invocation` are `&self` methods.
    /// Trimmed to the last 20 entries on each `record_invocation` to
    /// bound memory growth in long sessions.
    turn_history: Mutex<Vec<(String, String)>>,

    /// Per-tool single-turn risk patterns, keyed by tool name.
    ///
    /// Populated from [`unix_patterns`](Self::unix_patterns) by default
    /// (`Bash`, `Write`, `Edit`); extendable via the builder. The keyset
    /// also defines [`watched_tools`](ToolSafetyShield::watched_tools).
    patterns: HashMap<&'static str, Vec<RiskPattern>>,

    /// Combination rules for dangerous sequences.
    ///
    /// Populated from [`unix_combination_rules`](Self::unix_combination_rules)
    /// by default (write→execute, download→execute, chmod→write);
    /// extendable via the builder.
    combination_rules: Vec<CombinationRule>,
}

impl UnixShield {
    /// Create a shield with default Unix shell patterns and thresholds.
    ///
    /// - Warn at aggregate score ≥ 0.4
    /// - Block at aggregate score ≥ 0.7
    #[must_use]
    pub fn new() -> Self {
        Self {
            warn_threshold: 0.4,
            block_threshold: 0.7,
            turn_history: Mutex::new(Vec::new()),
            patterns: Self::unix_patterns(),
            combination_rules: Self::unix_combination_rules(),
        }
    }

    /// Override the default warn and block thresholds, builder-style.
    ///
    /// Both values are clamped to `[0.0, 1.0]` so an out-of-range
    /// configuration cannot produce a shield that never warns or never
    /// blocks. The shield does not enforce `block >= warn`; passing an
    /// inverted pair will produce surprising decisions, so callers
    /// should validate their own inputs.
    #[must_use]
    pub fn with_thresholds(mut self, warn: f32, block: f32) -> Self {
        self.warn_threshold = warn.clamp(0.0, 1.0);
        self.block_threshold = block.clamp(0.0, 1.0);
        self
    }

    /// Create a builder for a shield with custom patterns and rules.
    ///
    /// The builder is initialised with the default Unix patterns.
    /// Use [`UnixShieldBuilder::blank`] to start with an empty pattern
    /// database.
    #[must_use]
    pub fn builder() -> UnixShieldBuilder {
        UnixShieldBuilder::new()
    }

    /// Assess single-turn risk by matching the tool input against the
    /// patterns registered for that tool.
    ///
    /// Stringifies `input` and substring-matches it against every
    /// [`RiskPattern`] keyed under `tool_name`. When multiple patterns
    /// match, the highest `score` among them is returned (a single
    /// dangerous pattern dominates several minor ones). Returns `0.0`
    /// when the tool has no registered patterns or none of them match.
    ///
    /// This is the single-turn contribution to the aggregate score;
    /// [`evaluate`](ToolSafetyShield::evaluate) weights it at `1.0`
    /// (the highest-weighted dimension).
    pub fn assess_single_tool_risk(&self, tool_name: &str, input: &Value) -> f32 {
        let Some(patterns) = self.patterns.get(tool_name) else {
            return 0.0;
        };
        let input_str = input.to_string();
        let mut max_score = 0.0_f32;
        for p in patterns {
            if input_str.contains(p.pattern) {
                max_score = max_score.max(p.score);
            }
        }
        max_score
    }

    /// Assess multi-turn risk by counting prior calls to the same tool
    /// in the recorded history.
    ///
    /// The score graduates with repetition so that a single repeat is
    /// mild but a long run of the same tool flags as suspicious:
    /// Graduated: 0 calls = 0.0, 1 = 0.1, 2 = 0.3, 3+ = 0.6
    /// [`evaluate`](ToolSafetyShield::evaluate) weights this at `0.5`,
    /// so even at the saturated `0.6` it cannot alone push the
    /// aggregate past the warn threshold — but combined with a
    /// single-turn hit it adds up.
    pub fn assess_multi_turn(&self, ctx: &ShieldContext) -> f32 {
        let history = crate::error::recover_guard(self.turn_history.lock());
        let same_tool_count = history
            .iter()
            .filter(|(name, _)| name == &ctx.tool_name)
            .count();
        match same_tool_count {
            0 => 0.0,
            1 => 0.1,
            2 => 0.3,
            _ => 0.6,
        }
    }

    /// Assess combination risk by detecting dangerous sequences of
    /// tool calls in recorded history plus the current call.
    ///
    /// Each [`CombinationRule`] specifies an ordered sequence of
    /// triggers (`(tool_name, optional_substring)` pairs). The match
    /// walks the candidate sequence — recorded history followed by the
    /// current call — and advances a per-rule trigger pointer only
    /// when both the tool name matches and, if a substring is
    /// supplied, that substring appears in the call's stringified
    /// input. A rule fires when its pointer reaches the end of its
    /// trigger list, contributing its `score`.
    ///
    /// When multiple rules fire, the highest score wins. Returns `0.0`
    /// if no rule triggers. [`evaluate`](ToolSafetyShield::evaluate)
    /// weights this dimension at `0.3` — the lowest-weighted, but the
    /// only one that can detect adversarial *sequences* (download →
    /// execute, write → chmod +x).
    pub fn assess_combination(&self, ctx: &ShieldContext) -> f32 {
        let history = crate::error::recover_guard(self.turn_history.lock());
        let mut max_risk = 0.0_f32;

        // Build a chronological candidate sequence: history entries in
        // recorded order, followed by the current call.
        let current_event = (ctx.tool_name.as_str(), ctx.input.to_string());
        let candidates: Vec<(&str, String)> = history
            .iter()
            .map(|(name, input)| (name.as_str(), input.clone()))
            .chain(std::iter::once((current_event.0, current_event.1.clone())))
            .collect();

        for rule in &self.combination_rules {
            // Walk the candidate sequence, advancing a trigger pointer
            // only when the current trigger matches in order.
            let mut trigger_idx: usize = 0;
            for (name, input) in &candidates {
                let Some(&(tool, pattern)) = rule.triggers.get(trigger_idx) else {
                    break;
                };
                if *name == tool && pattern.is_none_or(|p| input.contains(p)) {
                    trigger_idx = trigger_idx.saturating_add(1);
                    if trigger_idx == rule.triggers.len() {
                        break;
                    }
                }
            }
            if trigger_idx == rule.triggers.len() {
                max_risk = max_risk.max(rule.score);
            }
        }
        max_risk
    }

    /// Map an aggregate risk score to a [`RiskLevel`] using this
    /// shield's configured thresholds.
    ///
    /// Bands (default thresholds): `< 0.2` → [`Safe`](RiskLevel::Safe),
    /// `[0.2, 0.4)` → [`Low`](RiskLevel::Low),
    /// `[warn_threshold, block_threshold)` →
    /// [`Medium`](RiskLevel::Medium),
    /// `[block_threshold, 0.9)` → [`High`](RiskLevel::High),
    /// `>= 0.9` → [`Critical`](RiskLevel::Critical).
    /// `0.9` is hardcoded for `Critical` so the most dangerous patterns
    /// always land in their own bucket regardless of threshold
    /// configuration.
    fn score_to_level(&self, score: f32) -> RiskLevel {
        if score >= 0.9 {
            RiskLevel::Critical
        } else if score >= self.block_threshold {
            RiskLevel::High
        } else if score >= self.warn_threshold {
            RiskLevel::Medium
        } else if score >= 0.2 {
            RiskLevel::Low
        } else {
            RiskLevel::Safe
        }
    }

    /// Default pattern database for Unix shell environments.
    ///
    /// Covers `Bash` (destructive commands, privilege escalation,
    /// pipe-to-shell), `Write` (system directories, SSH config), and
    /// `Edit` (system directories, SSH config).
    fn unix_patterns() -> HashMap<&'static str, Vec<RiskPattern>> {
        let mut m = HashMap::new();
        m.insert(
            "Bash",
            vec![
                RiskPattern {
                    name: "recursive_delete",
                    score: 0.9,
                    pattern: "rm -rf",
                },
                RiskPattern {
                    name: "force_delete",
                    score: 0.7,
                    pattern: "rm -f",
                },
                RiskPattern {
                    name: "sudo",
                    score: 0.6,
                    pattern: "sudo",
                },
                RiskPattern {
                    name: "chmod_777",
                    score: 0.5,
                    pattern: "chmod 777",
                },
                RiskPattern {
                    name: "shell_pipe_to_sh",
                    score: 0.8,
                    pattern: "| sh",
                },
                RiskPattern {
                    name: "curl_pipe_sh",
                    score: 0.9,
                    pattern: "curl",
                },
                RiskPattern {
                    name: "wget_pipe",
                    score: 0.7,
                    pattern: "wget",
                },
            ],
        );
        m.insert(
            "Write",
            vec![
                RiskPattern {
                    name: "write_bin",
                    score: 0.6,
                    pattern: "/usr/bin",
                },
                RiskPattern {
                    name: "write_bin_sbin",
                    score: 0.6,
                    pattern: "/usr/sbin",
                },
                RiskPattern {
                    name: "write_etc",
                    score: 0.7,
                    pattern: "/etc/",
                },
                RiskPattern {
                    name: "write_ssh",
                    score: 0.8,
                    pattern: ".ssh/",
                },
            ],
        );
        m.insert(
            "Edit",
            vec![
                RiskPattern {
                    name: "edit_etc",
                    score: 0.5,
                    pattern: "/etc/",
                },
                RiskPattern {
                    name: "edit_ssh",
                    score: 0.7,
                    pattern: ".ssh/",
                },
            ],
        );
        m
    }

    /// Default combination rules for Unix shell environments.
    ///
    /// Detects: write → execute, download → execute, and
    /// modify-permissions → write sequences.
    fn unix_combination_rules() -> Vec<CombinationRule> {
        vec![
            CombinationRule {
                description: "write then execute",
                score: 0.75,
                triggers: &[("Write", None), ("Bash", Some("chmod +x"))],
            },
            CombinationRule {
                description: "download then execute",
                score: 0.85,
                triggers: &[("Bash", Some("curl")), ("Bash", Some("| sh"))],
            },
            CombinationRule {
                description: "modify permissions then write",
                score: 0.65,
                triggers: &[("Bash", Some("chmod")), ("Write", None)],
            },
        ]
    }
}

impl Default for UnixShield {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolSafetyShield for UnixShield {
    fn evaluate(&self, ctx: &ShieldContext) -> SafetyDecision {
        let single = self.assess_single_tool_risk(&ctx.tool_name, &ctx.input);
        let multi = self.assess_multi_turn(ctx);
        let combo = self.assess_combination(ctx);
        let total = (single + multi * 0.5 + combo * 0.3).min(1.0);
        let level = self.score_to_level(total);

        match level {
            RiskLevel::Safe | RiskLevel::Low => SafetyDecision::allow(),
            RiskLevel::Medium => SafetyDecision::warn(
                format!("Moderate risk detected (score={total:.2})"),
                "safety_evaluation",
            ),
            RiskLevel::High | RiskLevel::Critical => SafetyDecision::block(
                format!("High risk blocked (score={total:.2}, category={level})"),
                "safety_evaluation",
            ),
        }
    }

    fn record_invocation(&self, tool_name: &str, input: &Value, _success: bool) {
        let mut history = crate::error::recover_guard(self.turn_history.lock());
        history.push((tool_name.to_string(), input.to_string()));
        // Trim to last 20 entries.
        if history.len() > 20 {
            let drain_until = history.len().saturating_sub(20);
            history.drain(..drain_until);
        }
    }

    fn watched_tools(&self) -> HashSet<String> {
        self.patterns.keys().map(|k| (*k).to_string()).collect()
    }
}

/// Builder for constructing a [`UnixShield`] with custom configuration.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::tool::shield::{UnixShield, RiskPattern, CombinationRule};
///
/// let shield = UnixShield::builder()
///     .with_warn_threshold(0.3)
///     .with_block_threshold(0.6)
///     .with_pattern("PowerShell", vec![
///         RiskPattern { name: "remove_item", score: 0.9, pattern: "Remove-Item" },
///     ])
///     .with_combination_rule(CombinationRule {
///         description: "download then run",
///         score: 0.8,
///         triggers: &[("PowerShell", Some("Invoke-WebRequest")), ("PowerShell", Some("Invoke-Expression"))],
///     })
///     .build();
/// ```
pub struct UnixShieldBuilder {
    /// Aggregate score at or above which the built shield will return
    /// [`SafetyAction::Warn`]. Defaults to `0.4`; override via
    /// [`with_warn_threshold`](Self::with_warn_threshold).
    warn_threshold: f32,

    /// Aggregate score at or above which the built shield will return
    /// [`SafetyAction::Block`]. Defaults to `0.7`; override via
    /// [`with_block_threshold`](Self::with_block_threshold).
    block_threshold: f32,

    /// Per-tool single-turn risk patterns, keyed by tool name.
    ///
    /// Populated from [`UnixShield::unix_patterns`] by
    /// [`new`](Self::new); empty under [`blank`](Self::blank); extended
    /// via [`with_pattern`](Self::with_pattern).
    patterns: HashMap<&'static str, Vec<RiskPattern>>,

    /// Combination rules for dangerous sequences.
    ///
    /// Populated from [`UnixShield::unix_combination_rules`] by
    /// [`new`](Self::new); empty under [`blank`](Self::blank); extended
    /// via [`with_combination_rule`](Self::with_combination_rule).
    ///
    /// [`UnixShield::unix_combination_rules`]: crate::tool::shield::UnixShield::unix_combination_rules
    combination_rules: Vec<CombinationRule>,
}

impl UnixShieldBuilder {
    /// Create a builder initialised with the default Unix patterns.
    ///
    /// Call [`with_pattern`](UnixShieldBuilder::with_pattern) to add additional
    /// patterns, or use [`blank`](UnixShieldBuilder::blank) to start
    /// from scratch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            warn_threshold: 0.4,
            block_threshold: 0.7,
            patterns: UnixShield::unix_patterns(),
            combination_rules: UnixShield::unix_combination_rules(),
        }
    }

    /// Create a builder with no patterns (blank slate).
    ///
    /// Useful when building a shield for a non-Unix environment (e.g.
    /// Windows `PowerShell`, cloud APIs) from the ground up.
    #[must_use]
    pub fn blank() -> Self {
        Self {
            warn_threshold: 0.4,
            block_threshold: 0.7,
            patterns: HashMap::new(),
            combination_rules: Vec::new(),
        }
    }

    /// Set the warn threshold. Clamped to `[0.0, 1.0]`.
    ///
    /// An aggregate score at or above this value produces a
    /// [`SafetyAction::Warn`] decision.
    #[must_use]
    pub fn with_warn_threshold(mut self, threshold: f32) -> Self {
        self.warn_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the block threshold. Clamped to `[0.0, 1.0]`.
    ///
    /// An aggregate score at or above this value produces a
    /// [`SafetyAction::Block`] decision.
    #[must_use]
    pub fn with_block_threshold(mut self, threshold: f32) -> Self {
        self.block_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Register single-turn risk patterns for a tool, builder-style.
    ///
    /// Appends to any patterns already registered for `tool_name`
    /// rather than replacing them — call repeatedly to assemble a
    /// tool's risk profile across multiple additions. The first call
    /// for a given `tool_name` also adds it to the built shield's
    /// [`watched_tools`](ToolSafetyShield::watched_tools) set.
    #[must_use]
    pub fn with_pattern(mut self, tool_name: &'static str, patterns: Vec<RiskPattern>) -> Self {
        self.patterns.entry(tool_name).or_default().extend(patterns);
        self
    }

    /// Register a combination rule for dangerous tool sequences,
    /// builder-style.
    ///
    /// The rule is evaluated on every call to
    /// [`UnixShield::evaluate`] and fires when all of its triggers
    /// appear in the session history in order. Call repeatedly to add
    /// multiple rules; rules are independent (the highest score among
    /// triggered rules wins).
    ///
    /// [`UnixShield::evaluate`]: crate::tool::shield::UnixShield::evaluate
    #[must_use]
    pub fn with_combination_rule(mut self, rule: CombinationRule) -> Self {
        self.combination_rules.push(rule);
        self
    }

    /// Consume the builder and return a configured [`UnixShield`].
    ///
    /// The shield's [`watched_tools`](ToolSafetyShield::watched_tools)
    /// set is derived from the pattern keys, so tools with no patterns
    /// are skipped during evaluation.
    #[must_use]
    pub fn build(self) -> UnixShield {
        UnixShield {
            warn_threshold: self.warn_threshold,
            block_threshold: self.block_threshold,
            turn_history: Mutex::new(Vec::new()),
            patterns: self.patterns,
            combination_rules: self.combination_rules,
        }
    }
}

impl Default for UnixShieldBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A no-op [`ToolSafetyShield`] that allows every call and watches no
/// tools.
///
/// Used as the default shield type when the `tool_shield` feature is
/// disabled, so engine code that holds `Arc<dyn ToolSafetyShield>`
/// compiles and runs without forcing the feature on downstream
/// consumers.
///
/// # Zero-cost
///
/// All three trait methods return constants with no field access
/// (`evaluate` returns a pre-built [`SafetyDecision::allow`],
/// `record_invocation` no-ops, `watched_tools` returns an empty
/// [`HashSet`]). Because the middleware short-circuits evaluation
/// entirely when `watched_tools` is empty, an installed `NullShield`
/// costs nothing at runtime — and the compiler can inline and elide
/// the constant returns.
pub struct NullShield;

impl ToolSafetyShield for NullShield {
    fn evaluate(&self, _ctx: &ShieldContext) -> SafetyDecision {
        SafetyDecision::allow()
    }

    fn record_invocation(&self, _tool_name: &str, _input: &Value, _success: bool) {
        // No-op — no state to update.
    }

    fn watched_tools(&self) -> HashSet<String> {
        // Empty set = no tools watched = middleware skips evaluation entirely.
        HashSet::new()
    }
}

impl Default for NullShield {
    fn default() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx(name: &str, input: Value, turn: usize) -> ShieldContext {
        ShieldContext {
            tool_name: name.to_string(),
            input,
            turn,
            recent_calls: vec![],
        }
    }

    #[test]
    fn null_shield_allows_everything() {
        let shield = NullShield;
        let decision = shield.evaluate(&ctx("Bash", json!("rm -rf /"), 0));
        assert!(decision.is_allowed());
        assert!(shield.watched_tools().is_empty());
    }

    #[test]
    fn null_shield_default_trait() {
        let shield = NullShield;
        let _ = &shield;
    }

    #[test]
    fn unix_shield_allows_safe_command() {
        let shield = UnixShield::new();
        let decision = shield.evaluate(&ctx("Bash", json!({ "command": "ls -la" }), 0));
        assert!(decision.is_allowed());
    }

    #[test]
    fn unix_shield_blocks_recursive_delete() {
        let shield = UnixShield::new();
        let decision = shield.evaluate(&ctx("Bash", json!({ "command": "rm -rf /" }), 0));
        assert!(decision.is_blocked());
    }

    #[test]
    fn unix_shield_warns_on_sudo() {
        let shield = UnixShield::new();
        let decision = shield.evaluate(&ctx("Bash", json!({ "command": "sudo apt update" }), 0));
        assert!(decision.is_warn() || decision.is_blocked());
    }

    #[test]
    fn unix_shield_allows_read() {
        let shield = UnixShield::new();
        let decision = shield.evaluate(&ctx("Read", json!({ "path": "/tmp/hello.txt" }), 0));
        assert!(decision.is_allowed());
    }

    #[test]
    fn unix_shield_records_and_trims_history() {
        let shield = UnixShield::new();
        for i in 0..25 {
            shield.record_invocation("Bash", &json!({ "command": "ls" }), true);
            let history = shield.turn_history.lock().unwrap();
            assert!(history.len() <= 20, "iteration {i}: history too long");
        }
    }

    #[test]
    fn unix_shield_watched_tools_derived_from_patterns() {
        let shield = UnixShield::new();
        let watched = shield.watched_tools();
        assert!(watched.contains("Bash"));
        assert!(watched.contains("Write"));
        assert!(watched.contains("Edit"));
    }

    #[test]
    fn unix_shield_default_trait() {
        let _shield = UnixShield::default();
    }

    #[test]
    fn combination_rule_requires_chronological_order() {
        // Use a blank shield with a single rule to avoid false matches
        // from other built-in rules (e.g. "modify permissions then write").
        let shield = UnixShieldBuilder::blank()
            .with_combination_rule(CombinationRule {
                description: "write then execute",
                score: 0.75,
                triggers: &[("Write", None), ("Bash", Some("chmod +x"))],
            })
            .build();

        // Record in REVERSE order: Bash(chmod +x) first, then Write.
        // The rule requires Write → Bash, so this should NOT match.
        shield.record_invocation("Bash", &json!({ "command": "chmod +x /tmp/payload" }), true);
        shield.record_invocation("Write", &json!({ "path": "/tmp/payload" }), true);

        let eval_ctx = ctx("Read", json!({ "path": "/tmp/data" }), 2);
        let combo = shield.assess_combination(&eval_ctx);
        assert!(
            (combo - 0.0).abs() < f32::EPSILON,
            "reversed order should not match"
        );

        // Now test correct order: Write first, then Bash(chmod +x) as the
        // current call.
        let shield2 = UnixShieldBuilder::blank()
            .with_combination_rule(CombinationRule {
                description: "write then execute",
                score: 0.75,
                triggers: &[("Write", None), ("Bash", Some("chmod +x"))],
            })
            .build();
        shield2.record_invocation("Write", &json!({ "path": "/tmp/payload" }), true);
        let eval_ctx2 = ctx("Bash", json!({ "command": "chmod +x /tmp/payload" }), 1);
        let combo2 = shield2.assess_combination(&eval_ctx2);
        assert!(combo2 > 0.0, "correct order should match");
    }

    #[test]
    fn builder_blank_has_no_patterns() {
        let shield = UnixShieldBuilder::blank().build();
        let watched = shield.watched_tools();
        assert!(watched.is_empty());
    }

    #[test]
    fn builder_custom_pattern() {
        let shield = UnixShieldBuilder::blank()
            .with_pattern(
                "PowerShell",
                vec![RiskPattern {
                    name: "remove_item",
                    score: 0.9,
                    pattern: "Remove-Item",
                }],
            )
            .build();
        let watched = shield.watched_tools();
        assert!(watched.contains("PowerShell"));
        assert!(!watched.contains("Bash"));

        let decision = shield.evaluate(&ctx(
            "PowerShell",
            json!({ "command": "Remove-Item -Recurse -Force C:\\" }),
            0,
        ));
        assert!(decision.is_blocked());
    }

    #[test]
    fn builder_custom_thresholds() {
        let shield = UnixShield::builder()
            .with_warn_threshold(0.1)
            .with_block_threshold(0.2)
            .build();
        // sudo scores 0.6, which is above block_threshold of 0.2
        let decision = shield.evaluate(&ctx("Bash", json!({ "command": "sudo ls" }), 0));
        assert!(decision.is_blocked());
    }

    #[test]
    fn builder_default_trait() {
        let _builder = UnixShieldBuilder::default();
    }

    #[test]
    fn risk_level_display() {
        assert_eq!(RiskLevel::Safe.to_string(), "safe");
        assert_eq!(RiskLevel::Critical.to_string(), "critical");
    }

    #[test]
    fn safety_decision_accessors() {
        let allow = SafetyDecision::allow();
        assert!(allow.is_allowed());
        assert!(!allow.is_blocked());
        assert!(!allow.is_warn());

        let warn = SafetyDecision::warn("test".into(), "cat");
        assert!(warn.is_warn());

        let block = SafetyDecision::block("test".into(), "cat");
        assert!(block.is_blocked());
    }
}

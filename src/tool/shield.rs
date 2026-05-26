//! Tool Safety Shield — multi-turn adversarial defense.
//!
//! This module provides the [`ToolSafetyShield`] trait — a generic,
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
//! ```text
//! ┌────────────────────────────────────────────┐
//! │          Agent Loop (middleware)            │
//! │                                            │
//! │  tool_call ──▶ ToolSafetyShield::evaluate()│
//! │                     │                      │
//! │            ┌────────┴────────┐             │
//! │            │  SafetyDecision │             │
//! │            │  Allow / Warn / │             │
//! │            │  Block          │             │
//! │            └────────┬────────┘             │
//! │                     │                      │
//! │  tool_result ──▶ record_invocation()       │
//! └────────────────────────────────────────────┘
//! ```
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

// ===================================================
// RiskLevel
// ===================================================

/// Risk classification produced by shield evaluation.
///
/// Each dimension of the shield (single-turn, multi-turn, combination)
/// scores into a float in `[0.0, 1.0]`. The aggregate score is then
/// mapped to a [`RiskLevel`] using configurable thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiskLevel {
    /// No risk detected. Aggregate score < 0.2.
    Safe,
    /// Low risk. Aggregate score in [0.2, `warn_threshold`).
    Low,
    /// Moderate risk. Aggregate score in [`warn_threshold`, `block_threshold`).
    Medium,
    /// High risk. Aggregate score in [`block_threshold`, 0.9).
    High,
    /// Critical risk. Aggregate score ≥ 0.9.
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

// ===================================================
// SafetyAction
// ===================================================

/// The action the shield recommends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SafetyAction {
    /// Allow the tool call to proceed.
    Allow,
    /// Allow but emit a warning.
    Warn,
    /// Block the tool call entirely.
    Block,
}

// ===================================================
// SafetyDecision
// ===================================================

/// The shield's decision for a single tool invocation.
///
/// Produced by [`ToolSafetyShield::evaluate`]. Carries the action, an
/// optional human-readable reason, and a machine-readable category tag.
#[derive(Debug, Clone)]
pub struct SafetyDecision {
    /// The recommended action.
    pub action: SafetyAction,
    /// Human-readable explanation (set for `Warn` and `Block`).
    pub reason: Option<String>,
    /// Machine-readable category (e.g. `"safety_evaluation"`, `"pattern_match"`).
    pub category: Option<String>,
}

impl SafetyDecision {
    /// Create an `Allow` decision with no attached metadata.
    #[must_use]
    pub fn allow() -> Self {
        Self {
            action: SafetyAction::Allow,
            reason: None,
            category: None,
        }
    }

    /// Create a `Warn` decision with reason and category.
    #[must_use]
    pub fn warn(reason: String, category: &str) -> Self {
        Self {
            action: SafetyAction::Warn,
            reason: Some(reason),
            category: Some(category.to_string()),
        }
    }

    /// Create a `Block` decision with reason and category.
    #[must_use]
    pub fn block(reason: String, category: &str) -> Self {
        Self {
            action: SafetyAction::Block,
            reason: Some(reason),
            category: Some(category.to_string()),
        }
    }

    /// Whether the decision is `Allow`.
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        self.action == SafetyAction::Allow
    }

    /// Whether the decision is `Block`.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.action == SafetyAction::Block
    }

    /// Whether the decision is `Warn`.
    #[must_use]
    pub fn is_warn(&self) -> bool {
        self.action == SafetyAction::Warn
    }
}

// ===================================================
// ShieldContext
// ===================================================

/// Context provided to the shield for evaluation.
///
/// Constructed by the middleware before each tool call. The shield uses
/// this to evaluate single-turn risk and multi-turn patterns.
#[derive(Debug, Clone)]
pub struct ShieldContext {
    /// Name of the tool being invoked.
    pub tool_name: String,
    /// The JSON input to the tool.
    pub input: Value,
    /// Current turn number in the agent session (0-indexed).
    pub turn: usize,
    /// Recent tool invocations in this session (tool name, turn).
    pub recent_calls: Vec<(String, usize)>,
}

// ===================================================
// RiskPattern
// ===================================================

/// A named risk pattern matched against tool input.
///
/// Patterns are stored per tool name. When a tool call is evaluated,
/// its stringified JSON input is checked against every pattern
/// registered for that tool name.
#[derive(Clone)]
pub struct RiskPattern {
    /// Human-readable name for diagnostics and log messages.
    pub name: &'static str,
    /// Score contribution if matched, in `[0.0, 1.0]`.
    pub score: f32,
    /// Substring to search for in the stringified input.
    pub pattern: &'static str,
}

// ===================================================
// CombinationRule
// ===================================================

/// A rule that scores a dangerous *sequence* of tool calls.
///
/// A combination rule triggers when **all** of its [`triggers`](CombinationRule::triggers)
/// appear in recent history or the current call. Each trigger is a
/// `(tool_name, optional_substring)` pair.
#[derive(Clone)]
pub struct CombinationRule {
    /// Human-readable description for diagnostics and log messages.
    pub description: &'static str,
    /// Score contribution if matched.
    pub score: f32,
    /// Pairs of `(tool_name, optional_substring)` that must all appear
    /// for the rule to trigger.
    pub triggers: &'static [(&'static str, Option<&'static str>)],
}

// ===================================================
// ToolSafetyShield trait
// ===================================================

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
    fn evaluate(&self, ctx: &ShieldContext) -> SafetyDecision;

    /// Called after the tool executes. Allows the shield to update
    /// internal state (e.g. call history for multi-turn analysis).
    fn record_invocation(&self, tool_name: &str, success: bool);

    /// Return the set of tool names this shield wants to inspect.
    ///
    /// Optimization: if the shield returns an empty set, the middleware
    /// can skip calling [`evaluate()`](ToolSafetyShield::evaluate) for
    /// tools that the shield has no rules for.
    fn watched_tools(&self) -> HashSet<String>;
}

// ===================================================
// UnixShield
// ===================================================

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
    /// Score at which [`SafetyAction::Warn`] is returned.
    warn_threshold: f32,
    /// Score at which [`SafetyAction::Block`] is returned.
    block_threshold: f32,
    /// Per-turn invocation history: `(tool_name, turn)`.
    turn_history: Mutex<Vec<(String, usize)>>,
    /// Per-tool single-turn risk patterns.
    patterns: HashMap<&'static str, Vec<RiskPattern>>,
    /// Combination rules for dangerous sequences.
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

    /// Create a shield with custom thresholds.
    ///
    /// Values are clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn with_thresholds(warn: f32, block: f32) -> Self {
        Self {
            warn_threshold: warn.clamp(0.0, 1.0),
            block_threshold: block.clamp(0.0, 1.0),
            turn_history: Mutex::new(Vec::new()),
            patterns: Self::unix_patterns(),
            combination_rules: Self::unix_combination_rules(),
        }
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

    /// Assess single-turn risk: match the tool input against known
    /// dangerous patterns for that tool.
    ///
    /// Returns the highest score among all matched patterns, or `0.0`
    /// if the tool has no registered patterns or none matched.
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

    /// Assess multi-turn risk: look for repeated calls to the same
    /// tool across recent turns.
    ///
    /// The score graduates with repetition: 0 calls → 0.0, 1 → 0.1,
    /// 2 → 0.3, 3+ → 0.6.
    pub fn assess_multi_turn(&self, ctx: &ShieldContext) -> f32 {
        let history = self
            .turn_history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let same_tool_count = history
            .iter()
            .filter(|(name, _)| name == &ctx.tool_name)
            .count();
        // Graduated: 0 calls = 0.0, 1 = 0.1, 2 = 0.3, 3+ = 0.6
        match same_tool_count {
            0 => 0.0,
            1 => 0.1,
            2 => 0.3,
            _ => 0.6,
        }
    }

    /// Assess combination risk: detect dangerous tool sequences in
    /// recent history.
    ///
    /// Each [`CombinationRule`] specifies a set of triggers that must
    /// all appear (either in history or the current call) for the rule
    /// to fire. Returns the highest score among all matched rules.
    pub fn assess_combination(&self, ctx: &ShieldContext) -> f32 {
        let history = self
            .turn_history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut max_risk = 0.0_f32;
        for rule in &self.combination_rules {
            let mut all_matched = true;
            for &(tool, pat) in rule.triggers {
                let found_in_history = history.iter().any(|(name, _)| {
                    if name != tool {
                        return false;
                    }
                    pat.is_none_or(|p| ctx.input.to_string().contains(p))
                });
                if !found_in_history {
                    let current_matches = ctx.tool_name == tool
                        && pat.is_none_or(|p| ctx.input.to_string().contains(p));
                    if !current_matches {
                        all_matched = false;
                        break;
                    }
                }
            }
            if all_matched {
                max_risk = max_risk.max(rule.score);
            }
        }
        max_risk
    }

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

    fn record_invocation(&self, tool_name: &str, _success: bool) {
        let mut history = self
            .turn_history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        history.push((tool_name.to_string(), 0));
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

// ===================================================
// UnixShieldBuilder
// ===================================================

/// Builder for constructing a [`UnixShield`] with custom configuration.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::tool::shield::{UnixShield, RiskPattern, CombinationRule};
///
/// let shield = UnixShield::builder()
///     .warn_threshold(0.3)
///     .block_threshold(0.6)
///     .pattern("PowerShell", vec![
///         RiskPattern { name: "remove_item", score: 0.9, pattern: "Remove-Item" },
///     ])
///     .combination_rule(CombinationRule {
///         description: "download then run",
///         score: 0.8,
///         triggers: &[("PowerShell", Some("Invoke-WebRequest")), ("PowerShell", Some("Invoke-Expression"))],
///     })
///     .build();
/// ```
pub struct UnixShieldBuilder {
    warn_threshold: f32,
    block_threshold: f32,
    patterns: HashMap<&'static str, Vec<RiskPattern>>,
    combination_rules: Vec<CombinationRule>,
}

impl UnixShieldBuilder {
    /// Create a builder initialised with the default Unix patterns.
    ///
    /// Call [`pattern`](UnixShieldBuilder::pattern) to add additional
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
    pub fn warn_threshold(mut self, threshold: f32) -> Self {
        self.warn_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the block threshold. Clamped to `[0.0, 1.0]`.
    ///
    /// An aggregate score at or above this value produces a
    /// [`SafetyAction::Block`] decision.
    #[must_use]
    pub fn block_threshold(mut self, threshold: f32) -> Self {
        self.block_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Register single-turn risk patterns for a tool.
    ///
    /// Appends to any existing patterns for that tool name.
    #[must_use]
    pub fn pattern(mut self, tool_name: &'static str, patterns: Vec<RiskPattern>) -> Self {
        self.patterns.entry(tool_name).or_default().extend(patterns);
        self
    }

    /// Register a combination rule.
    ///
    /// The rule is evaluated on every call to [`UnixShield::evaluate`]
    /// and fires when all of its triggers appear in the session history.
    #[must_use]
    pub fn combination_rule(mut self, rule: CombinationRule) -> Self {
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

// ===================================================
// NullShield
// ===================================================

/// A no-op shield that allows everything.
///
/// Used when the `tool_shield` feature is disabled. Because all methods
/// return constants with no field access, the compiler can inline and
/// eliminate all calls to this type, making it truly zero-cost.
pub struct NullShield;

impl ToolSafetyShield for NullShield {
    fn evaluate(&self, _ctx: &ShieldContext) -> SafetyDecision {
        SafetyDecision::allow()
    }

    fn record_invocation(&self, _tool_name: &str, _success: bool) {
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

    // ===================================================
    // NullShield tests
    // ===================================================

    #[test]
    fn null_shield_allows_everything() {
        let shield = NullShield;
        let decision = shield.evaluate(&ctx("Bash", json!("rm -rf /"), 0));
        assert!(decision.is_allowed());
        assert!(shield.watched_tools().is_empty());
    }

    #[test]
    fn null_shield_default_trait() {
        let _shield = NullShield::default();
    }

    // ===================================================
    // UnixShield tests
    // ===================================================

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
            shield.record_invocation("Bash", true);
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

    // ===================================================
    // Builder tests
    // ===================================================

    #[test]
    fn builder_blank_has_no_patterns() {
        let shield = UnixShieldBuilder::blank().build();
        let watched = shield.watched_tools();
        assert!(watched.is_empty());
    }

    #[test]
    fn builder_custom_pattern() {
        let shield = UnixShieldBuilder::blank()
            .pattern(
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
            .warn_threshold(0.1)
            .block_threshold(0.2)
            .build();
        // sudo scores 0.6, which is above block_threshold of 0.2
        let decision = shield.evaluate(&ctx("Bash", json!({ "command": "sudo ls" }), 0));
        assert!(decision.is_blocked());
    }

    #[test]
    fn builder_default_trait() {
        let _builder = UnixShieldBuilder::default();
    }

    // ===================================================
    // Type tests
    // ===================================================

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

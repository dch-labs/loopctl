//! Loop detection for agent operations — prevents infinite repetition.
//!
//! This module detects when the agent is stuck in a loop, repeating the same
//! operations without making forward progress. It maintains a sliding window
//! of recent tool invocations, hashes their results, and raises warnings or
//! hard stops when the same `(tool, primary_param, result_hash)` tuple
//! recurs beyond configurable thresholds.
//!
//! # Why Loop Detection Matters
//!
//! Autonomous agents can get trapped in repetitive cycles — for example,
//! repeatedly reading a file and then attempting the same edit that failed
//! before. Without detection, the agent wastes tokens and time until an
//! external timeout kicks in. This module provides an early-warning system
//! that flags loops after just a few repetitions, and can force-stop the
//! agent when the repetition count becomes dangerous.
//!
//! # Core Algorithm
//!
//! Every tool invocation is recorded as an [`Operation`] (tool name +
//! primary parameter + optional result hash). The detector maintains a
//! bounded sliding window of these operations. On each [`LoopDetector::check_loop`]
//! call, it counts occurrences of identical operations and compares the counts
//! against configurable thresholds. Operations producing *different* results
//! are treated as distinct, because changing output indicates forward progress.
//!
//! # Provided Types
//!
//! - **[`LoopDetector`]** — Thread-safe detector that records operations and
//!   evaluates loop conditions on every call.
//! - **[`LoopDetectorConfig`]** — Tunable thresholds for window size,
//!   repetition counts, per-turn limits, and file-read caps.
//! - **[`Operation`]** — A single recorded tool invocation (tool name +
//!   primary parameter + optional result hash).
//! - **[`LoopStatus`]** — The result of a [`LoopDetector::check_loop`] call,
//!   indicating whether a loop was found and whether the agent should stop.
//! - **[`ToolSignature`]** — Trait for injecting tool-specific parsing logic
//!   (e.g. extracting file paths from JSON, recognising recoverable errors).
//! - **[`NoOpToolSignature`]** — Default implementation that treats all tools
//!   generically.
//! - **[`hash_result`]** — Utility function that hashes tool output into a
//!   compact `u64` for result-aware comparison.
//! - **[`global_detector`]** — Lazy singleton for frameworks that don't
//!   manage their own detector instance.
//!
//! # Quick Start
//!
//! ```rust
//! use std::sync::Arc;
//! use loopctl::loop_control::loop_detector::{
//!     LoopDetector, LoopDetectorConfig, Operation, ToolSignature,
//! };
//!
//! // Create a detector with default settings and a no-op signature.
//! let detector = LoopDetector::default_detector();
//!
//! // Record some operations (normally called by the framework after each tool call).
//! detector.record(Operation::new("Read", "/src/main.rs"));
//! detector.record(Operation::new("Read", "/src/main.rs"));
//! detector.record(Operation::new("Read", "/src/main.rs"));
//!
//! // Check for loops.
//! let status = detector.check_loop();
//! if status.is_looping {
//!     println!("Warning: {}", status.warning.unwrap_or_default());
//! }
//! ```
//!
//! # Data Flow
//!
//! ```text
//! ┌──────────────┐     record()       ┌─────────────────────┐
//! │  Framework   │ ────────────────►  │  LoopDetector       │
//! │  (tool call) │                    │  ┌───────────────┐  │
//! └──────────────┘                    │  │ sliding window│  │
//!                                     │  │ (VecDeque)    │  │
//!                                     │  └───────────────┘  │
//!                                     │  ┌───────────────┐  │
//!                                     │  │ warned_ops    │  │
//!                                     │  │ (HashSet)     │  │
//!                                     │  └───────────────┘  │
//!                                     └─────────┬───────────┘
//!                                               │
//!                                   check_loop()/check_turn_limit()
//!                                               │
//!                                               ▼
//!                                     ┌─────────────────────┐
//!                                     │    LoopStatus       │
//!                                     │    is_looping       │
//!                                     │    should_stop      │
//!                                     │    warning          │
//!                                     └─────────────────────┘
//! ```
//!
//! # Edit-Recovery Workflow
//!
//! The detector includes special handling for the edit-recovery pattern.
//! When an edit tool fails (recoverable error) and the agent re-reads the
//! file to get updated contents, the loop warning for that file is cleared
//! because the agent is making progress. This prevents false positives
//! during normal edit-retry cycles.
//!
//! ```text
//!  Edit(file, old_text) → FAIL (recoverable)
//!    ↓
//!  Read(file) → check_and_reset_on_file_read() clears warning
//!    ↓
//!  Edit(file, new_text) → SUCCESS (not counted as a loop)
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Trait for extracting tool-specific information during loop detection.
///
/// The loop detector needs to understand *what* a tool invocation targets
/// (e.g. which file, which command) and *whether* an error is recoverable.
/// Because each agent framework has its own set of tools with different
/// JSON schemas, this logic is injected via the [`ToolSignature`] trait.
///
/// The framework provides a [`NoOpToolSignature`] default that treats all
/// tools generically — `extract_primary_param` returns an empty string,
/// no errors are considered recoverable, and no suggestions are offered.
/// Production crates (e.g. `dch-agent-tools`) implement their own
/// signature with real parsing logic.
///
/// # Design Rationale
///
/// Rather than hard-coding tool names and JSON paths inside the detector,
/// the signature pattern keeps the detector agnostic to any particular
/// tool set. This allows the same [`LoopDetector`] to work with any
/// collection of tools — just swap the signature implementation.
///
/// # Implementing
///
/// All methods have default no-op implementations, so implementors only
/// need to override the methods relevant to their tool set:
///
/// - [`extract_primary_param`](ToolSignature::extract_primary_param) —
///   Returns a string that uniquely identifies the operation's target.
///   **Override this for every tool that has a distinguishing parameter.**
/// - [`is_recoverable_error`](ToolSignature::is_recoverable_error) —
///   Returns `true` if the error means the agent is still making progress.
/// - [`get_suggestion`](ToolSignature::get_suggestion) —
///   Returns an optional tool-specific fix-it suggestion for loop warnings.
/// - [`file_path_for_reset`](ToolSignature::file_path_for_reset) —
///   Returns the file path if a tool invocation should reset loop state
///   for that file.
/// - [`is_file_read_tool`](ToolSignature::is_file_read_tool) —
///   Returns `true` if the tool reads file contents.
/// - [`is_file_edit_tool`](ToolSignature::is_file_edit_tool) —
///   Returns `true` if the tool modifies file contents.
/// - [`tool_thresholds`](ToolSignature::tool_thresholds) —
///   Returns tool-specific repetition thresholds that override the
///   generic [`LoopDetectorConfig::repetition_threshold`].
///
/// # Example
///
/// ```rust
/// use loopctl::loop_control::loop_detector::ToolSignature;
///
/// struct MyToolSignature;
///
/// impl ToolSignature for MyToolSignature {
///     fn extract_primary_param(&self, tool: &str, input: &serde_json::Value) -> String {
///         match tool {
///             "ReadFile" => input.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string(),
///             "Bash"     => input.get("command").and_then(|v| v.as_str()).unwrap_or("").to_string(),
///             _          => input.to_string(),
///         }
///     }
///
///     fn is_file_read_tool(&self, tool: &str) -> bool {
///         matches!(tool, "ReadFile" | "Bash")
///     }
///
///     fn is_file_edit_tool(&self, tool: &str) -> bool {
///         tool == "EditFile"
///     }
/// }
/// ```
///
/// # Thread Safety
///
/// Implementations must be [`Send`] + [`Sync`] because the trait object is
/// stored in an [`Arc`] inside [`LoopDetector`]. All methods take `&self`,
/// so the implementation should be stateless or use interior mutability.
///
pub trait ToolSignature: Send + Sync {
    /// Extract a primary parameter from tool input for loop comparison.
    ///
    /// The returned string should uniquely identify the operation's target
    /// (e.g., file path + content hash for Edit operations). Two operations
    /// with the same `(tool, primary_param)` are considered identical for
    /// loop detection purposes.
    ///
    /// # When Called
    ///
    /// Called by [`Operation::from_input_with_signature`] and
    /// [`LoopDetector::record_from_input`] when the framework records a
    /// new tool invocation.
    ///
    /// # Default
    ///
    /// Returns an empty string — no differentiation between invocations.
    ///
    fn extract_primary_param(&self, tool: &str, input: &serde_json::Value) -> String {
        let _ = (tool, input);
        String::new()
    }

    /// Check if a tool error indicates a recoverable condition (not a loop).
    ///
    /// For example, "old text not found" in an Edit tool means the file
    /// has changed — the agent is making progress, not looping. Returning
    /// `true` prevents the detector from flagging the sequence as a loop.
    ///
    /// This is one of the most impactful methods to override correctly.
    /// Without recoverable-error detection, the detector will flag every
    /// retry cycle as a loop, even when the agent is following a healthy
    /// read-fail-retry pattern.
    ///
    /// # When Called
    ///
    /// Called by the framework when a tool invocation returns an error,
    /// before deciding whether to count it toward repetition.
    ///
    /// # Default
    ///
    /// Returns `false` — all errors are treated as potential loop signals.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::loop_detector::ToolSignature;
    ///
    /// struct MySig;
    /// impl ToolSignature for MySig {
    ///     fn is_recoverable_error(&self, tool: &str, error: &str) -> bool {
    ///         match tool {
    ///             "Edit" => error.contains("old text not found"),
    ///             "Bash" => error.contains("permission denied"),
    ///             _ => false,
    ///         }
    ///     }
    /// }
    /// ```
    ///
    fn is_recoverable_error(&self, tool: &str, error: &str) -> bool {
        let _ = (tool, error);
        false
    }

    /// Get a tool-specific suggestion when a loop is detected.
    ///
    /// Returns `None` if no tool-specific advice is available. When a
    /// suggestion is returned it is appended to the
    /// [`LoopStatus::warning`] message, giving the agent (or the human
    /// operator) actionable guidance.
    ///
    /// # When Called
    ///
    /// Called by [`LoopDetector::check_loop`] when building the warning
    /// message for a detected loop.
    ///
    /// # Default
    ///
    /// Returns `None`.
    ///
    fn get_suggestion(&self, tool: &str) -> Option<String> {
        let _ = tool;
        None
    }

    /// Check if a Read operation to the same target should reset loop state
    /// after a previous failed Edit.
    ///
    /// Returns the file path to reset for, or `None` if no reset is needed.
    /// This supports the edit-recovery workflow: when an edit fails and the
    /// agent re-reads the file to get updated contents, the loop warning
    /// for that file is cleared because the agent is making progress.
    ///
    /// This mechanism prevents a common false-positive pattern where the
    /// agent edits → fails → reads → edits again in a legitimate retry
    /// cycle. Without the reset, the detector would incorrectly flag the
    /// second edit as a loop continuation.
    ///
    /// # When Called
    ///
    /// Called by the framework after each tool invocation to determine
    /// whether to reset loop state.
    ///
    /// # Default
    ///
    /// Returns `None` — no automatic resets.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::loop_detector::ToolSignature;
    ///
    /// struct MySig;
    /// impl ToolSignature for MySig {
    ///     fn file_path_for_reset(&self, tool: &str, input: &serde_json::Value) -> Option<String> {
    ///         if tool == "Read" {
    ///             input.get("file_path").and_then(|v| v.as_str()).map(|s| s.to_string())
    ///         } else {
    ///             None
    ///         }
    ///     }
    /// }
    /// ```
    ///
    fn file_path_for_reset(&self, tool: &str, input: &serde_json::Value) -> Option<String> {
        let _ = (tool, input);
        None
    }

    /// Check if the tool is a "read" type that accesses files.
    ///
    /// Used by [`LoopDetector::check_file_reads`] to count how many times
    /// a specific file has been read. If the count exceeds
    /// [`LoopDetectorConfig::max_same_file_reads`], a warning is raised.
    ///
    /// # When Called
    ///
    /// Called during [`LoopDetector::check_file_reads`] and
    /// [`LoopDetector::check_and_reset_on_file_read`].
    ///
    /// # Default
    ///
    /// Returns `false`.
    ///
    fn is_file_read_tool(&self, tool: &str) -> bool {
        let _ = tool;
        false
    }

    /// Check if the tool is an "edit" type that modifies files.
    ///
    /// Used by the edit-recovery logic: if an edit fails (recoverable
    /// error), subsequent reads to the same file clear the loop warning
    /// because the agent is re-reading to get the updated state.
    ///
    /// # When Called
    ///
    /// Called during [`LoopDetector::record`] to detect recoverable edits,
    /// and during [`LoopDetector::check_and_reset_on_file_read`] to
    /// identify failed edits in the operation history.
    ///
    /// # Default
    ///
    /// Returns `false`.
    ///
    fn is_file_edit_tool(&self, tool: &str) -> bool {
        let _ = tool;
        false
    }

    /// Return tool-specific repetition thresholds that override the generic
    /// [`LoopDetectorConfig::repetition_threshold`].
    ///
    /// For example, `Edit` and `MultiEdit` typically get `threshold = 2` because
    /// a failing edit repeating 3+ times is almost always a loop, whereas the
    /// generic threshold might be 3.
    ///
    /// # When Called
    ///
    /// Called by [`LoopDetectorConfig::threshold_for_tool`] when looking up
    /// the effective threshold for a given tool name.
    ///
    /// # Default
    ///
    /// Returns an empty `HashMap` — no tool-specific overrides.
    ///
    fn tool_thresholds(&self) -> HashMap<String, usize> {
        HashMap::new()
    }

    /// Normalize the primary parameter for file-path comparison.
    ///
    /// Some tools embed extra information in the primary parameter beyond
    /// the file path — for example, line-number anchors like
    /// `"src/main.rs#42"`. When comparing operations to decide whether
    /// they target the same file, the anchor should be stripped so that
    /// `"src/main.rs#42"` and `"src/main.rs#100"` are treated as the
    /// same file.
    ///
    /// Override this method if your tool set uses such conventions. The
    /// default implementation returns the parameter unchanged.
    ///
    /// # When Called
    ///
    /// Called during edit-recovery logic in
    /// [`LoopDetector::record_from_input_with_error`] to match edit
    /// operations to the same file, and during
    /// [`LoopDetector::check_and_reset_on_file_read`] to find prior
    /// warned edits for the file being read.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::loop_detector::ToolSignature;
    ///
    /// struct MySig;
    /// impl ToolSignature for MySig {
    ///     fn normalize_param_for_comparison(&self, _tool: &str, param: &str) -> String {
    ///         // Strip "#line_number" suffixes for file-path comparison.
    ///         param.split('#').next().unwrap_or(param).to_string()
    ///     }
    /// }
    ///
    /// let sig = MySig;
    /// assert_eq!(
    ///     sig.normalize_param_for_comparison("Edit", "src/main.rs#42"),
    ///     "src/main.rs",
    /// );
    /// ```
    fn normalize_param_for_comparison(&self, _tool: &str, param: &str) -> String {
        param.to_string()
    }
}

/// A no-op tool signature that provides generic, unopinionated behavior.
///
/// All tools are treated identically —
/// [`extract_primary_param`](ToolSignature::extract_primary_param) returns
/// an empty string, no recoverable errors are recognised, and no
/// suggestions are offered. Use this as a safe default or when
/// tool-specific behaviour is not needed (e.g. in tests or simple agents).
///
/// Because [`extract_primary_param`](ToolSignature::extract_primary_param)
/// always returns an empty string, *every* invocation of the same tool
/// with the same result hash is considered identical. This means the
/// detector can still catch loops — it just can't distinguish between
/// different targets within the same tool.
///
/// # When to Use
///
/// - **Testing** — When you need a detector but don't care about
///   tool-specific behaviour.
/// - **Simple agents** — When all tools are treated uniformly.
/// - **Prototyping** — Before investing in a custom [`ToolSignature`]
///   implementation.
///
/// # When to Replace
///
/// Switch to a custom signature when you need:
/// - Per-file loop detection (e.g. "Read `/a.rs`" vs "Read `/b.rs`")
/// - Recoverable-error recognition (e.g. "old text not found")
/// - Tool-specific suggestions in warning messages
/// - Different repetition thresholds per tool
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use loopctl::loop_control::loop_detector::{
///     LoopDetector, LoopDetectorConfig, NoOpToolSignature,
/// };
///
/// let detector = LoopDetector::new(
///     LoopDetectorConfig::default(),
///     Arc::new(NoOpToolSignature),
/// );
/// ```
///
pub struct NoOpToolSignature;

/// Blanket [`ToolSignature`] implementation for [`NoOpToolSignature`].
///
/// Inherits every default method — no overrides. All tools are treated as
/// opaque operations with no primary parameter, no recoverable errors, and
/// no suggestions. Because [`ToolSignature::extract_primary_param`] always
/// returns `""`, every invocation of the same tool with the same result hash
/// is considered identical for loop-detection purposes.
///
/// This implementation is intentionally empty — it relies entirely on the
/// default method bodies defined in the [`ToolSignature`] trait. See the
/// trait-level documentation for the semantics of each default.
///
impl ToolSignature for NoOpToolSignature {}

/// Configuration for the [`LoopDetector`] — controls sensitivity and limits.
///
/// Tune these values to adjust how aggressively the detector flags loops.
/// Stricter thresholds catch loops sooner but may produce false positives;
/// looser thresholds reduce noise but let the agent spin longer.
///
/// # Construction
///
/// Use [`LoopDetectorConfig::default`] for sensible out-of-the-box values,
/// then override individual fields as needed:
///
/// ```rust
/// use loopctl::loop_control::loop_detector::LoopDetectorConfig;
/// use std::collections::HashMap;
///
/// let config = LoopDetectorConfig {
///     window_size: 100,
///     repetition_threshold: 3,
///     stop_threshold: 8,
///     tool_thresholds: HashMap::from([
///         ("Edit".to_string(), 2),
///         ("MultiEdit".to_string(), 2),
///     ]),
///     ..Default::default()
/// };
/// ```
///
/// # Defaults
///
/// | Field                  | Default | Purpose                               |
/// |------------------------|---------|---------------------------------------|
/// | `window_size`          | 50      | Operations kept in the sliding window |
/// | `repetition_threshold` | 3       | Same-op count before "loop" flag      |
/// | `max_tools_per_turn`   | 9999    | Tool calls per turn before cap        |
/// | `max_same_file_reads`  | 5       | Same-file reads before warning        |
/// | `stop_threshold`       | 10      | Repetitions before forced stop        |
/// | `tool_thresholds`      | empty   | Per-tool override map                 |
///
/// # Tuning Guide
///
/// - **Reduce `repetition_threshold`** (e.g. 2) for agents that should be
///   stopped quickly when they repeat, at the cost of more false positives.
/// - **Increase `window_size`** (e.g. 100) for agents that run long
///   sessions with many interleaved operations, so the detector has
///   enough history to spot slow loops.
/// - **Lower `max_tools_per_turn`** (e.g. 30) to prevent runaway tool
///   usage within a single agent turn.
/// - **Set `stop_threshold` to 0** to disable forced stops entirely — the
///   detector will still issue warnings but never halt the agent.
/// - **Add `tool_thresholds`** for tools that are especially sensitive to
///   repetition (e.g. `"Edit" → 2`) or particularly noisy (e.g.
///   `"Grep" → 5`).
///
#[derive(Debug, Clone)]
pub struct LoopDetectorConfig {
    /// Maximum number of operations kept in the sliding window.
    ///
    /// The [`LoopDetector`] maintains a [`VecDeque`] of recent operations.
    /// When the deque reaches this size, the oldest entry is evicted before
    /// a new one is appended. A larger window detects slower loops; a
    /// smaller window is more memory-efficient and focuses on recent
    /// activity.
    ///
    /// **Default:** `50`.
    ///
    pub window_size: usize,

    /// Number of identical repetitions required to flag a loop.
    ///
    /// An operation is considered "looping" when it appears at least this
    /// many times (with the same [`Operation::result_hash`]) within the
    /// current window. Can be overridden per tool via
    /// [`tool_thresholds`](LoopDetectorConfig::tool_thresholds).
    ///
    /// **Default:** `3`.
    ///
    pub repetition_threshold: usize,

    /// Maximum number of tool calls allowed in a single turn.
    ///
    /// Once the per-turn call count reaches this limit,
    /// [`LoopDetector::check_turn_limit`] returns `true`. The turn counter
    /// is reset by [`LoopDetector::reset_turn`], which the framework calls
    /// at the start of each new turn.
    ///
    /// **Default:** `9999` (effectively unlimited).
    ///
    pub max_tools_per_turn: usize,

    /// Maximum number of identical file reads before a warning is raised.
    ///
    /// Checked by [`LoopDetector::check_file_reads`]. When a single file
    /// path appears in more than this many read-type operations within the
    /// window, the method returns `true`.
    ///
    /// **Default:** `5`.
    ///
    pub max_same_file_reads: usize,

    /// Number of repetitions required to force-stop the agent.
    ///
    /// When [`LoopDetector::check_loop`] detects repetitions ≥ this value,
    /// it sets [`LoopStatus::should_stop`] to `true` and includes
    /// `"STOPPING to prevent infinite loop"` in the warning message. Set
    /// to `0` to disable forced stops entirely (the detector will still
    /// issue warnings).
    ///
    /// **Default:** `10`.
    ///
    pub stop_threshold: usize,

    /// Tool-specific repetition thresholds that override
    /// [`repetition_threshold`](LoopDetectorConfig::repetition_threshold).
    ///
    /// Map keys are tool names (e.g. `"Edit"`, `"MultiEdit"`) and values
    /// are the repetition count that triggers a loop for that tool. Looked
    /// up by [`LoopDetectorConfig::threshold_for_tool`]. Consumers should
    /// populate this with their tool names and desired thresholds. The
    /// framework default is an empty map (no overrides).
    ///
    /// **Default:** empty `HashMap`.
    ///
    pub tool_thresholds: HashMap<String, usize>,
}

/// Produce a [`LoopDetectorConfig`] with sensible out-of-the-box values.
///
/// The defaults are tuned for typical single-agent sessions. See the field
/// documentation for the exact values and their rationale.
///
/// The returned config balances sensitivity against noise:
/// - A [`repetition_threshold`](LoopDetectorConfig::repetition_threshold) of `3`
///   catches loops quickly without false-flagging double-checks.
/// - A [`stop_threshold`](LoopDetectorConfig::stop_threshold) of `10` gives the
///   agent plenty of runway before a forced stop.
/// - A [`window_size`](LoopDetectorConfig::window_size) of `50` covers
///   interleaved multi-tool sequences without excessive memory use.
/// - A [`max_tools_per_turn`](LoopDetectorConfig::max_tools_per_turn) of `9999`
///   is effectively unlimited for normal sessions.
/// - A [`max_same_file_reads`](LoopDetectorConfig::max_same_file_reads) of `5`
///   tolerates re-reads during debugging without triggering warnings.
/// - An empty [`tool_thresholds`](LoopDetectorConfig::tool_thresholds) map
///   means the generic threshold applies to all tools uniformly.
///
/// # Example
///
/// ```rust
/// use loopctl::loop_control::loop_detector::LoopDetectorConfig;
///
/// let config = LoopDetectorConfig::default();
/// assert_eq!(config.window_size, 50);
/// assert_eq!(config.repetition_threshold, 3);
/// assert_eq!(config.max_tools_per_turn, 9999);
/// assert_eq!(config.max_same_file_reads, 5);
/// assert_eq!(config.stop_threshold, 10);
/// assert!(config.tool_thresholds.is_empty());
/// ```
///
/// # See Also
///
/// - [`LoopDetector::new`] — constructs a detector from a config.
/// - [`LoopDetectorConfig::threshold_for_tool`] — per-tool threshold lookup.
///
impl Default for LoopDetectorConfig {
    /// Build a config with the default values described in the trait-level docs.
    ///
    /// All fields are set to their documented defaults. The operation window
    /// is pre-allocated with capacity matching
    /// [`window_size`](LoopDetectorConfig::window_size).
    ///
    fn default() -> Self {
        Self {
            window_size: 50,
            repetition_threshold: 3,
            max_tools_per_turn: 9999,
            max_same_file_reads: 5,
            stop_threshold: 10,
            tool_thresholds: HashMap::new(),
        }
    }
}

/// Accessor methods for [`LoopDetectorConfig`].
///
/// This `impl` block provides helpers for looking up configuration values
/// with fallback behaviour (e.g. per-tool thresholds that delegate to the
/// generic default when no override is set).
///
/// # Methods
///
/// - [`threshold_for_tool`](LoopDetectorConfig::threshold_for_tool) — Returns
///   the effective repetition threshold for a given tool, checking per-tool
///   overrides first and falling back to the generic
///   [`repetition_threshold`](LoopDetectorConfig::repetition_threshold).
///
impl LoopDetectorConfig {
    /// Get the effective repetition threshold for a specific tool.
    ///
    /// Looks up `tool_name` in [`tool_thresholds`](LoopDetectorConfig::tool_thresholds).
    /// If a per-tool override exists it is returned; otherwise falls back to
    /// the generic [`repetition_threshold`](LoopDetectorConfig::repetition_threshold).
    ///
    /// # When Called
    ///
    /// Called by [`LoopDetector::check_loop`] when evaluating each
    /// operation's repetition count against the appropriate threshold.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::loop_detector::LoopDetectorConfig;
    ///
    /// let mut config = LoopDetectorConfig::default();
    /// config.tool_thresholds.insert("Edit".to_string(), 2);
    ///
    /// assert_eq!(config.threshold_for_tool("Edit"), 2);   // override
    /// assert_eq!(config.threshold_for_tool("Read"), 3);   // generic default
    /// ```
    ///
    #[must_use]
    pub fn threshold_for_tool(&self, tool_name: &str) -> usize {
        self.tool_thresholds
            .get(tool_name)
            .copied()
            .unwrap_or(self.repetition_threshold)
    }
}

/// A single recorded tool invocation, used as the unit of loop analysis.
///
/// Each [`Operation`] captures the tool name, a primary parameter string
/// (e.g. file path or command), and an optional hash of the tool result
/// content. Two operations with the same `(tool, primary_param, result_hash)`
/// are considered identical for loop detection: the same tool was called
/// on the same target and produced the same output.
///
/// # Equality Semantics
///
/// [`Operation`] derives [`PartialEq`] and [`Hash`], so two operations are
/// equal only when all three fields match exactly. This means the *same*
/// command producing *different* results is treated as two distinct
/// operations — a key design choice that prevents false positives when
/// the agent is making progress.
///
/// # Derives
///
/// - [`Debug`] — For logging and diagnostics.
/// - [`Clone`] — Operations are stored in [`std::collections::HashSet`]s and [`VecDeque`]s
///   which may require cloning during retention scans.
/// - [`PartialEq`] + [`Eq`] — For equality comparison in loop counting.
/// - [`Hash`] — For use as keys in [`HashMap`]s during repetition counting.
///
/// # Construction
///
/// ```rust
/// use loopctl::loop_control::loop_detector::{Operation, hash_result};
///
/// // Simple construction (no result hash):
/// let op = Operation::new("Read", "/src/main.rs");
///
/// // With a result hash:
/// let hash = hash_result("file contents here");
/// let op = Operation::new("Read", "/src/main.rs").with_result_hash(hash);
/// ```
///
/// # Role in Loop Detection
///
/// The [`LoopDetector`] stores a sliding window of [`Operation`] values.
/// During [`LoopDetector::check_loop`], it counts how many times each
/// distinct operation appears. If the count exceeds the configured
/// threshold, a loop is reported. The optional `result_hash` ensures that
/// operations producing *different* outputs are not counted as repetitions.
///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Operation {
    /// Name of the tool that was invoked (e.g. `"Read"`, `"Edit"`, `"Bash"`).
    ///
    /// Used as the first component of the loop-detection key. Together with
    /// [`primary_param`](Operation::primary_param) and
    /// [`result_hash`](Operation::result_hash) it uniquely identifies a
    /// repeated invocation pattern.
    ///
    pub tool: String,

    /// Primary parameter that identifies the operation's target.
    ///
    /// Extracted by [`ToolSignature::extract_primary_param`]. Typically a
    /// file path (for Read/Edit) or a command string (for Bash). Two
    /// operations with the same `tool` but different `primary_param` are
    /// *not* considered a loop (they target different resources).
    ///
    pub primary_param: String,

    /// Hash of the tool result content, for result-aware loop detection.
    ///
    /// When `Some(hash)`, two operations are only considered identical if
    /// they produced the same result. When `None`, results are not taken
    /// into account — the detector relies solely on `(tool, primary_param)`.
    /// Set via [`Operation::with_result_hash`] or the constructor
    /// [`Operation::from_input_with_result_and_signature`].
    ///
    /// Generated by the free function [`hash_result`].
    ///
    pub result_hash: Option<u64>,
}

/// Constructors and builder methods for [`Operation`].
///
/// Operations can be created in several ways depending on what information
/// is available at call sites:
///
/// - **[`Operation::new`]** — Simplest path: tool name + primary param.
/// - **[`Operation::from_input_with_signature`]** — Parses the primary
///   parameter from raw JSON using a [`ToolSignature`].
/// - **[`Operation::from_input_with_result_and_signature`]** — Full
///   construction with result hash, used after a tool invocation completes.
/// - **[`Operation::with_result_hash`]** — Builder-style attachment of a
///   result hash to an existing operation.
///
/// # Construction Decision Tree
///
/// ```text
/// Do you have the result yet?
///   ├── NO  → Operation::from_input_with_signature(tool, input, sig)
///   │        or Operation::new(tool, param)
///   └── YES → Operation::from_input_with_result_and_signature(
///              tool, input, hash, sig)
///            or Operation::new(tool, param).with_result_hash(hash)
/// ```
///
impl Operation {
    /// Create a new operation with the given tool name and primary parameter.
    ///
    /// The [`result_hash`](Operation::result_hash) is set to `None`. Use
    /// [`with_result_hash`](Operation::with_result_hash) to attach one
    /// after construction.
    ///
    /// Accepts `impl Into<String>` for both arguments, so you can pass
    /// `&str`, `String`, or any type that converts into `String`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::loop_detector::Operation;
    ///
    /// let op = Operation::new("Read", "/src/main.rs");
    /// assert_eq!(op.tool, "Read");
    /// assert_eq!(op.primary_param, "/src/main.rs");
    /// assert_eq!(op.result_hash, None);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Operation::from_input_with_signature`] — when you have raw JSON input.
    /// - [`Operation::with_result_hash`] — to attach a result hash after creation.
    ///
    pub fn new(tool: impl Into<String>, primary_param: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            primary_param: primary_param.into(),
            result_hash: None,
        }
    }

    /// Create an operation from a tool name, JSON input, and a tool signature.
    ///
    /// Delegates to [`ToolSignature::extract_primary_param`] to pull the
    /// identifying parameter out of the JSON `input`. The
    /// [`result_hash`](Operation::result_hash) is set to `None`.
    ///
    /// This constructor is useful when you have the raw tool input but
    /// have not yet executed the tool (so no result hash is available).
    /// After the tool finishes, call [`Operation::with_result_hash`] to
    /// attach the hash.
    ///
    /// # When Called
    ///
    /// Called by the framework when it has the raw tool input but not yet
    /// a result hash (e.g. before the tool has finished executing).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::loop_detector::{Operation, ToolSignature};
    ///
    /// struct MyToolSignature;
    /// impl ToolSignature for MyToolSignature {
    ///     fn extract_primary_param(&self, tool: &str, input: &serde_json::Value) -> String {
    ///         input.get("file_path").and_then(|v| v.as_str()).unwrap_or("").to_string()
    ///     }
    /// }
    ///
    /// let sig = MyToolSignature;
    /// let op = Operation::from_input_with_signature(
    ///     "Read",
    ///     &serde_json::json!({"file_path": "/src/main.rs"}),
    ///     &sig,
    /// );
    /// assert_eq!(op.primary_param, "/src/main.rs");
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Operation::from_input_with_result_and_signature`] — full construction with hash.
    /// - [`ToolSignature::extract_primary_param`] — the parsing logic.
    ///
    pub fn from_input_with_signature(
        tool: &str,
        input: &serde_json::Value,
        signature: &dyn ToolSignature,
    ) -> Self {
        let primary_param = signature.extract_primary_param(tool, input);
        Self::new(tool, primary_param)
    }

    /// Create an operation from tool name, JSON input, result hash, and signature.
    ///
    /// Combines [`ToolSignature::extract_primary_param`] with an explicit
    /// result hash into a single constructor call. This is the most complete
    /// construction path, used when both the input and the result are known.
    ///
    /// The `result_hash` should be computed by [`hash_result`] or set to
    /// `None` if the tool produced no output.
    ///
    /// # When Called
    ///
    /// Called by [`LoopDetector::record_from_input`] after a tool invocation
    /// completes and the result has been hashed via [`hash_result`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::loop_detector::{Operation, ToolSignature, hash_result};
    ///
    /// struct MyToolSignature;
    /// impl ToolSignature for MyToolSignature {
    ///     fn extract_primary_param(&self, tool: &str, input: &serde_json::Value) -> String {
    ///         input.get("file_path").and_then(|v| v.as_str()).unwrap_or("").to_string()
    ///     }
    /// }
    ///
    /// let sig = MyToolSignature;
    /// let hash = hash_result("file contents");
    /// let op = Operation::from_input_with_result_and_signature(
    ///     "Read",
    ///     &serde_json::json!({"file_path": "/src/main.rs"}),
    ///     hash,
    ///     &sig,
    /// );
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Operation::from_input_with_signature`] — same but without result hash.
    /// - [`LoopDetector::record_from_input`] — the primary caller.
    ///
    pub fn from_input_with_result_and_signature(
        tool: &str,
        input: &serde_json::Value,
        result_hash: Option<u64>,
        signature: &dyn ToolSignature,
    ) -> Self {
        let primary_param = signature.extract_primary_param(tool, input);
        Self {
            tool: tool.to_string(),
            primary_param,
            result_hash,
        }
    }

    /// Attach a result hash to this operation (builder style).
    ///
    /// Consumes `self` and returns a new [`Operation`] with the given hash.
    /// Used when the result is not available at construction time — for example,
    /// when the operation was created via [`Operation::from_input_with_signature`]
    /// before the tool finished executing.
    ///
    /// If `hash` is `None`, the operation's
    /// [`result_hash`](Operation::result_hash) is set to `None`, meaning the
    /// detector will compare operations solely on `(tool, primary_param)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::loop_detector::{Operation, hash_result};
    ///
    /// let hash = hash_result("file contents");
    /// let op = Operation::new("Read", "/src/main.rs").with_result_hash(hash);
    /// assert_eq!(op.result_hash, hash);
    ///
    /// // Without a hash:
    /// let op2 = Operation::new("Read", "/src/main.rs").with_result_hash(None);
    /// assert_eq!(op2.result_hash, None);
    /// ```
    ///
    /// # See Also
    ///
    /// - [`hash_result`] — computes the hash from tool output.
    /// - [`Operation::result_hash`] — the field this sets.
    ///
    #[must_use]
    pub fn with_result_hash(mut self, hash: Option<u64>) -> Self {
        self.result_hash = hash;
        self
    }
}

/// Hash tool result content into a compact `u64` for loop detection.
///
/// Uses [`std::collections::hash_map::DefaultHasher`] to produce a fast,
/// non-cryptographic hash of the content string. Two identical content
/// strings will always produce the same hash, making it suitable for
/// equality comparison inside [`Operation::result_hash`].
///
/// Returns `None` if `content` is empty (no meaningful hash to compute).
/// This is intentional: an empty result usually means the tool produced
/// no output, and hashing it would add noise without value.
///
/// # Use in Loop Detection
///
/// The hash is stored in [`Operation::result_hash`] and used by
/// [`LoopDetector::check_loop`] to distinguish between operations that
/// produce the same output (likely a loop) versus different output
/// (likely progress). Without this hashing, the detector would rely
/// solely on `(tool, primary_param)` equality, which cannot tell whether
/// the agent is making forward progress.
///
/// # Performance
///
/// [`std::collections::hash_map::DefaultHasher`] is a fast, non-cryptographic hasher suitable for
/// runtime use. It is *not* suitable for security purposes (e.g. storing
/// passwords), but it is perfect for quick equality checks in hot paths.
/// The cost is O(n) in the length of `content`, which is acceptable because
/// tool results are typically bounded in size.
///
/// # Determinism
///
/// The hash is deterministic within a single process invocation. It is
/// *not* guaranteed to be stable across different versions of the Rust
/// standard library, so do not persist these hashes to disk.
///
/// # Example
///
/// ```rust
/// use loopctl::loop_control::loop_detector::hash_result;
///
/// let h1 = hash_result("same output");
/// let h2 = hash_result("same output");
/// let h3 = hash_result("different output");
///
/// assert_eq!(h1, h2);     // same content → same hash
/// assert_ne!(h1, h3);     // different content → different hash
/// assert_eq!(hash_result(""), None);  // empty → None
/// ```
///
/// # See Also
///
/// - [`Operation::with_result_hash`] — attaches the hash to an operation.
/// - [`Operation::result_hash`] — the field that stores the hash.
///
#[must_use]
pub fn hash_result(content: &str) -> Option<u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    if content.is_empty() {
        return None;
    }

    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    Some(hasher.finish())
}

/// Result of a [`LoopDetector::check_loop`] analysis.
///
/// Describes whether a loop was detected, which operations are repeating,
/// how many repetitions were observed, and whether the agent should be
/// force-stopped. Also carries an optional human-readable
/// [`warning`](LoopStatus::warning) message.
///
/// # Fields Overview
///
/// - [`is_looping`](LoopStatus::is_looping) — `true` if any operation
///   exceeded its repetition threshold.
/// - [`repeated_operations`](LoopStatus::repeated_operations) — The
///   specific [`Operation`] values that triggered the loop.
/// - [`repetition_count`](LoopStatus::repetition_count) — How many times
///   the most-repeated operation appeared in the window.
/// - [`warning`](LoopStatus::warning) — Human-readable message for the
///   agent (or operator). `None` if no loop or already warned.
/// - [`should_stop`](LoopStatus::should_stop) — `true` when the
///   framework should halt the agent immediately.
///
/// # Lifecycle
///
/// ```text
/// check_loop() returns LoopStatus
///   │
///   ├── is_looping = false
///   │   └── No action needed. Agent continues normally.
///   │
///   └── is_looping = true
///       ├── should_stop = false
///       │   └── Warning emitted (if not already warned).
///       │       Agent should adjust behaviour.
///       │
///       └── should_stop = true
///           └── STOPPING message included in warning.
///               Framework should halt the agent.
/// ```
///
/// # Warning Deduplication
///
/// The [`warning`](LoopStatus::warning) field is `None` when:
/// 1. No loop was detected ([`is_looping`](LoopStatus::is_looping) is `false`), or
/// 2. The same operation was already warned about and the agent hasn't
///    hit the [`stop_threshold`](LoopDetectorConfig::stop_threshold).
///
/// This prevents the agent's context from being flooded with identical
/// loop messages across consecutive [`check_loop`](LoopDetector::check_loop)
/// calls. When the agent makes progress (result hash changes) or the
/// detector is reset, the warning is re-enabled.
///
/// # Example
///
/// ```rust
/// use loopctl::loop_control::loop_detector::LoopDetector;
///
/// let detector = LoopDetector::default_detector();
/// let status = detector.check_loop();
/// if status.is_looping {
///     eprintln!("Loop! count={}", status.repetition_count);
///     if let Some(msg) = &status.warning {
///         eprintln!("  {}", msg);
///     }
///     if status.should_stop {
///         // Halt the agent.
///     }
/// }
/// ```
///
/// # Default
///
/// The [`Default`] implementation produces a "no loop" status:
/// all boolean fields are `false`, `repetition_count` is `0`,
/// `repeated_operations` is empty, and `warning` is `None`.
///
/// # Derives
///
/// Derives [`Debug`] and [`Clone`] for logging and error propagation.
/// Does *not* derive [`PartialEq`] because [`Option<String>`] comparison
/// is rarely useful for status objects.
///
#[derive(Debug, Clone, Default)]
pub struct LoopStatus {
    /// Whether a loop was detected.
    ///
    /// `true` when at least one operation repeats beyond the applicable
    /// threshold (either the generic
    /// [`repetition_threshold`](LoopDetectorConfig::repetition_threshold)
    /// or a per-tool override from
    /// [`tool_thresholds`](LoopDetectorConfig::tool_thresholds)).
    ///
    pub is_looping: bool,

    /// Operations that triggered the loop detection.
    ///
    /// Contains all operations whose repetition count equals the maximum
    /// observed count and exceeds the threshold. When multiple operations
    /// tie for the highest repetition count, all of them are included.
    /// Empty when [`is_looping`](LoopStatus::is_looping) is `false`.
    ///
    pub repeated_operations: Vec<Operation>,

    /// Number of repetitions of the most-repeated operation.
    ///
    /// Equals the count of the first entry in
    /// [`repeated_operations`](LoopStatus::repeated_operations). Zero when
    /// no loop was detected.
    ///
    pub repetition_count: usize,

    /// Human-readable warning message describing the detected loop.
    ///
    /// Contains the operation description, repetition count, a "STOPPING"
    /// notice if [`should_stop`](LoopStatus::should_stop) is true, and a
    /// tool-specific suggestion from
    /// [`ToolSignature::get_suggestion`]. Set to `None` when no loop is
    /// detected, or when the loop has already been warned about (to avoid
    /// spamming the agent with duplicate warnings).
    ///
    pub warning: Option<String>,

    /// Whether the agent should be force-stopped due to severe looping.
    ///
    /// `true` when [`repetition_count`](LoopStatus::repetition_count)
    /// reaches or exceeds [`LoopDetectorConfig::stop_threshold`] (and the
    /// stop threshold is non-zero). The framework should halt the agent's
    /// event loop when this is `true`.
    ///
    pub should_stop: bool,
}

/// Thread-safe loop detector that tracks tool invocations and flags loops.
///
/// The detector maintains a sliding window of recent [`Operation`] records
/// (bounded by [`LoopDetectorConfig::window_size`]) and a set of
/// already-warned operations to avoid duplicate warnings. All internal
/// state is protected by [`Mutex`] so the detector can be shared across
/// threads (e.g. between the agent loop and an observer).
///
/// # Construction
///
/// ```rust
/// use std::sync::Arc;
/// use loopctl::loop_control::loop_detector::{
///     LoopDetector, LoopDetectorConfig, NoOpToolSignature,
/// };
///
/// // With default config and no-op signature:
/// let detector = LoopDetector::default_detector();
///
/// // With custom config:
/// let config = LoopDetectorConfig { window_size: 100, ..Default::default() };
/// let detector = LoopDetector::new(config, Arc::new(NoOpToolSignature));
///
/// // With custom tool signature:
/// // let detector = LoopDetector::with_signature(Arc::new(MyToolSignature));
/// ```
///
/// # Lifecycle
///
/// ```text
/// ┌─────────────────────────────────────────────────────┐
/// │  For each tool call:                                │
/// │    1. record_from_input(tool, input, result_hash)   │
/// │    2. check_loop() → LoopStatus                     │
/// │    3. check_turn_limit() → bool                     │
/// │                                                     │
/// │  At turn boundary:                                  │
/// │    4. reset_turn()                                  │
/// │                                                     │
/// │  At task boundary:                                  │
/// │    5. reset()                                       │
/// └─────────────────────────────────────────────────────┘
/// ```
///
/// # Thread Safety
///
/// All mutable state is wrapped in [`Mutex`]. Methods lock each field
/// independently and handle lock poisoning gracefully (by skipping the
/// operation rather than panicking). This means the detector degrades
/// gracefully under contention but never blocks the agent loop.
///
/// # Design Decisions
///
/// - **Sliding window (not global history):** A bounded [`VecDeque`] keeps
///   memory usage predictable and focuses detection on *recent* behaviour,
///   avoiding false positives from operations far in the past.
/// - **Result-aware comparison:** Two invocations of the same tool on the
///   same target are only considered identical if they produced the same
///   result hash. This prevents the detector from flagging operations
///   where the agent is genuinely making progress.
/// - **Warning deduplication:** Once an operation has triggered a warning,
///   subsequent warnings are suppressed unless the result changes or the
///   stop threshold is reached. This prevents the agent's context from
///   being flooded with identical loop messages.
/// - **Per-turn limits:** Separate from repetition detection, the turn
///   counter catches runaway tool usage regardless of whether the tools
///   are repeating. This catches scenarios like "try 50 different bash
///   commands" that wouldn't trigger repetition-based detection.
///
/// # Interior Fields
///
/// The detector holds five pieces of internal state, each in its own
/// [`Mutex`] to minimise lock contention:
///
/// | Field                | Type                      | Purpose                        |
/// |----------------------|---------------------------|--------------------------------|
/// | `operations`         | `Mutex<VecDeque<Op>>`     | Sliding window of history      |
/// | `config`             | `LoopDetectorConfig`      | Thresholds (immutable)         |
/// | `turn_count`         | `Mutex<usize>`            | Per-turn call counter          |
/// | `warned_operations`  | `Mutex<HashSet<Op>>`      | Already-warned dedup set       |
/// | `signature`          | `Arc<dyn ToolSignature>`  | Tool-specific parsing logic    |
///
pub struct LoopDetector {
    /// Sliding window of recent [`Operation`] records.
    ///
    /// Bounded by [`LoopDetectorConfig::window_size`]. When full, the
    /// oldest operation is evicted before a new one is appended. The
    /// window is scanned by [`LoopDetector::check_loop`] to find repeated
    /// operations.
    ///
    operations: Mutex<VecDeque<Operation>>,

    /// Configuration controlling thresholds and limits.
    ///
    /// Set at construction time via [`LoopDetector::new`]. Immutable for
    /// the lifetime of the detector.
    ///
    config: LoopDetectorConfig,

    /// Count of tool invocations in the current turn.
    ///
    /// Incremented by [`LoopDetector::record`] and reset to zero by
    /// [`LoopDetector::reset_turn`]. Checked against
    /// [`LoopDetectorConfig::max_tools_per_turn`] by
    /// [`LoopDetector::check_turn_limit`].
    ///
    turn_count: Mutex<usize>,

    /// Set of operations that have already triggered a warning.
    ///
    /// Once an operation appears in this set, subsequent calls to
    /// [`LoopDetector::check_loop`] will suppress the warning message
    /// (returning `None` in [`LoopStatus::warning`]) to avoid spamming
    /// the agent. Entries are cleared when the result changes (indicating
    /// progress) or when [`LoopDetector::clear`] / [`LoopDetector::reset`]
    /// is called.
    ///
    warned_operations: Mutex<std::collections::HashSet<Operation>>,

    /// Tool signature for extracting tool-specific parameters.
    ///
    /// Wrapped in [`Arc`] because it is shared between the detector and
    /// any code that needs to inspect the signature via
    /// [`LoopDetector::signature`]. The trait object is `Send + Sync` so
    /// it can be used from any thread.
    ///
    signature: Arc<dyn ToolSignature>,
}

/// Core methods for [`LoopDetector`].
///
/// This `impl` block provides all the public API for recording operations,
/// checking for loops, managing turn state, and resetting the detector.
/// Methods are designed to be called from the framework's main loop in a
/// specific order (see the lifecycle diagram in the struct-level docs).
///
/// # Error Handling
///
/// All methods handle [`Mutex`] poisoning gracefully — if another thread
/// panicked while holding the lock, the method returns a sensible default
/// (empty status, `false`, zero count) rather than propagulating the panic.
/// This ensures the detector never crashes the agent loop.
///
impl LoopDetector {
    /// Create a new loop detector with the given configuration and tool signature.
    ///
    /// Initialises an empty operation window, zero turn count, and an empty
    /// warned-operations set. The `signature` is stored in an [`Arc`] for
    /// shared access.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use loopctl::loop_control::loop_detector::{
    ///     LoopDetector, LoopDetectorConfig, NoOpToolSignature,
    /// };
    ///
    /// let config = LoopDetectorConfig { window_size: 100, ..Default::default() };
    /// let detector = LoopDetector::new(config, Arc::new(NoOpToolSignature));
    /// ```
    ///
    pub fn new(config: LoopDetectorConfig, signature: Arc<dyn ToolSignature>) -> Self {
        Self {
            operations: Mutex::new(VecDeque::with_capacity(config.window_size)),
            config,
            turn_count: Mutex::new(0),
            warned_operations: Mutex::new(std::collections::HashSet::new()),
            signature,
        }
    }

    /// Create a detector with default configuration and a no-op tool signature.
    ///
    /// Convenience constructor for simple use cases that don't need
    /// tool-specific logic. Equivalent to:
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use loopctl::loop_control::loop_detector::{LoopDetector, LoopDetectorConfig, NoOpToolSignature};
    ///
    /// LoopDetector::new(LoopDetectorConfig::default(), Arc::new(NoOpToolSignature));
    /// ```
    ///
    #[must_use]
    pub fn default_detector() -> Self {
        Self::new(LoopDetectorConfig::default(), Arc::new(NoOpToolSignature))
    }

    /// Create a detector with default configuration and a specific tool signature.
    ///
    /// Convenience constructor when you have a custom signature but are
    /// happy with the default thresholds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::loop_control::loop_detector::LoopDetector;
    ///
    /// // With a custom signature, you would write:
    /// // let detector = LoopDetector::with_signature(Arc::new(MyToolSignature));
    /// ```
    ///
    pub fn with_signature(signature: Arc<dyn ToolSignature>) -> Self {
        Self::new(LoopDetectorConfig::default(), signature)
    }

    /// Get a reference to the configured tool signature.
    ///
    /// Returns a `&dyn ToolSignature` borrowed from the internal [`Arc`].
    /// Useful when external code needs to query the same tool-specific
    /// logic (e.g. to extract parameters for logging).
    ///
    pub fn signature(&self) -> &dyn ToolSignature {
        self.signature.as_ref()
    }

    /// Record a tool invocation from raw inputs (convenience method).
    ///
    /// Uses the configured [`ToolSignature`] to extract the primary
    /// parameter from `input`, constructs an [`Operation`], and delegates
    /// to [`record`](LoopDetector::record). This is a convenience wrapper
    /// around [`LoopDetector::record_from_input_with_error`] that passes `None` for the
    /// error parameter.
    ///
    /// Use [`LoopDetector::record_from_input_with_error`] instead when you have the tool
    /// error string available — it calls
    /// [`ToolSignature::is_recoverable_error`] for accurate recovery
    /// detection instead of relying on the heuristic in [`LoopDetector::record`].
    ///
    /// # Parameters
    ///
    /// - `tool` — Name of the tool that was invoked.
    /// - `input` — The JSON input that was passed to the tool.
    /// - `result_hash` — Optional hash of the tool result, typically
    ///   generated by [`hash_result`].
    ///
    pub fn record_from_input(
        &self,
        tool: &str,
        input: &serde_json::Value,
        result_hash: Option<u64>,
    ) {
        self.record_from_input_with_error(tool, input, result_hash, None);
    }

    /// Record a tool invocation with an optional error string.
    ///
    /// This is the primary entry point for the framework. It uses the
    /// configured [`ToolSignature`] to extract the primary parameter from
    /// `input`, constructs an [`Operation`], and records it.
    ///
    /// Unlike [`LoopDetector::record_from_input`], this method accepts an optional error
    /// string and calls [`ToolSignature::is_recoverable_error`] to accurately
    /// determine whether the failure is recoverable. When the error is
    /// recoverable (e.g. "old text not found" on an edit tool), warnings for
    /// prior edits to the same file are cleared because the agent is making
    /// progress.
    ///
    /// # Parameters
    ///
    /// - `tool` — Name of the tool that was invoked.
    /// - `input` — The JSON input that was passed to the tool.
    /// - `result_hash` — Optional hash of the tool result, typically
    ///   generated by [`hash_result`].
    /// - `error` — Optional error string returned by the tool. When
    ///   `Some`, this is passed to
    ///   [`ToolSignature::is_recoverable_error`] for accurate detection.
    ///   When `None`, falls back to the heuristic in [`LoopDetector::record`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use loopctl::loop_control::loop_detector::{
    ///     LoopDetector, LoopDetectorConfig, ToolSignature,
    /// };
    ///
    /// struct MySig;
    /// impl ToolSignature for MySig {
    ///     fn is_recoverable_error(&self, tool: &str, error: &str) -> bool {
    ///         tool == "Edit" && error.contains("old text not found")
    ///     }
    /// }
    ///
    /// let detector = LoopDetector::new(LoopDetectorConfig::default(), Arc::new(MySig));
    ///
    /// // Record a failed edit with the error string.
    /// detector.record_from_input_with_error(
    ///     "Edit",
    ///     &serde_json::json!({"file_path": "/src/main.rs"}),
    ///     None,
    ///     Some("Error: old text not found at line 5"),
    /// );
    /// ```
    ///
    pub fn record_from_input_with_error(
        &self,
        tool: &str,
        input: &serde_json::Value,
        result_hash: Option<u64>,
        error: Option<&str>,
    ) {
        let op = Operation::from_input_with_result_and_signature(
            tool,
            input,
            result_hash,
            self.signature.as_ref(),
        );

        // Determine recoverability accurately via the trait method.
        let is_recoverable =
            error.is_some_and(|err| self.signature.is_recoverable_error(tool, err));

        if is_recoverable {
            self.clear_warnings_for_recoverable_edit(&op);
        }

        self.record(op);
    }

    /// Record an operation into the sliding window.
    ///
    /// Appends `operation` to the internal [`VecDeque`], evicting the
    /// oldest entry if the window is full. Also increments the per-turn
    /// counter and performs a clean-up pass on the warned-operations
    /// set: if a previously warned operation now has a different
    /// `result_hash` (the agent made progress), the warning is cleared.
    ///
    /// # Recoverable Errors
    ///
    /// This method does **not** handle recoverable-error detection because
    /// it only receives an [`Operation`] (no error string). Recoverable-error
    /// clearing is done by [`LoopDetector::record_from_input_with_error`],
    /// which calls [`ToolSignature::is_recoverable_error`] with the actual
    /// error string.
    ///
    /// # When Called
    ///
    /// Called by the framework after every tool invocation, either
    /// directly, via [`LoopDetector::record_from_input`], or via
    /// [`LoopDetector::record_from_input_with_error`].
    ///
    pub fn record(&self, operation: Operation) {
        // Check if this operation was previously warned with a different result
        if let Ok(mut warned) = self.warned_operations.lock() {
            warned.retain(|warned_op| {
                !(warned_op.tool == operation.tool
                    && warned_op.primary_param == operation.primary_param
                    && warned_op.result_hash != operation.result_hash)
            });
        }

        if let Ok(mut ops) = self.operations.lock() {
            if ops.len() >= self.config.window_size {
                ops.pop_front();
            }
            ops.push_back(operation);
        }

        if let Ok(mut count) = self.turn_count.lock() {
            *count = count.saturating_add(1);
        }
    }

    /// Clear loop warnings for prior edit operations targeting the same file.
    ///
    /// Called when a recoverable error is detected by
    /// [`LoopDetector::record_from_input_with_error`]. Removes all prior
    /// warnings for edit operations whose primary parameter resolves to
    /// the same file path, and also removes those operations from the
    /// sliding window so they won't re-trigger a loop detection on the
    /// next [`LoopDetector::check_loop`] call.
    ///
    /// # File Path Matching
    ///
    /// Uses [`ToolSignature::normalize_param_for_comparison`] to strip
    /// tool-specific suffixes (e.g. `#line_number`) before comparing
    /// file paths.
    fn clear_warnings_for_recoverable_edit(&self, operation: &Operation) {
        let current_file = self
            .signature
            .normalize_param_for_comparison(&operation.tool, &operation.primary_param);
        let sig = &self.signature;

        // Remove warned operations for the same file.
        if let Ok(mut warned) = self.warned_operations.lock() {
            warned.retain(|warned_op| {
                if sig.is_file_edit_tool(&warned_op.tool) {
                    let warned_file = sig
                        .normalize_param_for_comparison(&warned_op.tool, &warned_op.primary_param);
                    warned_file != current_file
                } else {
                    true
                }
            });
        }

        // Remove the edit operations from the sliding window so they don't
        // re-trigger loop detection on the next check_loop() call.
        if let Ok(mut ops) = self.operations.lock() {
            ops.retain(|op| {
                if sig.is_file_edit_tool(&op.tool) {
                    let op_file = sig.normalize_param_for_comparison(&op.tool, &op.primary_param);
                    op_file != current_file
                } else {
                    true
                }
            });
        }
    }

    /// Analyse the operation window for loops and return a [`LoopStatus`].
    ///
    /// Scans the sliding window, counts occurrences of each distinct
    /// [`Operation`], and flags any operation whose count meets or exceeds
    /// the applicable threshold (from
    /// [`LoopDetectorConfig::threshold_for_tool`]). Only operations with
    /// the *maximum* repetition count are reported.
    ///
    /// **Result-aware:** two invocations with the same `(tool, primary_param)`
    /// but different [`result_hash`](Operation::result_hash) are treated
    /// as distinct operations — the agent is making progress, not looping.
    ///
    /// **Deduplication:** if a warning was already issued for the same
    /// operation (tracked in the warned-operations set) and
    /// [`should_stop`](LoopStatus::should_stop) is `false`, the
    /// [`warning`](LoopStatus::warning) field is suppressed to avoid
    /// spamming the agent.
    ///
    /// # Returns
    ///
    /// A [`LoopStatus`] describing the detected loop (or a default "no
    /// loop" status).
    ///
    /// # When Called
    ///
    /// Called by the framework after each tool invocation, typically
    /// immediately after [`record`](LoopDetector::record).
    ///
    pub fn check_loop(&self) -> LoopStatus {
        let Ok(ops) = self.operations.lock() else {
            return LoopStatus::default();
        };

        let mut repeated_operations = Vec::new();
        let mut max_repetitions = 0;
        let mut op_counts: HashMap<Operation, usize> = HashMap::new();
        for op in ops.iter() {
            op_counts
                .entry(op.clone())
                .and_modify(|c| *c = c.saturating_add(1))
                .or_insert(1);
        }

        for (op, count) in op_counts {
            let threshold = self.config.threshold_for_tool(&op.tool);
            if count >= threshold {
                if count > max_repetitions {
                    max_repetitions = count;
                    repeated_operations.clear();
                    repeated_operations.push(op);
                } else if count == max_repetitions {
                    repeated_operations.push(op);
                }
            }
        }

        let is_looping = !repeated_operations.is_empty();
        let should_stop =
            self.config.stop_threshold > 0 && max_repetitions >= self.config.stop_threshold;
        let warning = if is_looping {
            let already_warned = if let Some(first_op) = repeated_operations.first() {
                if let Ok(warned) = self.warned_operations.lock() {
                    warned.contains(first_op)
                } else {
                    false
                }
            } else {
                false
            };

            if already_warned && !should_stop {
                None
            } else {
                if let Some(first_op) = repeated_operations.first()
                    && let Ok(mut warned) = self.warned_operations.lock()
                {
                    warned.insert(first_op.clone());
                }

                let stop_msg = if should_stop {
                    " STOPPING to prevent infinite loop."
                } else {
                    ""
                };

                let tool_name = repeated_operations.first().map_or("", |o| o.tool.as_str());

                let suggestion = self
                    .signature
                    .get_suggestion(tool_name)
                    .unwrap_or_else(|| "Consider a different approach or tool.".to_string());

                Some(format!(
                    "Loop detected: Operation '{}' repeated {} times with same result.{} {}",
                    repeated_operations
                        .first()
                        .map(|o| format!("{}({})", o.tool, o.primary_param))
                        .unwrap_or_default(),
                    max_repetitions,
                    stop_msg,
                    suggestion
                ))
            }
        } else {
            None
        };

        LoopStatus {
            is_looping,
            repeated_operations,
            repetition_count: max_repetitions,
            warning,
            should_stop,
        }
    }

    /// Check whether the per-turn tool-call limit has been reached.
    ///
    /// Returns `true` when [`turn_count`](LoopDetector::turn_count) is
    /// greater than or equal to
    /// [`max_tools_per_turn`](LoopDetectorConfig::max_tools_per_turn).
    /// The framework uses this to decide whether to allow additional tool
    /// calls in the current turn.
    ///
    /// # When Called
    ///
    /// Called by the framework before dispatching each tool call within a
    /// turn. If `true`, the framework should stop the turn.
    ///
    pub fn check_turn_limit(&self) -> bool {
        match self.turn_count.lock() {
            Ok(count) => *count >= self.config.max_tools_per_turn,
            Err(_) => false,
        }
    }

    /// Get the current number of tool calls recorded in this turn.
    ///
    /// The count increments with each call to [`record`](LoopDetector::record)
    /// and resets to zero when [`reset_turn`](LoopDetector::reset_turn) is
    /// called.
    ///
    /// # Returns
    ///
    /// The turn-local call count, or `0` if the lock is poisoned.
    ///
    pub fn turn_count(&self) -> usize {
        self.turn_count.lock().map_or(0, |c| *c)
    }

    /// Reset the per-turn tool-call counter to zero.
    ///
    /// Called by the framework at the start of each new agent turn so the
    /// per-turn limit can be re-evaluated from scratch.
    ///
    /// # When Called
    ///
    /// At the beginning of every new turn, before any tool calls are
    /// dispatched.
    ///
    pub fn reset_turn(&self) {
        if let Ok(mut count) = self.turn_count.lock() {
            *count = 0;
        }
    }

    /// Check whether a specific file has been read too many times.
    ///
    /// Scans the operation window for read-type tools (as identified by
    /// [`ToolSignature::is_file_read_tool`]) whose
    /// [`primary_param`](Operation::primary_param) contains `file_path`.
    /// Returns `true` if the count meets or exceeds
    /// [`LoopDetectorConfig::max_same_file_reads`].
    ///
    /// # When Called
    ///
    /// Called by the framework before dispatching a file-read tool call,
    /// to guard against excessive re-reading of the same file.
    ///
    /// # Parameters
    ///
    /// - `file_path` — The file path (or a substring of it) to check.
    ///
    /// # Returns
    ///
    /// `true` if the file has been read ≥ `max_same_file_reads` times,
    /// `false` otherwise (including if the lock is poisoned).
    ///
    pub fn check_file_reads(&self, file_path: &str) -> bool {
        let Ok(ops) = self.operations.lock() else {
            return false;
        };

        let sig = &self.signature;
        let read_count = ops
            .iter()
            .filter(|o| sig.is_file_read_tool(&o.tool) && o.primary_param.contains(file_path))
            .count();

        read_count >= self.config.max_same_file_reads
    }

    /// Reset loop state when a file-read follows a failed edit to the same file.
    ///
    /// If `tool` is a read-type tool (per
    /// [`ToolSignature::is_file_read_tool`]) and the operation window
    /// contains a recent edit to `file_path` (per
    /// [`ToolSignature::is_file_edit_tool`]), this method removes all
    /// edit operations for that file from the window *and* from the
    /// warned-operations set. The rationale is that the agent is
    /// re-reading the file to get updated contents after a failed edit,
    /// which constitutes progress rather than a loop.
    ///
    /// # When Called
    ///
    /// Called by the framework when a read tool is dispatched after a
    /// failed edit, to prevent the edit-read cycle from being flagged as
    /// a loop.
    ///
    /// # Parameters
    ///
    /// - `tool` — Name of the tool being dispatched.
    /// - `file_path` — The file being read.
    ///
    pub fn check_and_reset_on_file_read(&self, tool: &str, file_path: &str) {
        if !self.signature.is_file_read_tool(tool) {
            return;
        }

        let sig = &self.signature;
        if let Ok(mut ops) = self.operations.lock() {
            let has_recent_failed_edit = ops.iter().any(|op| {
                sig.is_file_edit_tool(&op.tool)
                    && sig.normalize_param_for_comparison(&op.tool, &op.primary_param) == file_path
            });

            if has_recent_failed_edit {
                ops.retain(|op| {
                    let op_file = sig.normalize_param_for_comparison(&op.tool, &op.primary_param);
                    !(sig.is_file_edit_tool(&op.tool) && op_file == file_path)
                });
            }
        }

        if let Ok(mut warned) = self.warned_operations.lock() {
            let sig = &self.signature;
            warned.retain(|op| {
                let op_file = sig.normalize_param_for_comparison(&op.tool, &op.primary_param);
                !(sig.is_file_edit_tool(&op.tool) && op_file == file_path)
            });
        }
    }

    /// Clear all recorded operations and warned-operation state.
    ///
    /// Removes every entry from the sliding window and the warned-operations
    /// set, but does *not* reset the turn counter. Use
    /// [`reset`](LoopDetector::reset) for a full reset, or this method
    /// when you want to keep the turn count but start fresh on loop
    /// analysis.
    ///
    /// # When Called
    ///
    /// Called when the agent wants to clear loop history without resetting
    /// turn state — for example, after a successful corrective action.
    ///
    pub fn clear(&self) {
        if let Ok(mut ops) = self.operations.lock() {
            ops.clear();
        }
        if let Ok(mut warned) = self.warned_operations.lock() {
            warned.clear();
        }
    }

    /// Reset the detector to its initial state (full reset).
    ///
    /// Clears the operation window, zeroes the turn counter, and empties
    /// the warned-operations set. After calling this, the detector behaves
    /// as if freshly constructed (with the same config and signature).
    ///
    /// # When Called
    ///
    /// Called at task boundaries — when the agent finishes one task and
    /// starts another — so that loop state from the previous task doesn't
    /// bleed into the next one.
    ///
    pub fn reset(&self) {
        if let Ok(mut ops) = self.operations.lock() {
            ops.clear();
        }
        if let Ok(mut count) = self.turn_count.lock() {
            *count = 0;
        }
        if let Ok(mut warned) = self.warned_operations.lock() {
            warned.clear();
        }
    }
}

/// Produce a [`LoopDetector`] with default configuration and a no-op signature.
///
/// Delegates to [`LoopDetector::default_detector`]. This is the same as
/// calling `LoopDetector::default_detector()` and is provided for
/// ergonomic compatibility with generic code that uses `Default`.
///
/// The resulting detector uses [`LoopDetectorConfig::default`] thresholds
/// and a [`NoOpToolSignature`] — suitable for simple agents and tests but
/// not for production use where tool-specific parsing is needed.
///
/// # Example
///
/// ```rust
/// use loopctl::loop_control::loop_detector::LoopDetector;
///
/// let detector = LoopDetector::default();
/// let status = detector.check_loop();
/// assert!(!status.is_looping);
/// ```
///
/// # See Also
///
/// - [`LoopDetector::new`] — for custom configuration.
/// - [`LoopDetector::with_signature`] — for custom tool signatures.
/// - [`LoopDetector::default_detector`] — the method this delegates to.
///
impl Default for LoopDetector {
    /// Build a detector with [`LoopDetectorConfig::default`] and [`NoOpToolSignature`].
    ///
    /// Equivalent to `LoopDetector::default_detector()`. The internal state
    /// is empty (no operations recorded, zero turn count, no warnings).
    ///
    fn default() -> Self {
        Self::default_detector()
    }
}

/// Global loop detector instance, lazily initialised via [`std::sync::OnceLock`].
///
/// Provides a process-wide [`LoopDetector`] that can be accessed from
/// anywhere via [`global_detector`]. The detector is created exactly once
/// with default configuration ([`LoopDetectorConfig::default`]) and a
/// [`NoOpToolSignature`]. This is useful for simple agents that don't
/// need tool-specific loop detection logic.
///
/// For production use with custom tool signatures, prefer constructing a
/// dedicated [`LoopDetector`] via [`LoopDetector::new`] or
/// [`LoopDetector::with_signature`] instead of relying on this global.
///
/// # Thread Safety
///
/// [`OnceLock`] guarantees safe concurrent access from multiple threads
/// without additional synchronisation.
///
static GLOBAL_DETECTOR: std::sync::OnceLock<Arc<LoopDetector>> = std::sync::OnceLock::new();

/// Get a reference-counted handle to the global [`LoopDetector`] singleton.
///
/// Returns a cloned [`Arc`] pointing to the shared instance. On the first
/// call the detector is initialised with [`LoopDetector::default_detector`];
/// subsequent calls return the same instance (no re-initialisation).
///
/// # When Called
///
/// Called by the framework's main loop when it needs a detector but
/// doesn't have a task-specific one. Also useful in tests or simple
/// scripts that want loop detection without wiring up a detector manually.
///
/// # Example
///
/// ```rust
/// use loopctl::loop_control::loop_detector::{global_detector, Operation};
///
/// let detector = global_detector();
/// detector.record(Operation::new("Read", "/src/main.rs"));
/// let status = detector.check_loop();
/// assert!(!status.is_looping); // first read, no loop yet
/// ```
///
/// # See Also
///
/// - [`LoopDetector::new`] — for custom configuration.
/// - [`LoopDetector::with_signature`] — for custom tool signatures.
///
pub fn global_detector() -> Arc<LoopDetector> {
    Arc::clone(GLOBAL_DETECTOR.get_or_init(|| Arc::new(LoopDetector::default_detector())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A test tool signature that knows about "Read" and "Bash" tools.
    ///
    /// Implements [`ToolSignature`] for unit tests within this module. Extracts
    /// the `file_path` parameter for `Read` calls and the `command` parameter
    /// for `Bash` calls, falling back to an empty string for unknown tools.
    ///
    struct TestToolSignature;

    impl ToolSignature for TestToolSignature {
        fn extract_primary_param(&self, tool: &str, input: &serde_json::Value) -> String {
            match tool {
                "Read" => input
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                "Glob" => {
                    let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
                    let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
                    if path.is_empty() {
                        pattern.to_string()
                    } else {
                        format!("{}:{}", path, pattern)
                    }
                }
                "Bash" => input
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                _ => input.to_string(),
            }
        }

        fn get_suggestion(&self, tool: &str) -> Option<String> {
            match tool {
                "Bash" => Some("Check the command output or try a different approach.".to_string()),
                "Read" => Some("Try using Glob to find files first.".to_string()),
                _ => None,
            }
        }

        fn is_file_read_tool(&self, tool: &str) -> bool {
            matches!(tool, "Read" | "Bash")
        }

        fn is_file_edit_tool(&self, tool: &str) -> bool {
            tool == "Edit"
        }
    }

    fn test_detector() -> LoopDetector {
        LoopDetector::new(LoopDetectorConfig::default(), Arc::new(TestToolSignature))
    }

    #[test]
    fn test_operation_from_input() {
        let sig = TestToolSignature;
        let op = Operation::from_input_with_signature(
            "Read",
            &json!({"file_path": "/test/file.txt"}),
            &sig,
        );
        assert_eq!(op.tool, "Read");
        assert_eq!(op.primary_param, "/test/file.txt");
    }

    #[test]
    fn test_operation_from_input_glob() {
        let sig = TestToolSignature;
        let op = Operation::from_input_with_signature("Glob", &json!({"pattern": "*.rs"}), &sig);
        assert_eq!(op.tool, "Glob");
        assert_eq!(op.primary_param, "*.rs");

        let op = Operation::from_input_with_signature(
            "Glob",
            &json!({"pattern": "**/*.rs", "path": "/src"}),
            &sig,
        );
        assert_eq!(op.tool, "Glob");
        assert_eq!(op.primary_param, "/src:**/*.rs");
    }

    #[test]
    fn test_loop_detection() {
        let detector = test_detector();

        for _ in 0..5 {
            detector.record(Operation::new("Read", "/test/file.txt"));
        }

        let status = detector.check_loop();
        assert!(status.is_looping);
        assert!(status.repetition_count >= 3);
    }

    #[test]
    fn test_no_loop() {
        let detector = test_detector();

        detector.record(Operation::new("Read", "/file1.txt"));
        detector.record(Operation::new("Read", "/file2.txt"));
        detector.record(Operation::new("Bash", "ls -la"));

        let status = detector.check_loop();
        assert!(!status.is_looping);
    }

    #[test]
    fn test_turn_limit() {
        let config = LoopDetectorConfig {
            max_tools_per_turn: 5,
            ..Default::default()
        };
        let detector = LoopDetector::new(config, Arc::new(TestToolSignature));

        for i in 0..4 {
            detector.record(Operation::new("Read", format!("/file{i}.txt")));
            assert!(!detector.check_turn_limit());
        }

        detector.record(Operation::new("Read", "/file5.txt"));
        assert!(detector.check_turn_limit());

        detector.reset_turn();
        assert!(!detector.check_turn_limit());
    }

    #[test]
    fn test_file_read_limit() {
        let config = LoopDetectorConfig {
            max_same_file_reads: 3,
            ..Default::default()
        };
        let detector = LoopDetector::new(config, Arc::new(TestToolSignature));

        for _ in 0..2 {
            detector.record(Operation::new("Read", "/test.txt"));
            assert!(!detector.check_file_reads("/test.txt"));
        }

        detector.record(Operation::new("Read", "/test.txt"));
        assert!(detector.check_file_reads("/test.txt"));
    }

    #[test]
    fn test_should_stop_below_threshold() {
        let config = LoopDetectorConfig {
            stop_threshold: 10,
            repetition_threshold: 3,
            ..Default::default()
        };
        let detector = LoopDetector::new(config, Arc::new(TestToolSignature));

        for _ in 0..5 {
            detector.record(Operation::new("Bash", "git status"));
        }

        let status = detector.check_loop();
        assert!(status.is_looping);
        assert!(!status.should_stop);
    }

    #[test]
    fn test_should_stop_at_threshold() {
        let config = LoopDetectorConfig {
            stop_threshold: 10,
            repetition_threshold: 3,
            ..Default::default()
        };
        let detector = LoopDetector::new(config, Arc::new(TestToolSignature));

        for _ in 0..10 {
            detector.record(Operation::new("Bash", "git status"));
        }

        let status = detector.check_loop();
        assert!(status.is_looping);
        assert!(status.should_stop);
        assert!(status.warning.as_ref().unwrap().contains("STOPPING"));
    }

    #[test]
    fn test_should_stop_disabled() {
        let config = LoopDetectorConfig {
            stop_threshold: 0,
            repetition_threshold: 3,
            ..Default::default()
        };
        let detector = LoopDetector::new(config, Arc::new(TestToolSignature));

        for _ in 0..20 {
            detector.record(Operation::new("Bash", "git status"));
        }

        let status = detector.check_loop();
        assert!(status.is_looping);
        assert!(!status.should_stop);
    }

    #[test]
    fn test_different_operations_no_loop() {
        let detector = test_detector();

        for i in 0..20 {
            detector.record(Operation::new("Bash", format!("git status {}", i)));
        }

        let status = detector.check_loop();
        assert!(!status.is_looping);
        assert!(!status.should_stop);
    }

    #[test]
    fn test_loop_detection_with_same_results() {
        let detector = test_detector();

        let result_hash = hash_result("same output");
        for _ in 0..5 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(result_hash));
        }

        let status = detector.check_loop();
        assert!(status.is_looping);
        assert!(status.repetition_count >= 3);
        assert!(
            status
                .warning
                .as_ref()
                .unwrap()
                .contains("with same result")
        );
    }

    #[test]
    fn test_loop_detection_with_different_results() {
        let detector = test_detector();

        for i in 0..5 {
            let result_hash = hash_result(&format!("output {}", i));
            detector.record(Operation::new("Bash", "git status").with_result_hash(result_hash));
        }

        let status = detector.check_loop();
        assert!(!status.is_looping);
    }

    #[test]
    fn test_record_from_input_with_recoverable_error_clears_warnings() {
        use std::sync::Arc;

        // Signature that recognises "old text not found" as recoverable for Edit.
        struct RecoverableEditSig;
        impl ToolSignature for RecoverableEditSig {
            fn extract_primary_param(&self, tool: &str, input: &serde_json::Value) -> String {
                if tool == "Edit" {
                    input
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                } else {
                    String::new()
                }
            }

            fn is_recoverable_error(&self, tool: &str, error: &str) -> bool {
                tool == "Edit" && error.contains("old text not found")
            }

            fn is_file_edit_tool(&self, tool: &str) -> bool {
                tool == "Edit"
            }
        }

        let detector =
            LoopDetector::new(LoopDetectorConfig::default(), Arc::new(RecoverableEditSig));

        // Record enough Edit operations to trigger a loop warning.
        let hash = hash_result("same output");
        for _ in 0..4 {
            detector.record(Operation::new("Edit", "/src/main.rs").with_result_hash(hash));
        }

        let status = detector.check_loop();
        assert!(status.is_looping, "should detect loop after repeated edits");

        // Now record a failed edit via record_from_input_with_error with a
        // recoverable error string. This should clear the warning state.
        detector.record_from_input_with_error(
            "Edit",
            &serde_json::json!({"file_path": "/src/main.rs"}),
            None,
            Some("Error: old text not found at line 5"),
        );

        // The warning should have been cleared — check_loop should report
        // no warning for this operation now (it was removed from warned set).
        let status2 = detector.check_loop();
        assert!(
            status2.warning.is_none(),
            "warning should be cleared after recoverable error, got: {:?}",
            status2.warning
        );
    }

    #[test]
    fn test_record_from_input_with_non_recoverable_error_keeps_warnings() {
        use std::sync::Arc;

        struct PartialSig;
        impl ToolSignature for PartialSig {
            fn is_recoverable_error(&self, _tool: &str, _error: &str) -> bool {
                false // nothing is recoverable
            }
        }

        let detector = LoopDetector::new(LoopDetectorConfig::default(), Arc::new(PartialSig));

        // Record enough operations to trigger a warning.
        let hash = hash_result("same output");
        for _ in 0..4 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(hash));
        }

        let status = detector.check_loop();
        assert!(status.is_looping);

        // Record with an error that is NOT recoverable.
        detector.record_from_input_with_error(
            "Bash",
            &serde_json::json!({"command": "git status"}),
            None,
            Some("permission denied"),
        );

        // Warning should still be present — non-recoverable error doesn't clear it.
        let status2 = detector.check_loop();
        assert!(
            status2.is_looping,
            "loop should still be detected after non-recoverable error"
        );
    }

    #[test]
    fn test_warning_not_repeated_for_same_operation() {
        let detector = test_detector();

        let result_hash = hash_result("same output");
        for _ in 0..5 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(result_hash));
        }

        let status1 = detector.check_loop();
        assert!(status1.is_looping);
        assert!(status1.warning.is_some());

        let status2 = detector.check_loop();
        assert!(status2.is_looping);
        assert!(status2.warning.is_none());
    }

    #[test]
    fn test_reset_clears_all_state() {
        let detector = test_detector();

        let hash1 = hash_result("same output");
        for _ in 0..5 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(hash1));
        }

        let status1 = detector.check_loop();
        assert!(status1.is_looping);
        assert!(status1.repetition_count >= 5);
        assert!(status1.warning.is_some());
        assert!(detector.turn_count() > 0);

        detector.reset();

        assert_eq!(detector.turn_count(), 0);
        let status2 = detector.check_loop();
        assert!(!status2.is_looping);
        assert!(status2.warning.is_none());
    }

    #[test]
    fn test_operations_with_none_result_hash() {
        let detector = test_detector();

        for _ in 0..5 {
            detector.record(Operation::new("Bash", "git status"));
        }

        let status = detector.check_loop();
        assert!(status.is_looping);
    }

    #[test]
    fn test_hash_result_empty_string() {
        assert_eq!(hash_result(""), None);
    }

    #[test]
    fn test_hash_result_non_empty() {
        let hash1 = hash_result("some content");
        let hash2 = hash_result("some content");
        let hash3 = hash_result("different content");

        assert!(hash1.is_some());
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_noop_tool_signature() {
        let sig = NoOpToolSignature;
        assert_eq!(
            sig.extract_primary_param("Read", &json!({"file_path": "/test"})),
            ""
        );
        assert!(!sig.is_recoverable_error("Edit", "old text not found"));
        assert!(sig.get_suggestion("Edit").is_none());
    }

    #[test]
    fn test_record_from_input() {
        let detector = test_detector();
        detector.record_from_input("Read", &json!({"file_path": "/test.txt"}), None);
        assert_eq!(detector.turn_count(), 1);
    }

    #[test]
    fn test_tool_specific_threshold_in_config() {
        let mut config = LoopDetectorConfig::default();

        // Default config has no tool-specific thresholds
        assert_eq!(config.threshold_for_tool("Edit"), 3);
        assert_eq!(config.threshold_for_tool("Bash"), 3);

        // Add tool-specific thresholds
        config.tool_thresholds.insert("Edit".to_string(), 2);
        assert_eq!(config.threshold_for_tool("Edit"), 2);
        assert_eq!(config.threshold_for_tool("Bash"), 3);
    }

    #[test]
    fn test_detector_with_custom_signature() {
        let detector = LoopDetector::with_signature(Arc::new(TestToolSignature));

        detector.record(Operation::new("Read", "/test.txt"));
        detector.record(Operation::new("Read", "/test.txt"));
        detector.record(Operation::new("Read", "/test.txt"));

        let status = detector.check_loop();
        assert!(status.is_looping);
    }

    #[test]
    fn test_warning_shown_again_after_clear() {
        let detector = test_detector();

        let hash1 = hash_result("output 1");
        for _ in 0..3 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(hash1));
        }

        let status1 = detector.check_loop();
        assert!(status1.warning.is_some());

        detector.clear();

        for _ in 0..3 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(hash1));
        }

        let status2 = detector.check_loop();
        assert!(status2.warning.is_some());
    }

    #[test]
    fn test_warning_cleared_when_result_changes() {
        let detector = test_detector();

        let hash1 = hash_result("same output");
        for _ in 0..3 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(hash1));
        }

        let status1 = detector.check_loop();
        assert!(status1.warning.is_some());

        let hash2 = hash_result("different output - progress!");
        detector.record(Operation::new("Bash", "git status").with_result_hash(hash2));

        for _ in 0..3 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(hash1));
        }

        let status2 = detector.check_loop();
        assert!(status2.warning.is_some());
    }

    #[test]
    fn test_result_hash_differentiates_operations() {
        let detector = test_detector();

        for i in 0..5 {
            let hash = hash_result(&format!("output {}", i));
            detector.record(Operation::new("Bash", "git status").with_result_hash(hash));
        }

        let status = detector.check_loop();
        assert!(!status.is_looping, "Different results should not be a loop");
    }

    #[test]
    fn test_same_operation_same_result_is_loop() {
        let detector = test_detector();

        let hash = hash_result("same output every time");
        for _ in 0..5 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(hash));
        }

        let status = detector.check_loop();
        assert!(
            status.is_looping,
            "Same command with same result should be a loop"
        );
    }

    #[test]
    fn test_reset_allows_new_warnings_for_same_operations() {
        let detector = test_detector();

        let hash1 = hash_result("same output");
        for _ in 0..3 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(hash1));
        }
        let status1 = detector.check_loop();
        assert!(status1.warning.is_some());

        let status2 = detector.check_loop();
        assert!(status2.warning.is_none());

        detector.reset();

        for _ in 0..3 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(hash1));
        }

        let status3 = detector.check_loop();
        assert!(status3.warning.is_some());
    }

    #[test]
    fn test_mixed_same_and_different_results() {
        let detector = test_detector();

        let hash1 = hash_result("output 1");
        let hash2 = hash_result("output 2");
        detector.record(Operation::new("Bash", "git status").with_result_hash(hash1));
        detector.record(Operation::new("Bash", "git status").with_result_hash(hash2));

        let hash3 = hash_result("same output");
        detector.record(Operation::new("Bash", "git status").with_result_hash(hash3));
        detector.record(Operation::new("Bash", "git status").with_result_hash(hash3));
        detector.record(Operation::new("Bash", "git status").with_result_hash(hash3));

        let status = detector.check_loop();
        assert!(status.is_looping);
        assert_eq!(status.repetition_count, 3);
    }

    #[test]
    fn test_suggestion_from_signature() {
        let detector = test_detector();

        for _ in 0..3 {
            detector.record(Operation::new("Bash", "git status"));
        }

        let status = detector.check_loop();
        assert!(status.warning.is_some());
        let warning = status.warning.unwrap();
        assert!(
            warning.contains("Check the command"),
            "Bash warning should contain signature suggestion: {}",
            warning
        );
    }
}

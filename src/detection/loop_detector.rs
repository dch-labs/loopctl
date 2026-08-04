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
//! external timeout kicks in. An early-warning system
//! flags loops early and can force-stop the
//! agent when the repetition count exceeds a threshold.
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
//! use loopctl::detection::loop_detector::{
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
//! # Edit-Recovery Workflow
//!
//! The detector includes special handling for the edit-recovery pattern.
//! When an edit tool fails (recoverable error) and the agent re-reads the
//! file to get updated contents, the loop warning for that file is cleared
//! because the agent is making progress. This prevents false positives
//! during normal edit-retry cycles.

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
/// collection of tools — swap the signature implementation as needed.
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
/// use loopctl::detection::loop_detector::ToolSignature;
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
    /// Overriding this method correctly is critical.
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
    /// use loopctl::detection::loop_detector::ToolSignature;
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
    /// use loopctl::detection::loop_detector::ToolSignature;
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
    /// Called during [`LoopDetector::check_file_reads`].
    ///
    /// # Default
    ///
    /// Returns `false`.
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
    /// Called during [`LoopDetector::record`] to detect recoverable edits.
    ///
    /// # Default
    ///
    /// Returns `false`.
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
    /// operations to the same file.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::detection::loop_detector::ToolSignature;
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
/// detector can still catch loops — it cannot distinguish between
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
/// use loopctl::detection::loop_detector::{
///     LoopDetector, LoopDetectorConfig, NoOpToolSignature,
/// };
///
/// let detector = LoopDetector::new(
///     LoopDetectorConfig::default(),
///     Arc::new(NoOpToolSignature),
/// );
/// ```
pub struct NoOpToolSignature;

/// Blanket [`ToolSignature`] implementation for [`NoOpToolSignature`].
///
/// Inherits every default method — no overrides. All tools are treated as
/// opaque operations with no primary parameter, no recoverable errors, and
/// no suggestions. Because [`ToolSignature::extract_primary_param`] always
/// returns `""`, every invocation of the same tool with the same result hash
/// is considered identical for loop-detection purposes.
///
/// Empty implementation — relies entirely on the
/// [`ToolSignature`] trait defaults. See the
/// trait-level documentation for the semantics of each default.
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
/// use loopctl::detection::loop_detector::LoopDetectorConfig;
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
#[derive(Debug, Clone)]
pub struct LoopDetectorConfig {
    /// Number of recent operations kept in the sliding window.
    ///
    /// **Default:** `50`.
    pub window_size: usize,

    /// Identical repetitions required to flag a loop. Overridden per tool by `tool_thresholds`.
    ///
    /// **Default:** `3`.
    pub repetition_threshold: usize,

    /// Max tool calls per turn. Checked by [`check_turn_limit`](LoopDetector::check_turn_limit).
    ///
    /// **Default:** `9999` (effectively unlimited).
    pub max_tools_per_turn: usize,

    /// Max identical file reads before a warning. Checked by [`check_file_reads`](LoopDetector::check_file_reads).
    ///
    /// **Default:** `5`.
    pub max_same_file_reads: usize,

    /// Repetitions required to force-stop the agent. Set to `0` to disable forced stops.
    ///
    /// **Default:** `10`.
    pub stop_threshold: usize,

    /// Per-tool repetition thresholds. Looked up by [`threshold_for_tool`](LoopDetectorConfig::threshold_for_tool).
    ///
    /// **Default:** empty `HashMap`.
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
/// use loopctl::detection::loop_detector::LoopDetectorConfig;
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
impl Default for LoopDetectorConfig {
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
    /// use loopctl::detection::loop_detector::LoopDetectorConfig;
    ///
    /// let mut config = LoopDetectorConfig::default();
    /// config.tool_thresholds.insert("Edit".to_string(), 2);
    ///
    /// assert_eq!(config.threshold_for_tool("Edit"), 2);   // override
    /// assert_eq!(config.threshold_for_tool("Read"), 3);   // generic default
    /// ```
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
/// [`Operation`] derives [`Eq`] and [`Hash`], so two operations are
/// equal only when all three fields match exactly. This means the *same*
/// command producing *different* results is treated as two distinct
/// operations — a key design choice that prevents false positives when
/// the agent is making progress.
///
/// # Construction
///
/// ```rust
/// use loopctl::detection::loop_detector::{Operation, hash_result};
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Operation {
    /// Tool name (e.g. `"Read"`, `"Edit"`, `"Bash"`).
    ///
    /// The identifier the tool was registered under, used as the first
    /// component of the `(tool, primary_param, result_hash)` loop signature.
    pub tool: String,

    /// Operation target (file path, command, etc.). Extracted by [`ToolSignature::extract_primary_param`].
    ///
    /// A string that uniquely identifies what the operation acted on —
    /// typically a file path or shell command — so two calls to the same tool
    /// on different targets are treated as distinct operations.
    pub primary_param: String,

    /// Hash of the tool result. `None` means results are ignored. Generated by [`hash_result`].
    ///
    /// Compact digest of the tool's output. Two operations with identical
    /// `tool` and `primary_param` but different result hashes are treated as
    /// distinct (the agent is making progress); `None` disables that
    /// result-aware comparison.
    pub result_hash: Option<u64>,
}

/// Constructors and builder methods for [`Operation`].
///
/// Operations can be created in several ways depending on what information
/// is available:
///
/// - **[`Operation::new`]** — Simplest path: tool name + primary param.
/// - **[`Operation::from_input_with_signature`]** — Parses the primary
///   parameter from raw JSON using a [`ToolSignature`].
/// - **[`Operation::from_input_with_result_and_signature`]** — Full
///   construction with result hash, used after a tool invocation completes.
/// - **[`Operation::with_result_hash`]** — Builder-style attachment of a
///   result hash to an existing operation.
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
    /// use loopctl::detection::loop_detector::Operation;
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
    /// use loopctl::detection::loop_detector::{Operation, ToolSignature};
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
    /// result hash into a single constructor call. Most complete
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
    /// use loopctl::detection::loop_detector::{Operation, ToolSignature, hash_result};
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
    /// use loopctl::detection::loop_detector::{Operation, hash_result};
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
/// An empty result usually means the tool produced
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
/// The cost is O(n) in the length of `content`.
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
/// use loopctl::detection::loop_detector::hash_result;
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
/// use loopctl::detection::loop_detector::LoopDetector;
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
#[derive(Debug, Clone, Default)]
pub struct LoopStatus {
    /// `true` when an operation repeats beyond the configured threshold.
    ///
    /// The headline flag: set when at least one operation's occurrence count
    /// reaches the applicable (generic or per-tool) threshold within the
    /// sliding window, signalling a probable repetitive loop.
    pub is_looping: bool,

    /// Operations tied for highest repetition. Empty when `is_looping` is `false`.
    ///
    /// Every operation sharing the maximum repetition count, so callers can
    /// see the full set of culprits when several operations loop in lockstep.
    pub repeated_operations: Vec<Operation>,

    /// Repetitions of the most-repeated operation. Zero when no loop detected.
    ///
    /// The count attained by the entries in
    /// [`repeated_operations`](Self::repeated_operations); compared against
    /// [`LoopDetectorConfig::stop_threshold`] to decide on a forced stop.
    pub repetition_count: usize,

    /// Loop description with count and suggestion. `None` when not looping or already warned.
    ///
    /// Human-readable message naming the looping operation, its repetition
    /// count, and any tool-specific suggestion. Suppressed (set to `None`)
    /// once a warning has been emitted for the same operation, to avoid
    /// flooding the agent's context with duplicates.
    pub warning: Option<String>,

    /// `true` when `repetition_count >= stop_threshold` (non-zero).
    ///
    /// Escalation flag indicating the loop has persisted long enough that the
    /// caller should halt the agent rather than merely warn. Inert when the
    /// configured stop threshold is `0` (forced stops disabled).
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
/// use loopctl::detection::loop_detector::{
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
/// // let detector = LoopDetector::new_with_signature(Arc::new(MyToolSignature));
/// ```
///
/// # Thread Safety
///
/// All mutable state is wrapped in [`Mutex`]. Methods lock each field
/// independently and handle lock poisoning gracefully (by skipping the
/// operation rather than panicking). This means the detector degrades
/// gracefully under contention but never blocks the agent loop.
pub struct LoopDetector {
    /// Sliding window of recent [`Operation`] records.
    ///
    /// Bounded to [`LoopDetectorConfig::window_size`] entries, ordered
    /// oldest to newest. Each recorded tool invocation pushes onto the
    /// back and evicts from the front once the window is full; loop
    /// detection scans this window to count repetitions of the same
    /// operation. Guarded by a [`Mutex`] so the window can be updated
    /// from the agent loop while being read by an observer thread.
    operations: Mutex<VecDeque<Operation>>,

    /// Detector configuration: thresholds, window size, and limits.
    ///
    /// Held immutably for the lifetime of the detector (it is set in
    /// the constructor and never mutated). Storing it by value keeps
    /// threshold checks a single field access with no indirection.
    config: LoopDetectorConfig,

    /// Number of tool calls recorded in the current turn.
    ///
    /// Incremented by every
    /// [`record`](LoopDetector::record) call and reset to zero by
    /// [`reset_turn`](LoopDetector::reset_turn). Drives the per-turn
    /// tool-call budget check so a runaway turn can be flagged before
    /// the sliding window fills. Guarded by a [`Mutex`].
    turn_count: Mutex<usize>,

    /// Operations that have already triggered a warning.
    ///
    /// Each entry is an [`Operation`] that crossed the warning
    /// threshold, recorded so the detector does not re-warn on the
    /// same pattern. Entries are removed when an operation's result
    /// changes (progress observed) and the whole set is cleared by
    /// [`reset`](LoopDetector::reset). Guarded by a [`Mutex`].
    warned_operations: Mutex<std::collections::HashSet<Operation>>,

    /// Tool signature extractor used to derive each tool's primary parameter.
    ///
    /// A trait object held behind [`Arc`] so it can be shared cheaply
    /// across clones of the detector. The signature decides which
    /// argument identifies an operation as "the same" (e.g. a file
    /// path for `Read`, a command for `Bash`), making repetition
    /// detection tool-aware rather than string-comparing raw JSON.
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
impl LoopDetector {
    /// Create a new loop detector with the given configuration and tool signature.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use loopctl::detection::loop_detector::{
    ///     LoopDetector, LoopDetectorConfig, NoOpToolSignature,
    /// };
    ///
    /// let config = LoopDetectorConfig { window_size: 100, ..Default::default() };
    /// let detector = LoopDetector::new(config, Arc::new(NoOpToolSignature));
    /// ```
    pub fn new(mut config: LoopDetectorConfig, signature: Arc<dyn ToolSignature>) -> Self {
        if config.window_size == 0 {
            tracing::warn!(
                "LoopDetectorConfig.window_size was 0; clamping to 1 (the smallest sensible window)"
            );
            config.window_size = 1;
        }
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
    /// Equivalent to:
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use loopctl::detection::loop_detector::{LoopDetector, LoopDetectorConfig, NoOpToolSignature};
    ///
    /// LoopDetector::new(LoopDetectorConfig::default(), Arc::new(NoOpToolSignature));
    /// ```
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
    /// use loopctl::detection::loop_detector::LoopDetector;
    ///
    /// // With a custom signature, you would write:
    /// // let detector = LoopDetector::new_with_signature(Arc::new(MyToolSignature));
    /// ```
    pub fn new_with_signature(signature: Arc<dyn ToolSignature>) -> Self {
        Self::new(LoopDetectorConfig::default(), signature)
    }

    /// Get a reference to the configured tool signature.
    ///
    /// Returns a `&dyn ToolSignature` borrowed from the internal [`Arc`].
    /// Useful when external code needs to query the same tool-specific
    /// logic (e.g. to extract parameters for logging).
    pub fn signature(&self) -> &dyn ToolSignature {
        self.signature.as_ref()
    }

    /// Record a tool invocation from raw inputs (convenience method).
    ///
    /// Uses the configured [`ToolSignature`] to extract the primary
    /// parameter from `input`, constructs an [`Operation`], and delegates
    /// to [`record`](LoopDetector::record). Convenience wrapper
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
    /// Primary entry point for the framework. Uses the
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
    /// use loopctl::detection::loop_detector::{
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
    pub fn record(&self, operation: Operation) {
        let should_clear_history = {
            let mut clear = false;
            if let Ok(mut warned) = self.warned_operations.lock() {
                warned.retain(|warned_op| {
                    let matches = warned_op.tool == operation.tool
                        && warned_op.primary_param == operation.primary_param
                        && warned_op.result_hash != operation.result_hash;
                    if matches {
                        clear = true;
                    }
                    !matches
                });
            }
            clear
        };

        // Remove stale operations from the sliding window so they don't
        // re-trigger loop detection on the next check_loop() call.
        if should_clear_history && let Ok(mut ops) = self.operations.lock() {
            ops.retain(|op| {
                !(op.tool == operation.tool
                    && op.primary_param == operation.primary_param
                    && op.result_hash != operation.result_hash)
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
    pub fn check_loop(&self) -> LoopStatus {
        let Ok(ops) = self.operations.lock() else {
            return LoopStatus::default();
        };

        let (repeated_operations, max_repetitions) =
            Self::find_repeated(&ops, |tool| self.config.threshold_for_tool(tool));
        let is_looping = !repeated_operations.is_empty();
        let should_stop =
            self.config.stop_threshold > 0 && max_repetitions >= self.config.stop_threshold;
        let warning = self.build_warning(
            &repeated_operations,
            max_repetitions,
            is_looping,
            should_stop,
        );

        LoopStatus {
            is_looping,
            repeated_operations,
            repetition_count: max_repetitions,
            warning,
            should_stop,
        }
    }

    /// Count occurrences of each operation in the deque.
    ///
    /// Iterates over every operation in `ops` and builds a [`HashMap`] where
    /// each key is a cloned [`Operation`] and the value is the number of
    /// times it appears. Counts are capped at `usize::MAX` via saturating
    /// addition to avoid overflow on extremely long deques.
    fn count_operations(ops: &VecDeque<Operation>) -> HashMap<Operation, usize> {
        let mut counts: HashMap<Operation, usize> = HashMap::new();
        for op in ops {
            counts
                .entry(op.clone())
                .and_modify(|c| *c = c.saturating_add(1))
                .or_insert(1);
        }
        counts
    }

    /// Find operations exceeding their per-tool threshold.
    ///
    /// Only the operations with the highest repetition count are
    /// returned (ties included). The `threshold` closure maps a tool
    /// name to its configured threshold.
    fn find_repeated(
        ops: &VecDeque<Operation>,
        threshold: impl Fn(&str) -> usize,
    ) -> (Vec<Operation>, usize) {
        let counts = Self::count_operations(ops);
        let mut repeated = Vec::new();
        let mut max = 0;

        for (op, count) in counts {
            let t = threshold(&op.tool);
            if count >= t {
                if count > max {
                    max = count;
                    repeated.clear();
                    repeated.push(op);
                } else if count == max {
                    repeated.push(op);
                }
            }
        }

        // Sort by tool then primary_param for deterministic ordering
        // when multiple operations share the same repetition count.
        repeated.sort_by(|a, b| {
            a.tool
                .cmp(&b.tool)
                .then_with(|| a.primary_param.cmp(&b.primary_param))
        });

        (repeated, max)
    }

    /// Build the warning string for repeated operations.
    ///
    /// Returns `None` when not looping, or when already warned and
    /// not stopping.  When a new warning is produced, the first
    /// repeated operation is recorded in `warned_operations`.
    fn build_warning(
        &self,
        repeated_operations: &[Operation],
        max_repetitions: usize,
        is_looping: bool,
        should_stop: bool,
    ) -> Option<String> {
        if !is_looping {
            return None;
        }

        let first_op = repeated_operations.first()?;
        let already_warned = self
            .warned_operations
            .lock()
            .is_ok_and(|w| w.contains(first_op));

        if already_warned && !should_stop {
            return None;
        }

        if !already_warned && let Ok(mut warned) = self.warned_operations.lock() {
            warned.insert(first_op.clone());
        }

        let stop_msg = if should_stop {
            " STOPPING to prevent infinite loop."
        } else {
            ""
        };

        let suggestion = self
            .signature
            .get_suggestion(&first_op.tool)
            .unwrap_or_else(|| "Consider a different approach or tool.".to_string());

        Some(format!(
            "Loop detected: Operation '{}({})' repeated {} times with same result.{} {}",
            first_op.tool, first_op.primary_param, max_repetitions, stop_msg, suggestion
        ))
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
    /// # Returns
    ///
    /// `true` if the file has been read ≥ `max_same_file_reads` times,
    /// `false` otherwise (including if the lock is poisoned).
    pub fn check_file_reads(&self, file_path: &str) -> bool {
        let Ok(ops) = self.operations.lock() else {
            return false;
        };

        let sig = &self.signature;
        let read_count = ops
            .iter()
            .filter(|o| {
                if !sig.is_file_read_tool(&o.tool) {
                    return false;
                }
                let normalized_op = sig.normalize_param_for_comparison(&o.tool, &o.primary_param);
                let normalized_query = sig.normalize_param_for_comparison(&o.tool, file_path);
                normalized_op.contains(normalized_query.as_str())
                    || normalized_query.contains(normalized_op.as_str())
            })
            .count();

        read_count >= self.config.max_same_file_reads
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
/// Delegates to [`LoopDetector::default_detector`]. Same as
/// calling `LoopDetector::default_detector()` and provided for
/// ergonomic compatibility with generic code that uses `Default`.
///
/// The resulting detector uses [`LoopDetectorConfig::default`] thresholds
/// and a [`NoOpToolSignature`] — suitable for simple agents and tests but
/// not for production use where tool-specific parsing is needed.
///
/// # Example
///
/// ```rust
/// use loopctl::detection::loop_detector::LoopDetector;
///
/// let detector = LoopDetector::default();
/// let status = detector.check_loop();
/// assert!(!status.is_looping);
/// ```
///
/// # See Also
///
/// - [`LoopDetector::new`] — for custom configuration.
/// - [`LoopDetector::new_with_signature`] — for custom tool signatures.
/// - [`LoopDetector::default_detector`] — the method this delegates to.
impl Default for LoopDetector {
    fn default() -> Self {
        Self::default_detector()
    }
}

/// Global loop detector instance, lazily initialised via [`std::sync::OnceLock`].
///
/// Provides a process-wide [`LoopDetector`] that can be accessed from
/// anywhere via [`global_detector`]. The detector is created exactly once
/// with default configuration ([`LoopDetectorConfig::default`]) and a
/// [`NoOpToolSignature`]. Useful for simple agents that don't
/// need tool-specific loop detection logic.
///
/// For production use with custom tool signatures, prefer constructing a
/// dedicated [`LoopDetector`] via [`LoopDetector::new`] or
/// [`LoopDetector::new_with_signature`] instead of relying on this global.
///
/// # Thread Safety
///
/// [`OnceLock`] guarantees safe concurrent access from multiple threads
/// without additional synchronisation.
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
/// use loopctl::detection::loop_detector::{global_detector, Operation};
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
/// - [`LoopDetector::new_with_signature`] — for custom tool signatures.
pub fn global_detector() -> Arc<LoopDetector> {
    Arc::clone(GLOBAL_DETECTOR.get_or_init(|| Arc::new(LoopDetector::default_detector())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
                        format!("{path}:{pattern}")
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
            detector.record(Operation::new("Bash", format!("git status {i}")));
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
            let result_hash = hash_result(&format!("output {i}"));
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
        let detector = LoopDetector::new_with_signature(Arc::new(TestToolSignature));

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

        // After recording with a different hash, both the warned set and
        // the sliding window are cleared, so the warning should be gone.
        let status_cleared = detector.check_loop();
        assert!(
            status_cleared.warning.is_none(),
            "warning should be cleared when result hash changes"
        );

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
            let hash = hash_result(&format!("output {i}"));
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
    fn test_warning_not_cleared_when_result_stays_same() {
        let detector = test_detector();

        let hash1 = hash_result("same output");
        for _ in 0..3 {
            detector.record(Operation::new("Bash", "git status").with_result_hash(hash1));
        }

        let status1 = detector.check_loop();
        assert!(status1.warning.is_some());

        // Recording again with the SAME hash should NOT clear the history.
        // check_loop() suppresses already-warned ops (returns None), but the
        // loop is still detected (is_looping = true) and the history is intact.
        detector.record(Operation::new("Bash", "git status").with_result_hash(hash1));

        let status2 = detector.check_loop();
        assert!(
            status2.is_looping,
            "loop should still be detected when result hash stays the same"
        );
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
            "Bash warning should contain signature suggestion: {warning}",
        );
    }

    fn make_ops(pairs: &[(&str, &str)]) -> VecDeque<Operation> {
        pairs
            .iter()
            .map(|&(tool, param)| Operation::new(tool, param))
            .collect()
    }

    fn make_ops_hashed(pairs: &[(&str, &str, &str)]) -> VecDeque<Operation> {
        pairs
            .iter()
            .map(|&(tool, param, result)| {
                Operation::new(tool, param).with_result_hash(hash_result(result))
            })
            .collect()
    }

    #[test]
    fn test_count_operations_empty() {
        let ops: VecDeque<Operation> = VecDeque::new();
        let counts = LoopDetector::count_operations(&ops);
        assert!(counts.is_empty());
    }

    #[test]
    fn test_count_operations_single_op() {
        let ops = make_ops(&[("Bash", "ls"), ("Bash", "ls"), ("Bash", "ls")]);
        let counts = LoopDetector::count_operations(&ops);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts.get(&Operation::new("Bash", "ls")), Some(&3));
    }

    #[test]
    fn test_count_operations_distinct_ops() {
        let ops = make_ops(&[("Bash", "ls"), ("Read", "file.txt"), ("Bash", "ls")]);
        let counts = LoopDetector::count_operations(&ops);
        assert_eq!(counts.len(), 2);
        assert_eq!(counts.get(&Operation::new("Bash", "ls")), Some(&2));
        assert_eq!(counts.get(&Operation::new("Read", "file.txt")), Some(&1));
    }

    #[test]
    fn test_count_operations_different_hashes_are_distinct() {
        let ops = make_ops_hashed(&[("Bash", "ls", "output_a"), ("Bash", "ls", "output_b")]);
        let counts = LoopDetector::count_operations(&ops);
        // Different result hashes → different operations
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn test_count_operations_same_hashes_are_grouped() {
        let ops = make_ops_hashed(&[
            ("Bash", "ls", "same_output"),
            ("Bash", "ls", "same_output"),
            ("Bash", "ls", "same_output"),
        ]);
        let counts = LoopDetector::count_operations(&ops);
        assert_eq!(counts.len(), 1);
        let key = Operation::new("Bash", "ls").with_result_hash(hash_result("same_output"));
        assert_eq!(counts.get(&key), Some(&3));
    }

    #[test]
    fn test_find_repeated_none() {
        let ops = make_ops(&[("Bash", "ls"), ("Read", "f.txt")]);
        let (repeated, max) = LoopDetector::find_repeated(&ops, |_tool| 3);
        assert!(repeated.is_empty());
        assert_eq!(max, 0);
    }

    #[test]
    fn test_find_repeated_single_above_threshold() {
        let ops = make_ops(&[("Bash", "ls"); 5]);
        let (repeated, max) = LoopDetector::find_repeated(&ops, |_tool| 3);
        assert_eq!(max, 5);
        assert_eq!(repeated.len(), 1);
        assert_eq!(repeated[0], Operation::new("Bash", "ls"));
    }

    #[test]
    fn test_find_repeated_tie_keeps_both() {
        let mut ops = VecDeque::new();
        for _ in 0..3 {
            ops.push_back(Operation::new("Bash", "ls"));
        }
        for _ in 0..3 {
            ops.push_back(Operation::new("Read", "f.txt"));
        }
        let (repeated, max) = LoopDetector::find_repeated(&ops, |_tool| 3);
        assert_eq!(max, 3);
        assert_eq!(repeated.len(), 2);
    }

    #[test]
    fn test_find_repeated_per_tool_threshold() {
        let mut ops = VecDeque::new();
        for _ in 0..2 {
            ops.push_back(Operation::new("Bash", "ls"));
        }
        for _ in 0..5 {
            ops.push_back(Operation::new("Read", "f.txt"));
        }
        // Bash threshold = 3, Read threshold = 4
        let (repeated, max) =
            LoopDetector::find_repeated(&ops, |tool| if tool == "Bash" { 3 } else { 4 });
        // Bash(2) < 3 → excluded. Read(5) >= 4 → included.
        assert_eq!(max, 5);
        assert_eq!(repeated.len(), 1);
        assert_eq!(repeated[0].tool, "Read");
    }

    #[test]
    fn test_find_repeated_higher_count_wins() {
        let mut ops = VecDeque::new();
        for _ in 0..5 {
            ops.push_back(Operation::new("Bash", "ls"));
        }
        for _ in 0..3 {
            ops.push_back(Operation::new("Read", "f.txt"));
        }
        let (repeated, max) = LoopDetector::find_repeated(&ops, |_tool| 2);
        // Only Bash(5) wins — Read(3) is below max
        assert_eq!(max, 5);
        assert_eq!(repeated.len(), 1);
        assert_eq!(repeated[0].tool, "Bash");
    }

    #[test]
    fn test_build_warning_not_looping() {
        let detector = test_detector();
        let warning = detector.build_warning(&[], 0, false, false);
        assert!(warning.is_none());
    }

    #[test]
    fn test_build_warning_first_warning() {
        let detector = test_detector();
        let ops = vec![Operation::new("Bash", "ls")];
        let warning = detector.build_warning(&ops, 3, true, false);
        assert!(warning.is_some());
        let msg = warning.unwrap();
        assert!(msg.contains("Bash(ls)"));
        assert!(msg.contains("3 times"));
        assert!(!msg.contains("STOPPING"));
    }

    #[test]
    fn test_build_warning_includes_stop_message() {
        let detector = test_detector();
        let ops = vec![Operation::new("Bash", "ls")];
        let warning = detector.build_warning(&ops, 5, true, true);
        assert!(warning.is_some());
        let msg = warning.unwrap();
        assert!(msg.contains("STOPPING"));
    }

    #[test]
    fn test_build_warning_suppresses_duplicate() {
        let detector = test_detector();
        let ops = vec![Operation::new("Bash", "ls")];

        // First call produces a warning and records the op as warned
        let w1 = detector.build_warning(&ops, 3, true, false);
        assert!(w1.is_some());

        // Second call suppresses because already warned
        let w2 = detector.build_warning(&ops, 3, true, false);
        assert!(w2.is_none());
    }

    #[test]
    fn test_build_warning_duplicate_not_suppressed_when_stopping() {
        let detector = test_detector();
        let ops = vec![Operation::new("Bash", "ls")];

        // First warning
        let w1 = detector.build_warning(&ops, 3, true, false);
        assert!(w1.is_some());

        // Second call with should_stop=true still produces a warning
        let w2 = detector.build_warning(&ops, 3, true, true);
        assert!(w2.is_some());
        assert!(w2.unwrap().contains("STOPPING"));
    }

    #[test]
    fn test_build_warning_includes_suggestion() {
        let detector = test_detector();
        let ops = vec![Operation::new("Bash", "git status")];
        let warning = detector.build_warning(&ops, 3, true, false);
        assert!(warning.is_some());
        let msg = warning.unwrap();
        assert!(
            msg.contains("Check the command"),
            "Should contain tool signature suggestion: {msg}"
        );
    }

    /// Signature that flags the `Read` tool as a file-read tool and prefixes
    /// the primary parameter with the tool name during comparison, so tests
    /// can verify which tool name `check_file_reads` passes to
    /// `normalize_param_for_comparison`.
    struct ToolNameNormalizingSig;

    impl ToolSignature for ToolNameNormalizingSig {
        fn is_file_read_tool(&self, tool: &str) -> bool {
            tool == "read"
        }

        fn normalize_param_for_comparison(&self, tool: &str, param: &str) -> String {
            format!("{tool}:{param}")
        }
    }

    #[test]
    fn check_file_reads_matches_containment_both_directions() {
        // Use the shared TestToolSignature which treats "Read" as a file-read
        // tool and leaves params unchanged under normalization. Lower the
        // threshold so a single recorded read trips the check.
        let config = LoopDetectorConfig {
            max_same_file_reads: 1,
            ..Default::default()
        };
        let mk = || LoopDetector::new(config.clone(), Arc::new(TestToolSignature));

        // Forward direction: op path "/a/b/c" contains query "/a/b".
        let detector = mk();
        detector.record(Operation::new("Read", "/a/b/c"));
        assert!(
            detector.check_file_reads("/a/b"),
            "op path containing the query path should match"
        );

        // Reverse direction: query "/a/b/c" contains op path "/a".
        let detector2 = mk();
        detector2.record(Operation::new("Read", "/a"));
        assert!(
            detector2.check_file_reads("/a/b/c"),
            "query path containing the op path should match"
        );
    }

    #[test]
    fn check_file_reads_uses_real_tool_name() {
        // ToolNameNormalizingSig prepends the tool name during normalization,
        // so the match only succeeds if check_file_reads passes the recorded
        // op's tool name ("read") rather than an empty string.
        let detector = LoopDetector::new(
            LoopDetectorConfig {
                max_same_file_reads: 1,
                ..Default::default()
            },
            Arc::new(ToolNameNormalizingSig),
        );

        detector.record(Operation::new("read", "/file"));
        // normalize_param_for_comparison("read", "/file") == "read:/file"
        // and the query is normalized with the same tool name, so the
        // containment check only holds if both use "read".
        assert!(
            detector.check_file_reads("/file"),
            "check_file_reads must normalize under the recorded op's tool name"
        );
    }

    #[test]
    fn window_size_zero_clamps_to_one() {
        let config = LoopDetectorConfig {
            window_size: 0,
            ..Default::default()
        };
        let detector = LoopDetector::new(config, Arc::new(TestToolSignature));
        assert_eq!(
            detector.config.window_size, 1,
            "a window_size of 0 must be clamped to 1"
        );
    }
}

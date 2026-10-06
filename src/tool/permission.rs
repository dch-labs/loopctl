//! Permission checking for tool invocations.
//!
//! [`PermissionCheck`] — the result type returned by
//! the agent loop's permission gate before a tool is allowed to execute.
//! See the [`PermissionCheck`] documentation for the full decision tree.
//!
//! [`GateDecision`] — the serializable record of one gate verdict,
//! carried on the dispatch result and emitted to observers so asks,
//! plan mode, audit, and cassette replay all read one decision shape.

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

/// Result of a permission check before tool execution.
///
/// Before invoking [`Tool::call`](super::Tool::call), the agent loop can run a
/// permission gate that returns one of four outcomes: allow, deny, ask the user,
/// or modify the input. This lets host applications enforce safety
/// policies without modifying individual tool implementations.
///
/// # Example
///
/// ```rust,ignore
/// let check = PermissionCheck::deny("dangerous operation");
/// if check.is_deny() {
///     return Err(ToolError::Permission("blocked by policy".into()));
/// }
/// ```
#[derive(Debug, Clone)]
pub enum PermissionCheck {
    /// Allow the tool to execute unmodified.
    ///
    /// The agent loop proceeds with the original input and context.
    Allow,

    /// Deny execution with a human-readable reason.
    ///
    /// The agent loop should return
    /// [`ToolError::Permission`](super::ToolError::Permission) with the given
    /// `reason` so the LLM can react accordingly.
    Deny {
        /// Explanation forwarded to the LLM as part of the error.
        ///
        /// Surfaced inside [`ToolError::Permission`](super::ToolError::Permission)
        /// so the model can see *why* its call was rejected and adjust
        /// its next action. Keep it concrete and actionable — for
        /// example `"shell execution is disabled"` rather than just
        /// `"denied"`.
        reason: String,
    },

    /// Prompt the user for approval before proceeding.
    ///
    /// In interactive sessions the agent loop should present `prompt` to
    /// the user and then treat the response as either [`Allow`](PermissionCheck::Allow)
    /// or [`Deny`](PermissionCheck::Deny).
    Ask {
        /// Prompt text to present to the user for approval.
        ///
        /// Should clearly describe the action and its potential side
        /// effects so the user can make an informed decision — for
        /// example `"Allow write to /etc/config.yaml?"`. In
        /// non-interactive sessions the loop treats an `Ask` as a
        /// `Deny`, so this string is primarily for interactive hosts.
        prompt: String,
    },

    /// Modify the tool's input before execution.
    ///
    /// The agent loop should invoke [`Tool::call`](super::Tool::call) with
    /// `modified_input` instead of the original input. Useful for sanitising
    /// paths, redacting secrets, or injecting default values.
    Modify {
        /// Rewritten input to pass to [`Tool::call`](super::Tool::call)
        /// in place of the original.
        ///
        /// Must conform to the tool's
        /// [`ToolSchema::input_schema`](super::ToolSchema::input_schema)
        /// — the loop does not re-validate it. Use this to sanitise
        /// paths, redact secrets, or inject default values before the
        /// tool sees the input.
        modified_input: Value,
    },
}

impl PermissionCheck {
    /// Create an [`Allow`](PermissionCheck::Allow) result.
    ///
    /// Signals that the tool invocation may proceed without changes.
    /// The `#[must_use]` attribute reminds callers to check the result
    /// rather than silently discarding it.
    ///
    /// # When returned
    ///
    /// The permission gate returns this variant when the requested
    /// operation is within the configured safety policy — for example,
    /// a read-only tool invocation or an operation on an allowed path.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let check = PermissionCheck::allow();
    /// assert!(check.is_allow());
    /// ```
    #[must_use]
    pub fn allow() -> Self {
        Self::Allow
    }

    /// Create a [`Deny`](PermissionCheck::Deny) result with a reason.
    ///
    /// The `reason` string will be forwarded to the LLM as part of the
    /// error message, helping it understand why the invocation was
    /// rejected and adjust its next action.
    ///
    /// # When returned
    ///
    /// The permission gate returns this variant when the requested
    /// operation violates a hard safety rule — for example, executing
    /// a shell command when shell access is disabled.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let check = PermissionCheck::deny("shell execution is disabled");
    /// assert!(check.is_deny());
    /// ```
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny {
            reason: reason.into(),
        }
    }

    /// Create an [`Ask`](PermissionCheck::Ask) result with a prompt.
    ///
    /// The agent loop should present the `prompt` to the user (in
    /// interactive mode) and then proceed based on the user's response.
    ///
    /// # When returned
    ///
    /// The permission gate returns this variant for operations that are
    /// potentially dangerous but not outright prohibited — for example,
    /// writing to a file for the first time. The user's decision is then
    /// converted to [`Allow`](PermissionCheck::Allow) or
    /// [`Deny`](PermissionCheck::Deny).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let check = PermissionCheck::ask("Allow write to /etc/config.yaml?");
    /// assert!(check.is_ask());
    /// ```
    pub fn ask(prompt: impl Into<String>) -> Self {
        Self::Ask {
            prompt: prompt.into(),
        }
    }

    /// Create a [`Modify`](PermissionCheck::Modify) result with rewritten input.
    ///
    /// The agent loop should replace the original tool input with the
    /// provided `modified_input` before invoking [`Tool::call`](super::Tool::call).
    /// Useful for sanitising paths, redacting secrets, or injecting default
    /// values.
    ///
    /// # When returned
    ///
    /// The permission gate returns this variant when the requested
    /// operation is acceptable but the input needs adjustment — for
    /// example, resolving a relative path to an absolute one within
    /// the allowed directory tree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    /// use serde_json::json;
    ///
    /// let check = PermissionCheck::modify(json!({"path": "/safe/dir/file.txt"}));
    /// assert!(check.is_modify());
    /// ```
    #[must_use]
    pub fn modify(modified_input: Value) -> Self {
        Self::Modify { modified_input }
    }

    /// Returns `true` if this is an [`Allow`](PermissionCheck::Allow).
    ///
    /// Convenience predicate for the most common happy-path check.
    /// Used by the agent loop to test whether to proceed with
    /// [`Tool::call`](super::Tool::call) without further processing.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if check.is_allow() {
    ///     let result = tool.call(input, &ctx).await;
    /// }
    /// ```
    #[must_use]
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// Returns `true` if this is a [`Deny`](PermissionCheck::Deny).
    ///
    /// When `true`, the agent loop should *not* invoke the tool and
    /// should instead return a permission error to the LLM. The denial
    /// reason can be extracted by destructuring the variant or by
    /// converting to [`ToolError::Permission`](super::ToolError::Permission).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if check.is_deny() {
    ///     return Err(ToolError::Permission("blocked by policy".into()));
    /// }
    /// ```
    #[must_use]
    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Deny { .. })
    }

    /// Returns `true` if this is an [`Ask`](PermissionCheck::Ask).
    ///
    /// When `true`, the agent loop should prompt the user before
    /// deciding whether to allow or deny the invocation. In
    /// non-interactive mode ([`ToolContext::is_non_interactive`](super::ToolContext::is_non_interactive)),
    /// the loop typically treats an [`Ask`](PermissionCheck::Ask) as a
    /// [`Deny`](PermissionCheck::Deny).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if check.is_ask() {
    ///     println!("Tool requests approval: {}", prompt);
    /// }
    /// ```
    #[must_use]
    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask { .. })
    }

    /// Returns `true` if this is a [`Modify`](PermissionCheck::Modify).
    ///
    /// When `true`, the agent loop should replace the original input
    /// with the modified version before calling the tool. The modified
    /// input can be extracted by matching the variant.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let PermissionCheck::Modify { modified_input } = check {
    ///     let result = tool.call(modified_input, &ctx).await;
    /// }
    /// ```
    #[must_use]
    pub fn is_modify(&self) -> bool {
        matches!(self, Self::Modify { .. })
    }
}

/// The outcome a gate reached, as data.
///
/// One variant per path the permission middleware can take: the four
/// pre-resolution answers (the same four [`PermissionCheck`] yields)
/// plus the resolutions only an executing dispatch observes — the
/// user's answer to an ask, the headless denial of an unanswered ask,
/// and a prompt the cancel signal cut short. A dry-run
/// ([`GateDecision`]'s evaluation path) never produces the resolved
/// variants, because no prompt is ever shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateVerdict {
    /// The gate passed the call through unmodified.
    ///
    /// The dispatch proceeded with the input exactly as the model
    /// sent it — no rule rewrote anything.
    Allow,

    /// The gate rewrote the input, then passed the call through.
    ///
    /// The call executed, but with sanitized or defaulted arguments.
    /// The record's digest is of the original input, because it
    /// identifies the call the model made.
    AllowModified,

    /// The gate refused the call.
    ///
    /// The dispatch never ran; the refusal surfaced to the model as
    /// a soft error carrying the deciding rule's reason.
    Deny,

    /// The gate wants a user decision the caller has not given yet.
    ///
    /// The pre-resolution answer of an ask — what a dry-run reports
    /// and what an executing dispatch holds only until the resolver
    /// answers.
    Ask,

    /// The user approved the ask.
    ///
    /// The decision the engine acted on to dispatch the call — one
    /// record, not two: the intermediate ask is a step, and this
    /// verdict is its resolution. Whether the call ultimately ran is
    /// read from the surrounding tool events, not from this verdict:
    /// a downstream gate's record can displace it on the dispatch
    /// result, and a call refused or cancelled after the approval
    /// records this verdict at the exit that stopped it.
    AskAllowed,

    /// The user refused the ask; the call was denied.
    ///
    /// The decision is the user's, so the record's provenance is the
    /// resolver rather than the rule that asked.
    AskDenied,

    /// No resolver was configured, so the ask denied the call.
    ///
    /// The headless path: the gate wanted a decision and had no way
    /// to get one.
    AskUnresolved,

    /// The parked ask outlived its deadline and denied the call.
    ///
    /// An external approval was being awaited when the configured
    /// deadline passed; the policy denies. Distinct from a refusal
    /// (the approver said no) and from an unresolved ask (no approver
    /// existed) — the wait itself expired.
    AskExpired,

    /// The cancel signal ended the ask before anyone answered.
    ///
    /// Distinct from a refusal: the run stopped, the user did not say
    /// no.
    Cancelled,
}

/// Where the rule that produced a verdict came from.
///
/// Provenance for [`GateDecision`]: a host auditing decisions (or a
/// manifest referencing rule ids) can tell a middleware policy from a
/// per-call claim from the user's own answer to a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateRuleSource {
    /// The permission middleware's own check function decided.
    ///
    /// The path every configured gate takes, named or unnamed — the
    /// record's rule id is the check's own, or `"middleware"` when
    /// it carries none.
    Middleware,

    /// The dispatch context's per-call claim decided.
    ///
    /// The path taken when no check function is configured and the
    /// gate reads [`PermissionCheck`] from the context.
    Context,

    /// The user's answer to an ask decided.
    ///
    /// The rule asked; this source resolved.
    AskResolver,

    /// The engine's own ask policy decided.
    ///
    /// The outcome a parked ask reaches without an approver's answer —
    /// headless denial, deadline expiry, or cancellation — decided by
    /// the engine's policy rather than by any rule or resolver.
    Engine,
}

/// The serializable record of one permission-gate verdict.
///
/// Every decision the gate makes — allow, rewrite, refuse, ask, and
/// each way an ask can resolve — produces one of these, carried on
/// the [`ToolDispatchResult`](crate::tool::ToolDispatchResult) the
/// middleware returns and emitted to observers by the engine. The
/// arguments are **not** stored: only a digest, so the record is safe
/// to persist and replay while remaining matchable against a known
/// call. `ts` is stamped by whoever holds a clock — the engine at
/// emission — so a record read before emission (and every
/// dry-run record) carries `0`.
///
/// # Example
///
/// ```
/// use loopctl::tool::permission::{GateDecision, GateVerdict, GateRuleSource};
///
/// let decision = GateDecision::new(
///     "deploy",
///     GateVerdict::Deny,
///     "deny-write-etc",
///     GateRuleSource::Middleware,
/// )
/// .with_reason("etc is off limits");
///
/// assert_eq!(decision.verdict, GateVerdict::Deny);
/// assert_eq!(decision.rule_id, "deny-write-etc");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GateDecision {
    /// The tool the gate decided about.
    ///
    /// The name as the model called it, matching the dispatch the
    /// record rode out on.
    pub tool: String,

    /// A stable digest of the call's arguments.
    ///
    /// FNV-1a 64 over the input's canonical JSON — every object's keys
    /// sorted, recursively, by the digest's own canonicalization —
    /// rendered as 16 lowercase hex characters: the same fixed
    /// algorithm the demotion sink tags persisted memories with, so
    /// digests never churn across Rust releases. Key order never
    /// moves it, whichever `serde_json` map backend the host's graph
    /// resolves to; any argument change produces a different digest
    /// except under a collision of the 64-bit hash. The arguments
    /// themselves are never stored.
    pub args_digest: String,

    /// The outcome the gate reached.
    ///
    /// See [`GateVerdict`] for the full path inventory.
    pub verdict: GateVerdict,

    /// The identifier of the rule that decided.
    ///
    /// Stable across a session so manifests (and audit roll-ups) can
    /// reference it: `"middleware"` for an unnamed check function,
    /// the id a [`with_named_check`](crate::middleware::PermissionMiddleware::with_named_check)
    /// rule was given, `"context"` for a per-call claim, `"ask"` for
    /// a resolver's answer, `"hook"` for the engine's parked-ask
    /// policy outcomes (an expired, cancelled, or headless hook ask).
    pub rule_id: String,

    /// Where the deciding rule came from.
    ///
    /// One of the middleware's own check, the dispatch context's
    /// per-call claim, or the user's answer to an ask — an audit
    /// roll-up groups on this field to tell policy outcomes from user
    /// outcomes. See [`GateRuleSource`].
    pub rule_source: GateRuleSource,

    /// The pattern a rule matched, when the rule is pattern-shaped.
    ///
    /// `None` on every built-in path — the shipped rules decide as
    /// functions and claims, never by pattern. A gate implementation
    /// whose rules match by pattern populates the field with the
    /// matched pattern text.
    pub matched_pattern: Option<String>,

    /// Why the gate decided as it did, when it said so.
    ///
    /// A denial's reason, an ask's prompt — the text the deciding
    /// rule produced. `None` when the verdict needs no explanation
    /// (a plain allow).
    pub reason: Option<String>,

    /// When the decision was made, in Unix epoch milliseconds.
    ///
    /// Stamped from the engine's clock seam at emission; a record
    /// read straight off a dispatch result before the engine emits —
    /// and every dry-run record — carries `0`, because the middleware
    /// that minted it has no clock.
    pub ts: u64,
}

impl GateDecision {
    /// Build a record from its decision facts.
    ///
    /// The construction path for code outside the crate — the type is
    /// `#[non_exhaustive]`, so struct literals compile only inside
    /// it. `matched_pattern`, `reason`, and `ts` default to
    /// `None`/`None`/`0` and have their own builders.
    #[must_use]
    pub fn new(
        tool: impl Into<String>,
        verdict: GateVerdict,
        rule_id: impl Into<String>,
        rule_source: GateRuleSource,
    ) -> Self {
        Self {
            tool: tool.into(),
            args_digest: String::new(),
            verdict,
            rule_id: rule_id.into(),
            rule_source,
            matched_pattern: None,
            reason: None,
            ts: 0,
        }
    }

    /// Digest a call's arguments for the record.
    ///
    /// The canonical-JSON FNV-1a 64 digest in 16 hex characters:
    /// independent of key order and of the `serde_json` map backend,
    /// and changed by any argument change except under a collision of
    /// the 64-bit hash — the digest canonicalizes the input itself
    /// (every object's keys sorted, recursively) instead of serializing
    /// it verbatim, because a host's dependency graph can switch
    /// `serde_json` to insertion-ordered maps.
    #[must_use]
    pub fn args_digest(input: &Value) -> String {
        format!(
            "{:016x}",
            crate::compact::demote::fnv1a64(super::canonical_json(input).as_bytes())
        )
    }

    /// Attach the digested arguments of the call this record is about.
    ///
    /// Digests `input` with [`args_digest`](Self::args_digest) and
    /// stores the result.
    #[must_use]
    pub fn for_args(mut self, input: &Value) -> Self {
        self.args_digest = Self::args_digest(input);
        self
    }

    /// Set the matched pattern.
    ///
    /// Populates [`matched_pattern`](Self::matched_pattern) with the
    /// pattern text a pattern-shaped rule matched — a built-in
    /// record leaves it unset.
    #[must_use]
    pub fn with_matched_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.matched_pattern = Some(pattern.into());
        self
    }

    /// Set the deciding reason.
    ///
    /// The text the deciding rule produced — a denial's reason or an
    /// ask's prompt.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Set the decision timestamp.
    ///
    /// Called by the holder of a clock — the engine, at emission.
    #[must_use]
    pub fn with_ts(mut self, ts: u64) -> Self {
        self.ts = ts;
        self
    }
}

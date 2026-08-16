//! Tool trait and registry — the framework-level abstraction for agent tools.
//!
//! Tools are the primary way agents interact with the outside world. This
//! module defines the core [`Tool`] trait that every concrete tool must
//! implement, along with the supporting types that make up the tool
//! ecosystem: schemas, results, errors, context, permissions, and a
//! dynamic registry.
//!
//! # Provided Types
//!
//! - **[`Tool`]** — The trait every tool implements. Downstream crates
//!   provide concrete implementations.
//! - **[`FnTool`]** — An adapter that wraps a plain function pointer as a
//!   [`Tool`] trait object, wrapping plain functions as trait implementations.
//! - **[`ToolRegistry`]** — A name → tool map used by the agent loop for
//!   dynamic dispatch.
//! - **[`ToolSchema`]** — JSON Schema descriptor sent to the LLM for
//!   tool discovery.
//! - **[`ToolOutput`] / [`ToolError`]** — Success and error result types.
//! - **[`ToolContext`]** — Session-level context passed to every invocation.
//! - **[`PermissionCheck`]** — Pre-execution permission gate.
//!
//! # Middleware Pipeline
//!
//! The [`engine::middleware`](crate::middleware) module provides a composable
//! middleware chain for tool dispatch with cross-cutting concerns
//! (timeouts, output limiting, etc.).
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use loopctl::tool::{Tool, ToolContext, ToolOutput, ToolError, ToolSchema};
//! use serde_json::{Value, json};
//! use std::pin::Pin;
//! use std::future::Future;
//!
//! struct EchoTool;
//!
//! impl Tool for EchoTool {
//!     fn name(&self) -> &str { "echo" }
//!     fn description(&self) -> &str { "Echoes back the input" }
//!     fn schema(&self) -> ToolSchema {
//!         ToolSchema {
//!             tool: "echo".into(),
//!             description: "Echoes back the input".into(),
//!             input_schema: json!({
//!                 "type": "object",
//!                 "properties": { "message": { "type": "string" } },
//!                 "required": ["message"]
//!             }),
//!         }
//!     }
//!
//!     fn call(&self, input: Value, _ctx: &ToolContext)
//!         -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>>
//!     {
//!         let msg = input.get("message").and_then(|v| v.as_str()).unwrap_or("").to_string();
//!         Box::pin(async move { Ok(ToolOutput::text(msg)) })
//!     }
//! }
//! ```

#[cfg(feature = "tool_health")]
pub mod health;
#[cfg(feature = "tool_shield")]
pub mod shield;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::message::ToolContent as MessageToolContent;

pub mod permission;
pub mod registry;

pub use permission::PermissionCheck;
pub use registry::{FnTool, ToolRegistry};

/// Schema descriptor for a tool, used for LLM API tool definitions.
///
/// When the agent loop sends a list of available tools to the LLM, each
/// tool is represented by a [`ToolSchema`] instance. The LLM uses the
/// schema's `tool`, `description`, and `input_schema` to decide which
/// tool to call and how to format its arguments.
///
/// # Construction
///
/// Typically produced by [`Tool::schema`] inside each tool implementation:
///
/// ```rust,ignore
/// fn schema(&self) -> ToolSchema {
///     ToolSchema {
///         tool: "read_file".into(),
///         description: "Read a file from disk".into(),
///         input_schema: json!({
///             "type": "object",
///             "properties": {
///                 "path": { "type": "string", "description": "File path" }
///             },
///             "required": ["path"]
///         }),
///     }
/// }
/// ```
///
/// # Serialization
///
/// [`ToolSchema`] derives [`Serialize`] and [`Deserialize`] so it can be
/// embedded directly in LLM API request payloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// The tool's unique name identifier.
    ///
    /// Must match [`Tool::name`] exactly. Used by the LLM to select which
    /// tool to invoke and by [`ToolRegistry`] as the lookup key.
    pub tool: String,

    /// Human-readable description of what the tool does.
    ///
    /// Sent to the LLM as part of the tool definition. A clear, concise
    /// description helps the model choose the right tool and provide
    /// correct arguments.
    pub description: String,

    /// JSON Schema describing the tool's input parameters.
    ///
    /// Conforms to JSON Schema Draft 07. The LLM uses this to construct
    /// valid `input` objects for [`Tool::call`].
    pub input_schema: Value,
}

/// Advisory rendering hint attached to a [`ToolOutput`].
///
/// Tells presentation layers (TUI, headless console, loggers) *how* the tool
/// author intends the payload to be rendered. It is **purely advisory**: the
/// agent loop, compaction, loop-detection hashing, and message serialization
/// never read it. A consumer that does not understand a variant MUST fall back
/// to plain-text rendering — `None` and [`DisplayHint::Text`] are equivalent.
///
/// `#[non_exhaustive]` so loopctl can add rendering strategies (e.g. `Table`,
/// `Image`) in a later minor release without breaking downstream `match`es
/// that carry a `_ =>` arm. Downstream code that matches on `DisplayHint`
/// MUST include a wildcard arm.
///
/// # Serialization
///
/// Derives [`Serialize`] and [`Deserialize`] (using `#[serde(tag = "type")]`,
/// matching the [`MessagePart`](crate::message::MessagePart) convention) so
/// observer-event sinks and debug pretty-printers can carry the hint. The hint
/// is still never part of the conversation message model — it terminates at
/// [`ToolPostContext`](crate::observer::ToolPostContext) and is not added to
/// [`MessagePart::ToolResult`](crate::message::MessagePart) — so this derive
/// does not by itself leak the hint into API payloads or saved sessions.
///
/// # When to set
///
/// Set the hint via [`ToolOutput::with_hint`] when the tool knows its payload
/// has a structure richer than plain text. Tools that emit ordinary prose
/// leave the hint as `None` (the default). See each variant's doc for when it
/// applies.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::tool::{ToolOutput, DisplayHint};
///
/// // An Edit tool that produces a unified diff:
/// let result = ToolOutput::text(diff_text).with_hint(DisplayHint::Diff);
///
/// // A JSON tool that wants pretty-printing:
/// let result = ToolOutput::text(json_string).with_hint(DisplayHint::Json);
///
/// // A Read tool whose output is the file body — suppress the one-line preview:
/// let result = ToolOutput::text(file_contents).with_hint(DisplayHint::Suppress);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DisplayHint {
    /// Plain text — the default. `None` ≡ `Text`.
    ///
    /// Render the payload as-is. This is what every tool gets when it does not
    /// call [`ToolOutput::with_hint`].
    Text,

    /// A unified diff — render with `+`/`-` line coloring.
    ///
    /// Intended for Edit/MultiEdit/Patch tools whose payload is a
    /// `diff`-formatted string. A consumer that supports diff rendering parses
    /// the payload as a unified diff; one that does not falls back to `Text`.
    Diff,

    /// Structured JSON — pretty-print / syntax-highlight.
    ///
    /// The payload is a JSON-encoded string the consumer may parse and
    /// re-format. If parsing fails the consumer falls back to `Text`.
    Json,

    /// Syntax-highlighted source code in the given language.
    ///
    /// `language` is a free-form hint (e.g. `"rust"`, `"python"`, `"json"`),
    /// matching the identifiers used by common highlighters. A consumer that
    /// does not highlight renders the payload as plain `Text`.
    Code {
        /// The source language identifier (lowercase convention, e.g. `"rust"`).
        language: String,
    },

    /// Noise to humans — suppress surface-level previews.
    ///
    /// The payload is still sent to the model in full (loop semantics are
    /// unaffected); only *presentation* layers should elide it (e.g. omit the
    /// one-line output preview in Normal verbosity, or collapse the block).
    /// Intended for tools whose output duplicates content the user already saw
    /// (a Read echoing a large file) or is machine-only.
    Suppress,

    /// Structured Markdown the tool authored as Markdown.
    ///
    /// Use when the payload is *intentionally* Markdown the tool constructed
    /// (a formatted report, a rendered table-as-markdown, a summary with
    /// emphasis), so the consumer can skip heuristic detection and render it
    /// confidently. For plain prose, prefer [`Text`](Self::Text) (the default)
    /// — any text *could* be rendered as Markdown, but `Markdown` signals the
    /// tool's positive intent. A consumer that does not render Markdown falls
    /// back to `Text`.
    Markdown,
}

/// Result from a tool invocation.
///
/// Every [`Tool::call`] returns a `Result<ToolOutput, ToolError>`. On
/// success, [`ToolOutput`] wraps the output content (plain text or
/// structured multi-part content) and a flag indicating whether the
/// content represents an error message — this lets tools report *soft*
/// failures (e.g., "file not found") without raising a hard [`ToolError`].
///
/// # Construction
///
/// Use the named constructors rather than building the struct directly:
///
/// ```rust,ignore
/// // Success
/// let ok = ToolOutput::text("hello world");
/// let ok = ToolOutput::success(structured_content);
///
/// // Soft error (tool ran, but the result is an error message)
/// let err = ToolOutput::error_text("file not found");
/// let err = ToolOutput::error(structured_content);
/// ```
///
/// # Content types
///
/// The [`payload`](ToolOutput::payload) field holds a
/// [`MessageToolContent`] which is either a simple [`String`] or a
/// structured list of [`ToolContentPart`](crate::message::ToolContentPart)
/// elements. Use [`text_content`](ToolOutput::text_content) to extract
/// plain text regardless of which variant is stored.
///
/// # Conversion
///
/// [`ToolOutput`] implements [`From<String>`] and [`From<&str>`] so
/// simple string values can be converted implicitly:
///
/// ```rust
/// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
///
/// let result: ToolOutput = "done".into();
/// ```
///
/// # Data flow
///
/// ```text
/// Tool::call(input, ctx)
///   → Ok(ToolOutput { content, is_error: false })
///   → Ok(ToolOutput { content, is_error: true  })   [soft failure]
///   → Err(ToolError)                                [hard failure]
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolOutput {
    /// The payload returned by the tool.
    ///
    /// Holds either a plain-text string or structured multi-part content.
    /// Use [`ToolOutput::text_content`] to extract text regardless of the
    /// variant.
    pub payload: MessageToolContent,

    /// Whether this result represents an error.
    ///
    /// When `true`, the agent loop treats the result as a soft failure —
    /// the tool executed without panicking, but the output describes what
    /// went wrong. Defaults to `false` for results created via
    /// [`ToolOutput::success`] or [`ToolOutput::text`].
    pub is_error: bool,

    /// Advisory rendering hint. `None` ≡ [`DisplayHint::Text`].
    ///
    /// **Never affects loop semantics** — not read by compaction, loop-detection
    /// hashing, or message serialization. Forwarded to
    /// [`ToolDispatchResult`] and onward to
    /// [`ToolPostContext`](crate::observer::ToolPostContext) so presentation
    /// layers can read it. Set via [`with_hint`](ToolOutput::with_hint).
    pub display_hint: Option<DisplayHint>,
}

impl ToolOutput {
    /// Create a successful result with the given content.
    ///
    /// Called by tool implementations when the invocation succeeds and
    /// the output is a structured [`MessageToolContent`] value.
    /// Sets [`is_error`](ToolOutput::is_error) to `false`.
    ///
    /// Most general success constructor — accepts any type
    /// that converts into [`MessageToolContent`]. For simple text results,
    /// prefer the more concise [`ToolOutput::text`] helper.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    /// use loopctl::message::ToolContent as MessageToolContent;
    ///
    /// let result = ToolOutput::success(MessageToolContent::Text("done".into()));
    /// assert!(!result.is_error);
    /// ```
    pub fn success(payload: impl Into<MessageToolContent>) -> Self {
        Self {
            payload: payload.into(),
            is_error: false,
            display_hint: None,
        }
    }

    /// Create an error result with the given content.
    ///
    /// Called by tool implementations that want to report a *soft* failure
    /// — the tool ran to completion, but the output describes an error
    /// condition (e.g., "file not found"). Sets
    /// [`is_error`](ToolOutput::is_error) to `true`.
    ///
    /// Soft failures are distinct from hard [`ToolError`] returns — the
    /// tool did not panic and the result can still be processed by the
    /// agent loop and forwarded to the LLM.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let result = ToolOutput::error("permission denied");
    /// assert!(result.is_error);
    /// ```
    pub fn error(payload: impl Into<MessageToolContent>) -> Self {
        Self {
            payload: payload.into(),
            is_error: true,
            display_hint: None,
        }
    }

    /// Create a successful plain-text result.
    ///
    /// Convenience wrapper around [`ToolOutput::success`] that converts
    /// the input string into a [`MessageToolContent::Text`] variant.
    /// Most common constructor for simple tool outputs.
    ///
    /// # When to use
    ///
    /// Use this when the tool produces a simple string response — for
    /// example, file contents, a computed value, or a status message.
    /// For structured multi-part responses, use [`ToolOutput::success`]
    /// directly with a [`MessageToolContent::Multipart`] value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let result = ToolOutput::text("42");
    /// assert_eq!(result.text_content(), "42");
    /// ```
    pub fn text(text: impl Into<String>) -> Self {
        Self::success(text.into())
    }

    /// Create an error plain-text result.
    ///
    /// Convenience wrapper around [`ToolOutput::error`] that converts
    /// the input string into a [`MessageToolContent::Text`] variant.
    ///
    /// Use this for simple error messages like "file not found" or
    /// "permission denied". For structured error content, use
    /// [`ToolOutput::error`] directly.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let result = ToolOutput::error_text("disk full");
    /// assert!(result.is_error);
    /// assert_eq!(result.text_content(), "disk full");
    /// ```
    pub fn error_text(text: impl Into<String>) -> Self {
        Self::error(text.into())
    }

    /// Attach an advisory [`DisplayHint`]. Fluent builder.
    ///
    /// Does not mutate [`payload`](ToolOutput::payload) or
    /// [`is_error`](ToolOutput::is_error); the hint travels alongside the
    /// content for presentation layers to read. Loop semantics are
    /// unaffected.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::tool::{ToolOutput, DisplayHint};
    ///
    /// let out = ToolOutput::text(diff).with_hint(DisplayHint::Diff);
    /// assert_eq!(out.display_hint, Some(DisplayHint::Diff));
    /// ```
    #[must_use]
    pub fn with_hint(mut self, hint: DisplayHint) -> Self {
        self.display_hint = Some(hint);
        self
    }

    /// Extract all text content from the result, regardless of structure.
    ///
    /// Inspects the [`payload`](ToolOutput::payload) field and returns a
    /// flat [`String`]:
    /// - For [`MessageToolContent::Text`], returns the string directly.
    /// - For [`MessageToolContent::Multipart`], joins all
    ///   [`ToolContentPart::Text`](crate::message::ToolContentPart::Text)
    ///   parts with newlines, discarding non-text parts.
    ///
    /// Useful when the consumer only cares about the textual payload and
    /// does not need to handle structured content.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    /// use loopctl::message::ToolContent as MessageToolContent;
    /// use loopctl::message::ToolContentPart;
    ///
    /// let result = ToolOutput::text("hello world");
    /// assert_eq!(result.text_content(), "hello world");
    ///
    /// // Works with structured content too:
    /// let structured = ToolOutput::success(MessageToolContent::Multipart(vec![
    ///     ToolContentPart::Text { text: "line 1".into() },
    ///     ToolContentPart::Text { text: "line 2".into() },
    /// ]));
    /// assert_eq!(structured.text_content(), "line 1\nline 2");
    /// ```
    ///
    /// # See also
    ///
    /// - [`ToolOutput::payload`] — access the raw content without flattening.
    #[must_use]
    pub fn text_content(&self) -> String {
        match &self.payload {
            MessageToolContent::Text(s) => s.clone(),
            MessageToolContent::Multipart(parts) => {
                use crate::message::ToolContentPart;
                parts
                    .iter()
                    .filter_map(|p| match p {
                        ToolContentPart::Text { text } => Some(text.clone()),
                        ToolContentPart::Image { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
    }

    /// Construct a successful result from any serializable value.
    ///
    /// The value is JSON-serialized into the
    /// [`ToolContent::Text`](crate::message::ToolContent::Text) payload and
    /// flagged so [`structured_value`](Self::structured_value) can recover it.
    /// `T` need not implement [`StructuredOutput`](crate::structured::StructuredOutput)
    /// — any `Serialize` works —
    /// but types that *do* get a round-trippable accessor via
    /// [`structured_as`](Self::structured_as).
    ///
    /// If serialization fails (should not happen for normal structs), returns
    /// an error result rather than panicking.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::ToolOutput;
    /// use serde::Serialize;
    ///
    /// #[derive(Serialize)]
    /// struct Data { count: u32 }
    ///
    /// let out = ToolOutput::structured(&Data { count: 42 });
    /// assert!(!out.is_error);
    /// assert!(out.structured_value().is_some());
    /// ```
    pub fn structured<T: serde::Serialize>(value: &T) -> Self {
        match serde_json::to_string(value) {
            Ok(json) => Self::success(json),
            Err(e) => Self::error_text(format!("structured serialization failed: {e}")),
        }
    }

    /// Parse the payload back into a [`serde_json::Value`], if it is valid JSON.
    ///
    /// Returns `None` for multipart payloads or non-JSON text. Use this when
    /// the consumer does not know the concrete type at compile time (e.g. an
    /// observer that re-serializes for a TUI).
    #[must_use]
    pub fn structured_value(&self) -> Option<serde_json::Value> {
        let MessageToolContent::Text(s) = &self.payload else {
            return None;
        };
        serde_json::from_str(s).ok()
    }

    /// Parse the payload into a concrete `T: StructuredOutput`.
    ///
    /// Round-trips a value produced by [`structured`](Self::structured) or any
    /// tool whose text output happens to conform to `T`'s schema. Returns
    /// `None` if the payload is not valid JSON for `T`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::tool::ToolOutput;
    /// use loopctl::structured::StructuredOutput;
    ///
    /// let out = ToolOutput::structured(&action);
    /// let back: Action = out.structured_as().unwrap();
    /// ```
    #[must_use]
    pub fn structured_as<T: crate::structured::StructuredOutput + DeserializeOwned>(
        &self,
    ) -> Option<T> {
        self.structured_value().and_then(|v| T::from_value(v).ok())
    }
}

impl From<String> for ToolOutput {
    fn from(s: String) -> Self {
        Self::text(s)
    }
}

impl From<&str> for ToolOutput {
    fn from(s: &str) -> Self {
        Self::text(s)
    }
}

/// The outcome of a single tool invocation.
///
/// Produced after the framework dispatches a tool call and collects
/// the tool's output. Used throughout the middleware pipeline, the
/// engine dispatch layer, and returned to callers.
///
/// # Fields
///
/// | Field                  | Source                              |
/// |------------------------|-------------------------------------|
/// | `tool_call_id`         | Set by the engine after dispatch    |
/// | `output`               | From [`ToolOutput::payload`]        |
/// | `is_error`             | From [`ToolOutput::is_error`]       |
/// | `duration`             | Measured by the dispatch layer      |
/// | `resolved_tool_name`   | Set by middleware or engine         |
///
/// # Construction
///
/// Middlewares build results with [`ToolDispatchResult::ok`],
/// [`ToolDispatchResult::err`], or [`From<ToolOutput>`] combined with
/// builder methods. The engine layer attaches the `tool_call_id` via
/// [`ToolDispatchResult::with_call_id`] after the middleware pipeline
/// returns.
///
/// ```
/// use std::time::Duration;
/// use loopctl::tool::{ToolDispatchResult, ToolOutput};
///
/// let output = ToolOutput::text("done");
/// let result = ToolDispatchResult::from(output)
///     .with_tool_name("bash")
///     .with_duration(Duration::from_millis(42))
///     .with_call_id("call_abc123");
///
/// assert_eq!(result.tool_call_id, "call_abc123");
/// assert_eq!(result.resolved_tool_name, "bash");
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolDispatchResult {
    /// The ID the model assigned to this tool call.
    ///
    /// Copied from the inbound `ToolCall` by the engine so the caller
    /// can correlate the result back to the request. Set via
    /// [`with_call_id`](Self::with_call_id) on the builder; empty when
    /// constructed via [`From<ToolOutput>`](Self#impl-From<ToolOutput>).
    pub tool_call_id: String,

    /// The content returned by the tool.
    ///
    /// A [`ToolContent::Text`](crate::message::ToolContent::Text) on the
    /// common path (plain-text result or an error message wrapped in
    /// text). A
    /// [`ToolContent::Multipart`](crate::message::ToolContent::Multipart)
    /// when the tool produces mixed content (text + images) or multiple
    /// text segments. Middlewares may rewrite this post-execution (e.g.
    /// `OutputLimitMiddleware` truncates, `VerifyMiddleware` appends a
    /// `[verify]` block).
    pub output: crate::message::ToolContent,

    /// Whether the tool dispatch resulted in an error.
    ///
    /// `false` on success; `true` when the tool returned an error or the
    /// dispatch itself failed (timeout, panic caught via `catch_unwind`,
    /// permission denied). Middlewares and the recovery loop consult
    /// this flag — for example, `VerifyMiddleware` skips verification
    /// and `MemoizingMiddleware` skips caching when `is_error` is `true`.
    pub is_error: bool,

    /// Wall-clock execution time of the tool call.
    ///
    /// Measured from dispatch start to completion by the engine. Zero on
    /// cached results returned by `MemoizingMiddleware` (the cached call
    /// didn't execute this turn). Useful for metrics, slow-tool
    /// detection, and timeout diagnostics.
    pub duration: Duration,

    /// The tool name the call actually ran under.
    ///
    /// Usually identical to the requested name. May differ when a
    /// routing middleware redirected the call (e.g. aliasing an
    /// unknown tool name to the closest match). Middlewares that need
    /// to reason about *which tool executed* (rather than what the
    /// model requested) should read this field instead of the
    /// pre-dispatch `ToolDispatchContext::tool_name`.
    pub resolved_tool_name: String,

    /// Advisory rendering hint forwarded from the originating [`ToolOutput`].
    ///
    /// `None` for error/panic/blocked paths that have no `ToolOutput` to read
    /// from. Carried into [`ToolPostContext`](crate::observer::ToolPostContext)
    /// for observers to read; never read by loop semantics.
    pub display_hint: Option<DisplayHint>,
}

impl ToolDispatchResult {
    /// Create a successful result with text output.
    ///
    /// Constructor for the common case where a tool
    /// produces a plain-text response.
    #[must_use]
    pub fn ok(tool_name: &str, output: String, duration: Duration) -> Self {
        Self {
            tool_call_id: String::new(),
            output: crate::message::ToolContent::Text(output),
            is_error: false,
            duration,
            resolved_tool_name: tool_name.to_string(),
            display_hint: None,
        }
    }

    /// Create an error result with a message.
    ///
    /// Used when a middleware short-circuits or the tool reports failure.
    #[must_use]
    pub fn err(tool_name: &str, message: String, duration: Duration) -> Self {
        Self {
            tool_call_id: String::new(),
            output: crate::message::ToolContent::Text(message),
            is_error: true,
            duration,
            resolved_tool_name: tool_name.to_string(),
            display_hint: None,
        }
    }

    /// Create a result from a [`ToolOutput`].
    ///
    /// Converts the tool's output struct into a dispatch result,
    /// preserving the error flag and content payload.
    #[must_use]
    pub fn from_tool_output(tool_name: &str, output: ToolOutput, duration: Duration) -> Self {
        Self::from(output)
            .with_tool_name(tool_name)
            .with_duration(duration)
    }

    /// Builder: attach the [`tool_call_id`](Self::tool_call_id).
    ///
    /// Called by the engine layer after the middleware pipeline returns
    /// to correlate this result with the original tool call.
    #[must_use]
    pub fn with_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = id.into();
        self
    }

    /// Set the [`resolved_tool_name`](Self::resolved_tool_name).
    ///
    /// Part of the builder chain when constructing a
    /// `ToolDispatchResult` from [`From<ToolOutput>`].
    #[must_use]
    pub fn with_tool_name(mut self, name: &str) -> Self {
        name.clone_into(&mut self.resolved_tool_name);
        self
    }

    /// Set the [`duration`](Self::duration).
    ///
    /// Part of the builder chain when constructing a
    /// `ToolDispatchResult` from [`From<ToolOutput>`].
    #[must_use]
    pub fn with_duration(mut self, dur: Duration) -> Self {
        self.duration = dur;
        self
    }

    /// Create a result from a [`ToolError`].
    ///
    /// Converts the tool's error into a dispatch result with `is_error`
    /// set to `true`.
    #[must_use]
    pub fn from_tool_error(tool_name: &str, error: &ToolError, duration: Duration) -> Self {
        Self {
            tool_call_id: String::new(),
            output: crate::message::ToolContent::Text(error.to_string()),
            is_error: true,
            duration,
            resolved_tool_name: tool_name.to_string(),
            display_hint: None,
        }
    }

    /// Create a result from a tool call outcome.
    ///
    /// Covers the common `Result<ToolOutput, ToolError>` pattern produced by
    /// [`Tool::call()`](Tool::call). Maps [`Ok`] through
    /// [`from_tool_output`](Self::from_tool_output) and [`Err`] through
    /// [`from_tool_error`](Self::from_tool_error).
    #[must_use]
    pub fn from_result(
        tool_name: &str,
        result: Result<ToolOutput, ToolError>,
        duration: Duration,
    ) -> Self {
        match result {
            Ok(output) => Self::from_tool_output(tool_name, output, duration),
            Err(e) => Self::from_tool_error(tool_name, &e, duration),
        }
    }
}

/// Conversion from a bare [`ToolOutput`].
///
/// Produces a `ToolDispatchResult` with no call ID, [`Duration::ZERO`],
/// and an empty `resolved_tool_name`. Chain builder methods to complete
/// the fields:
///
/// ```
/// use std::time::Duration;
/// use loopctl::tool::{ToolDispatchResult, ToolOutput};
///
/// let result = ToolDispatchResult::from(ToolOutput::text("ok"))
///     .with_call_id("call_1")
///     .with_tool_name("echo")
///     .with_duration(Duration::from_millis(5));
/// ```
impl From<ToolOutput> for ToolDispatchResult {
    fn from(output: ToolOutput) -> Self {
        Self {
            tool_call_id: String::new(),
            output: output.payload,
            is_error: output.is_error,
            duration: Duration::ZERO,
            resolved_tool_name: String::new(),
            display_hint: output.display_hint,
        }
    }
}

/// Error type for tool invocations.
///
/// Covers the full range of failure modes a tool can encounter — from
/// "tool not found" and invalid input, to I/O failures, timeouts, and
/// permission denials. Each variant carries enough context for the agent
/// loop (or the LLM) to decide how to recover.
///
/// The [`thiserror::Error`] derive provides [`Display`](std::fmt::Display)
/// and [`Error`](std::error::Error) implementations with human-readable
/// messages.
///
/// # Recovery strategy
///
/// ```text
/// ToolError::NotFound      → inform LLM of available tools, retry
/// ToolError::InvalidInput  → ask LLM to fix arguments, retry
/// ToolError::Permission    → inform LLM, cannot retry without user approval
/// ToolError::Timeout       → optionally retry with longer timeout
/// ToolError::Execution     → generic retry or abort
/// ToolError::Io            → typically non-retryable
/// ToolError::Json          → typically non-retryable (bug in tool)
/// ToolError::Cancelled     → session shutting down, do not retry
/// ToolError::FileNotFound  → inform LLM, may adjust path and retry
/// ```
///
/// # Conversion from std errors
///
/// [`ToolError`] implements `From<std::io::Error>` and
/// `From<serde_json::Error>` so the `?` operator works naturally
/// inside tool implementations:
///
/// ```rust,ignore
/// let data = tokio::fs::read_to_string(&path).await?;  // io::Error → ToolError
/// let parsed: Value = serde_json::from_str(&data)?;    // json::Error → ToolError
/// ```
///
/// # Example
///
/// ```rust
/// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
///
/// let err = ToolError::not_found("grep", &["read_file", "bash"]);
/// assert!(err.to_string().contains("grep"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The requested tool was not found in the registry.
    ///
    /// The first field is the requested tool name; the second is a
    /// comma-separated list of available tool names (or "none registered").
    /// Produced by [`ToolError::not_found`].
    ///
    /// # Recovery
    ///
    /// The agent loop should inform the LLM which tools *are* available
    /// so it can retry with a valid tool name.
    #[error("Tool not found: {0}. Available: {1}")]
    NotFound(String, String),

    /// The tool input failed validation.
    ///
    /// Carries a human-readable description of what was wrong — for
    /// example "missing required field `path`" or "expected integer".
    ///
    /// # Recovery
    ///
    /// The agent loop should ask the LLM to fix its arguments and retry.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// An execution error occurred inside the tool.
    ///
    /// A catch-all for runtime failures that are not covered by the more
    /// specific variants (e.g., a subprocess exited with a non-zero code).
    ///
    /// # Recovery
    ///
    /// The agent loop may retry the invocation or report the error to
    /// the LLM for an alternative approach.
    #[error("Execution error: {0}")]
    Execution(String),

    /// Permission denied for the requested operation.
    ///
    /// Raised when the tool (or the agent loop's permission gate) rejects
    /// an invocation — for example, writing outside the allowed directory.
    ///
    /// # Recovery
    ///
    /// Cannot be retried without user approval or a change in the
    /// permission policy.
    #[error("Permission denied: {0}")]
    Permission(String),

    /// The target file or resource was not found.
    ///
    /// Distinct from [`NotFound`](ToolError::NotFound) which refers to a
    /// missing *tool*. This variant means the tool was found but the file
    /// it tried to access does not exist.
    ///
    /// # Recovery
    ///
    /// The agent loop should inform the LLM so it can adjust the file
    /// path and retry.
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// The tool execution exceeded its time limit.
    ///
    /// The `u64` field is the timeout in seconds. The agent loop may
    /// choose to retry or report the timeout to the LLM.
    ///
    /// # Recovery
    ///
    /// Optionally retry with a longer timeout or suggest the LLM simplify
    /// its request.
    #[error("Timeout after {0}s")]
    Timeout(u64),

    /// The tool was explicitly cancelled.
    ///
    /// Set when the user or framework aborts an in-flight tool call —
    /// for example when the agent session is shutting down.
    ///
    /// # Recovery
    ///
    /// Do not retry. The session is likely winding down.
    #[error("Cancelled")]
    Cancelled,

    /// An underlying I/O error.
    ///
    /// Automatically created via `?` when a `std::io::Error` propagates
    /// out of a tool implementation. Common causes include file-not-found,
    /// permission denied at the OS level, or broken pipes.
    ///
    /// # Recovery
    ///
    /// Typically non-retryable. The agent loop should report the error to
    /// the LLM.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A JSON serialization / deserialization error.
    ///
    /// Automatically created via `?` when a `serde_json::Error` propagates
    /// out of a tool implementation. Usually indicates a bug in the tool's
    /// input parsing logic.
    ///
    /// # Recovery
    ///
    /// Typically non-retryable — the tool code needs to be fixed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl ToolError {
    /// Create a [`NotFound`](ToolError::NotFound) error with an available-tools list.
    ///
    /// Formats the `available` slice into a human-friendly string:
    /// - Empty → `"none registered"`
    /// - ≤ 10 tools → comma-separated names
    /// - \> 10 tools → first 10 names + `"... (and N more)"`
    ///
    /// Called by the agent loop when a tool name requested by the LLM is
    /// not present in the [`ToolRegistry`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let err = ToolError::not_found("grep", &["bash", "read_file"]);
    /// assert!(matches!(err, ToolError::NotFound(..)));
    /// ```
    pub fn not_found(tool: impl Into<String>, available: &[&str]) -> Self {
        let tool = tool.into();
        let available_str = if available.is_empty() {
            "none registered".to_string()
        } else if available.len() <= 10 {
            available.join(", ")
        } else {
            let count = available.len().saturating_sub(10);
            format!(
                "{}... (and {count} more)",
                available
                    .iter()
                    .take(10)
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        };
        Self::NotFound(tool, available_str)
    }
}

/// Session-level context provided to every tool invocation.
///
/// The agent loop constructs a [`ToolContext`] at session start and passes
/// a reference to each [`Tool::call`]. Tools use it to discover the
/// working directory, session ID, temp directory, and any
/// domain-specific extensions the host application has attached.
///
/// # Extensions
///
/// The [`extensions`](ToolContext::extensions) field lets downstream
/// crates inject typed, arbitrary data without coupling the framework to
/// any particular use case. Use [`ToolContext::set_extension`] and
/// [`ToolContext::get_extension`] to store and retrieve values keyed by
/// their Rust type.
///
/// # Passing host state to tools
///
/// Host state (a working directory, configuration, channels — anything the
/// embedding application owns that a tool needs but the model does not send)
/// reaches a tool through one of two supported options, depending on who
/// dispatches the tool.
///
/// **Option A — the engine dispatches (`BareLoop`): a middleware injector.**
/// Each dispatch's `ToolContext` is built fresh by the engine with only
/// `session_id` set — `cwd`, `is_non_interactive`, and `extensions` all start
/// at their defaults, and host code never holds that value. The one place it
/// can be augmented is a [`ToolMiddleware`], which receives `&mut`
/// [`ToolDispatchContext`] — whose public `tool_context` field exists for
/// exactly this — before the pipeline core invokes the tool. Install with
/// [`BareLoop::set_pipeline`] (which also shares the loop's own tool registry
/// with the pipeline core), registering the injector first so later
/// middlewares see the enriched context. Without a pipeline installed,
/// engine-dispatched tools observe no host state at all.
///
/// **Option B — the host dispatches: build the context yourself.**
/// When your code calls [`Tool::call`] directly (tests, scripts, simple
/// integrations), construct the `ToolContext`, set fields and extensions, and
/// pass it in — you own the value end to end, so nothing else is required.
///
/// Also possible, but discouraged: registering a wrapper tool that clones the
/// incoming context, installs the extension, and delegates (works because
/// `ToolContext` is `Clone`, but costs per-tool wiring plus a `Tool`-trait
/// forwarding layer), and capturing state in the tool struct at construction
/// (which bypasses the context entirely). Prefer the middleware — it is the
/// same idea at the engine's single sanctioned interception point. The
/// `host-state` example in the repository's `examples/` directory demonstrates
/// both options end to end.
///
/// [`BareLoop`]: crate::engine::BareLoop
/// [`BareLoop::set_pipeline`]: crate::engine::BareLoop::set_pipeline
/// [`Tool::call`]: crate::tool::Tool::call
/// [`ToolDispatchContext`]: crate::middleware::ToolDispatchContext
/// [`ToolDispatchContext::tool_context`]: crate::middleware::ToolDispatchContext::tool_context
/// [`ToolMiddleware`]: crate::middleware::ToolMiddleware
///
/// # Example — Option B: host-built context (runnable)
///
/// ```
/// use loopctl::tool::ToolContext;
///
/// #[derive(Clone)]
/// struct MyConfig {
///     verbose: bool,
/// }
///
/// let mut ctx = ToolContext::default();
/// ctx.set_extension(MyConfig { verbose: true });
/// assert!(ctx.get_extension::<MyConfig>().expect("just set").verbose);
/// ```
///
/// # Example — Option A: middleware injector (compiles, does not run)
///
/// ```rust,no_run
/// use std::future::Future;
/// use std::pin::Pin;
/// use std::sync::Arc;
/// use loopctl::config::SessionConfig;
/// use loopctl::engine::BareLoop;
/// use loopctl::middleware::ToolDispatchContext;
/// use loopctl::middleware::ToolDispatchResult;
/// use loopctl::middleware::ToolMiddleware;
/// use loopctl::middleware::ToolPipeline;
/// # use loopctl::testing::MockApiClient;
/// # use loopctl::tool::ToolRegistry;
///
/// #[derive(Clone)]
/// struct MyConfig {
///     verbose: bool,
/// }
///
/// struct Injector;
///
/// impl ToolMiddleware for Injector {
///     fn name(&self) -> &str {
///         "my-injector"
///     }
///     fn dispatch<'a>(
///         &'a self,
///         ctx: &'a mut ToolDispatchContext,
///         next: &'a ToolPipeline,
///     ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
///         ctx.tool_context.set_extension(MyConfig { verbose: true });
///         Box::pin(async move { next.dispatch(ctx).await })
///     }
/// }
///
/// let client = MockApiClient::new("demo");
/// let mut agent =
///     BareLoop::new(Arc::new(client), ToolRegistry::new(), SessionConfig::default());
/// agent
///     .set_pipeline(ToolPipeline::builder().with_middleware(Injector))
///     .expect("static pipeline composition");
/// ```
#[derive(Clone)]
pub struct ToolContext {
    /// Current working directory for the agent session.
    ///
    /// Tools that interact with the filesystem (e.g., read, write, glob)
    /// should resolve relative paths against this directory. Defaults to
    /// `"."` (the process's current directory).
    pub cwd: String,

    /// Unique identifier for the agent session.
    ///
    /// A UUID v4 generated when the context is created. Useful for
    /// correlating log entries, naming temp files, or isolating
    /// per-session state.
    pub session_id: uuid::Uuid,

    /// Path to the session's temporary directory.
    ///
    /// Each session gets its own temp directory so tools can write
    /// intermediate files without colliding with other sessions. Defaults
    /// to [`std::env::temp_dir`].
    pub temp_dir: String,

    /// Whether the agent is running in non-interactive (headless) mode.
    ///
    /// When `true`, tools should avoid prompting the user for input or
    /// confirmation and instead use sensible defaults or fail with
    /// [`ToolError::Permission`]. Defaults to `false`.
    pub is_non_interactive: bool,

    /// Additional key-value context supplied by the caller.
    ///
    /// The host application can populate this map with arbitrary string
    /// data (e.g., project name, environment, user ID) that tools can
    /// read at invocation time. Defaults to empty.
    pub user_context: HashMap<String, String>,

    /// Domain-specific extensions for downstream crates.
    ///
    /// A type-map (`TypeId` → `Arc<dyn Any + Send + Sync>`) that lets
    /// tool authors attach structured data without modifying the
    /// [`ToolContext`] struct. Use [`ToolContext::set_extension`] to
    /// insert and [`ToolContext::get_extension`] to retrieve.
    pub extensions: HashMap<std::any::TypeId, Arc<dyn std::any::Any + Send + Sync>>,
}

impl ToolContext {
    /// Store a typed extension value in the context.
    ///
    /// Inserts `val` keyed by its [`TypeId`](std::any::TypeId). If a value
    /// of the same type already exists it is replaced. Call this during
    /// session setup, before tools are invoked.
    ///
    /// Extensions are the recommended way to pass host-application-specific
    /// configuration to tools without coupling the framework to any
    /// particular use case.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// ctx.set_extension(Arc::new(my_callback_fn));
    /// ctx.set_extension(MyConfig { max_retries: 3 });
    /// ```
    pub fn set_extension<T: 'static + Send + Sync>(&mut self, val: T) {
        self.extensions
            .insert(std::any::TypeId::of::<T>(), Arc::new(val));
    }

    /// Retrieve a previously stored extension by type.
    ///
    /// Returns `Some(&T)` if [`set_extension`](ToolContext::set_extension)
    /// was called with a value of the same type `T`, or `None` otherwise.
    /// Called from inside [`Tool::call`] implementations.
    ///
    /// The returned reference borrows from the [`ToolContext`] and remains
    /// valid for as long as the context is alive.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(cb) = ctx.get_extension::<MyCallback>() {
    ///     cb.invoke();
    /// }
    /// ```
    #[must_use]
    pub fn get_extension<T: 'static>(&self) -> Option<&T> {
        self.extensions
            .get(&std::any::TypeId::of::<T>())
            .and_then(|arc| arc.downcast_ref::<T>())
    }
}

impl fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolContext")
            .field("cwd", &self.cwd)
            .field("session_id", &self.session_id)
            .field("temp_dir", &self.temp_dir)
            .field("is_non_interactive", &self.is_non_interactive)
            .field("user_context", &self.user_context)
            .field("extensions", &format!("{} entries", self.extensions.len()))
            .finish()
    }
}

/// Produce a context with sensible defaults.
///
/// Creates a ready-to-use [`ToolContext`] suitable for most agent
/// sessions. The defaults are:
///
/// | Field               | Default                              |
/// |---------------------|--------------------------------------|
/// | `cwd`               | `"."` (process current directory)    |
/// | `session_id`        | new UUID v4                          |
/// | `temp_dir`          | [`std::env::temp_dir`]               |
/// | `is_non_interactive`| `false`                              |
/// | `user_context`      | empty [`HashMap`]                    |
/// | `extensions`        | empty [`HashMap`]                    |
///
/// # Example
///
/// ```rust
/// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
///
/// let ctx = ToolContext::default();
/// assert_eq!(ctx.cwd, ".");
/// assert!(!ctx.is_non_interactive);
/// ```
impl Default for ToolContext {
    fn default() -> Self {
        Self {
            cwd: ".".to_string(),
            session_id: uuid::Uuid::new_v4(),
            temp_dir: std::env::temp_dir().to_string_lossy().to_string(),
            is_non_interactive: false,
            user_context: HashMap::new(),
            extensions: HashMap::new(),
        }
    }
}

/// The trait that all agent tools must implement.
///
/// Framework-level tool interface. Concrete tools (defined in
/// downstream crates like `dch-tools`) implement this trait, and the
/// [`ToolRegistry`] manages dynamic lookup by name.
///
/// The trait uses a `Pin<Box<dyn Future>>` return type for [`call`](Tool::call)
/// to be maximally compatible with both `async fn` bodies and manually
/// constructed futures, keeping the trait object-safe and free of
/// lifetime issues that `async fn` in traits can introduce.
///
/// # Lifecycle
///
/// ```text
/// registry.register(tool)
///   → tool.schema()            [sent to LLM as tool definition]
///   → tool.call(input, ctx)    [invoked when LLM selects this tool]
///   → tool.system_prompt()     [optional: injected into system message]
/// ```
///
/// # Data flow
///
/// ```text
/// LLM selects tool "read_file"
///   → ToolRegistry::get("read_file")
///   → PermissionCheck gate
///     → Allow → Tool::call(input, ctx)
///                → Ok(ToolOutput) or Err(ToolError)
///     → Deny  → Err(ToolError::Permission)
/// ```
///
/// # Implementing
///
/// At a minimum you must provide [`name`](Tool::name),
/// [`description`](Tool::description), [`schema`](Tool::schema), and
/// [`call`](Tool::call). The trait supplies default implementations for
/// [`is_concurrency_safe`](Tool::is_concurrency_safe),
/// [`is_safe_for_concurrent_execution`](Tool::is_safe_for_concurrent_execution),
/// [`is_read_only`](Tool::is_read_only), and
/// [`system_prompt`](Tool::system_prompt).
///
/// # Required methods
///
/// | Method               | Purpose                                    |
/// |----------------------|--------------------------------------------|
/// | `name`               | Unique identifier used for registry lookup |
/// | `description`        | Human-readable summary sent to the LLM     |
/// | `schema`             | JSON Schema for input validation           |
/// | `call`               | Core execution logic                       |
///
/// # Provided methods
///
/// | Method                              | Default   | Purpose                             |
/// |-------------------------------------|-----------|-------------------------------------|
/// | `is_concurrency_safe`               | `false`   | Static concurrency flag             |
/// | `is_safe_for_concurrent_execution`  | delegates | Per-input concurrency check         |
/// | `is_read_only`                      | `false`   | Side-effect flag for permission     |
/// | `system_prompt`                     | `None`    | Extra LLM context for this tool     |
///
/// # Example
///
/// ```rust,ignore
/// struct ReadFileTool;
///
/// impl Tool for ReadFileTool {
///     fn name(&self) -> &str { "read_file" }
///     fn description(&self) -> &str { "Read a file from disk" }
///     fn schema(&self) -> ToolSchema {
///         ToolSchema {
///             tool: "read_file".into(),
///             description: "Read a file from disk".into(),
///             input_schema: json!({
///                 "type": "object",
///                 "properties": { "path": { "type": "string" } },
///                 "required": ["path"]
///             }),
///         }
///     }
///     fn call(&self, input: Value, ctx: &ToolContext)
///         -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>>
///     {
///         let path = input["path"].as_str().unwrap_or_default().to_string();
///         Box::pin(async move {
///             let full = std::path::Path::new(&ctx.cwd).join(&path);
///             let content = tokio::fs::read_to_string(full).await?;
///             Ok(ToolOutput::text(content))
///         })
///     }
///     fn is_read_only(&self) -> bool { true }
/// }
/// ```
pub trait Tool: Send + Sync {
    /// The tool's unique name (e.g., `"read_file"`, `"bash"`).
    ///
    /// Used as the lookup key in [`ToolRegistry`] and as the `name` field
    /// in the [`ToolSchema`] sent to the LLM. Must be non-empty and
    /// stable across the lifetime of the session.
    ///
    /// # Invariants
    ///
    /// Implementations must ensure the returned value is:
    /// - Non-empty and free of leading/trailing whitespace.
    /// - Unique within a given [`ToolRegistry`].
    /// - Stable — calling this method multiple times yields the same value.
    fn name(&self) -> &str;

    /// Human-readable description for the LLM.
    ///
    /// Sent to the LLM alongside [`name`](Tool::name) and
    /// [`schema`](Tool::schema). A clear description helps the model
    /// choose the right tool and construct correct arguments.
    ///
    /// # Guidelines
    ///
    /// - Keep it to one or two sentences.
    /// - Describe *what* the tool does, not how it's implemented.
    /// - Include constraints (e.g., "reads files under 10 MB") when relevant.
    fn description(&self) -> &str;

    /// Return the JSON Schema descriptor for this tool.
    ///
    /// Called by the agent loop to assemble the tool list sent to the LLM.
    /// The returned [`ToolSchema`] must have a `name` matching
    /// [`Tool::name`].
    ///
    /// # Validation
    ///
    /// The `input_schema` field should conform to JSON Schema Draft 07.
    /// While the framework does not enforce schema validity, an invalid
    /// schema will cause the LLM to produce malformed tool calls.
    fn schema(&self) -> ToolSchema;

    /// Invoke the tool with the given input and context.
    ///
    /// Main execution entry point. The `input` is a
    /// [`Value`] (typically a JSON object) matching the tool's
    /// [`ToolSchema::input_schema`]. The [`ToolContext`] provides
    /// session-level data such as working directory and extensions.
    ///
    /// Called by the agent loop when the LLM selects this tool. Returns
    /// `Ok(ToolOutput)` on success or `Err(ToolError)` on failure.
    ///
    /// # Return type
    ///
    /// The `Pin<Box<dyn Future>>` return type maximises compatibility
    /// with both `async fn` bodies and manually constructed futures,
    /// keeping the trait object-safe and free of lifetime issues.
    ///
    /// # Errors
    ///
    /// Implementations should return [`ToolError`] variants that
    /// accurately describe the failure mode so the agent loop can decide
    /// on an appropriate recovery strategy.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// fn call(&self, input: Value, ctx: &ToolContext)
    ///     -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>>
    /// {
    ///     let path = input["path"].as_str().unwrap_or_default().to_string();
    ///     let cwd = ctx.cwd.clone();
    ///     Box::pin(async move {
    ///         let full = std::path::Path::new(&cwd).join(&path);
    ///         let content = tokio::fs::read_to_string(full).await?;
    ///         Ok(ToolOutput::text(content))
    ///     })
    /// }
    /// ```
    fn call(
        &self,
        input: Value,
        context: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>>;

    /// Whether this tool is safe to run concurrently with itself.
    ///
    /// Return `true` if the tool has no mutable shared state and can be
    /// safely invoked in parallel (e.g., a pure read-only tool). The
    /// agent loop uses this to decide which tools can run simultaneously.
    /// Defaults to `false`.
    ///
    /// # When called
    ///
    /// Queried by [`ToolRegistry::concurrent_safe_tools`] during session
    /// setup and by the agent loop's parallel execution planner before
    /// dispatching multiple tool calls in a single turn.
    ///
    /// # See also
    ///
    /// - [`is_safe_for_concurrent_execution`](Tool::is_safe_for_concurrent_execution)
    ///   — input-dependent version of this check.
    fn is_concurrency_safe(&self) -> bool {
        false
    }

    /// Dynamic concurrency check based on the specific input.
    ///
    /// Override to provide input-dependent safety checks — for example,
    /// a file-write tool might allow concurrent writes to *different*
    /// files but not to the same file. Falls back to
    /// [`is_concurrency_safe`](Tool::is_concurrency_safe) when not
    /// overridden.
    ///
    /// # When called
    ///
    /// Called by the agent loop's parallel execution planner immediately
    /// before dispatching each tool call. Receives the same `input`
    /// [`Value`] that will be passed to [`Tool::call`].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // A write tool that allows concurrent writes to different files:
    /// fn is_safe_for_concurrent_execution(&self, input: &Value) -> bool {
    ///     // Track which files are in-flight; only block on same-file writes
    ///     true // simplified — real impl checks the `path` field
    /// }
    /// ```
    fn is_safe_for_concurrent_execution(&self, _input: &Value) -> bool {
        self.is_concurrency_safe()
    }

    /// A stable key identifying the resource this call touches, for conflict
    /// detection during parallel dispatch.
    ///
    /// Two eligible calls whose `resource_key` returns **equal `Some(_)`** are
    /// treated as conflicting and serialized (put in different waves). Returning
    /// `None` (the default) means "no declarable resource — conflict only on the
    /// static [`is_concurrency_safe`](Tool::is_concurrency_safe) flag."
    ///
    /// Implement this for tools that touch a named resource (a file path, a
    /// shell working directory, a job id). Return the resource identifier, e.g.
    /// the canonical `path` for a file tool. The framework never calls this for
    /// a call whose [`is_safe_for_concurrent_execution`](Tool::is_safe_for_concurrent_execution)
    /// returned `false` — such calls are serialized unconditionally.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// fn resource_key(&self, input: &Value) -> Option<String> {
    ///     input.get("path").and_then(|p| p.as_str()).map(String::from)
    /// }
    /// ```
    fn resource_key(&self, _input: &Value) -> Option<String> {
        None
    }

    /// Whether this tool only reads data and has no side effects.
    ///
    /// Read-only tools (e.g., file readers, search tools) can be
    /// auto-approved by permission gates. Defaults to `false`.
    ///
    /// # When called
    ///
    /// Checked by the agent loop's permission gate before prompting the
    /// user. If this returns `true`, the gate may skip the user
    /// confirmation step and return [`PermissionCheck::Allow`] directly.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // A grep-like tool that only reads files:
    /// fn is_read_only(&self) -> bool { true }
    /// ```
    fn is_read_only(&self) -> bool {
        false
    }

    /// Optional extra system prompt injected when this tool is available.
    ///
    /// Some tools benefit from giving the LLM additional context — for
    /// example, a shell tool might include "Prefer bash one-liners". The
    /// agent loop appends this to the system message when the tool is
    /// registered. Returns `None` by default.
    ///
    /// # When called
    ///
    /// Queried once during session setup when the agent loop assembles
    /// the system prompt. The returned string (if any) is appended after
    /// the base prompt and before any per-turn dynamic instructions.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // A bash tool that advises the LLM on style:
    /// fn system_prompt(&self) -> Option<String> {
    ///     Some("Prefer single-line bash commands over multi-line scripts.".into())
    /// }
    /// ```
    fn system_prompt(&self) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "testing")]
    use crate::engine::RunConfig;
    #[cfg(feature = "testing")]
    use crate::engine::core::Loop;
    use serde_json::json;

    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "Echoes back the input"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "echo".into(),
                description: "Echoes back the input".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "message": { "type": "string" } },
                    "required": ["message"]
                }),
            }
        }
        fn call(
            &self,
            input: Value,
            _context: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            let msg = input
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Box::pin(async move { Ok(ToolOutput::text(msg)) })
        }
        fn is_concurrency_safe(&self) -> bool {
            true
        }
        fn is_read_only(&self) -> bool {
            true
        }
    }

    struct FailTool;

    impl Tool for FailTool {
        fn name(&self) -> &'static str {
            "fail"
        }
        fn description(&self) -> &'static str {
            "Always fails"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: "fail".into(),
                description: "Always fails".into(),
                input_schema: json!({ "type": "object", "properties": {} }),
            }
        }
        fn call(
            &self,
            _input: Value,
            _context: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            Box::pin(async { Err(ToolError::Execution("always fails".into())) })
        }
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolOutput::text("hello");
        assert!(!result.is_error);
        assert_eq!(result.text_content(), "hello");
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolOutput::error_text("something went wrong");
        assert!(result.is_error);
        assert_eq!(result.text_content(), "something went wrong");
    }

    #[test]
    fn test_tool_result_from_string() {
        let result: ToolOutput = "hello".into();
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_result_from_str() {
        let result: ToolOutput = "hello".into();
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_error_not_found() {
        let err = ToolError::not_found("missing", &["tool_a", "tool_b"]);
        assert!(err.to_string().contains("missing"));
        assert!(err.to_string().contains("tool_a"));
    }

    #[test]
    fn test_tool_error_not_found_empty() {
        let err = ToolError::not_found("missing", &[]);
        assert!(err.to_string().contains("none registered"));
    }

    #[test]
    fn test_tool_error_not_found_many() {
        let tools: Vec<&str> = (0..15)
            .map(|i| Box::leak(format!("tool_{i}").into_boxed_str()) as &str)
            .collect();
        let err = ToolError::not_found("missing", &tools);
        assert!(err.to_string().contains("and 5 more"));
    }

    #[test]
    fn test_tool_context_default() {
        let ctx = ToolContext::default();
        assert_eq!(ctx.cwd, ".");
        assert!(!ctx.is_non_interactive);
    }

    #[test]
    fn test_permission_check_allow() {
        let check = PermissionCheck::allow();
        assert!(check.is_allow());
        assert!(!check.is_deny());
    }

    #[test]
    fn test_permission_check_deny() {
        let check = PermissionCheck::deny("unsafe");
        assert!(check.is_deny());
        assert!(!check.is_allow());
    }

    #[test]
    fn test_permission_check_ask() {
        let check = PermissionCheck::ask("Run this?");
        assert!(check.is_ask());
    }

    #[test]
    fn test_permission_check_modify() {
        let check = PermissionCheck::modify(json!({"safe": true}));
        assert!(check.is_modify());
    }

    #[test]
    fn test_tool_schema_serialization() {
        let schema = ToolSchema {
            tool: "test".into(),
            description: "A test tool".into(),
            input_schema: json!({"type": "object"}),
        };
        let json = serde_json::to_string(&schema).unwrap();
        let back: ToolSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema.tool, back.tool);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = ToolRegistry::new();
        assert!(registry.is_empty());

        registry.register(EchoTool);
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("echo"));

        let tool = registry.get("echo").unwrap();
        assert_eq!(tool.name(), "echo");
    }

    #[test]
    fn test_registry_get_missing() {
        let registry = ToolRegistry::new();
        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn test_registry_all_schemas() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        let schemas = registry.all_schemas();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].tool, "echo");
    }

    #[test]
    fn test_registry_tool_names() {
        let mut registry = ToolRegistry::new();
        registry.register(FailTool);
        registry.register(EchoTool);
        let names = registry.tool_names();
        assert_eq!(names, vec!["echo", "fail"]);
    }

    #[test]
    fn test_registry_concurrent_safe_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry.register(FailTool);
        let safe = registry.concurrent_safe_tools();
        assert_eq!(safe.len(), 1);
        assert_eq!(safe[0].name(), "echo");
    }

    #[tokio::test]
    async fn test_echo_tool_call() {
        let tool = EchoTool;
        let ctx = ToolContext::default();
        let result = tool.call(json!({"message": "hello"}), &ctx).await;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.text_content(), "hello");
    }

    #[tokio::test]
    async fn test_fail_tool_call() {
        let tool = FailTool;
        let ctx = ToolContext::default();
        let result = tool.call(json!({}), &ctx).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_trait_concurrency_default() {
        let tool = FailTool;
        assert!(!tool.is_concurrency_safe());
        assert!(!tool.is_safe_for_concurrent_execution(&json!({})));
    }

    #[test]
    fn test_tool_trait_read_only_default() {
        let tool = FailTool;
        assert!(!tool.is_read_only());
    }

    #[test]
    fn test_tool_trait_system_prompt_default() {
        let tool = EchoTool;
        assert!(tool.system_prompt().is_none());
    }

    #[test]
    fn tool_output_structured_round_trip() {
        use crate::structured::StructuredOutput;

        #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
        struct Action {
            tool: String,
            args: serde_json::Value,
        }

        impl StructuredOutput for Action {
            fn name() -> &'static str {
                "action"
            }
            fn schema() -> serde_json::Value {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "tool": { "type": "string" },
                        "args": {}
                    },
                    "required": ["tool", "args"]
                })
            }
        }

        let action = Action {
            tool: "write".to_string(),
            args: serde_json::json!({"path": "/a"}),
        };
        let out = ToolOutput::structured(&action);
        assert!(!out.is_error);
        assert!(out.structured_value().is_some());
        let back: Action = out.structured_as().expect("should round-trip");
        assert_eq!(back, action);
    }

    #[test]
    fn tool_output_structured_value_none_for_non_json() {
        let out = ToolOutput::text("not json");
        assert!(out.structured_value().is_none());
    }

    #[test]
    fn tool_output_structured_with_plain_serialize() {
        #[derive(serde::Serialize)]
        struct Count {
            n: u32,
        }

        let out = ToolOutput::structured(&Count { n: 7 });
        assert!(!out.is_error);
        let v = out.structured_value().expect("should be valid JSON");
        assert_eq!(v["n"], 7);
    }

    #[test]
    fn tool_output_structured_primitive() {
        let out = ToolOutput::structured(&42u32);
        assert!(!out.is_error);
        let v = out.structured_value().expect("should parse");
        assert_eq!(v, 42);
    }

    #[test]
    fn tool_output_structured_value_for_multipart_is_none() {
        use crate::message::{ToolContent, ToolContentPart};
        let out = ToolOutput::success(ToolContent::Multipart(vec![ToolContentPart::Text {
            text: "a".into(),
        }]));
        assert!(out.structured_value().is_none());
    }

    #[test]
    fn tool_output_structured_as_returns_none_when_type_mismatches() {
        use crate::structured::StructuredOutput;

        #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
        struct Target {
            name: String,
        }

        impl StructuredOutput for Target {
            fn name() -> &'static str {
                "target"
            }
            fn schema() -> serde_json::Value {
                serde_json::json!({
                    "type": "object",
                    "properties": { "name": { "type": "string" } },
                    "required": ["name"]
                })
            }
        }

        // Feed JSON that has a different shape than Target expects.
        let out = ToolOutput::text(r#"{"count": 5}"#);
        let result: Option<Target> = out.structured_as();
        assert!(result.is_none(), "mismatched shape should not deserialize");
    }

    #[test]
    fn display_hint_with_hint_sets_field() {
        let out = ToolOutput::text("x").with_hint(DisplayHint::Json);
        assert_eq!(out.display_hint, Some(DisplayHint::Json));
    }

    #[test]
    fn display_hint_default_is_none_for_all_constructors() {
        assert_eq!(ToolOutput::text("x").display_hint, None);
        assert_eq!(ToolOutput::error_text("x").display_hint, None);
        assert_eq!(
            ToolOutput::success(MessageToolContent::Text("x".to_string())).display_hint,
            None
        );
        assert_eq!(
            ToolOutput::error(MessageToolContent::Text("x".to_string())).display_hint,
            None
        );
        let from_string: ToolOutput = String::from("x").into();
        assert_eq!(from_string.display_hint, None);
        let from_str: ToolOutput = "x".into();
        assert_eq!(from_str.display_hint, None);
    }

    #[test]
    fn display_hint_with_hint_does_not_mutate_payload_or_error_flag() {
        let out = ToolOutput::error_text("boom").with_hint(DisplayHint::Suppress);
        assert!(out.is_error, "is_error untouched");
        assert_eq!(out.text_content(), "boom", "payload untouched");
        assert_eq!(out.display_hint, Some(DisplayHint::Suppress));
    }

    #[test]
    fn display_hint_code_carries_language() {
        let out = ToolOutput::text("fn main() {}").with_hint(DisplayHint::Code {
            language: "rust".into(),
        });
        match &out.display_hint {
            Some(DisplayHint::Code { language }) => assert_eq!(language, "rust"),
            other => panic!("expected Code{{language}}, got {other:?}"),
        }
    }

    #[test]
    fn display_hint_markdown_round_trips() {
        let out = ToolOutput::text("# hi").with_hint(DisplayHint::Markdown);
        assert_eq!(out.display_hint, Some(DisplayHint::Markdown));
    }

    #[test]
    fn display_hint_is_clone_and_eq() {
        let diff = DisplayHint::Diff;
        assert_eq!(diff, DisplayHint::Diff);
        let code = DisplayHint::Code {
            language: "py".into(),
        };
        assert_eq!(
            code,
            DisplayHint::Code {
                language: "py".into()
            }
        );
        assert_ne!(DisplayHint::Text, DisplayHint::Diff);
    }

    #[test]
    fn display_hint_serde_round_trip_tagged() {
        let json = serde_json::to_string(&DisplayHint::Diff).expect("Diff serializes");
        assert_eq!(json, "{\"type\":\"diff\"}");
        let parsed: DisplayHint = serde_json::from_str(&json).expect("Diff deserializes");
        assert_eq!(parsed, DisplayHint::Diff);

        let code_json = serde_json::to_string(&DisplayHint::Code {
            language: "rust".into(),
        })
        .expect("");
        assert_eq!(code_json, "{\"type\":\"code\",\"language\":\"rust\"}");
        let code_parsed: DisplayHint = serde_json::from_str(&code_json).expect("");
        assert_eq!(
            code_parsed,
            DisplayHint::Code {
                language: "rust".into()
            }
        );
    }

    #[test]
    fn from_tool_output_forwards_display_hint() {
        let out = ToolOutput::text("diff body").with_hint(DisplayHint::Diff);
        let result: ToolDispatchResult = out.into();
        assert_eq!(result.display_hint, Some(DisplayHint::Diff));
    }

    #[test]
    fn tool_dispatch_result_ok_err_from_tool_error_carry_none() {
        use std::time::Duration;
        assert_eq!(
            ToolDispatchResult::ok("t", "x".into(), Duration::ZERO).display_hint,
            None
        );
        assert_eq!(
            ToolDispatchResult::err("t", "x".into(), Duration::ZERO).display_hint,
            None
        );
    }

    #[test]
    fn from_tool_output_constructor_forwards_hint() {
        use std::time::Duration;
        let out = ToolOutput::text("json body").with_hint(DisplayHint::Json);
        let result = ToolDispatchResult::from_tool_output("t", out, Duration::ZERO);
        assert_eq!(result.display_hint, Some(DisplayHint::Json));
    }

    #[test]
    fn from_result_forwards_hint_on_ok_and_none_on_err() {
        use std::time::Duration;
        let ok: Result<ToolOutput, ToolError> =
            Ok(ToolOutput::text("x").with_hint(DisplayHint::Markdown));
        assert_eq!(
            ToolDispatchResult::from_result("t", ok, Duration::ZERO).display_hint,
            Some(DisplayHint::Markdown)
        );

        let err: Result<ToolOutput, ToolError> = Err(ToolError::not_found("t", &[]));
        assert_eq!(
            ToolDispatchResult::from_result("t", err, Duration::ZERO).display_hint,
            None
        );
    }

    #[test]
    fn soft_error_preserves_display_hint() {
        let soft_err = ToolOutput::error_text("conflict in foo.rs").with_hint(DisplayHint::Diff);
        assert!(soft_err.is_error, "fixture is a soft error");
        let result: ToolDispatchResult = soft_err.into();
        assert_eq!(
            result.display_hint,
            Some(DisplayHint::Diff),
            "soft-error outputs preserve their hint (only hard errors / panics drop it)"
        );
    }

    /// A stub tool that returns a fixed hinted `ToolOutput`.
    #[cfg(feature = "testing")]
    struct HintedTool {
        name: &'static str,
        output_text: &'static str,
        hint: Option<DisplayHint>,
    }

    #[cfg(feature = "testing")]
    impl Tool for HintedTool {
        fn name(&self) -> &'static str {
            self.name
        }
        fn description(&self) -> &'static str {
            "stub for display-hint threading tests"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: self.name.into(),
                description: "stub for display-hint threading tests".into(),
                input_schema: json!({"type": "object", "properties": {}}),
            }
        }
        fn call(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, ToolError>> + Send + '_>>
        {
            let out = match self.hint.clone() {
                Some(h) => ToolOutput::text(self.output_text).with_hint(h),
                None => ToolOutput::text(self.output_text),
            };
            Box::pin(async move { Ok(out) })
        }
    }

    /// Captures every `on_tool_post` snapshot for later assertion.
    #[cfg(feature = "testing")]
    #[derive(Default)]
    struct PostCapture {
        posts: std::sync::Mutex<Vec<crate::observer::ToolPostContext>>,
    }

    #[cfg(feature = "testing")]
    impl crate::observer::LoopObserver for PostCapture {
        fn name(&self) -> &'static str {
            "post-capture"
        }
        fn on_tool_post(&self, ctx: &crate::observer::ToolPostContext) {
            crate::error::recover_guard(self.posts.lock()).push(ctx.clone());
        }
    }

    #[cfg(feature = "testing")]
    fn hinted_dispatch_setup(
        tool_name: &'static str,
        output_text: &'static str,
        hint: Option<DisplayHint>,
    ) -> (
        crate::engine::BareLoop<crate::testing::MockApiClient>,
        std::sync::Arc<PostCapture>,
    ) {
        use crate::testing::{MockApiClient, MockResponse, MockToolCall};

        let client = MockApiClient::new("test-model").with_responses(vec![
            MockResponse {
                text: String::new(),
                tool_call: Some(MockToolCall {
                    id: "call_1".into(),
                    name: tool_name.into(),
                    input: json!({}),
                }),
                stop_reason: "tool_use".into(),
            },
            MockResponse {
                text: "done".into(),
                tool_call: None,
                stop_reason: "end_turn".into(),
            },
        ]);

        let mut registry = crate::tool::ToolRegistry::new();
        registry.register(HintedTool {
            name: tool_name,
            output_text,
            hint,
        });

        let mut agent = crate::engine::BareLoop::new(
            std::sync::Arc::new(client),
            registry,
            crate::config::SessionConfig::default(),
        );
        let capture = std::sync::Arc::new(PostCapture::default());
        agent.register_observer(capture.clone());
        (agent, capture)
    }

    #[cfg(feature = "testing")]
    #[tokio::test]
    async fn hint_reaches_observer_on_normal_path() {
        let (mut agent, capture) = hinted_dispatch_setup(
            "diff_tool",
            "@@ -1 +1 @@\n-old\n+new",
            Some(DisplayHint::Diff),
        );
        agent
            .run("edit the file", &RunConfig::default())
            .await
            .unwrap();

        let posts = crate::error::recover_guard(capture.posts.lock()).clone();
        assert_eq!(posts.len(), 1, "exactly one tool call this turn");
        assert_eq!(
            posts[0].display_hint,
            Some(DisplayHint::Diff),
            "the hint set by the tool must reach the observer"
        );
        assert_eq!(posts[0].tool, "diff_tool");
    }

    #[cfg(feature = "testing")]
    #[tokio::test]
    async fn no_hint_tool_yields_none_at_observer() {
        let (mut agent, capture) = hinted_dispatch_setup("plain", "just text", None);
        agent.run("go", &RunConfig::default()).await.unwrap();

        let posts = crate::error::recover_guard(capture.posts.lock()).clone();
        assert_eq!(posts.len(), 1);
        assert_eq!(
            posts[0].display_hint, None,
            "no hint set → None at observer"
        );
    }

    #[cfg(feature = "testing")]
    #[tokio::test]
    async fn suppress_hint_keeps_full_payload_into_conversation() {
        let (mut agent, capture) =
            hinted_dispatch_setup("reader", "the quick brown fox", Some(DisplayHint::Suppress));
        agent.run("read it", &RunConfig::default()).await.unwrap();

        let posts = crate::error::recover_guard(capture.posts.lock()).clone();
        assert_eq!(posts[0].display_hint, Some(DisplayHint::Suppress));

        // The full payload reached the conversation (loop semantics unaffected).
        let conv = agent.conversation();
        let tool_result_text: String = conv
            .iter()
            .flat_map(|m| {
                m.parts.iter().filter_map(|p| match p {
                    crate::message::MessagePart::ToolResult { output, .. } => {
                        Some(output.to_string())
                    }
                    _ => None,
                })
            })
            .collect();
        assert!(
            tool_result_text.contains("the quick brown fox"),
            "Suppress hint must not strip the payload from the conversation: {tool_result_text}"
        );
    }
}

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

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::message::ToolResult as MessageToolResult;

// ===================================================
// ToolSchema
// ===================================================

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

// ===================================================
// ToolOutput
// ===================================================

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
/// [`MessageToolResult`] which is either a simple [`String`] or a
/// structured list of [`ToolResultPart`](crate::message::ToolResultPart)
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
///   → Err(ToolError)                                 [hard failure]
/// ```
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// The payload returned by the tool.
    ///
    /// Holds either a plain-text string or structured multi-part content.
    /// Use [`ToolOutput::text_content`] to extract text regardless of the
    /// variant.
    pub payload: MessageToolResult,

    /// Whether this result represents an error.
    ///
    /// When `true`, the agent loop treats the result as a soft failure —
    /// the tool executed without panicking, but the output describes what
    /// went wrong. Defaults to `false` for results created via
    /// [`ToolOutput::success`] or [`ToolOutput::text`].
    pub is_error: bool,
}

impl ToolOutput {
    /// Create a successful result with the given content.
    ///
    /// Called by tool implementations when the invocation succeeds and
    /// the output is a structured [`MessageToolResult`] value.
    /// Sets [`is_error`](ToolOutput::is_error) to `false`.
    ///
    /// This is the most general success constructor — it accepts any type
    /// that converts into [`MessageToolResult`]. For simple text results,
    /// prefer the more concise [`ToolOutput::text`] helper.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    /// use loopctl::message::ToolResult as MessageToolResult;
    ///
    /// let result = ToolOutput::success(MessageToolResult::Text("done".into()));
    /// assert!(!result.is_error);
    /// ```
    pub fn success(payload: impl Into<MessageToolResult>) -> Self {
        Self {
            payload: payload.into(),
            is_error: false,
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
    pub fn error(payload: impl Into<MessageToolResult>) -> Self {
        Self {
            payload: payload.into(),
            is_error: true,
        }
    }

    /// Create a successful plain-text result.
    ///
    /// Convenience wrapper around [`ToolOutput::success`] that converts
    /// the input string into a [`MessageToolResult::Text`] variant.
    /// This is the most common constructor for simple tool outputs.
    ///
    /// # When to use
    ///
    /// Use this when the tool produces a simple string response — for
    /// example, file contents, a computed value, or a status message.
    /// For structured multi-part responses, use [`ToolOutput::success`]
    /// directly with a [`MessageToolResult::Multipart`] value.
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
    /// the input string into a [`MessageToolResult::Text`] variant.
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

    /// Extract all text content from the result, regardless of structure.
    ///
    /// Inspects the [`payload`](ToolOutput::payload) field and returns a
    /// flat [`String`]:
    /// - For [`MessageToolResult::Text`], returns the string directly.
    /// - For [`MessageToolResult::Multipart`], joins all
    ///   [`ToolResultPart::Text`](crate::message::ToolResultPart::Text)
    ///   parts with newlines, discarding non-text parts.
    ///
    /// Useful when the consumer only cares about the textual payload and
    /// does not need to handle structured content.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    /// use loopctl::message::ToolResult as MessageToolResult;
    /// use loopctl::message::ToolResultPart;
    ///
    /// let result = ToolOutput::text("hello world");
    /// assert_eq!(result.text_content(), "hello world");
    ///
    /// // Works with structured content too:
    /// let structured = ToolOutput::success(MessageToolResult::Multipart(vec![
    ///     ToolResultPart::Text { text: "line 1".into() },
    ///     ToolResultPart::Text { text: "line 2".into() },
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
            MessageToolResult::Text(s) => s.clone(),
            MessageToolResult::Multipart(parts) => {
                use crate::message::ToolResultPart;
                parts
                    .iter()
                    .filter_map(|p| match p {
                        ToolResultPart::Text { text } => Some(text.clone()),
                        ToolResultPart::Image { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
    }
}

impl From<String> for ToolOutput {
    /// Convert a [`String`] into a successful text [`ToolOutput`].
    ///
    /// Enables `let result: ToolOutput = s.into()` where `s: String`.
    /// Equivalent to calling [`ToolOutput::text`]. Sets
    /// [`is_error`](ToolOutput::is_error) to `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let s = String::from("hello");
    /// let result: ToolOutput = s.into();
    /// assert_eq!(result.text_content(), "hello");
    /// ```
    fn from(s: String) -> Self {
        Self::text(s)
    }
}

impl From<&str> for ToolOutput {
    /// Convert a `&str` into a successful text [`ToolOutput`].
    ///
    /// Enables `let result: ToolOutput = "ok".into()`. Equivalent to
    /// calling [`ToolOutput::text`]. Sets
    /// [`is_error`](ToolOutput::is_error) to `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let result: ToolOutput = "ok".into();
    /// assert_eq!(result.text_content(), "ok");
    /// ```
    fn from(s: &str) -> Self {
        Self::text(s)
    }
}

// ===================================================
// ToolError
// ===================================================

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
///
/// The framework selects the appropriate variant based on runtime
/// conditions and the agent's current state.
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

// ===================================================
// ToolContext
// ===================================================

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
/// # Example
///
/// ```rust,ignore
/// let mut ctx = ToolContext::default();
/// ctx.cwd = "/tmp/workspace".into();
/// ctx.set_extension(MyConfig { verbose: true });
///
/// // Inside a tool:
/// if let Some(cfg) = ctx.get_extension::<MyConfig>() {
///     println!("verbose={}", cfg.verbose);
/// }
/// ```
///
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
    /// Format the context for debug output, summarizing extensions.
    ///
    /// Produces a human-readable struct dump. The `extensions` field is
    /// rendered as a count (e.g., `"3 entries"`) rather than printing every
    /// entry, because extensions can be arbitrarily large and their types
    /// may not implement [`Debug`](std::fmt::Debug).
    ///
    /// # Example output
    ///
    /// ```text
    /// ToolContext { cwd: "/tmp/ws", session_id: 0123-4567-..., temp_dir: "/tmp", is_non_interactive: false, user_context: {}, extensions: 2 entries }
    /// ```
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

impl Default for ToolContext {
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
// ===================================================
// PermissionCheck
// ===================================================

/// Result of a permission check before tool execution.
///
/// Before invoking [`Tool::call`], the agent loop can run a permission
/// gate that returns one of four outcomes: allow, deny, ask the user,
/// or modify the input. This lets host applications enforce safety
/// policies without modifying individual tool implementations.
///
/// # Lifecycle
///
/// ```text
/// tool.call(input, ctx)
///   → permission_gate(input)
///     → Allow          [proceed with original input]
///     → Deny           [return ToolError::Permission]
///     → Ask { prompt } [prompt user, then Allow or Deny]
///     → Modify { .. }  [proceed with modified input]
/// ```
///
/// # Decision tree
///
/// ```text
///           ┌─────────────────┐
///           │ Permission gate │
///           └───────┬─────────┘
///            ┌──────┼──────────┐
///            ▼      ▼          ▼
///         Allow   Deny      ┌ Ask ──┐
///           │      │        ▼       │
///           │      │       user     │
///           │      │      approves  │
///           │      │       |        │
///           │      │       ▼        ▼
///           ▼      ▼     Allow     Deny
///      Tool::call  Err(Permission)
/// ```
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
    /// [`ToolError::Permission`] with the given
    /// `reason` so the LLM can react accordingly.
    Deny {
        /// Explanation of why the invocation was blocked.
        ///
        /// Forwarded to the LLM as part of the error message so it can
        /// adjust its next action.
        reason: String,
    },

    /// Prompt the user for approval before proceeding.
    ///
    /// In interactive sessions the agent loop should present `prompt` to
    /// the user and then treat the response as either [`Allow`](PermissionCheck::Allow)
    /// or [`Deny`](PermissionCheck::Deny).
    ///
    Ask {
        /// The question to present to the user.
        ///
        /// Should clearly describe the action the tool is about to take
        /// and any potential side effects.
        prompt: String,
    },

    /// Modify the tool's input before execution.
    ///
    /// The agent loop should invoke [`Tool::call`] with `modified_input`
    /// instead of the original input. Useful for sanitising paths,
    /// redacting secrets, or injecting default values.
    Modify {
        /// The sanitized or rewritten input to pass to [`Tool::call`].
        ///
        /// Must conform to the tool's [`ToolSchema::input_schema`].
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
    /// provided `modified_input` before invoking [`Tool::call`]. Useful
    /// for sanitising paths, redacting secrets, or injecting default
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
    /// [`Tool::call`] without further processing.
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
    /// converting to [`ToolError::Permission`].
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
    /// non-interactive mode ([`ToolContext::is_non_interactive`]), the
    /// loop typically treats an [`Ask`](PermissionCheck::Ask) as a
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

// ===================================================
// Tool trait
// ===================================================

/// The trait that all agent tools must implement.
///
/// This is the framework-level tool interface. Concrete tools (defined in
/// downstream crates like `dch-tools`) implement this trait, and the
/// [`ToolRegistry`] manages dynamic lookup by name.
///
/// The trait uses a `Pin<Box<dyn Future>>` return type for [`call`](Tool::call)
/// to be maximally compatible with both `async fn` and manual `Future`
/// implementations, without requiring `async_fn_in_trait` stabilisation.
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
/// | Method                              | Default | Purpose                            |
/// |-------------------------------------|---------|------------------------------------|
/// | `is_concurrency_safe`             | `false` | Static concurrency flag            |
/// | `is_safe_for_concurrent_execution`| delegates| Per-input concurrency check        |
/// | `is_read_only`                    | `false` | Side-effect flag for permission    |
/// | `system_prompt`                   | `None`  | Extra LLM context for this tool    |
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
///
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
    /// This is the main execution entry point. The `input` is a
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
    /// without requiring `async_fn_in_trait` stabilisation.
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

// ===================================================
// ToolRegistry
// ===================================================

/// Registry of available tools for dynamic lookup by name.
///
/// The agent loop creates a [`ToolRegistry`] at session start, registers
/// all available tools via [`register`](ToolRegistry::register), and then
/// uses [`get`](ToolRegistry::get) to dispatch invocations when the LLM
/// selects a tool by name. The registry also provides bulk accessors for
/// tool schemas and concurrency-safe tool lists.
///
/// # Data flow
///
/// ```text
/// ┌──────────────┐
/// │ Session init │
/// └──────┬───────┘
///        ▼
/// ToolRegistry::new()
///   → register(tool_1)
///   → register(tool_2)
///   → ...
///        │
///        ▼
/// all_schemas()  ──→  LLM API request (tool definitions)
/// get("name")    ──→  Tool::call(input, ctx)  (dispatch)
/// concurrent_safe_tools() ──→ parallel execution planner
/// ```
///
/// # Thread safety
///
/// The registry itself is not `Sync` — it is created once during session
/// setup and then accessed immutably during tool dispatch. If you need
/// cross-thread sharing, wrap it in an `Arc<RwLock<ToolRegistry>>`.
///
/// # Example
///
/// ```rust,ignore
/// let mut registry = ToolRegistry::new();
/// registry.register(ReadFileTool);
/// registry.register(WriteFileTool);
///
/// // Dispatch an invocation
/// let tool = registry.get("read_file").expect("tool exists");
/// let result = tool.call(input, &ctx).await;
///
/// // Send schemas to the LLM
/// let schemas = registry.all_schemas();
/// ```
pub struct ToolRegistry {
    /// Internal name → tool map.
    ///
    /// Each entry is a `Box<dyn Tool>` keyed by its [`Tool::name`].
    /// Populated by [`register`](ToolRegistry::register) and queried by
    /// [`get`](ToolRegistry::get).
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    ///
    /// The registry starts with no tools. Use [`register`](ToolRegistry::register)
    /// to add tools before the agent loop begins processing turns.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool, replacing any previous tool with the same name.
    ///
    /// Called during session setup, before any turns are processed. If a
    /// tool with the same [`Tool::name`] already exists it is silently
    /// replaced.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// registry.register(ReadFileTool);
    /// registry.register(WriteFileTool);
    /// ```
    pub fn register(&mut self, tool: impl Tool + 'static) {
        let name = tool.name().to_string();
        self.tools.insert(name, Box::new(tool));
    }

    /// Look up a tool by name.
    ///
    /// Returns `Some(&dyn Tool)` if a tool with the given name was
    /// previously [`register`](ToolRegistry::register)ed, or `None`
    /// otherwise. Called by the agent loop when dispatching an LLM tool
    /// call.
    ///
    /// The returned reference borrows from the registry and is valid for
    /// as long as the registry is alive.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(tool) = registry.get("read_file") {
    ///     let result = tool.call(input, &ctx).await;
    /// }
    /// ```
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(std::convert::AsRef::as_ref)
    }

    /// Check whether a tool with the given name is registered.
    ///
    /// Useful for pre-flight validation before attempting
    /// [`get`](ToolRegistry::get). Returns `true` if the name maps to a
    /// registered tool.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if registry.contains("bash") {
    ///     // Safe to call registry.get("bash")
    /// }
    /// ```
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Collect [`ToolSchema`] descriptors for all registered tools.
    ///
    /// Called by the agent loop to build the tool list sent to the LLM
    /// at the start of each session (or turn, if the tool set changes).
    /// The order is unspecified.
    ///
    /// Each schema is freshly constructed via [`Tool::schema`], so the
    /// caller does not need to worry about stale data.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let schemas = registry.all_schemas();
    /// for schema in &schemas {
    ///     println!("  - {}: {}", schema.name, schema.description);
    /// }
    /// ```
    #[must_use]
    pub fn all_schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|t| t.schema()).collect()
    }

    /// Return all registered tool names, sorted alphabetically.
    ///
    /// Useful for diagnostics, logging, and building error messages
    /// in [`ToolError::not_found`].
    #[must_use]
    pub fn tool_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of registered tools.
    ///
    /// Used by the framework and by
    /// [`is_empty`](ToolRegistry::is_empty). Returns `0` for a freshly
    /// created registry.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// assert_eq!(registry.len(), 3); // three tools registered
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry contains no tools.
    ///
    /// Defaults to `self.len() == 0`. The agent loop typically checks
    /// this during startup to ensure at least one tool is available.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let registry = ToolRegistry::new();
    /// assert!(registry.is_empty());
    /// registry.register(MyTool);
    /// assert!(!registry.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Return references to all tools that are concurrency-safe.
    ///
    /// Filters by [`Tool::is_concurrency_safe`] returning `true`. Used
    /// by the agent loop to decide which tools can be invoked in parallel
    /// during a single turn.
    #[must_use]
    pub fn concurrent_safe_tools(&self) -> Vec<&dyn Tool> {
        self.tools
            .values()
            .map(std::convert::AsRef::as_ref)
            .filter(|t| t.is_concurrency_safe())
            .collect()
    }
}

impl Default for ToolRegistry {
    /// Produce an empty registry (equivalent to [`ToolRegistry::new`]).
    ///
    /// Allows `ToolRegistry` to be used in contexts that require
    /// [`Default`], such as struct initialization with `..Default::default()`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::tool::{ToolOutput, ToolError, ToolSchema, ToolContext, PermissionCheck, ToolRegistry};
    ///
    /// let registry = ToolRegistry::default();
    /// assert!(registry.is_empty());
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================
// FnTool adapter
// ===================================================

/// Type alias for an async tool function pointer.
///
/// Matches the signature used by concrete tools in downstream crates:
/// `fn(Value, &ToolContext) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'static>>`.
///
/// Stored in the `f` field of [`FnTool`] to adapt function-pointer-based
/// tool definitions to the [`Tool`] trait.
pub type ToolFn =
    fn(
        Value,
        &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'static>>;

/// Type alias for a dynamic concurrency check function.
///
/// Takes a reference to the tool input [`Value`] and returns `true` if
/// the tool is safe to run concurrently with that specific input. Used
/// by [`FnTool::with_concurrency_check`] to override the static
/// [`Tool::is_concurrency_safe`] flag on a per-call basis.
pub type ConcurrencyCheckFn = fn(&Value) -> bool;

/// Adapter that wraps a function pointer as a [`Tool`] trait implementation.
///
/// Use [`FnTool`] when you have a standalone async function that implements
/// tool logic and want to register it without defining a dedicated struct.
/// The adapter wraps the function pointer so it can be stored in a
/// [`ToolRegistry`] alongside any other [`Tool`] implementation.
///
/// For complex tools with internal state, implement [`Tool`] directly on a
/// struct instead.
///
/// # Builder API
///
/// [`FnTool`] supports a builder pattern for optional properties:
///
/// ```rust,ignore
/// let tool = FnTool::new("my_tool".into(), "Does a thing".into(),
///     json!({"type": "object", "properties": {"text": {"type": "string"}}}),
///     my_tool as ToolFn)
///     .concurrency_safe()                 // mark as safe for parallel execution
///     .read_only()                        // mark as side-effect-free
///     .with_system_prompt("...".into());  // inject extra LLM context
///
/// let mut registry = ToolRegistry::new();
/// registry.register(tool);
/// ```
///
/// # Builder flow
///
/// ```text
/// FnTool::new(name, desc, schema, f)
///   │
///   ├─ .concurrency_safe()         → sets is_concurrency_safe = true
///   ├─ .with_concurrency_check(fn) → sets per-input check
///   ├─ .read_only()                → sets is_read_only = true
///   └─ .with_system_prompt(s)      → sets system_prompt = Some(s)
/// ```
///
/// # Example
///
/// ```rust,ignore
/// fn my_tool(input: Value, _ctx: &ToolContext)
///     -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'static>>
/// {
///     let text = input.get("text").unwrap().to_string();
///     Box::pin(async move { Ok(ToolOutput::text(text)) })
/// }
///
/// let tool = FnTool::new("my_tool".into(), "Does a thing".into(),
///     json!({"type": "object", "properties": {"text": {"type": "string"}}}),
///     my_tool as ToolFn)
///     .concurrency_safe()
///     .read_only();
///
/// let mut registry = ToolRegistry::new();
/// registry.register(tool);
/// ```
pub struct FnTool {
    /// The tool's unique name identifier.
    ///
    /// Must match the name used in the [`ToolSchema`] and serves as the
    /// [`ToolRegistry`] lookup key. Set at construction time via
    /// [`FnTool::new`].
    pub name: String,

    /// Human-readable description for the LLM.
    ///
    /// Sent to the LLM as part of the [`ToolSchema`]. A clear
    /// description improves tool selection accuracy.
    pub description: String,

    /// JSON Schema describing the tool's input parameters.
    ///
    /// Must be a valid JSON Schema object. Embedded in the
    /// [`ToolSchema`] returned by [`Tool::schema`].
    pub input_schema: Value,

    /// The function pointer that implements the tool's core logic.
    ///
    /// Called by [`Tool::call`] with the LLM-supplied input and the
    /// session's [`ToolContext`]. Must return a pinned, `Send` future
    /// producing a `Result<ToolOutput, ToolError>`.
    pub tool_fn: ToolFn,

    /// Whether this tool is safe to run concurrently with itself.
    ///
    /// Set via the [`concurrency_safe`](FnTool::concurrency_safe) builder
    /// method. Defaults to `false`. When `true`, the agent loop may
    /// invoke this tool in parallel with other concurrent-safe tools.
    pub is_concurrency_safe: bool,

    /// Optional dynamic concurrency check function.
    ///
    /// When set via [`with_concurrency_check`](FnTool::with_concurrency_check),
    /// this function is called with the tool input to decide per-invocation
    /// concurrency safety. Overrides the static
    /// [`is_concurrency_safe`](FnTool::is_concurrency_safe) flag when present.
    pub concurrency_check_fn: Option<ConcurrencyCheckFn>,

    /// Whether this tool only reads data (no side effects).
    ///
    /// Set via the [`read_only`](FnTool::read_only) builder method.
    /// Defaults to `false`. Read-only tools can be auto-approved by
    /// permission gates.
    pub is_read_only: bool,

    /// Optional extra system prompt injected when this tool is available.
    ///
    /// Set via [`with_system_prompt`](FnTool::with_system_prompt).
    /// The agent loop appends this to the system message. Defaults to
    /// `None`.
    pub system_prompt: Option<String>,
}

impl FnTool {
    /// Create a new function-pointer tool with the given name, description,
    /// schema, and implementation function.
    ///
    /// All optional properties default to their "off" values:
    /// `is_concurrency_safe → false`, `concurrency_check_fn → None`,
    /// `is_read_only → false`, `system_prompt → None`. Use the builder
    /// methods to enable them.
    ///
    /// # Arguments
    ///
    /// | Argument        | Type      | Description                                    |
    /// |-----------------|-----------|------------------------------------------------|
    /// | `name`          | `String`  | Unique tool identifier, used as registry key    |
    /// | `description`   | `String`  | Human-readable summary sent to the LLM          |
    /// | `input_schema`  | `Value`   | JSON Schema for the tool's parameters           |
    /// | `f`             | [`ToolFn`] | The async function implementing tool logic     |
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tool = FnTool::new(
    ///     "grep".into(),
    ///     "Search files for a pattern".into(),
    ///     json!({"type": "object", "properties": {"pattern": {"type": "string"}}}),
    ///     my_grep_fn as ToolFn,
    /// );
    /// ```
    pub fn new(name: String, description: String, input_schema: Value, tool_fn: ToolFn) -> Self {
        Self {
            name,
            description,
            input_schema,
            tool_fn,
            is_concurrency_safe: false,
            concurrency_check_fn: None,
            is_read_only: false,
            system_prompt: None,
        }
    }

    /// Builder: mark this tool as concurrency-safe.
    ///
    /// Sets [`is_concurrency_safe`](FnTool::is_concurrency_safe) to
    /// `true`, signalling that the agent loop may invoke this tool in
    /// parallel with other concurrent-safe tools.
    ///
    /// # When to use
    ///
    /// Call this for tools that are pure functions or read-only — for
    /// example, a file-reading tool or a math calculator. Do *not* call
    /// this for tools that mutate shared state or write to the filesystem.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tool = FnTool::new(/* ... */)
    ///     .concurrency_safe();
    /// ```
    #[must_use]
    pub fn concurrency_safe(mut self) -> Self {
        self.is_concurrency_safe = true;
        self
    }

    /// Builder: set a dynamic concurrency check function.
    ///
    /// The provided function is called with the tool input on each
    /// invocation. If it returns `true`, the tool may run concurrently
    /// for that specific input. Overrides the static
    /// [`is_concurrency_safe`](FnTool::is_concurrency_safe) flag.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// fn can_run_concurrently(input: &Value) -> bool {
    ///     // Only safe if writing to different files
    ///     input.get("append").is_none()
    /// }
    /// let tool = FnTool::new(/* ... */).with_concurrency_check(can_run_concurrently);
    /// ```
    #[must_use]
    pub fn with_concurrency_check(mut self, check_fn: ConcurrencyCheckFn) -> Self {
        self.concurrency_check_fn = Some(check_fn);
        self
    }

    /// Builder: mark this tool as read-only (no side effects).
    ///
    /// Sets [`is_read_only`](FnTool::is_read_only) to `true`. Read-only
    /// tools can be auto-approved by permission gates and are generally
    /// safe to run without user confirmation.
    ///
    /// # When to use
    ///
    /// Call this for tools that only read data — file readers, search
    /// tools, calculators. Do *not* call this for tools that write files,
    /// execute commands, or modify external state.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tool = FnTool::new(/* ... */)
    ///     .read_only();
    /// ```
    #[must_use]
    pub fn read_only(mut self) -> Self {
        self.is_read_only = true;
        self
    }

    /// Builder: set an optional extra system prompt for this tool.
    ///
    /// The agent loop appends this string to the system message when the
    /// tool is registered, giving the LLM additional context about how
    /// to use the tool effectively.
    ///
    /// # When to use
    ///
    /// Use this when a tool benefits from usage hints or style guidance
    /// — for example, a shell tool might set a prompt like "Prefer
    /// single-line bash commands" to steer the LLM's behavior.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tool = FnTool::new(/* ... */)
    ///     .with_system_prompt("Always use absolute paths.".into());
    /// ```
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = Some(prompt);
        self
    }
}

/// [`Tool`] trait implementation for [`FnTool`].
///
/// Delegates each trait method to the corresponding field or function
/// pointer stored in the [`FnTool`] adapter. This is the glue that lets
/// function-pointer-based tools participate in the trait system without
/// any wrapper overhead.
///
/// # Delegation map
///
/// | Trait method                               | Delegates to                                 |
/// |--------------------------------------------|----------------------------------------------|
/// | [`Tool::name`]                             | [`FnTool::name`] field accessor              |
/// | [`Tool::description`]                      | [`FnTool::description`] accessor             |
/// | [`Tool::schema`]                           | Clones fields into [`ToolSchema`]            |
/// | [`Tool::call`]                             | internal function pointer                    |
/// | [`Tool::is_concurrency_safe`]              | [`FnTool::is_concurrency_safe`]              |
/// | [`Tool::is_safe_for_concurrent_execution`] | [`FnTool::concurrency_check_fn`] or fallback |
/// | [`Tool::is_read_only`]                     | [`FnTool::is_read_only`]                     |
/// | [`Tool::system_prompt`]                    | [`FnTool::system_prompt`] clone              |
impl Tool for FnTool {
    /// Return the tool name from the [`FnTool::name`] field.
    ///
    /// This is a trivial field accessor — the name was set at construction
    /// time via [`FnTool::new`] and does not change for the lifetime of
    /// the adapter. The returned slice borrows from `self`.
    fn name(&self) -> &str {
        &self.name
    }

    /// Return the tool description from the [`FnTool::description`] field.
    ///
    /// Like [`name`](FnTool::name), this is a field accessor for the value
    /// provided at construction time. The description is sent to the LLM as
    /// part of the [`ToolSchema`] to help it choose the right tool.
    fn description(&self) -> &str {
        &self.description
    }

    /// Build a [`ToolSchema`] from the stored fields.
    ///
    /// Assembles the [`FnTool::name`], [`FnTool::description`], and
    /// [`FnTool::input_schema`] into a [`ToolSchema`] suitable for
    /// sending to the LLM. The fields are cloned so the returned schema
    /// is independent of `self`.
    ///
    /// # When called
    ///
    /// Invoked by the agent loop when assembling the list of tool
    /// definitions to send in an LLM API request. Typically called once
    /// per session (or per turn if the tool set changes dynamically).
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }

    /// Delegate execution to the stored function pointer.
    ///
    /// Invokes the stored function pointer with the provided `input` and `context`,
    /// returning the pinned future directly. The function pointer owns
    /// the full future lifecycle — the `'static` bound on [`ToolFn`]
    /// ensures the future does not borrow from the tool adapter itself.
    ///
    /// # When called
    ///
    /// Called by the agent loop after the LLM selects this tool by name
    /// and the permission gate (if any) returns
    /// [`PermissionCheck::Allow`].
    fn call(
        &self,
        input: Value,
        context: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        (self.tool_fn)(input, context)
    }

    /// Return the static concurrency-safety flag.
    ///
    /// Reads [`FnTool::is_concurrency_safe`], which is set via the
    /// [`concurrency_safe`](FnTool::concurrency_safe) builder method.
    ///
    /// This is a *static* flag — it does not consider the specific input.
    /// For input-dependent checks, see
    /// [`is_safe_for_concurrent_execution`](Tool::is_safe_for_concurrent_execution).
    fn is_concurrency_safe(&self) -> bool {
        self.is_concurrency_safe
    }

    /// Dynamic concurrency check using the optional check function.
    ///
    /// If [`FnTool::concurrency_check_fn`] is set (via
    /// [`with_concurrency_check`](FnTool::with_concurrency_check)),
    /// delegates to it and returns its result. Otherwise falls back to
    /// the static [`is_concurrency_safe`](Tool::is_concurrency_safe) flag.
    ///
    /// This allows tools to express fine-grained concurrency policies —
    /// for example, allowing parallel reads to different files while
    /// serializing writes to the same file.
    fn is_safe_for_concurrent_execution(&self, input: &Value) -> bool {
        self.concurrency_check_fn
            .map_or(self.is_concurrency_safe, |f| f(input))
    }

    /// Return the read-only flag from [`FnTool::is_read_only`].
    ///
    /// Set via the [`read_only`](FnTool::read_only) builder method.
    /// When `true`, the agent loop's permission gate may auto-approve
    /// invocations without prompting the user, since the tool has no
    /// observable side effects.
    fn is_read_only(&self) -> bool {
        self.is_read_only
    }

    /// Clone and return the optional system prompt from [`FnTool::system_prompt`].
    ///
    /// The agent loop appends this string to the system message when the
    /// tool is registered, giving the LLM additional context about how to
    /// use the tool effectively. Returns `None` if no prompt was set via
    /// [`with_system_prompt`](FnTool::with_system_prompt).
    fn system_prompt(&self) -> Option<String> {
        self.system_prompt.clone()
    }
}

// ===================================================
// Tests
// ===================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
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
        fn name(&self) -> &str {
            "fail"
        }
        fn description(&self) -> &str {
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
}

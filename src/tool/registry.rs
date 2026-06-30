//! Tool registry and function-pointer adapter.
//!
//! [`ToolRegistry`] for dynamic tool lookup by name,
//! and [`FnTool`] (along with [`ToolFn`] and [`ConcurrencyCheckFn`]) for
//! wrapping plain async function pointers as [`Tool`] trait
//! implementations.

use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use super::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};

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
    /// `Box<dyn Tool>` keyed by [`Tool::name`].
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
    /// tool with the same [`Tool::name`] already exists
    /// it is silently replaced.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// registry.register(ReadFileTool);
    /// registry.register(WriteFileTool);
    /// ```
    pub fn register(&mut self, tool: impl Tool + 'static) {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            tracing::warn!(
                tool = %name,
                "overwriting previously registered tool with the same name"
            );
        }
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
    /// Each schema is freshly constructed via [`Tool::schema`],
    /// so the caller does not need to worry about stale data.
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
    /// Filters by [`Tool::is_concurrency_safe`]
    /// returning `true`. Used by the agent loop to decide which tools can
    /// be invoked in parallel during a single turn.
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
/// [`Tool::is_concurrency_safe`] flag
/// on a per-call basis.
pub type ConcurrencyCheckFn = fn(&Value) -> bool;

/// Adapter that wraps a function pointer as a [`Tool`] trait implementation.
///
/// Use [`FnTool`] when you have a standalone async function that implements
/// tool logic and want to register it without defining a dedicated struct.
/// The adapter wraps the function pointer so it can be stored in a
/// [`ToolRegistry`] alongside any other [`Tool`] implementation.
///
/// For complex tools with internal state, implement [`Tool`]
/// directly on a struct instead.
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
    /// Must match the name used in [`ToolSchema`] and [`ToolRegistry`] lookup.
    pub name: String,
    /// Sent to the LLM as part of the [`ToolSchema`].
    pub description: String,
    /// Must be a valid JSON Schema object.
    pub input_schema: Value,
    /// Called by [`Tool::call`] with the LLM-supplied
    /// input and session's [`ToolContext`].
    pub tool_fn: ToolFn,
    /// Set via [`concurrency_safe`](FnTool::concurrency_safe). Defaults to `false`.
    pub is_concurrency_safe: bool,
    /// When set, overrides the static [`is_concurrency_safe`](FnTool::is_concurrency_safe) flag.
    pub concurrency_check_fn: Option<ConcurrencyCheckFn>,
    /// Set via [`read_only`](FnTool::read_only). Defaults to `false`.
    pub is_read_only: bool,
    /// Set via [`with_system_prompt`](FnTool::with_system_prompt). Defaults to `None`.
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
    /// - `name` — Unique tool identifier, used as the registry key.
    /// - `description` — Human-readable summary sent to the LLM.
    /// - `input_schema` — JSON Schema describing the tool's parameters.
    /// - `tool_fn` — The async function implementing the tool logic.
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
/// pointer stored in the [`FnTool`] adapter.
impl Tool for FnTool {
    /// Returns the tool's unique identifier stored in [`name`](FnTool::name).
    fn name(&self) -> &str {
        &self.name
    }

    /// Returns the human-readable description stored in
    /// [`description`](FnTool::description).
    fn description(&self) -> &str {
        &self.description
    }

    /// Builds a [`ToolSchema`] from the stored name, description, and
    /// input schema.
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }

    /// Delegates execution to the stored [`tool_fn`](FnTool::tool_fn)
    /// function pointer, forwarding the input and context unchanged.
    fn call(
        &self,
        input: Value,
        context: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        (self.tool_fn)(input, context)
    }

    /// Returns the static concurrency-safety flag set via
    /// [`concurrency_safe`](FnTool::concurrency_safe).
    fn is_concurrency_safe(&self) -> bool {
        self.is_concurrency_safe
    }

    /// Delegates to [`concurrency_check_fn`](FnTool::concurrency_check_fn)
    /// when set, otherwise falls back to the static
    /// [`is_concurrency_safe`](Tool::is_concurrency_safe) flag.
    fn is_safe_for_concurrent_execution(&self, input: &Value) -> bool {
        self.concurrency_check_fn
            .map_or(self.is_concurrency_safe, |f| f(input))
    }

    /// Returns whether this tool only reads data and has no side effects,
    /// as configured via [`read_only`](FnTool::read_only).
    fn is_read_only(&self) -> bool {
        self.is_read_only
    }

    /// Returns the optional system prompt set via
    /// [`with_system_prompt`](FnTool::with_system_prompt).
    fn system_prompt(&self) -> Option<String> {
        self.system_prompt.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple tool function for testing duplicate registration (L1 fix).
    fn test_tool_fn(
        _input: Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'static>> {
        Box::pin(async { Ok(ToolOutput::text("ok")) })
    }

    /// A simple tool for testing duplicate registration (L1 fix).
    fn make_tool(name: &str) -> FnTool {
        FnTool::new(
            name.into(),
            "test tool".into(),
            serde_json::json!({"type": "object"}),
            test_tool_fn,
        )
    }

    #[test]
    fn register_duplicate_overwrites_previous() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("my_tool"));
        assert_eq!(registry.len(), 1);

        // Registering with the same name should overwrite, not add.
        registry.register(make_tool("my_tool"));
        assert_eq!(
            registry.len(),
            1,
            "duplicate registration should not increase count"
        );
    }

    #[test]
    fn register_different_names_adds_both() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("tool_a"));
        registry.register(make_tool("tool_b"));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn register_overwrite_uses_new_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("my_tool"));

        // After overwriting, the tool should still be callable.
        registry.register(make_tool("my_tool"));
        assert!(registry.get("my_tool").is_some());
    }
}

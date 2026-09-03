//! Structured output — request guaranteed-schema JSON responses from the model.
//!
//! This module provides:
//!
//! - [`StructuredOutput`] — a type-level trait that names a type, exposes its
//!   JSON Schema, and deserializes from a `serde_json::Value`.
//! - [`ResponseFormat`] + [`RequestOptions`] — the request-side carrier that
//!   tells the provider to constrain output to the schema.
//! - [`StructuredError`] — errors raised by the structured-output machinery.
//! - [`request_structured`] — a convenience helper that hides the
//!   options/extraction dance behind a single generic call.
//! - [`request_structured_prompted`] — the prompted fallback for
//!   providers that reject or ignore `response_format` (most local
//!   models): the schema rides the system prompt, the output is parsed
//!   leniently, and one corrective retry is made on failure.
//!
//! # Choosing an extraction path
//!
//! | Path | Mechanism | Use when |
//! |---|---|---|
//! | [`request_structured`] | provider-native `response_format` | the provider enforces schemas (OpenAI strict mode; Anthropic forced tool) |
//! | [`request_structured_prompted`] | schema in the system prompt + lenient parse + one corrective retry | the provider rejects or ignores `response_format` — most local models behind Ollama/vLLM |
//! | tool-forced output | a forced tool call carrying the schema | not provided by this crate |
//!
//! Native enforcement beats prompt guidance; prefer
//! [`request_structured`] wherever the provider supports it.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use loopctl::structured::{StructuredOutput, request_structured};
//! use serde::{Deserialize, Serialize};
//! use serde_json::json;
//!
//! #[derive(Debug, Serialize, Deserialize, PartialEq)]
//! pub struct Action {
//!     pub tool: String,
//!     pub args: serde_json::Value,
//! }
//!
//! impl StructuredOutput for Action {
//!     fn name() -> &'static str { "action" }
//!     fn schema() -> serde_json::Value {
//!         json!({
//!             "type": "object",
//!             "properties": {
//!                 "tool": { "type": "string" },
//!                 "args": {}
//!             },
//!             "required": ["tool", "args"],
//!             "additionalProperties": false
//!         })
//!     }
//! }
//!
//! // let action: Action = request_structured(&client, messages, system).await?;
//! ```

use serde::de::Error as _;
use tracing::Instrument as _;

use crate::api::ApiClient;
use crate::message::Message;

/// A type that can be requested from the model as a JSON-schema-conformant
/// response, and deserialized from the model's output.
///
/// This is a *type-level* trait (like `serde::Serialize`), not a provider
/// trait — it is never used as a trait object. Implement it on any `Sized +
/// Send + 'static` type that also implements `serde::de::DeserializeOwned`.
///
/// The schema returned by [`schema`](Self::schema) is injected into the
/// provider request (OpenAI `response_format` / Anthropic forced tool); the
/// model's output is parsed back via [`from_value`](Self::from_value).
///
/// # Manual schema vs derive
///
/// By default, implement `schema()` by returning a `serde_json::json!`
/// literal — no extra dependency, matching how
/// [`ToolSchema::input_schema`](crate::tool::ToolSchema::input_schema) is
/// authored today.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::structured::StructuredOutput;
/// use serde::{Deserialize, Serialize};
/// use serde_json::json;
///
/// #[derive(Debug, Serialize, Deserialize)]
/// pub struct Action {
///     pub tool: String,
///     pub args: serde_json::Value,
/// }
///
/// impl StructuredOutput for Action {
///     fn name() -> &'static str { "action" }
///     fn schema() -> serde_json::Value {
///         json!({
///             "type": "object",
///             "properties": {
///                 "tool": { "type": "string" },
///                 "args": {}
///             },
///             "required": ["tool", "args"],
///             "additionalProperties": false
///         })
///     }
/// }
/// ```
pub trait StructuredOutput: Sized + Send + 'static {
    /// Logical name for the schema.
    ///
    /// Used verbatim as the OpenAI `json_schema.name` field and as the
    /// synthesized Anthropic forced-tool name. Expected to match
    /// `^[a-zA-Z0-9_-]+$` (alphanumeric, underscore, hyphen only) — the
    /// convention OpenAI's schema identifiers follow; neither the crate
    /// nor the providers enforce it.
    fn name() -> &'static str;

    /// The JSON Schema (Draft 07) describing the desired output object.
    ///
    /// The schema is injected into the provider request: OpenAI emits it as
    /// `response_format.json_schema.schema`; Anthropic uses it as the
    /// forced tool's `input_schema`. The model's output is expected to
    /// conform to this schema — the [`from_value`](Self::from_value) method
    /// then deserializes it into `Self`.
    ///
    /// Implement this by returning a `serde_json::json!({ … })` literal
    /// (matching the pattern used by
    /// [`ToolSchema::input_schema`](crate::tool::ToolSchema::input_schema)).
    fn schema() -> serde_json::Value;

    /// Deserialize an instance from the model's JSON output.
    ///
    /// The default implementation is `serde_json::from_value::<Self>(v)`,
    /// which is correct for any `Self: DeserializeOwned`. Override only for
    /// post-processing (e.g. trimming, defaults, cross-field validation).
    ///
    /// # Errors
    ///
    /// Returns [`StructuredError::Deserialize`] if the value does not match
    /// the type (and, by construction, the schema).
    fn from_value(v: serde_json::Value) -> Result<Self, StructuredError>
    where
        Self: serde::de::DeserializeOwned,
    {
        serde_json::from_value(v).map_err(StructuredError::Deserialize)
    }
}

/// A request to constrain the model's output to a named JSON schema.
///
/// Construct with [`ResponseFormat::from_type`] from any [`StructuredOutput`]
/// type, or manually from a raw schema + name via [`ResponseFormat::new`].
/// Passed to the provider via [`RequestOptions`].
#[derive(Debug, Clone)]
pub struct ResponseFormat {
    /// Logical name for the schema.
    ///
    /// Copied from [`StructuredOutput::name`] when constructed via
    /// [`from_type`](Self::from_type). Used as the OpenAI `json_schema.name`
    /// and the Anthropic forced-tool name. The `^[a-zA-Z0-9_-]+$` shape is
    /// the expected convention, not an enforced one.
    pub name: String,

    /// The JSON Schema the model's output must satisfy.
    ///
    /// Injected into the provider request verbatim: OpenAI emits it as
    /// `response_format.json_schema.schema`; Anthropic uses it as the
    /// forced tool's `input_schema`.
    pub schema: serde_json::Value,

    /// Whether to enforce the schema server-side ("strict" mode).
    ///
    /// When `true` (the default), OpenAI guarantees the output conforms to
    /// the schema. Anthropic and Gemini cannot express strict mode and
    /// reject a strict request before sending it
    /// ([`ApiError::config_validation`](crate::api::error::ApiError::config_validation)).
    pub strict: bool,
}

impl ResponseFormat {
    /// Build a [`ResponseFormat`] from a [`StructuredOutput`] type.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use loopctl::structured::{ResponseFormat, StructuredOutput};
    ///
    /// let rf = ResponseFormat::from_type::<MyOutput>();
    /// assert_eq!(rf.name, MyOutput::name());
    /// assert_eq!(rf.schema, MyOutput::schema());
    /// assert!(rf.strict);
    /// ```
    #[must_use]
    pub fn from_type<T: StructuredOutput>() -> Self {
        Self {
            name: T::name().to_string(),
            schema: T::schema(),
            strict: true,
        }
    }

    /// Build a [`ResponseFormat`] from raw parts.
    ///
    /// Use this when the schema is constructed dynamically (e.g. at runtime
    /// from config or a database) rather than from a static type. For the
    /// common case of deriving the format from a `StructuredOutput` type,
    /// prefer [`from_type`](Self::from_type).
    ///
    /// Sets `strict: true` by default — the model is constrained server-side
    /// where supported (OpenAI strict mode, Anthropic ignores the flag).
    #[must_use]
    pub fn new(name: impl Into<String>, schema: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            schema,
            strict: true,
        }
    }
}

/// How tightly the provider must constrain tool-call output to the
/// registered schemas.
///
/// Default [`ToolConstraint::None`] reproduces the behaviour of versions
/// prior to this field: the provider advertises tool schemas as-is and the
/// model's tool calls are unconstrained. [`ToolConstraint::Strict`] asks
/// the provider to make malformed tool calls structurally impossible using
/// its native strict-tool mode (OpenAI `strict: true` with schema
/// tightening; Anthropic / Gemini tightened `input_schema` / `parameters`).
///
/// The enum is `#[non_exhaustive]`: future variants may be added
/// non-breakingly, and the `Grammar` variant is only present under the
/// `grammar` feature.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub enum ToolConstraint {
    /// No constraint — the provider advertises tool schemas as-is.
    ///
    /// This is the default: every tool's schema is forwarded to the
    /// provider verbatim, with no `additionalProperties: false`, no
    /// expanded `required`, and no `strict` flag. The model is free to
    /// emit any JSON it likes for a tool call, including hallucinated
    /// fields, and malformed tool calls are detected (and retried) rather
    /// than prevented.
    ///
    /// Choose this when you trust the model to emit well-formed tool calls
    /// (frontier models, warm-up turns, or any path where you'd rather
    /// surface a malformed call than have the provider reject the request).
    #[default]
    None,

    /// Use the provider's native strict-tool mode.
    ///
    /// On OpenAI this sets `strict: true` on each tool's `function` schema
    /// after tightening it (recursive `additionalProperties: false` and
    /// full `required`). On Anthropic and Gemini it tightens each tool's
    /// `input_schema` / `parameters` the same way.
    Strict,

    /// Use a grammar compiled from the tool schemas, passed to a
    /// grammar-aware sampler (vLLM `guided_json`).
    ///
    /// Only available when the `grammar` feature is enabled. The
    /// [`ToolGrammarProvider`](crate::provider::grammar::ToolGrammarProvider)
    /// trait is the extension point for other server dialects.
    #[cfg(feature = "grammar")]
    Grammar(std::sync::Arc<dyn crate::provider::grammar::ToolGrammarProvider>),
}

/// Optional per-request knobs layered on top of a `stream_messages` /
/// `create_message` call.
///
/// Additive and forward-compatible: every field has a default that
/// reproduces prior behaviour, and a field a client cannot honor is
/// rejected with a config error rather than silently ignored (see the
/// `ApiClient` trait's `*_with_options` defaults). Carries
/// [`response_format`](Self::response_format) (constrain the model's
/// free-text output to a schema),
/// [`tool_constraint`](Self::tool_constraint) (constrain the model's tool
/// calls to the registered schemas), and [`model`](Self::model) (serve
/// this one request with a different model).
///
/// The two paths are independent: setting `response_format` suppresses
/// `tools` (and therefore makes `tool_constraint` a no-op for that
/// request), while setting `tool_constraint` constrains the `tools` path
/// itself.
///
/// Per-request sampling knobs (temperature, top-p, stop sequences,
/// max response tokens) are deliberately out of scope: requests
/// without them are served with the provider-side defaults.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    /// If set, ask the model to return JSON conforming to this schema.
    ///
    /// When `Some`, the provider injects the schema into the request
    /// (OpenAI `response_format` / Anthropic forced tool). When `None`,
    /// the model's output is unconstrained — the default behaviour.
    pub response_format: Option<ResponseFormat>,

    /// How strictly the model's tool-call output must follow the
    /// registered tool schemas. Default [`ToolConstraint::None`] is a
    /// no-op. See [`ToolConstraint`] for the modes.
    pub tool_constraint: ToolConstraint,

    /// Serve the request with the named model, overriding the client's
    /// current model.
    ///
    /// `None` — the default — uses the client's model, so requests without
    /// an override behave exactly as before. Set via
    /// [`with_model`](Self::with_model); the fallback machinery uses this
    /// seam to route requests to the active fallback model without
    /// mutating shared client state.
    pub model: Option<String>,
}

impl RequestOptions {
    /// Create empty options with no response format set.
    ///
    /// Equivalent to [`RequestOptions::default`]. Use
    /// [`with_response_format`](Self::with_response_format) to chain a format
    /// builder-style.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a per-request model override.
    ///
    /// The provider serves this one request with the named model instead
    /// of the client's current model; the client itself is untouched, so
    /// concurrent loops over one shared client cannot cross-wire their
    /// models. An empty or whitespace-only name is ignored (the override
    /// stays unset) — a nameless model override is never what a caller
    /// means, and the providers reject it on the wire anyway.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        if model.trim().is_empty() {
            return self;
        }
        self.model = Some(model);
        self
    }

    /// Set the response format, builder-style.
    ///
    /// When set, the provider constrains the model's output to the schema.
    /// When left `None` (the default), the model's output is unconstrained.
    #[must_use]
    pub fn with_response_format(mut self, rf: ResponseFormat) -> Self {
        self.response_format = Some(rf);
        self
    }

    /// Set the tool-call constraint, builder-style.
    ///
    /// Default [`ToolConstraint::None`] advertises tool schemas as-is.
    /// [`ToolConstraint::Strict`] makes malformed tool calls structurally
    /// impossible via the provider's native strict mode.
    #[must_use]
    pub fn with_tool_constraint(mut self, c: ToolConstraint) -> Self {
        self.tool_constraint = c;
        self
    }
}

/// Errors raised by the structured-output machinery.
#[derive(Debug, thiserror::Error)]
pub enum StructuredError {
    /// The model's output did not deserialize into the target type.
    ///
    /// When raised from parsing a provider response, carries the
    /// underlying `serde_json::Error` with its exact location
    /// (line/column within the JSON). The prompted path
    /// ([`request_structured_prompted`]) reuses this variant after a
    /// failed corrective retry with a synthesized error whose message
    /// carries the failure reason and the truncated last output.
    #[error("structured output did not match the expected schema: {0}")]
    Deserialize(#[from] serde_json::Error),

    /// The provider API call failed (HTTP error, auth failure, timeout, rate
    /// limit).
    ///
    /// Carries the underlying [`ApiError`](crate::api::error::ApiError).
    #[error("API error during structured output request: {0}")]
    Api(crate::api::error::ApiError),
}

/// Parse a string as JSON, with a lenient fallback that finds the outermost
/// `{ ... }` or `[ ... ]` substring.
///
/// This is the single biggest lever for hitting the ≥95% schema-valid bar on
/// real-world providers that wrap JSON in markdown fences or prefix it with
/// prose.
///
/// Returns `None` if the content cannot be parsed as JSON (even after the
/// lenient rescue).
pub(crate) fn parse_json_lenient(text: &str) -> Option<serde_json::Value> {
    if let Ok(v) = serde_json::from_str(text) {
        return Some(v);
    }
    // Lenient rescue: find the outermost { ... } or [ ... ].
    extract_json_substring(text)
}

/// Find and parse the outermost JSON object or array in a string.
///
/// Scans for the first `{` or `[`, tracks independent brace and bracket
/// depths, and extracts the substring up to the close of the outermost
/// container — a candidate completes only when both depths return to zero,
/// so an inner container of the other kind (an array inside an object, or
/// an object inside an array) is a plain depth change rather than a
/// mismatch. A closing delimiter that arrives while its own depth is zero
/// cannot belong to balanced JSON, so the candidate is abandoned and the
/// scan resumes — a later, well-formed value may still be found. The
/// outermost candidate that parses wins. String-aware: braces/brackets
/// inside JSON string literals (`"..."`) do not affect depth, and `\"`
/// escapes are honored.
pub(crate) fn extract_json_substring(text: &str) -> Option<serde_json::Value> {
    let bytes = text.as_bytes();
    let mut start = None;
    let mut brace_depth: usize = 0;
    let mut bracket_depth: usize = 0;
    let mut in_string = false;
    let mut escaped = false;

    for (i, &byte) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }

        match byte {
            b'"' => {
                in_string = true;
            }
            b'{' | b'[' => {
                if start.is_none() {
                    start = Some(i);
                }
                if byte == b'{' {
                    brace_depth = brace_depth.saturating_add(1);
                } else {
                    bracket_depth = bracket_depth.saturating_add(1);
                }
            }
            b'}' | b']' => {
                let Some(s) = start else {
                    continue;
                };
                let depth = if byte == b'}' {
                    &mut brace_depth
                } else {
                    &mut bracket_depth
                };
                if *depth == 0 {
                    // Unbalanced closer inside the candidate — not JSON.
                    start = None;
                    brace_depth = 0;
                    bracket_depth = 0;
                } else {
                    *depth = depth.saturating_sub(1);
                    if brace_depth == 0 && bracket_depth == 0 {
                        let slice = text.get(s..=i).unwrap_or(text);
                        if let Ok(v) = serde_json::from_str(slice) {
                            return Some(v);
                        }
                        start = None;
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Tighten a JSON Schema for strict-mode submission.
///
/// Recursively, on every `type: "object"` subschema that has a `properties`
/// map: set `additionalProperties` to `false` and set `required` to the
/// union of any pre-existing entries and the full list of property keys —
/// every property becomes required, and author-declared entries naming keys
/// without a matching property survive. Non-object subschemas (`string`,
/// `number`, etc.) are returned unchanged; the implementation recurses into
/// object properties, `array` `items`, and the values of `allOf` / `anyOf` /
/// `oneOf` arrays, but leaves `$ref`, `if`/`then`/`else`, and other
/// combinator shapes untouched (best-effort).
///
/// Idempotent: passing an already-tight schema through it again yields the
/// same value.
///
/// # Example
///
/// ```rust,ignore
/// use loopctl::structured::tighten_json_schema;
/// use serde_json::json;
///
/// let schema = json!({
///     "type": "object",
///     "properties": {
///         "q": {"type": "string"},
///         "limit": {"type": "number"},
///         "filter": {
///             "type": "object",
///             "properties": {"lang": {"type": "string"}}
///         }
///     }
/// });
///
/// let tightened = tighten_json_schema(&schema);
///
/// // Top-level object: closed and fully required.
/// assert_eq!(tightened["additionalProperties"], false);
/// assert_eq!(tightened["required"], json!(["filter", "limit", "q"]));
///
/// // Nested object is tightened independently — its `required` lists only
/// // its own keys, not the parent's.
/// assert_eq!(tightened["properties"]["filter"]["additionalProperties"], false);
/// assert_eq!(tightened["properties"]["filter"]["required"], json!(["lang"]));
/// ```
#[cfg(any(
    feature = "anthropic",
    feature = "grammar",
    feature = "openai",
    feature = "gemini"
))]
pub(crate) fn tighten_json_schema(schema: &serde_json::Value) -> serde_json::Value {
    let mut out = schema.clone();
    tighten_in_place(&mut out);
    out
}

/// Recursively tighten a JSON Schema in place.
///
/// At each `type: "object"` node with a `properties` map, enforces
/// strictness by:
///
/// 1. setting `additionalProperties: false` — rejects extra/hallucinated
///    fields the model invents beyond the declared properties,
/// 2. setting `required` to the union of any pre-existing entries (kept in
///    their original order) and that object's own property keys (appended
///    when not already listed) — rejects missing fields the model omitted
///    without silently dropping an author-declared requirement that has no
///    matching property.
///
/// Together these make every object schema *closed* (no extra keys) and
/// *fully mandatory* (no optional keys). The two are complementary: neither
/// alone covers the other, and both are required for OpenAI's strict mode
/// to accept the schema without a `400`.
///
/// # Recursion rules
///
/// The walk descends into:
/// - each value of an object's `properties` map (child object schemas),
/// - an array schema's `items` subschema,
/// - each member of `allOf` / `anyOf` / `oneOf` combinator arrays,
/// - each named definition in a local `$defs` / `definitions` map.
///
/// It does **not** descend into or rewrite:
/// - `$ref` references themselves (not followed — would need a registry;
///   but the definitions they point to under `$defs` / `definitions` *are*
///   visited, so a local reference's target still gets tightened),
/// - `if` / `then` / `else` conditional subschemas,
/// - non-object typed schemas (`string`, `number`, `boolean`, …), which
///   are returned unchanged.
///
/// Idempotent: re-running on an already-tight schema leaves it unchanged.
#[cfg(any(
    feature = "anthropic",
    feature = "grammar",
    feature = "openai",
    feature = "gemini"
))]
fn tighten_in_place(schema: &mut serde_json::Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };

    // Tighten this object's own property subschemas first.
    if let Some(properties) = obj
        .get_mut("properties")
        .and_then(serde_json::Value::as_object_mut)
    {
        for child in properties.values_mut() {
            tighten_in_place(child);
        }
    }

    // Recurse into array `items`.
    if let Some(items) = obj.get_mut("items") {
        tighten_in_place(items);
    }

    // Recurse into combinator subschemas.
    for key in ["allOf", "anyOf", "oneOf"] {
        if let Some(arr) = obj.get_mut(key).and_then(serde_json::Value::as_array_mut) {
            for child in arr {
                tighten_in_place(child);
            }
        }
    }

    // Recurse into local named definitions so a `$ref: "#/$defs/..."`
    // target receives the same tightening. `$defs` is the Draft 2019-09+
    // keyword; `definitions` is the older Draft 07 keyword. Both are
    // object maps of subschemas keyed by definition name.
    for key in ["$defs", "definitions"] {
        if let Some(defs) = obj.get_mut(key).and_then(serde_json::Value::as_object_mut) {
            for child in defs.values_mut() {
                tighten_in_place(child);
            }
        }
    }

    // Only enforce object strictness on explicit `type: "object"` schemas.
    // Schemas without a type (or with another type) are left structurally
    // alone so we don't impose object semantics on, e.g., a free-form value.
    let is_object = obj
        .get("type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|t| t == "object");
    if !is_object {
        return;
    }

    obj.insert(
        "additionalProperties".to_string(),
        serde_json::Value::Bool(false),
    );

    // Enumerate `required` from the current properties (or empty when there
    // are none). Preserves any pre-existing required entries that have no
    // matching property (the model author may know best in odd cases).
    let property_keys: Vec<String> = obj
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .map(|props| props.keys().cloned().collect())
        .unwrap_or_default();
    let mut required: Vec<serde_json::Value> = obj
        .get("required")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let already_listed: std::collections::HashSet<String> = required
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect();
    for key in property_keys {
        if !already_listed.contains(key.as_str()) {
            required.push(serde_json::Value::String(key));
        }
    }
    obj.insert("required".to_string(), serde_json::Value::Array(required));
}

/// Request a typed, schema-conformant value from the model.
///
/// This is the ergonomic entry point callers use. It:
/// 1. Builds [`RequestOptions`] with the [`ResponseFormat`] for `T`.
/// 2. Calls [`create_message_with_options`](ApiClient::create_message_with_options)
///    on the client.
/// 3. Extracts the structured value from the provider's response.
/// 4. Deserializes it into `T` via [`StructuredOutput::from_value`].
///
/// # Errors
///
/// Returns [`StructuredError`] if the provider call fails, the response
/// cannot be parsed as JSON, or the JSON does not match `T`'s schema.
pub async fn request_structured<T: StructuredOutput + serde::de::DeserializeOwned>(
    client: &dyn ApiClient,
    messages: Vec<Message>,
    system: Option<String>,
) -> Result<T, StructuredError> {
    let opts = RequestOptions::new().with_response_format(ResponseFormat::from_type::<T>());
    let request = crate::api::StreamRequest {
        messages,
        system,
        tools: None,
    };
    let response = client
        .create_message_with_options(&request, opts)
        .await
        .map_err(StructuredError::Api)?;
    let value = client.extract_structured(&response.message);
    T::from_value(value)
}

/// The exact system-prompt envelope used by
/// [`request_structured_prompted`], public for hosts composing their own
/// flows.
///
/// The wording is part of the feature: imperative, short sentences, one
/// minified schema, and explicit "no fences" instructions. The lenient
/// parser tolerates fences and prose anyway, but asking for none keeps
/// the response free of clutter tokens. A caller-supplied system prompt
/// is composed ahead of this envelope, never replaced by it.
///
/// The schema is embedded minified (`serde_json::to_string`) — token
/// cheap, and small models read minified schemas fine; the wrapper text
/// carries the explanation. That serialization is total for any
/// `serde_json::Value` constructible in practice, so the guarded
/// failure arm below cannot fire today; it warns instead of failing
/// silently so the impossible stays observable if it ever stops being
/// impossible.
///
/// # Example
///
/// ```
/// use loopctl::structured::{prompted_system_prefix, StructuredOutput};
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Answer {
///     text: String,
/// }
///
/// impl StructuredOutput for Answer {
///     fn name() -> &'static str {
///         "answer"
///     }
///     fn schema() -> serde_json::Value {
///         serde_json::json!({
///             "type": "object",
///             "properties": {"text": {"type": "string"}},
///             "required": ["text"],
///             "additionalProperties": false
///         })
///     }
/// }
///
/// let prefix = prompted_system_prefix::<Answer>();
/// assert!(prefix.contains("You will answer with JSON only."));
/// assert!(prefix.contains(r#""required":["text"]"#));
/// ```
#[must_use]
pub fn prompted_system_prefix<T: StructuredOutput>() -> String {
    let schema_text = match serde_json::to_string(&T::schema()) {
        Ok(schema) => schema,
        Err(error) => {
            tracing::warn!(
                target: "loopctl::structured",
                error = %error,
                "the structured-output schema could not be serialized; the prompted envelope carries an empty schema"
            );
            String::new()
        }
    };
    format!(
        "You will answer with JSON only.\n\n\
         The JSON must exactly match this schema:\n\
         {schema_text}\n\n\
         Rules:\n\
         - Output ONE JSON object and nothing else — no prose, no markdown fences.\n\
         - Include every required field with the correct type.\n\
         - Strings must be valid JSON strings (escape newlines and quotes).",
    )
}

/// Compose the caller's system prompt with the prompted envelope.
///
/// The caller's prompt comes first — it carries the host's persona and
/// task framing — and the envelope follows as the last, most immediate
/// instruction.
fn compose_prompted_system<T: StructuredOutput>(system: Option<String>) -> String {
    let prefix = prompted_system_prefix::<T>();
    match system {
        Some(user_system) => format!("{user_system}\n\n{prefix}"),
        None => prefix,
    }
}

/// The corrective retry message fed back to the model after a failed
/// extraction.
///
/// The concrete error text is what makes one retry effective — serde
/// messages ("missing field `tool` at line 1 column 5") are
/// model-legible — but it is bounded before riding the retry wire: a
/// long prose failure embeds the whole failed answer in the serde error
/// text, and echoing it back whole could push a small local model's
/// retry over its context window, corrupting the recovery path's own
/// success condition.
fn corrective_message(error: &StructuredError) -> String {
    format!(
        "Your previous answer was not valid JSON for the schema:\n\
         {}\n\n\
         Answer again with the corrected JSON object only.",
        truncate_for_error(&error.to_string(), 2_000),
    )
}

/// Render a response's output as plain text.
///
/// Text content plus, when the reply carried tool-call parts, their
/// inputs: a reply whose only content is a schema-invalid tool-call
/// input has empty text, and without the inputs neither the failure
/// error nor the retry conversation would say anything about the
/// output that failed. The rendering is never empty — a reply with no
/// text and no tool parts renders as `(empty reply)` — so the failure
/// error stays informative and the corrective retry can replay this
/// rendering as an assistant turn on every provider: a replayed turn
/// still carrying the tool-call parts would be rejected by providers
/// that require a tool result after every tool call, and an empty
/// assistant turn by those that validate content at all.
fn last_output_text(message: &Message) -> String {
    use std::fmt::Write as _;
    let mut output = message.text_content();
    for (_, name, input) in message.tool_call_parts() {
        let _ignored = write!(output, "\n[tool call {name} input: {input}]");
    }
    if output.is_empty() {
        return "(empty reply)".to_string();
    }
    output
}

/// Bound an error-embedded text to a human-readable size.
///
/// The raw output rides the error for humans and tests, not for
/// re-feeding; bounding it keeps a pathological response from bloating
/// the error while keeping its informative head. Truncation counts by
/// `char` and never splits a multi-byte character.
fn truncate_for_error(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit).collect();
    format!("{head}…[truncated]")
}

/// The per-attempt telemetry span for the prompted extraction path.
///
/// Named `structured.extract` with `mode=prompted`, the zero-based
/// `attempt` number, an `ok` field left empty until the attempt's
/// outcome is known, and the OTel-convention
/// `gen_ai.operation.name` field. The span is attached to the attempt's
/// request future via [`Instrument`](tracing::Instrument) — never
/// entered around an await — so a subscriber sees the attempt decorated
/// and observes the recorded `ok` on close.
fn extract_attempt_span(attempt: u64) -> tracing::Span {
    tracing::debug_span!(
        target: "loopctl::structured",
        "structured.extract",
        mode = "prompted",
        attempt,
        ok = tracing::field::Empty,
        "gen_ai.operation.name" = "structured_extract",
    )
}

/// The per-attempt usage counters, when the response carried usage.
///
/// Two metric-targeted debug events on
/// `loopctl.structured.prompted.tokens` with `direction=in|out` and the
/// provider-reported token figures as `value`; silent when the response
/// carried no usage, so consumers can sum the counter without zero
/// noise from providers that do not report.
///
/// Returns the number of events emitted — zero when there was no
/// usage, two otherwise — so the silence contract is testable without
/// a capturing subscriber.
fn emit_usage(usage: Option<crate::stream::Usage>) -> usize {
    let Some(usage) = usage else {
        return 0;
    };
    tracing::debug!(
        target: "loopctl::metrics",
        metric = "loopctl.structured.prompted.tokens",
        direction = "in",
        value = usage.input_tokens,
        "structured prompted tokens in"
    );
    tracing::debug!(
        target: "loopctl::metrics",
        metric = "loopctl.structured.prompted.tokens",
        direction = "out",
        value = usage.output_tokens,
        "structured prompted tokens out"
    );
    2
}

/// The settle event: how many attempts the extraction took and how it
/// ended.
///
/// Emits the `loopctl.structured.prompted` counter with
/// `outcome=first_try|after_retry|failed|api_error` and the
/// `loopctl.structured.prompted.attempts_count` event carrying the
/// attempt count, so a consumer summing the counter sees every
/// extraction — including the provider-failure cases where the metric
/// matters most.
fn emit_settled(outcome: &str, attempts: u64) {
    tracing::debug!(
        target: "loopctl::metrics",
        metric = "loopctl.structured.prompted",
        outcome,
        "structured prompted extraction settled"
    );
    tracing::debug!(
        target: "loopctl::metrics",
        metric = "loopctl.structured.prompted.attempts_count",
        value = attempts,
        "structured prompted attempts"
    );
}

/// One attempt of the prompted extraction: send the request, emit the
/// usage counters, extract and parse — all inside the attempt's
/// telemetry span.
///
/// The span is attached with [`Instrument`](tracing::Instrument), so
/// the awaited request and the parse run decorated and the recorded
/// `ok` is observable on the span's close. Returns the parse outcome
/// together with the response message (the caller needs the failed
/// answer for the retry conversation); an API failure marks the span
/// unsuccessful and surfaces as [`StructuredError::Api`].
///
/// # Errors
///
/// [`StructuredError::Api`] when the underlying request fails; the
/// parse failure itself is returned in the `Ok` arm, left for the
/// caller to settle.
async fn prompted_attempt<T>(
    client: &dyn ApiClient,
    request: &crate::api::StreamRequest,
    span: &tracing::Span,
) -> Result<(Result<T, StructuredError>, Message), StructuredError>
where
    T: StructuredOutput + serde::de::DeserializeOwned,
{
    let attempt = async {
        let response = client.create_message(request).await?;
        emit_usage(response.usage);
        let parsed = T::from_value(client.extract_structured(&response.message));
        Ok::<_, crate::api::error::ApiError>((response.message, parsed))
    };
    match attempt.instrument(span.clone()).await {
        Ok((message, parsed)) => {
            span.record("ok", parsed.is_ok());
            Ok((parsed, message))
        }
        Err(error) => {
            span.record("ok", false);
            Err(StructuredError::Api(error))
        }
    }
}

/// Prompt-guided structured output for providers without native
/// `response_format` support (most local models).
///
/// Unlike [`request_structured`] (which sets a provider
/// [`ResponseFormat`]), this function embeds `T`'s JSON Schema in the
/// system prompt with strict-JSON output instructions (see
/// [`prompted_system_prefix`]), sends a plain request — no
/// [`RequestOptions`] at all, so there is nothing for a local server to
/// reject — and extracts the value leniently (fenced or prose-wrapped
/// JSON is accepted). If the response does not parse or fails `T`'s
/// schema, exactly one corrective retry is made with the concrete error
/// fed back to the model; a second failure returns
/// [`StructuredError::Deserialize`] carrying the failure and the last
/// model output (truncated).
///
/// It works on every provider by construction: the request is an
/// ordinary `create_message`. On providers with native support this is
/// simply the worse option — prefer [`request_structured`] there, since
/// native enforcement beats prompt guidance.
///
/// # Example
///
/// ```rust,no_run
/// use loopctl::api::ApiClient;
/// use loopctl::message::Message;
/// use loopctl::structured::request_structured_prompted;
/// use serde::Deserialize;
///
/// #[derive(Deserialize, Debug)]
/// struct RouteDecision {
///     route: String,
///     confidence: f64,
/// }
///
/// impl loopctl::structured::StructuredOutput for RouteDecision {
///     fn name() -> &'static str {
///         "route_decision"
///     }
///     fn schema() -> serde_json::Value {
///         serde_json::json!({
///             "type": "object",
///             "properties": {
///                 "route": {"type": "string"},
///                 "confidence": {"type": "number"}
///             },
///             "required": ["route", "confidence"],
///             "additionalProperties": false
///         })
///     }
/// }
///
/// async fn demo(
///     client: &dyn ApiClient,
/// ) -> Result<(), loopctl::structured::StructuredError> {
///     let decision: RouteDecision = request_structured_prompted(
///         client,
///         vec![Message::user("route this")],
///         None,
///     )
///     .await?;
///     println!("{decision:?}");
///     Ok(())
/// }
/// ```
///
/// # Errors
///
/// [`StructuredError::Api`] if either request fails;
/// [`StructuredError::Deserialize`] with the parse/schema error and the
/// truncated last output if both attempts fail.
pub async fn request_structured_prompted<T: StructuredOutput + serde::de::DeserializeOwned>(
    client: &dyn ApiClient,
    messages: Vec<Message>,
    system: Option<String>,
) -> Result<T, StructuredError> {
    let system = compose_prompted_system::<T>(system);
    let mut request = crate::api::StreamRequest {
        messages,
        system: Some(system),
        tools: None,
    };

    let span = extract_attempt_span(0);
    let (first_parsed, failed_answer) = match prompted_attempt::<T>(client, &request, &span).await {
        Ok(outcome) => outcome,
        Err(error) => {
            emit_settled("api_error", 1);
            return Err(error);
        }
    };
    let first_error = match first_parsed {
        Ok(value) => {
            emit_settled("first_try", 1);
            return Ok(value);
        }
        Err(error) => error,
    };

    request
        .messages
        .push(Message::assistant(last_output_text(&failed_answer)));
    request
        .messages
        .push(Message::user(corrective_message(&first_error)));

    let span = extract_attempt_span(1);
    let (second_parsed, second_answer) = match prompted_attempt::<T>(client, &request, &span).await
    {
        Ok(outcome) => outcome,
        Err(error) => {
            emit_settled("api_error", 2);
            return Err(error);
        }
    };
    match second_parsed {
        Ok(value) => {
            emit_settled("after_retry", 2);
            Ok(value)
        }
        Err(second_error) => {
            emit_settled("failed", 2);
            let raw_output = last_output_text(&second_answer);
            Err(StructuredError::Deserialize(serde_json::Error::custom(
                format!(
                    "{}; the corrective retry also failed. Last output: {}",
                    truncate_for_error(&second_error.to_string(), 300),
                    truncate_for_error(&raw_output, 2_000),
                ),
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;

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
                "required": ["tool", "args"],
                "additionalProperties": false
            })
        }
    }

    fn fixture_action() -> serde_json::Value {
        serde_json::json!({
            "tool": "write",
            "args": { "path": "/tmp/test.txt" }
        })
    }

    #[test]
    fn with_model_ignores_empty_and_whitespace_names() {
        let opts = RequestOptions::default().with_model("");
        assert!(
            opts.model.is_none(),
            "an empty model name leaves the override unset — providers reject nameless models"
        );
        let opts = RequestOptions::default().with_model("   ");
        assert!(
            opts.model.is_none(),
            "a whitespace-only model name leaves the override unset"
        );
        let opts = RequestOptions::default().with_model("fallback-model");
        assert_eq!(opts.model.as_deref(), Some("fallback-model"));
    }

    #[test]
    fn structured_output_round_trip() {
        let v = fixture_action();
        let action: Action = Action::from_value(v).expect("should deserialize");
        assert_eq!(action.tool, "write");
        assert_eq!(action.args, serde_json::json!({ "path": "/tmp/test.txt" }));
    }

    #[test]
    fn response_format_from_type() {
        let rf = ResponseFormat::from_type::<Action>();
        assert_eq!(rf.name, "action");
        assert_eq!(rf.schema, Action::schema());
        assert!(rf.strict);
    }

    #[test]
    fn request_options_builder() {
        let opts = RequestOptions::new();
        assert!(opts.response_format.is_none());

        let rf = ResponseFormat::from_type::<Action>();
        let opts = RequestOptions::new().with_response_format(rf);
        assert!(opts.response_format.is_some());
        assert_eq!(opts.response_format.as_ref().unwrap().name, "action");
    }

    #[test]
    fn parse_json_lenient_plain_json() {
        let v = parse_json_lenient(r#"{"a": 1}"#).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn parse_json_lenient_with_prefix() {
        let v = parse_json_lenient(r#"Here is the JSON: {"a": 1}"#).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn parse_json_lenient_markdown_fences() {
        let v = parse_json_lenient("```json\n{\"a\": 1}\n```").unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn parse_json_lenient_array() {
        let v = parse_json_lenient(r#"prefix [1, 2, 3] suffix"#).unwrap();
        assert_eq!(v[0], 1);
    }

    #[test]
    fn parse_json_lenient_no_json() {
        let result = parse_json_lenient("just prose, nothing here");
        assert!(result.is_none());
    }

    #[test]
    fn structured_error_displays() {
        let json_err = serde_json::from_str::<serde_json::Value>("bad").unwrap_err();
        let err = StructuredError::Deserialize(json_err);
        assert!(err.to_string().contains("schema"));
    }

    #[test]
    fn parse_json_lenient_brace_inside_string() {
        let v = parse_json_lenient(r#"prefix {"a": "}"} suffix"#).unwrap();
        assert_eq!(v["a"], "}");
    }

    #[test]
    fn parse_json_lenient_bracket_inside_string() {
        let v = parse_json_lenient(r#"before {"x": "]"} after"#).unwrap();
        assert_eq!(v["x"], "]");
    }

    #[test]
    fn parse_json_lenient_escaped_quote_in_string() {
        let v = parse_json_lenient(r#"here {"a": "he said \"hi\""} there"#).unwrap();
        assert_eq!(v["a"], "he said \"hi\"");
    }

    #[test]
    fn parse_json_lenient_nested_objects_in_prose() {
        let v = parse_json_lenient(r#"result: {"outer": {"inner": 42}}"#).unwrap();
        assert_eq!(v["outer"]["inner"], 42);
    }

    #[test]
    fn parse_json_lenient_mismatched_delimiter_then_valid() {
        let v = parse_json_lenient(r#"{oops] then {"a":1}"#).unwrap();
        assert_eq!(v["a"], 1);
    }

    struct PlainMockClient;
    impl crate::api::ApiClient for PlainMockClient {
        fn model(&self) -> String {
            "test".to_string()
        }
        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn futures::Stream<
                        Item = Result<crate::stream::StreamEvent, crate::api::error::ApiError>,
                    > + Send
                    + 'static,
            >,
        > {
            Box::pin(futures::stream::empty())
        }
        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::api::NonStreamingResponse,
                            crate::api::error::ApiError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant(""),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
    }

    #[tokio::test]
    async fn default_client_rejects_response_format() {
        let client = PlainMockClient;
        let opts =
            RequestOptions::new().with_response_format(ResponseFormat::from_type::<Action>());
        let request = crate::api::StreamRequest::new(vec![]);
        let result = client.create_message_with_options(&request, opts).await;
        assert!(
            result.is_err(),
            "client without structured-output support should reject response_format"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("does not support structured output"),
            "error should explain why: {err_msg}"
        );
    }

    #[tokio::test]
    async fn default_client_delegates_empty_options() {
        let client = PlainMockClient;
        let opts = RequestOptions::new();
        let request = crate::api::StreamRequest::new(vec![]);
        let result = client.create_message_with_options(&request, opts).await;
        assert!(result.is_ok(), "empty options should delegate normally");
    }

    struct StructuredMockClient;
    impl crate::api::ApiClient for StructuredMockClient {
        fn model(&self) -> String {
            "test".to_string()
        }
        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn futures::Stream<
                        Item = Result<crate::stream::StreamEvent, crate::api::error::ApiError>,
                    > + Send
                    + 'static,
            >,
        > {
            Box::pin(futures::stream::empty())
        }
        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::api::NonStreamingResponse,
                            crate::api::error::ApiError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant(""),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
        fn create_message_with_options(
            &self,
            _request: &crate::api::StreamRequest,
            _options: RequestOptions,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::api::NonStreamingResponse,
                            crate::api::error::ApiError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant(
                        r#"{"tool": "write", "args": {"path": "/test"}}"#,
                    ),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
    }

    #[tokio::test]
    async fn request_structured_end_to_end() {
        let client = StructuredMockClient;
        let action: Action = request_structured(&client, vec![], None)
            .await
            .expect("should succeed");
        assert_eq!(action.tool, "write");
    }

    struct ProseMockClient;
    impl crate::api::ApiClient for ProseMockClient {
        fn model(&self) -> String {
            "test".to_string()
        }
        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn futures::Stream<
                        Item = Result<crate::stream::StreamEvent, crate::api::error::ApiError>,
                    > + Send
                    + 'static,
            >,
        > {
            Box::pin(futures::stream::empty())
        }
        fn create_message(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::api::NonStreamingResponse,
                            crate::api::error::ApiError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant(""),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
        fn create_message_with_options(
            &self,
            _request: &crate::api::StreamRequest,
            _options: RequestOptions,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::api::NonStreamingResponse,
                            crate::api::error::ApiError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::api::NonStreamingResponse {
                    message: crate::message::Message::assistant("I cannot produce that."),
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
    }

    #[tokio::test]
    async fn request_structured_prose_returns_deserialize_error() {
        let client = ProseMockClient;
        let err = request_structured::<Action>(&client, vec![], None)
            .await
            .expect_err("should fail");
        // Prose is a valid JSON string but doesn't match Action's schema,
        // so deserialization fails.
        assert!(matches!(err, StructuredError::Deserialize(_)));
    }

    #[test]
    fn usageless_responses_emit_no_token_events() {
        assert_eq!(
            emit_usage(None),
            0,
            "a response without usage stays silent — no zero noise for consumers"
        );
        assert_eq!(
            emit_usage(Some(crate::stream::Usage::new(0, 0))),
            2,
            "a zero usage still reports both directions"
        );
    }

    #[test]
    fn prompted_prefix_carries_the_schema_and_rules() {
        let prefix = prompted_system_prefix::<Action>();
        assert!(prefix.contains("You will answer with JSON only."));
        assert!(
            prefix.contains(r#""required":["tool","args"]"#),
            "the schema is embedded minified, not pretty-printed"
        );
        assert!(prefix.contains("no prose, no markdown fences"));
        assert!(prefix.contains("Include every required field with the correct type."));
        assert!(prefix.contains("escape newlines and quotes"));
    }

    #[test]
    fn truncate_for_error_bounds_the_output() {
        assert_eq!(truncate_for_error("short", 2_000), "short");
        let oversized = "x".repeat(5_000);
        let bounded = truncate_for_error(&oversized, 2_000);
        assert!(
            bounded.chars().count() < 2_100,
            "an error-embedded output stays bounded"
        );
        assert!(bounded.ends_with("…[truncated]"));
        let tail_heavy = truncate_for_error(&oversized, 300);
        assert!(
            tail_heavy.chars().count() < 400,
            "a tighter limit bounds harder"
        );
    }

    #[test]
    fn prompted_prefix_composes_after_the_user_system() {
        let composed = compose_prompted_system::<Action>(Some("Be terse.".to_string()));
        let user = composed.find("Be terse.").expect("user system preserved");
        let prefix = composed
            .find("You will answer with JSON only.")
            .expect("envelope present");
        assert!(
            user < prefix,
            "the caller's system prompt comes first, the envelope after"
        );
    }

    #[test]
    fn tool_constraint_default_is_none() {
        assert!(matches!(ToolConstraint::default(), ToolConstraint::None));
        assert!(matches!(
            RequestOptions::default().tool_constraint,
            ToolConstraint::None
        ));
    }

    #[test]
    fn request_options_tool_constraint_builder() {
        let opts = RequestOptions::new().with_tool_constraint(ToolConstraint::Strict);
        assert!(matches!(opts.tool_constraint, ToolConstraint::Strict));
        // And response_format still composes on the same builder.
        let rf = ResponseFormat::from_type::<Action>();
        let opts = RequestOptions::new()
            .with_response_format(rf)
            .with_tool_constraint(ToolConstraint::Strict);
        assert!(opts.response_format.is_some());
        assert!(matches!(opts.tool_constraint, ToolConstraint::Strict));
    }

    #[test]
    fn tool_constraint_clone_compiles() {
        // RequestOptions derives Clone, which requires every field —
        // including tool_constraint — to be Clone. This test pins that by
        // cloning an options value that carries a constraint and asserting
        // both copies hold the same variant.
        let opts = RequestOptions::new().with_tool_constraint(ToolConstraint::Strict);
        let cloned = opts.clone();
        assert!(matches!(opts.tool_constraint, ToolConstraint::Strict));
        assert!(matches!(cloned.tool_constraint, ToolConstraint::Strict));
    }

    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[test]
    fn tighten_sets_additional_properties_false() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "string"}}
        });
        let tightened = tighten_json_schema(&schema);
        assert_eq!(tightened["additionalProperties"], false);
    }

    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[test]
    fn tighten_enumerates_required() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "number"}
            }
        });
        let tightened = tighten_json_schema(&schema);
        let required = tightened["required"].as_array().unwrap();
        assert_eq!(required.len(), 2);
        let keys: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(keys.contains(&"a"));
        assert!(keys.contains(&"b"));
    }

    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[test]
    fn tighten_recurses_into_nested_objects() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}}
                }
            }
        });
        let tightened = tighten_json_schema(&schema);
        assert_eq!(
            tightened["properties"]["inner"]["additionalProperties"],
            false
        );
        let inner_required = tightened["properties"]["inner"]["required"]
            .as_array()
            .unwrap();
        assert_eq!(inner_required.len(), 1);
        assert_eq!(inner_required[0], "x");
    }

    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[test]
    fn tighten_preserves_non_object_schemas() {
        let schema = serde_json::json!({"type": "string"});
        let tightened = tighten_json_schema(&schema);
        assert_eq!(tightened, schema);
        // Should not have gained additionalProperties / required.
        assert!(tightened.get("additionalProperties").is_none());
        assert!(tightened.get("required").is_none());
    }

    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[test]
    fn tighten_idempotent_on_already_strict() {
        let schema = serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {"a": {"type": "string"}},
            "required": ["a"]
        });
        let once = tighten_json_schema(&schema);
        let twice = tighten_json_schema(&once);
        assert_eq!(once, twice);
    }

    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[test]
    fn tighten_object_without_properties() {
        let schema = serde_json::json!({"type": "object"});
        let tightened = tighten_json_schema(&schema);
        assert_eq!(tightened["additionalProperties"], false);
        // `required` becomes an empty array, not absent.
        assert_eq!(tightened["required"].as_array().unwrap().len(), 0);
    }

    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[test]
    fn tighten_recurses_into_local_defs() {
        // A property that $refs a local definition: the reference itself
        // is not followed, but the definition under $defs must still be
        // tightened so a strict-mode server accepts it.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "filter": {"$ref": "#/$defs/Filter"}
            },
            "$defs": {
                "Filter": {
                    "type": "object",
                    "properties": {
                        "lang": {"type": "string"},
                        "limit": {"type": "number"}
                    }
                }
            }
        });
        let tightened = tighten_json_schema(&schema);

        // Top-level object: closed and fully required.
        assert_eq!(tightened["additionalProperties"], false);
        assert_eq!(tightened["required"], serde_json::json!(["filter"]));

        // The $defs/Filter definition is tightened: closed, and its
        // nested arguments are all required.
        let filter = &tightened["$defs"]["Filter"];
        assert_eq!(filter["additionalProperties"], false);
        let required = filter["required"].as_array().unwrap();
        assert_eq!(required.len(), 2);
        let keys: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(keys.contains(&"lang"));
        assert!(keys.contains(&"limit"));

        // The $ref reference itself is left in place (not rewritten).
        assert_eq!(tightened["properties"]["filter"]["$ref"], "#/$defs/Filter");
    }

    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[test]
    fn tighten_recurses_into_legacy_definitions() {
        // The Draft 07 keyword `definitions` should be walked the same way
        // as `$defs`.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"x": {"$ref": "#/definitions/X"}},
            "definitions": {
                "X": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}}
                }
            }
        });
        let tightened = tighten_json_schema(&schema);
        let def = &tightened["definitions"]["X"];
        assert_eq!(def["additionalProperties"], false);
        assert_eq!(def["required"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn parse_json_lenient_object_containing_array() {
        let v = parse_json_lenient(r#"prefix {"a": [1, 2]} suffix"#).unwrap();
        assert_eq!(v["a"], serde_json::json!([1, 2]));
    }

    #[test]
    fn parse_json_lenient_array_containing_object() {
        let v = parse_json_lenient(r#"prefix [{"a": 1}] suffix"#).unwrap();
        assert_eq!(v[0]["a"], 1);
    }

    #[test]
    fn parse_json_lenient_fenced_failure_analysis_shape() {
        let v = parse_json_lenient("```json\n{\"m\": {\"p\": [1]}}\n```").unwrap();
        assert!(v.is_object());
        assert_eq!(v["m"]["p"], serde_json::json!([1]));
    }

    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[test]
    fn tighten_preserves_required_entries_without_matching_property() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a", "meta"]
        });
        let tightened = tighten_json_schema(&schema);
        let required = tightened["required"].as_array().unwrap();
        assert!(
            required.iter().any(|v| v == "meta"),
            "required entry without a matching property must survive tightening: {required:?}"
        );
    }

    #[test]
    fn parse_json_lenient_deeply_mixed_nesting_extracts_outermost() {
        let v = parse_json_lenient(
            r#"analysis: {"is_recoverable":true,"correction":{"modified_input":{"path":["a"]}}}"#,
        )
        .unwrap();
        assert_eq!(
            v["correction"]["modified_input"]["path"],
            serde_json::json!(["a"])
        );
    }

    #[test]
    fn parse_json_lenient_object_with_array_of_objects() {
        let v = parse_json_lenient(r#"{"a": [{"b": 2}]}"#).unwrap();
        assert_eq!(v["a"][0]["b"], 2);
    }

    #[test]
    fn parse_json_lenient_stray_brace_in_array_resumes_scan() {
        let v = parse_json_lenient(r#"[1, 2} then {"a":1}"#).unwrap();
        assert_eq!(
            v["a"], 1,
            "a stray brace inside an array candidate aborts it; the later object still extracts"
        );
    }

    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[test]
    fn tighten_required_union_keeps_order_then_appends_properties() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"b": {"type": "number"}, "a": {"type": "string"}},
            "required": ["meta"]
        });
        let tightened = tighten_json_schema(&schema);
        let required = tightened["required"].as_array().unwrap();
        assert_eq!(
            required.first(),
            Some(&serde_json::json!("meta")),
            "pre-existing entries keep their position ahead of appended property keys"
        );
        assert_eq!(
            required.len(),
            3,
            "every property key is appended exactly once"
        );
        assert!(
            required.contains(&serde_json::json!("a"))
                && required.contains(&serde_json::json!("b")),
            "the unlisted property keys join the union, in map order: {required:?}"
        );
        let twice = tighten_json_schema(&tightened);
        assert_eq!(
            twice["required"], tightened["required"],
            "the union is idempotent: a second pass neither reorders nor duplicates"
        );
    }

    #[test]
    fn parse_json_lenient_first_valid_candidate_wins() {
        let v = parse_json_lenient(r#"{"a": 1} and {"b": 2}"#).unwrap();
        assert_eq!(
            v["a"], 1,
            "the outermost candidate that parses wins; a later sibling is not preferred"
        );
    }

    #[test]
    fn parse_json_lenient_unparseable_candidate_then_valid() {
        let v = parse_json_lenient(r#"{"a": } then {"b": 1}"#).unwrap();
        assert_eq!(
            v["b"], 1,
            "a balanced but invalid candidate is abandoned at its close; the scan resumes"
        );
    }

    #[test]
    fn parse_json_lenient_unterminated_json_returns_none() {
        let result = parse_json_lenient(r#"prefix {"a": 1"#);
        assert_eq!(
            result, None,
            "a candidate that never closes yields nothing, not a panic or a partial value"
        );
    }

    #[test]
    fn parse_json_lenient_nested_arrays() {
        let v = parse_json_lenient(r#"result [[1, 2], [3]]"#).unwrap();
        assert_eq!(v[0][1], 2);
        assert_eq!(v[1][0], 3);
    }

    #[test]
    fn parse_json_lenient_empty_containers_in_prose() {
        let empty_object = parse_json_lenient(r#"text {} more"#).unwrap();
        assert!(
            empty_object
                .as_object()
                .is_some_and(serde_json::Map::is_empty)
        );
        let empty_array = parse_json_lenient(r#"text [] more"#).unwrap();
        assert!(empty_array.as_array().is_some_and(Vec::is_empty));
    }

    #[test]
    fn parse_json_lenient_escaped_backslash_in_string() {
        let v = parse_json_lenient(r#"{"path": "c:\\"}"#).unwrap();
        assert_eq!(v["path"], "c:\\");
    }

    #[cfg(any(
        feature = "anthropic",
        feature = "grammar",
        feature = "openai",
        feature = "gemini"
    ))]
    #[test]
    fn tighten_required_union_applies_to_nested_objects() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "properties": {"lang": {"type": "string"}},
                    "required": ["secret"]
                }
            }
        });
        let tightened = tighten_json_schema(&schema);
        assert_eq!(
            tightened["properties"]["filter"]["required"],
            serde_json::json!(["secret", "lang"]),
            "the union applies at every object level, not just the root"
        );
    }
}

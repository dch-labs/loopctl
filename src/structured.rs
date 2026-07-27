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
    /// synthesized Anthropic forced-tool name. Must match
    /// `^[a-zA-Z0-9_-]+$` (alphanumeric, underscore, hyphen only) because
    /// both providers validate this identifier.
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
    /// and the Anthropic forced-tool name. Must match `^[a-zA-Z0-9_-]+$`.
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
    /// the schema. Providers that lack strict mode (Anthropic tool-forcing)
    /// ignore this flag.
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
/// non-breakingly, and the [`Grammar`](Self::Grammar) variant is only
/// present under the `grammar` feature.
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
/// reproduces prior behaviour, and providers that don't understand a field
/// ignore it. Carries [`response_format`](Self::response_format) (constrain
/// the model's free-text output to a schema) and
/// [`tool_constraint`](Self::tool_constraint) (constrain the model's tool
/// calls to the registered schemas).
///
/// The two paths are independent: setting `response_format` suppresses
/// `tools` (and therefore makes `tool_constraint` a no-op for that
/// request), while setting `tool_constraint` constrains the `tools` path
/// itself.
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
    /// Carries the underlying `serde_json::Error` with its exact location
    /// (line/column within the JSON). This typically means the model returned
    /// valid JSON but with missing fields, wrong types, or unexpected
    /// structure relative to `T`'s schema.
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
#[cfg(any(feature = "openai", feature = "gemini"))]
pub(crate) fn parse_json_lenient(text: &str) -> Option<serde_json::Value> {
    if let Ok(v) = serde_json::from_str(text) {
        return Some(v);
    }
    // Lenient rescue: find the outermost { ... } or [ ... ].
    extract_json_substring(text)
}

/// Find and parse the outermost JSON object or array in a string.
///
/// Scans for the first `{` or `[`, tracks brace/bracket depth, and extracts
/// the substring up to the matching close. String-aware: braces/brackets
/// inside JSON string literals (`"..."`) do not affect depth, and `\"`
/// escapes are honored.
#[cfg(any(feature = "openai", feature = "gemini"))]
pub(crate) fn extract_json_substring(text: &str) -> Option<serde_json::Value> {
    let bytes = text.as_bytes();
    let mut start = None;
    let mut depth: i32 = 0;
    let mut close = b'\0';
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
                    close = if byte == b'{' { b'}' } else { b']' };
                }
                depth = depth.saturating_add(1);
            }
            b'}' | b']' => {
                if let Some(s) = start {
                    if byte == close {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            let slice = text.get(s..=i).unwrap_or(text);
                            if let Ok(v) = serde_json::from_str(slice) {
                                return Some(v);
                            }
                            start = None;
                        }
                    } else {
                        // Mismatched closing delimiter — not valid JSON.
                        start = None;
                        depth = 0;
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
/// map: set `additionalProperties` to `false` and set `required` to the full
/// list of property keys. Non-object subschemas (`string`, `number`, etc.)
/// are returned unchanged; the implementation recurses into object
/// properties, `array` `items`, and the values of `allOf` / `anyOf` /
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
/// 2. setting `required` to the full list of that object's own property
///    keys — rejects missing fields the model omitted.
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
    obj.insert(
        "required".to_string(),
        serde_json::Value::Array(
            property_keys
                .into_iter()
                .map(serde_json::Value::String)
                .collect(),
        ),
    );
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
    let raw = client
        .create_message_with_options(&request, opts)
        .await
        .map_err(StructuredError::Api)?;
    let value = client.extract_structured(&raw);
    T::from_value(value)
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
                dyn Future<Output = Result<serde_json::Value, crate::api::error::ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async { Ok(serde_json::json!({})) })
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
                dyn Future<Output = Result<serde_json::Value, crate::api::error::ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async { Ok(serde_json::json!({})) })
        }
        fn create_message_with_options(
            &self,
            _request: &crate::api::StreamRequest,
            _options: RequestOptions,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<serde_json::Value, crate::api::error::ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(serde_json::json!({
                    "tool": "write",
                    "args": {"path": "/test"}
                }))
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
                dyn Future<Output = Result<serde_json::Value, crate::api::error::ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async { Ok(serde_json::json!({})) })
        }
        fn create_message_with_options(
            &self,
            _request: &crate::api::StreamRequest,
            _options: RequestOptions,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<serde_json::Value, crate::api::error::ApiError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async { Ok(serde_json::json!("I cannot produce that.")) })
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

    #[test]
    fn tighten_sets_additional_properties_false() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "string"}}
        });
        let tightened = tighten_json_schema(&schema);
        assert_eq!(tightened["additionalProperties"], false);
    }

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

    #[test]
    fn tighten_preserves_non_object_schemas() {
        let schema = serde_json::json!({"type": "string"});
        let tightened = tighten_json_schema(&schema);
        assert_eq!(tightened, schema);
        // Should not have gained additionalProperties / required.
        assert!(tightened.get("additionalProperties").is_none());
        assert!(tightened.get("required").is_none());
    }

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

    #[test]
    fn tighten_object_without_properties() {
        let schema = serde_json::json!({"type": "object"});
        let tightened = tighten_json_schema(&schema);
        assert_eq!(tightened["additionalProperties"], false);
        // `required` becomes an empty array, not absent.
        assert_eq!(tightened["required"].as_array().unwrap().len(), 0);
    }

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
}

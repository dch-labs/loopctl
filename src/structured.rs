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

/// Optional per-request knobs layered on top of a `stream_messages` /
/// `create_message` call.
///
/// Additive and forward-compatible: every field is optional, and providers
/// that don't understand a field ignore it. Today carries only
/// [`response_format`](Self::response_format); future tasks (max-tokens,
/// temperature, seed) extend it without touching the trait again.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    /// If set, ask the model to return JSON conforming to this schema.
    ///
    /// When `Some`, the provider injects the schema into the request
    /// (OpenAI `response_format` / Anthropic forced tool). When `None`,
    /// the model's output is unconstrained — the default behaviour.
    pub response_format: Option<ResponseFormat>,
}

impl RequestOptions {
    /// Create empty options with no response format set.
    ///
    /// Equivalent to [`RequestOptions::default`]. Use
    /// [`response_format`](Self::response_format) to chain a format
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
    pub fn response_format(mut self, rf: ResponseFormat) -> Self {
        self.response_format = Some(rf);
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
                    depth = depth.saturating_sub(1);
                    if depth == 0 && byte == close {
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
    let opts = RequestOptions::new().response_format(ResponseFormat::from_type::<T>());
    let raw = client
        .create_message_with_options(messages, system, None, opts)
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
        let opts = RequestOptions::new().response_format(rf);
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

    struct PlainMockClient;
    impl crate::api::ApiClient for PlainMockClient {
        fn model(&self) -> String {
            "test".to_string()
        }
        fn stream_messages(
            &self,
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<crate::tool::ToolSchema>>,
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
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<crate::tool::ToolSchema>>,
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
        let opts = RequestOptions::new().response_format(ResponseFormat::from_type::<Action>());
        let result = client
            .create_message_with_options(vec![], None, None, opts)
            .await;
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
        let result = client
            .create_message_with_options(vec![], None, None, opts)
            .await;
        assert!(result.is_ok(), "empty options should delegate normally");
    }

    struct StructuredMockClient;
    impl crate::api::ApiClient for StructuredMockClient {
        fn model(&self) -> String {
            "test".to_string()
        }
        fn stream_messages(
            &self,
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<crate::tool::ToolSchema>>,
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
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<crate::tool::ToolSchema>>,
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
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<crate::tool::ToolSchema>>,
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
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<crate::tool::ToolSchema>>,
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
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<crate::tool::ToolSchema>>,
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
            _messages: Vec<Message>,
            _system: Option<String>,
            _tools: Option<Vec<crate::tool::ToolSchema>>,
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
}

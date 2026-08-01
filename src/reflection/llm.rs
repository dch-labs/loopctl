//! LLM-powered failure reflection.
//!
//! [`LlmReflector`] asks the model to classify a failed tool call and
//! suggest a correction, returning a typed [`FailureAnalysis`] via the
//! [`StructuredOutput`](crate::structured::StructuredOutput) trait. This is
//! the first in-tree consumer of structured output, and it makes tool-error
//! recovery semantically intelligent instead of heuristic.
//!
//! # Construction
//!
//! ```rust,ignore
//! use loopctl::reflection::LlmReflector;
//! use std::sync::Arc;
//!
//! let reflector = LlmReflector::new(client);
//! agent.set_reflector(Arc::new(reflector));
//! ```
//!
//! # Latency and cost
//!
//! Each analyzed tool failure triggers one model round-trip. The reflector
//! is opt-in — the framework's default reflector is
//! [`NoopReflector`](super::NoopReflector), which performs no I/O. Only
//! callers that explicitly install an `LlmReflector` pay the per-failure
//! round-trip cost.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::api::ApiClient;
use crate::message::Message;
use crate::reflection::{FailureAnalysis, ReflectionContext, ReflectionError, Reflector};
use crate::structured::request_structured;
use crate::tool::ToolSchema;

/// The default system prompt used when the caller does not supply one via
/// [`LlmReflector::with_system_prompt`].
const DEFAULT_PROMPT: &str = "\
You are a tool-failure analyst for an LLM agent loop. Given a tool name, \
the JSON input that was passed, the tool's input schema (if provided), and \
the error that resulted, classify the failure and suggest a correction.\n\
\n\
Respond with a single JSON object matching this exact shape:\n\
{\n\
  \"is_recoverable\": <boolean>,\n\
  \"root_cause\": <string>,\n\
  \"severity\": \"low\" | \"medium\" | \"high\" | \"critical\",\n\
  \"correction\": {\n\
    \"correction_type\": \"input_fix\" | \"tool_change\" | \"prerequisite_fix\" | \"approach_change\" | \"escalate\",\n\
    \"description\": <string>,\n\
    \"modified_input\": <object or null>,\n\
    \"alternative_tool\": <string or null>,\n\
    \"guidance\": <string or null>\n\
  } | null,\n\
  \"context\": <string>\n\
}\n\
\n\
Only set \"modified_input\" when \"correction_type\" is \"input_fix\", and only \
when you can produce an input that conforms to the tool's schema. Prefer \
\"is_recoverable\": false over inventing a correction.";

/// A [`Reflector`] that asks an LLM to classify tool failures and suggest
/// corrections.
///
/// Holds a shared API client handle and an optional system-prompt override.
/// On each call to [`analyze`](Reflector::analyze) it builds a single user
/// message describing the failure, calls
/// [`request_structured::<FailureAnalysis>`](crate::structured::request_structured),
/// and returns the typed analysis.
///
/// When the `schema_validation` feature is enabled and the engine supplies
/// the failing tool's schema, the reflector validates the model's
/// `Correction::modified_input` against that schema and returns
/// [`ReflectionError::Internal`] on a mismatch.
pub struct LlmReflector {
    /// The shared API client used to make the structured-output call on each
    /// failure analysis.
    ///
    /// Held as an `Arc<dyn ApiClient>` so the reflector can outlive the
    /// borrow that [`Reflector::analyze`] is called under: the returned
    /// future borrows `&self`, but `request_structured` takes `&dyn
    /// ApiClient` — the `Arc` lets the future reach the client without
    /// capturing `&self`'s borrow chain. The trait's `'static + Send +
    /// Sync` bound rules out borrowing the client by lifetime.
    client: Arc<dyn ApiClient>,

    /// The system prompt sent with every analysis request, instructing the
    /// model to return JSON conforming to the [`FailureAnalysis`] schema.
    ///
    /// Defaults to [`DEFAULT_PROMPT`] when the reflector is constructed via
    /// [`new`](Self::new); replace it with
    /// [`with_system_prompt`](Self::with_system_prompt). Cloned per
    /// `analyze` call so the future owns its copy.
    ///
    /// [`FailureAnalysis`]: crate::reflection::FailureAnalysis
    system_prompt: String,
}

impl LlmReflector {
    /// Construct an `LlmReflector` backed by the given shared client.
    ///
    /// Uses the module's default system prompt. Override with
    /// [`with_system_prompt`](Self::with_system_prompt).
    #[must_use]
    pub fn new(client: Arc<dyn ApiClient>) -> Self {
        Self {
            client,
            system_prompt: DEFAULT_PROMPT.to_string(),
        }
    }

    /// Replace the default system prompt with a caller-supplied one.
    ///
    /// Builder-style; consumes and returns `self`. The prompt should still
    /// instruct the model to return JSON conforming to the
    /// `FailureAnalysis` schema, since the response is parsed via
    /// [`StructuredOutput`](crate::structured::StructuredOutput).
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }
}

impl std::fmt::Debug for LlmReflector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmReflector")
            .field("client", &"<dyn ApiClient>")
            .field(
                "system_prompt",
                &format!("{} chars", self.system_prompt.len()),
            )
            .finish()
    }
}

impl Reflector for LlmReflector {
    fn analyze(
        &self,
        error: &str,
        tool_name: &str,
        tool_input: &serde_json::Value,
        tool_schema: Option<&ToolSchema>,
        context: &ReflectionContext,
    ) -> Pin<Box<dyn Future<Output = Result<FailureAnalysis, ReflectionError>> + Send + '_>> {
        let schema_value = tool_schema.map(|s| s.input_schema.clone());
        let user_message =
            build_user_message(error, tool_name, tool_input, schema_value.as_ref(), context);
        let system = self.system_prompt.clone();
        let client = std::sync::Arc::clone(&self.client);

        Box::pin(async move {
            let analysis = request_structured::<FailureAnalysis>(
                &*client,
                vec![Message::user(user_message)],
                Some(system),
            )
            .await
            .map_err(|e| ReflectionError::Internal(format!("{e}")))?;

            #[cfg(feature = "schema_validation")]
            validate_modified_input(&analysis, schema_value.as_ref())?;
            Ok(analysis)
        })
    }
}

/// Build the single user message describing the failure.
///
/// Carries the error message, the tool name, the serialized tool input,
/// the tool's input schema (when available), and the task description
/// from the reflection context so the model has everything it needs to
/// produce a typed `FailureAnalysis`. The schema block is omitted
/// entirely when `tool_schema` is `None` (the engine passes `None` when
/// the tool isn't in the registry) so the model isn't shown a misleading
/// empty placeholder.
fn build_user_message(
    error: &str,
    tool_name: &str,
    tool_input: &serde_json::Value,
    tool_schema: Option<&serde_json::Value>,
    context: &ReflectionContext,
) -> String {
    let schema_line = match tool_schema {
        Some(schema) => format!("Schema: {schema}\n"),
        None => String::new(),
    };
    format!(
        "Tool: {tool_name}\n\
         Input: {tool_input}\n\
         {schema_line}\
         Error: {error}\n\
         Task: {task}\n\
         Attempt: {attempt} of {max}",
        tool_name = tool_name,
        tool_input = tool_input,
        schema_line = schema_line,
        error = error,
        task = context.task,
        attempt = context.attempt.saturating_add(1),
        max = context.max_attempts,
    )
}

/// Validate the model's suggested `modified_input` against the failing
/// tool's input schema.
///
/// Returns `Ok(())` when there is nothing to check (no correction, no
/// `modified_input`, or no schema supplied), or
/// [`ReflectionError::Internal`] when a supplied schema does not conform.
///
/// The whole function (and its single call site) is gated behind the
/// `schema_validation` feature: without it there is no validation to do,
/// so neither the function nor the call exist. This keeps the no-feature
/// build free of dead no-op stubs.
///
/// # Errors
///
/// Returns [`ReflectionError::Internal`] only when a schema is supplied
/// and `modified_input` does not conform to it.
#[cfg(feature = "schema_validation")]
fn validate_modified_input(
    analysis: &FailureAnalysis,
    tool_schema: Option<&serde_json::Value>,
) -> Result<(), ReflectionError> {
    let Some(correction) = &analysis.correction else {
        return Ok(());
    };
    let Some(modified_input) = &correction.modified_input else {
        return Ok(());
    };
    let Some(schema) = tool_schema else {
        return Ok(());
    };

    if !jsonschema::is_valid(schema, modified_input) {
        return Err(ReflectionError::Internal(
            "corrected input does not match the tool's schema".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::error::ApiError;
    use crate::message::MessagePart;
    use crate::reflection::{Correction, CorrectionType, FailureSeverity};
    use crate::structured::RequestOptions;
    use crate::tool::ToolSchema;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;

    /// A capture of what the reflector sent to the client, plus the canned
    /// response to return.
    #[derive(Clone)]
    struct Captured {
        system: Option<String>,
        user: Option<String>,
    }

    /// Mock returning a canned `FailureAnalysis`-shaped JSON value.
    struct CannedMock {
        response: serde_json::Value,
        captured: Arc<Mutex<Option<Captured>>>,
    }

    impl ApiClient for CannedMock {
        fn model(&self) -> String {
            "test".to_string()
        }
        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn futures::Stream<Item = Result<crate::stream::StreamEvent, ApiError>>
                    + Send
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
                dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_,
            >,
        > {
            let message = crate::message::Message::assistant(self.response.to_string());
            Box::pin(async move {
                Ok(crate::api::NonStreamingResponse {
                    message,
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
        fn create_message_with_options(
            &self,
            request: &crate::api::StreamRequest,
            _options: RequestOptions,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_,
            >,
        > {
            let crate::api::StreamRequest {
                messages,
                system,
                tools: _,
            } = request;
            let user = messages.first().and_then(|m| {
                m.parts.first().and_then(|p| match p {
                    MessagePart::Text { text } => Some(text.clone()),
                    _ => None,
                })
            });
            *self.captured.lock().unwrap() = Some(Captured {
                system: system.clone(),
                user,
            });
            let message = crate::message::Message::assistant(self.response.to_string());
            Box::pin(async move {
                Ok(crate::api::NonStreamingResponse {
                    message,
                    stop_reason: crate::stream::StreamStopReason::EndTurn,
                    usage: Some(crate::stream::Usage::default()),
                })
            })
        }
    }

    /// Mock returning a structured-output API error.
    struct ErrorMock;
    impl ApiClient for ErrorMock {
        fn model(&self) -> String {
            "test".to_string()
        }
        fn stream_messages(
            &self,
            _request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn futures::Stream<Item = Result<crate::stream::StreamEvent, ApiError>>
                    + Send
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
                dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_,
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
                dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_,
            >,
        > {
            Box::pin(async { Err(ApiError::http("upstream 500".to_string())) })
        }
    }

    /// Mock returning prose (a valid JSON string, but not an object matching
    /// the FailureAnalysis schema) to exercise the deserialize-error path.
    struct ProseMock(ErrorMock);
    impl ApiClient for ProseMock {
        fn model(&self) -> String {
            self.0.model()
        }
        fn stream_messages(
            &self,
            request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn futures::Stream<Item = Result<crate::stream::StreamEvent, ApiError>>
                    + Send
                    + 'static,
            >,
        > {
            self.0.stream_messages(request)
        }
        fn create_message(
            &self,
            request: &crate::api::StreamRequest,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_,
            >,
        > {
            self.0.create_message(request)
        }
        fn create_message_with_options(
            &self,
            _request: &crate::api::StreamRequest,
            _options: RequestOptions,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<crate::api::NonStreamingResponse, ApiError>> + Send + '_,
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

    fn fixture_analysis() -> serde_json::Value {
        serde_json::json!({
            "is_recoverable": true,
            "root_cause": "file not found",
            "severity": "medium",
            "correction": {
                "correction_type": "input_fix",
                "description": "fix the path",
                "modified_input": {"path": "/correct/path"},
                "alternative_tool": null,
                "guidance": null
            },
            "context": "open() call"
        })
    }

    fn ctx() -> ReflectionContext {
        ReflectionContext {
            task: "fix the bug".to_string(),
            attempt: 0,
            max_attempts: 3,
        }
    }

    #[tokio::test]
    async fn llm_reflector_returns_typed_analysis() {
        let captured = Arc::new(Mutex::new(None));
        let client = Arc::new(CannedMock {
            response: fixture_analysis(),
            captured: captured.clone(),
        });
        let reflector = LlmReflector::new(client);
        let analysis = reflector
            .analyze(
                "open: file not found",
                "read",
                &serde_json::json!({"path": "/wrong"}),
                None,
                &ctx(),
            )
            .await
            .expect("should succeed");
        assert!(analysis.is_recoverable);
        assert_eq!(analysis.root_cause, "file not found");
        assert_eq!(analysis.severity, FailureSeverity::Medium);
        let correction = analysis.correction.expect("correction present");
        assert_eq!(correction.correction_type, CorrectionType::InputFix);
        assert_eq!(correction.description, "fix the path");
        assert_eq!(
            correction.modified_input,
            Some(serde_json::json!({"path": "/correct/path"}))
        );
    }

    #[tokio::test]
    async fn llm_reflector_api_error_maps_to_internal() {
        let client: Arc<dyn ApiClient> = Arc::new(ErrorMock);
        let reflector = LlmReflector::new(client);
        let err = reflector
            .analyze("e", "t", &serde_json::json!({}), None, &ctx())
            .await
            .expect_err("should fail");
        assert!(
            matches!(err, ReflectionError::Internal(ref msg) if msg.contains("upstream 500")),
            "got: {err:?}"
        );
    }

    #[tokio::test]
    async fn llm_reflector_prose_maps_to_internal() {
        let client: Arc<dyn ApiClient> = Arc::new(ProseMock(ErrorMock));
        let reflector = LlmReflector::new(client);
        let err = reflector
            .analyze("e", "t", &serde_json::json!({}), None, &ctx())
            .await
            .expect_err("should fail");
        // Prose → deserialize error → Internal.
        assert!(matches!(err, ReflectionError::Internal(_)), "got: {err:?}");
    }

    #[tokio::test]
    async fn llm_reflector_uses_default_prompt() {
        let captured = Arc::new(Mutex::new(None));
        let client = Arc::new(CannedMock {
            response: fixture_analysis(),
            captured: captured.clone(),
        });
        let reflector = LlmReflector::new(client);
        let result = reflector
            .analyze("e", "t", &serde_json::json!({}), None, &ctx())
            .await;
        assert!(result.is_ok(), "analyze should succeed: {:?}", result.err());
        let cap = captured.lock().unwrap().clone().expect("captured");
        assert_eq!(cap.system.as_deref(), Some(DEFAULT_PROMPT));
    }

    #[tokio::test]
    async fn llm_reflector_with_system_prompt_overrides() {
        let captured = Arc::new(Mutex::new(None));
        let client = Arc::new(CannedMock {
            response: fixture_analysis(),
            captured: captured.clone(),
        });
        let reflector = LlmReflector::new(client).with_system_prompt("custom analyst prompt");
        let result = reflector
            .analyze("e", "t", &serde_json::json!({}), None, &ctx())
            .await;
        assert!(result.is_ok(), "analyze should succeed: {:?}", result.err());
        let cap = captured.lock().unwrap().clone().expect("captured");
        assert_eq!(cap.system.as_deref(), Some("custom analyst prompt"));
    }

    #[tokio::test]
    async fn llm_reflector_user_message_carries_all_fields() {
        let captured = Arc::new(Mutex::new(None));
        let client = Arc::new(CannedMock {
            response: fixture_analysis(),
            captured: captured.clone(),
        });
        let reflector = LlmReflector::new(client);
        let result = reflector
            .analyze(
                "the error text",
                "the_tool",
                &serde_json::json!({"k": "v"}),
                None,
                &ctx(),
            )
            .await;
        assert!(result.is_ok(), "analyze should succeed: {:?}", result.err());
        let cap = captured.lock().unwrap().clone().expect("captured");
        let user = cap.user.expect("user message");
        assert!(user.contains("the error text"), "user: {user}");
        assert!(user.contains("the_tool"), "user: {user}");
        assert!(user.contains("\"k\""), "user: {user}");
        assert!(user.contains("fix the bug"), "user: {user}");
    }

    #[tokio::test]
    async fn llm_reflector_user_message_includes_schema() {
        // When the engine supplies a tool schema, the reflector should
        // forward it in the user message so the model can produce a
        // schema-conforming modified_input.
        let captured = Arc::new(Mutex::new(None));
        let client = Arc::new(CannedMock {
            response: fixture_analysis(),
            captured: captured.clone(),
        });
        let reflector = LlmReflector::new(client);
        let tool_schema = ToolSchema {
            tool: "read".into(),
            description: "Read a file".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        };
        let result = reflector
            .analyze(
                "e",
                "read",
                &serde_json::json!({"path": "/wrong"}),
                Some(&tool_schema),
                &ctx(),
            )
            .await;
        assert!(result.is_ok(), "analyze should succeed: {:?}", result.err());
        let cap = captured.lock().unwrap().clone().expect("captured");
        let user = cap.user.expect("user message");
        assert!(
            user.contains("Schema:"),
            "expected the schema to appear in the user message: {user}"
        );
        assert!(
            user.contains("\"required\""),
            "expected schema content in the user message: {user}"
        );
    }

    #[cfg(feature = "schema_validation")]
    #[tokio::test]
    async fn llm_reflector_validates_modified_input_pass() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": false
        });
        let tool_schema = ToolSchema {
            tool: "read".into(),
            description: "Read a file".into(),
            input_schema: schema,
        };
        let client = Arc::new(CannedMock {
            response: fixture_analysis(),
            captured: Arc::new(Mutex::new(None)),
        });
        let reflector = LlmReflector::new(client);
        // modified_input is {"path": "/correct/path"}, which matches.
        let result = reflector
            .analyze(
                "e",
                "read",
                &serde_json::json!({}),
                Some(&tool_schema),
                &ctx(),
            )
            .await;
        assert!(
            result.is_ok(),
            "valid modified_input should pass: {:?}",
            result.err()
        );
    }

    #[cfg(feature = "schema_validation")]
    #[tokio::test]
    async fn llm_reflector_validates_modified_input_fail() {
        // Tool schema requires {path: string}, but the fixture's
        // modified_input is {"path": "/correct/path"} — that matches.
        // Construct a schema it *doesn't* match (requires a number).
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"n": {"type": "number"}},
            "required": ["n"],
            "additionalProperties": false
        });
        let tool_schema = ToolSchema {
            tool: "calc".into(),
            description: "Calculate".into(),
            input_schema: schema,
        };
        let client = Arc::new(CannedMock {
            response: fixture_analysis(),
            captured: Arc::new(Mutex::new(None)),
        });
        let reflector = LlmReflector::new(client);
        let err = reflector
            .analyze(
                "e",
                "calc",
                &serde_json::json!({}),
                Some(&tool_schema),
                &ctx(),
            )
            .await
            .expect_err("invalid modified_input should fail");
        assert!(
            matches!(err, ReflectionError::Internal(ref m) if m.contains("does not match")),
            "got: {err:?}"
        );
    }

    #[cfg(feature = "schema_validation")]
    #[tokio::test]
    async fn llm_reflector_skips_validation_when_no_schema() {
        // tool_schema = None; even if modified_input is weird, we return Ok.
        let client = Arc::new(CannedMock {
            response: fixture_analysis(),
            captured: Arc::new(Mutex::new(None)),
        });
        let reflector = LlmReflector::new(client);
        let result = reflector
            .analyze("e", "t", &serde_json::json!({}), None, &ctx())
            .await;
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(feature = "schema_validation")]
    fn validate_modified_input_noop_without_correction() {
        let analysis = FailureAnalysis {
            is_recoverable: false,
            root_cause: "x".to_string(),
            severity: FailureSeverity::Low,
            correction: None,
            context: String::new(),
        };
        // With a schema but no correction → Ok.
        assert!(validate_modified_input(&analysis, Some(&serde_json::json!({}))).is_ok());
    }

    #[test]
    #[cfg(feature = "schema_validation")]
    fn validate_modified_input_noop_without_modified_input() {
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "x".to_string(),
            severity: FailureSeverity::Low,
            correction: Some(Correction {
                correction_type: CorrectionType::ApproachChange,
                description: "no modified input".to_string(),
                modified_input: None,
                alternative_tool: None,
                guidance: None,
            }),
            context: String::new(),
        };
        assert!(validate_modified_input(&analysis, Some(&serde_json::json!({}))).is_ok());
    }

    #[test]
    #[cfg(feature = "schema_validation")]
    fn validate_modified_input_skips_when_no_schema() {
        // Even with a modified_input present, no schema → Ok (the engine
        // passes None when the tool isn't in the registry).
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "x".to_string(),
            severity: FailureSeverity::Low,
            correction: Some(Correction {
                correction_type: CorrectionType::InputFix,
                description: "fix".to_string(),
                modified_input: Some(serde_json::json!({"anything": true})),
                alternative_tool: None,
                guidance: None,
            }),
            context: String::new(),
        };
        assert!(validate_modified_input(&analysis, None).is_ok());
    }

    #[test]
    #[cfg(feature = "schema_validation")]
    fn validate_modified_input_rejects_mismatched_schema() {
        let analysis = FailureAnalysis {
            is_recoverable: true,
            root_cause: "x".to_string(),
            severity: FailureSeverity::Low,
            correction: Some(Correction {
                correction_type: CorrectionType::InputFix,
                description: "fix".to_string(),
                modified_input: Some(serde_json::json!({"wrong": "shape"})),
                alternative_tool: None,
                guidance: None,
            }),
            context: String::new(),
        };
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"n": {"type": "number"}},
            "required": ["n"],
            "additionalProperties": false
        });
        let result = validate_modified_input(&analysis, Some(&schema));
        assert!(
            matches!(result, Err(ReflectionError::Internal(_))),
            "with schema_validation the mismatch should fail: {result:?}"
        );
    }

    #[test]
    fn build_user_message_contains_all_fields() {
        let msg = build_user_message(
            "the error",
            "the_tool",
            &serde_json::json!({"k": "v"}),
            None,
            &ReflectionContext {
                task: "the task".to_string(),
                attempt: 1,
                max_attempts: 4,
            },
        );
        assert!(msg.contains("the_tool"), "tool name missing: {msg}");
        assert!(msg.contains("the error"), "error missing: {msg}");
        assert!(msg.contains("\"k\""), "input missing: {msg}");
        assert!(msg.contains("the task"), "task missing: {msg}");
    }

    #[test]
    fn build_user_message_attempt_is_one_indexed() {
        // attempt is 0-indexed in ReflectionContext; the message should
        // render it 1-indexed ("Attempt: 1 of N" for attempt=0).
        let msg = build_user_message(
            "e",
            "t",
            &serde_json::json!({}),
            None,
            &ReflectionContext {
                task: "x".to_string(),
                attempt: 0,
                max_attempts: 3,
            },
        );
        assert!(
            msg.contains("Attempt: 1 of 3"),
            "expected 1-indexed attempt in: {msg}"
        );
    }

    #[test]
    fn build_user_message_saturates_attempt_overflow() {
        // u32::MAX + 1 must saturate rather than panic.
        let msg = build_user_message(
            "e",
            "t",
            &serde_json::json!({}),
            None,
            &ReflectionContext {
                task: "x".to_string(),
                attempt: u32::MAX,
                max_attempts: u32::MAX,
            },
        );
        // Should contain u32::MAX (saturated, not overflowed).
        assert!(msg.contains(&u32::MAX.to_string()));
    }

    #[test]
    fn build_user_message_includes_schema_when_present() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        });
        let msg = build_user_message(
            "e",
            "read",
            &serde_json::json!({"path": "/wrong"}),
            Some(&schema),
            &ReflectionContext {
                task: "x".to_string(),
                attempt: 0,
                max_attempts: 3,
            },
        );
        assert!(
            msg.contains("Schema:"),
            "expected a Schema: line when schema is supplied: {msg}"
        );
        assert!(
            msg.contains("\"required\""),
            "schema content missing from message: {msg}"
        );
    }

    #[test]
    fn build_user_message_omits_schema_line_when_none() {
        let msg = build_user_message(
            "e",
            "t",
            &serde_json::json!({}),
            None,
            &ReflectionContext {
                task: "x".to_string(),
                attempt: 0,
                max_attempts: 3,
            },
        );
        assert!(
            !msg.contains("Schema:"),
            "no Schema: line expected when schema is None: {msg}"
        );
    }
}

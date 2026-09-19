//! Tool-call grammar providers for grammar-aware samplers.
//!
//! When a request opts into [`ToolConstraint::Grammar`](crate::structured::ToolConstraint::Grammar),
//! the provider serializes the grammar produced by a
//! [`ToolGrammarProvider`] into the request body (e.g. vLLM's `guided_json`).
//! This makes malformed tool calls structurally impossible at the sampler
//! level rather than relying on the model to emit valid JSON.
//!
//! # The two constraint envelopes
//!
//! Grammar constraining and
//! [`ToolConstraint::Strict`](crate::structured::ToolConstraint::Strict)
//! are different mechanisms that share a goal, and the docs keep them
//! apart:
//!
//! * **Whole-output guided decoding** (this module): the server's
//!   sampler constrains the *entire completion* to a single JSON
//!   object — the envelope [`JsonSchemaGrammar`] emits, keyed by tool
//!   name (`{"<tool>": <args>}`). A server that honors the field
//!   returns that one object as plain text; parsing it back into a
//!   tool call is the **consumer's** job — loopctl does not do it, so
//!   a grammar-honoring response surfaces as message text, not as a
//!   parsed tool call. Under this constraint the OpenAI-compatible
//!   request still advertises the tools natively alongside the
//!   grammar, so a server that ignores the guided field can answer
//!   with an ordinary native tool call instead.
//! * **API-native tool constraining**
//!   ([`ToolConstraint::Strict`](crate::structured::ToolConstraint::Strict)):
//!   the provider itself enforces the tool-call format (OpenAI strict
//!   function calling, Anthropic forced tools); the wire stays the
//!   provider's native tool-call shape.
//!
//! Provider support for the grammar envelope: OpenAI-compatible
//! servers that expose a guided-decoding field (vLLM's
//! `guided_json`, and local servers offering one) accept it;
//! Anthropic, Gemini, and Bedrock Converse expose no grammar field,
//! and requesting the grammar constraint on them fails fast with a
//! typed error instead of degrading to unconstrained decoding. On the
//! OpenAI-compatible path the grammar rides only when the request
//! carries tools — with an empty tool list the guided field is
//! omitted and the request goes out unconstrained, because an empty
//! grammar would force the completion into an empty tool set.
//! API-native constraining is a per-provider capability independent
//! of this module.
//!
//! The default implementation, [`JsonSchemaGrammar`], compiles a slice of
//! [`ToolSchema`] values into a single JSON object whose top-level
//! `properties` key maps each tool name to its tightened `input_schema`.
//! Implement [`ToolGrammarProvider`] directly to target a different sampler
//! dialect (GBNF, TGI grammar, etc.).

use crate::api::error::ApiError;
use crate::structured::tighten_json_schema;
use crate::tool::ToolSchema;

/// Compiles a tool registry's schemas into a grammar the sampler must obey.
///
/// A provider serializes the grammar string into its request body using
/// whatever field the upstream server expects (e.g. vLLM `guided_json`).
/// Implement this trait to target a different sampler dialect or to produce
/// a non-JSON grammar (e.g. GBNF for llama.cpp).
pub trait ToolGrammarProvider: Send + Sync + std::fmt::Debug {
    /// The compiled grammar string.
    ///
    /// Callers may invoke this per request; implementations precompute
    /// the grammar (or lazily cache it in `self`, as the borrowed return
    /// requires) so the call stays cheap.
    fn grammar(&self) -> &str;
}

/// A [`ToolGrammarProvider`] that compiles tool schemas into a JSON object.
///
/// The grammar is a JSON object whose `properties` map each registered tool
/// name to its (tightened) `input_schema`. Tightening applies the same
/// `additionalProperties: false` + full `required` transform the
/// [`ToolConstraint::Strict`](crate::structured::ToolConstraint::Strict)
/// path uses, so a sampler guided by this grammar emits the same shape a
/// strict-mode API would enforce.
#[derive(Debug)]
pub struct JsonSchemaGrammar {
    grammar: String,
}

impl JsonSchemaGrammar {
    /// Compile a grammar from the given tool schemas.
    ///
    /// Each tool's `input_schema` is tightened (recursive
    /// `additionalProperties: false` and full `required`) so the resulting
    /// grammar enforces the same strict shape across samplers. The grammar
    /// string references each tool by name. An empty slice yields a valid
    /// (empty-properties) JSON object rather than erroring.
    ///
    /// # Errors
    ///
    /// Returns a config-validation [`ApiError`] when a tool's
    /// `input_schema` does not explicitly declare the string
    /// `type: "object"` — a declared non-object type in either the
    /// string or the array form, a schema with no `type` at all (an
    /// `enum`- or `const`-only schema can emit a non-object value, and
    /// the tightening pass never injects the missing type), or a schema
    /// document that is not an object — because the grammar envelope
    /// maps tool names to object schemas and guarantees object-valued
    /// arguments only under that explicit declaration. A defensive
    /// envelope-serialization failure is typed too. Everything fails
    /// here, at construction, before any request can leave with a
    /// silently unconstrained grammar.
    ///
    /// The array form of `type` is rejected even when it names only
    /// `"object"` (`{"type": ["object"]}`): the tightening pass reads
    /// the string form, so an accepted array form would ride the wire
    /// untightened. Write the string form.
    pub fn from_schemas(schemas: &[ToolSchema]) -> Result<Self, ApiError> {
        let mut props = serde_json::Map::new();
        for schema in schemas {
            let tightened = tighten_json_schema(&schema.input_schema);
            if let Some(reason) = non_object_reason(&tightened) {
                return Err(ApiError::config_validation(format!(
                    "grammar: tool `{}` constrains its arguments to a \
                     non-object type ({reason}) — tool arguments are a JSON \
                     object, so the whole-output grammar cannot guide \
                     anything else",
                    schema.tool
                )));
            }
            props.insert(schema.tool.clone(), tightened);
        }
        let grammar = serde_json::json!({
            "type": "object",
            "properties": serde_json::Value::Object(props),
            "additionalProperties": false,
        });
        let grammar = serde_json::to_string(&grammar).map_err(|error| {
            ApiError::config_validation(format!("grammar: envelope failed to serialize: {error}"))
        })?;
        Ok(Self { grammar })
    }
}

/// Why a schema cannot guarantee object-valued tool arguments, as a
/// phrase for the rejection error — the grammar envelope's fail-loud
/// reason.
///
/// Only the explicit string `type: "object"` passes: the tightening
/// pass never injects a missing type, and the envelope's
/// object-valued-arguments guarantee holds only under that explicit
/// declaration — so absence, an `enum`/`const`-only schema, a
/// non-string `type` value, and every array form (even `["object"]`,
/// which would ride untightened) all reject, each naming what the
/// schema declares instead.
fn non_object_reason(schema: &serde_json::Value) -> Option<String> {
    match schema {
        serde_json::Value::Object(map) => match map.get("type") {
            Some(serde_json::Value::String(declared)) => {
                if declared == "object" {
                    None
                } else {
                    Some(format!("type \"{declared}\""))
                }
            }
            Some(serde_json::Value::Array(declared)) => {
                if declared.is_empty() {
                    Some("an empty type array".to_string())
                } else {
                    let kinds: Vec<&str> = declared
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .collect();
                    if kinds.len() == declared.len() {
                        Some(format!("type array-form [{}]", kinds.join(", ")))
                    } else {
                        Some("a type array with non-string members".to_string())
                    }
                }
            }
            None => Some("no `type` declaration".to_string()),
            Some(_) => Some("a non-string `type` value".to_string()),
        },
        serde_json::Value::Bool(_) => Some("a boolean schema document".to_string()),
        serde_json::Value::Array(_) => Some("an array schema document".to_string()),
        serde_json::Value::Number(_) => Some("a number schema document".to_string()),
        serde_json::Value::String(_) => Some("a string schema document".to_string()),
        serde_json::Value::Null => Some("a null schema document".to_string()),
    }
}

impl ToolGrammarProvider for JsonSchemaGrammar {
    fn grammar(&self) -> &str {
        &self.grammar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schemas() -> Vec<ToolSchema> {
        vec![
            ToolSchema {
                tool: "search".into(),
                description: "Search the web".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"q": {"type": "string"}}
                }),
            },
            ToolSchema {
                tool: "calc".into(),
                description: "Calculate".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"expr": {"type": "string"}}
                }),
            },
        ]
    }

    #[test]
    fn json_schema_grammar_contains_each_tool_name() {
        let grammar = JsonSchemaGrammar::from_schemas(&sample_schemas())
            .expect("well-formed schemas compile");
        let parsed: serde_json::Value =
            serde_json::from_str(grammar.grammar()).expect("grammar must be valid JSON");

        assert_eq!(parsed["type"], "object", "top-level must be an object");
        assert_eq!(
            parsed["additionalProperties"], false,
            "top-level must reject unknown tools"
        );

        let properties = parsed["properties"]
            .as_object()
            .expect("top-level must have a properties map");
        let keys: Vec<&str> = properties.keys().map(String::as_str).collect();
        assert!(
            keys.contains(&"search"),
            "tool names must be keys of the properties map, got: {keys:?}"
        );
        assert!(
            keys.contains(&"calc"),
            "tool names must be keys of the properties map, got: {keys:?}"
        );
    }

    #[test]
    fn json_schema_grammar_empty_schemas_no_panic() {
        let grammar = JsonSchemaGrammar::from_schemas(&[]).expect("an empty slice compiles");
        let s = grammar.grammar();
        // Should parse back as a JSON object.
        let parsed: serde_json::Value = serde_json::from_str(s).unwrap();
        assert!(parsed.is_object());
    }

    #[test]
    fn grammar_rejects_non_object_schemas_loudly() {
        let bad = vec![ToolSchema {
            tool: "search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({
                "type": "array",
                "items": {"type": "string"}
            }),
        }];
        let error = JsonSchemaGrammar::from_schemas(&bad).expect_err(
            "a non-object input schema must fail compilation, not ride the wire unconstrained",
        );
        match error {
            ApiError::Config(message) => {
                assert!(
                    message.contains("grammar") && message.contains("search"),
                    "the error must name the seam and the failing tool: {message}"
                );
            }
            other => panic!("the failure must be config-validation, got {other:?}"),
        }

        // A boolean schema document (`false` = "nothing matches") is the
        // other reachable rejection: not an object schema at all.
        let boolean = vec![ToolSchema {
            tool: "calc".into(),
            description: "Calculate".into(),
            input_schema: serde_json::json!(false),
        }];
        let error = JsonSchemaGrammar::from_schemas(&boolean)
            .expect_err("a boolean schema document must fail compilation too");
        assert!(
            matches!(error, ApiError::Config(ref message) if message.contains("calc")),
            "the boolean-schema rejection names its tool: {error:?}"
        );

        // JSON Schema's array form of `type` declares the same constraint
        // a string does; each direction must reject, naming the form.
        for (schema, expected_fragment) in [
            (
                serde_json::json!({"type": ["array"], "items": {"type": "string"}}),
                "array-form [array]",
            ),
            (
                serde_json::json!({"type": ["object", "null"]}),
                "array-form [object, null]",
            ),
            (
                // Even the all-object array form rides untightened (the
                // tightening pass reads the string form), so it rejects too.
                serde_json::json!({"type": ["object"]}),
                "array-form [object]",
            ),
            (
                // Degenerate array forms are declarations too — reject,
                // do not fall back to "unprovable".
                serde_json::json!({"type": []}),
                "an empty type array",
            ),
            (
                serde_json::json!({"type": [1]}),
                "a type array with non-string members",
            ),
            // A schema with no `type` at all can emit non-object values
            // (`{"enum": [1]}` permits the number 1), and tightening never
            // injects the missing type — so absence rejects, fail-closed.
            (serde_json::json!({"enum": [1]}), "no `type` declaration"),
            (serde_json::json!({"const": []}), "no `type` declaration"),
        ] {
            let rejection = vec![ToolSchema {
                tool: "search".into(),
                description: "Search the web".into(),
                input_schema: schema,
            }];
            let error = JsonSchemaGrammar::from_schemas(&rejection)
                .expect_err("every non-explicitly-object type declaration must fail compilation");
            assert!(
                matches!(error, ApiError::Config(ref message) if message.contains(expected_fragment)),
                "the rejection names the declared form ({expected_fragment}): {error:?}"
            );
        }
    }

    #[test]
    fn envelope_docs_match_the_wire() {
        let schema = ToolSchema {
            tool: "search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"q": {"type": "string"}}
            }),
        };
        let grammar = JsonSchemaGrammar::from_schemas(std::slice::from_ref(&schema))
            .expect("a well-formed schema compiles");
        let parsed: serde_json::Value =
            serde_json::from_str(grammar.grammar()).expect("the grammar is valid JSON");
        let expected = serde_json::json!({
            "type": "object",
            "properties": {
                "search": crate::structured::tighten_json_schema(&schema.input_schema)
            },
            "additionalProperties": false,
        });
        assert_eq!(
            parsed, expected,
            "the emitted grammar is exactly the documented whole-output envelope: \
             the entire completion is one object keyed by tool name"
        );
    }

    #[test]
    fn tool_grammar_provider_is_object_safe() {
        let g: Box<dyn ToolGrammarProvider> =
            Box::new(JsonSchemaGrammar::from_schemas(&[]).expect("an empty slice compiles"));
        let arc: std::sync::Arc<dyn ToolGrammarProvider> = std::sync::Arc::new(
            JsonSchemaGrammar::from_schemas(&sample_schemas())
                .expect("well-formed schemas compile"),
        );
        assert!(!arc.grammar().is_empty());
        // The Box also works through the trait object.
        assert!(!g.grammar().is_empty());
        // Debug is required by the bound — formatting through the trait
        // object proves the impl exists.
        let debug = format!("{arc:?}");
        assert!(debug.contains("JsonSchemaGrammar"));
    }

    /// Live smoke against a local 7B with grammar support (vLLM
    /// `guided_json`): every prompt on the 4-prompt corpus must produce
    /// exactly one valid tool call — either a native tool call (object
    /// arguments for a registered tool) or, when the server honors the
    /// grammar, a whole-output envelope (text that parses as one object
    /// keyed by a registered tool name). A statistical corpus — and any
    /// percentage claim — waits for a larger benchmark corpus; four
    /// prompts prove the path works, not a rate. Does not run in CI.
    ///
    /// Run with: `LOOPCTL_E2E=1 OPENAI_BASE_URL=http://localhost:8000/v1 \
    /// OPENAI_API_KEY=dummy OPENAI_MODEL=<your-model> cargo test \
    /// --features openai,grammar live_small_model_tool_call_validity -- --ignored`
    #[cfg(feature = "openai")]
    #[tokio::test]
    #[ignore = "requires a local 7B model server with grammar support; set LOOPCTL_E2E=1 to run"]
    async fn live_small_model_tool_call_validity() {
        use crate::api::ApiClient;
        use crate::structured::{RequestOptions, ToolConstraint};
        use futures::StreamExt;

        let client = crate::provider::OpenAiClient::from_env().expect("OPENAI_* env");
        let schemas = sample_schemas();
        let registered: Vec<String> = schemas.iter().map(|s| s.tool.clone()).collect();
        let grammar = std::sync::Arc::new(
            JsonSchemaGrammar::from_schemas(&schemas).expect("the corpus schemas compile"),
        );
        let opts = RequestOptions::new().with_tool_constraint(ToolConstraint::Grammar(grammar));

        // A small fixed corpus of prompts that should each produce exactly
        // one valid tool call; the smoke requires all of them.
        let corpus = [
            "Search the web for 'rust async'.",
            "Calculate 2 + 2.",
            "Find docs about trait objects.",
            "Compute 17 * 23.",
        ];

        let mut valid = 0usize;
        let mut total = 0usize;
        for prompt in corpus {
            let stream = client.stream_messages_with_options(
                &crate::api::StreamRequest::new(vec![crate::message::Message::user(prompt)])
                    .with_tools(Some(schemas.clone())),
                opts.clone(),
            );
            let events: Vec<Result<crate::stream::StreamEvent, crate::api::error::ApiError>> =
                stream.collect().await;
            let events: Vec<_> = events
                .into_iter()
                .collect::<Result<_, _>>()
                .expect("stream ok");
            // The oracle is the shared helper, unit-pinned beside it:
            // either serving shape counts, anything else does not.
            total = total.saturating_add(1);
            if one_valid_tool_call(&events, &registered) {
                valid = valid.saturating_add(1);
            }
        }

        assert!(
            valid == total,
            "every prompt on the 4-prompt local corpus must yield exactly one \
             valid tool call — native object arguments for a registered tool, \
             or a whole-output envelope keyed by one ({valid}/{total}); the \
             statistical bar waits for a larger benchmark corpus"
        );
    }

    /// Whether one prompt's stream shows exactly one valid tool call.
    ///
    /// The live smoke's validity oracle, extracted so its judgment is
    /// unit-pinnable rather than living only behind the `#[ignore]`d
    /// live test. Two serving shapes count: exactly **one** native tool
    /// call for a registered tool whose accumulated argument fragments
    /// parse as one JSON object, or — when the server honored the
    /// grammar — text that parses as one whole-output envelope: a
    /// single-key JSON object keyed by a registered tool name whose
    /// value is itself an object. Two tool starts, non-object
    /// arguments, a non-object envelope value, an unregistered tool,
    /// and plain text all fail.
    fn one_valid_tool_call(events: &[crate::stream::StreamEvent], registered: &[String]) -> bool {
        let mut native_args = String::new();
        let mut envelope_text = String::new();
        let mut registered_starts = 0usize;
        for event in events {
            match event {
                crate::stream::StreamEvent::IndexedDelta(d) => match &d.delta {
                    crate::stream::DeltaPart::InputJson { partial_json } => {
                        native_args.push_str(partial_json);
                    }
                    crate::stream::DeltaPart::Text { text } => {
                        envelope_text.push_str(text);
                    }
                    _ => {}
                },
                crate::stream::StreamEvent::PartStart(part) => {
                    if let Some(crate::message::MessagePart::ToolCall { name, .. }) = &part.part
                        && registered.contains(name)
                    {
                        registered_starts = registered_starts.saturating_add(1);
                    }
                }
                _ => {}
            }
        }
        let native_valid = registered_starts == 1
            && serde_json::from_str::<serde_json::Value>(&native_args)
                .is_ok_and(|value| value.is_object());
        let envelope_valid =
            serde_json::from_str::<serde_json::Value>(&envelope_text).is_ok_and(|value| {
                value.as_object().is_some_and(|object| {
                    object.len() == 1
                        && object
                            .iter()
                            .next()
                            .is_some_and(|(key, args)| registered.contains(key) && args.is_object())
                })
            });
        native_valid || envelope_valid
    }

    /// A `PartStart` opening a native tool-call lane for `name`.
    fn tool_start(name: &str) -> crate::stream::StreamEvent {
        crate::stream::StreamEvent::PartStart(crate::stream::PartStart {
            index: 0,
            part: Some(crate::message::MessagePart::tool_call(
                "call_test",
                name,
                serde_json::json!({}),
            )),
        })
    }

    /// An `IndexedDelta` carrying one fragment of native tool arguments.
    fn json_delta(fragment: &str) -> crate::stream::StreamEvent {
        crate::stream::StreamEvent::IndexedDelta(crate::stream::IndexedDelta {
            index: 0,
            delta: crate::stream::DeltaPart::InputJson {
                partial_json: fragment.to_string(),
            },
        })
    }

    /// An `IndexedDelta` carrying one fragment of completion text.
    fn text_delta(text: &str) -> crate::stream::StreamEvent {
        crate::stream::StreamEvent::IndexedDelta(crate::stream::IndexedDelta {
            index: 0,
            delta: crate::stream::DeltaPart::Text {
                text: text.to_string(),
            },
        })
    }

    #[test]
    fn the_smoke_oracle_accepts_one_valid_call_and_rejects_the_rest() {
        let registered = vec!["search".to_string(), "calc".to_string()];

        let events = vec![
            tool_start("search"),
            json_delta(r#"{"q":"#),
            json_delta(r#""rust"}"#),
        ];
        assert!(
            one_valid_tool_call(&events, &registered),
            "one native registered call with object arguments is valid"
        );

        // Two registered starts whose fragments concatenate into one
        // object: not one tool call, however the fragments merge.
        let events = vec![
            tool_start("search"),
            tool_start("calc"),
            json_delta(r#"{"a":"#),
            json_delta(r#"1}"#),
        ];
        assert!(
            !one_valid_tool_call(&events, &registered),
            "two tool starts cannot pass as one call"
        );

        let events = vec![tool_start("calc"), json_delta("42")];
        assert!(
            !one_valid_tool_call(&events, &registered),
            "non-object native arguments are not a valid call"
        );

        let events = vec![tool_start("unknown"), json_delta(r#"{"x":1}"#)];
        assert!(
            !one_valid_tool_call(&events, &registered),
            "an unregistered tool is not a valid call"
        );

        let events = vec![text_delta(r#"{"search": {"q":"#), text_delta(r#""rust"}}"#)];
        assert!(
            one_valid_tool_call(&events, &registered),
            "a well-formed whole-output envelope is valid"
        );

        let events = vec![text_delta(r#"{"search": 42}"#)];
        assert!(
            !one_valid_tool_call(&events, &registered),
            "an envelope value that is not an object is not a tool call"
        );

        let events = vec![text_delta(r#"{"unknown": {}}"#)];
        assert!(
            !one_valid_tool_call(&events, &registered),
            "an envelope keyed by an unregistered tool is not a valid call"
        );

        let events = vec![text_delta("hello")];
        assert!(
            !one_valid_tool_call(&events, &registered),
            "plain text is not a tool call"
        );
    }
}

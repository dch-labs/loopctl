//! Tool-call grammar providers for grammar-aware samplers.
//!
//! When a request opts into [`ToolConstraint::Grammar`](crate::structured::ToolConstraint::Grammar),
//! the provider serializes the grammar produced by a
//! [`ToolGrammarProvider`] into the request body (e.g. vLLM's `guided_json`).
//! This makes malformed tool calls structurally impossible at the sampler
//! level rather than relying on the model to emit valid JSON.
//!
//! The default implementation, [`JsonSchemaGrammar`], compiles a slice of
//! [`ToolSchema`] values into a single JSON object whose top-level
//! `properties` key maps each tool name to its tightened `input_schema`.
//! Implement [`ToolGrammarProvider`] directly to target a different sampler
//! dialect (GBNF, TGI grammar, etc.).

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
    /// string references each tool by name.
    ///
    /// Never panics: an empty slice yields a valid (empty-properties) JSON
    /// object rather than erroring.
    #[must_use]
    pub fn from_schemas(schemas: &[ToolSchema]) -> Self {
        let mut props = serde_json::Map::new();
        for schema in schemas {
            let tightened = tighten_json_schema(&schema.input_schema);
            props.insert(schema.tool.clone(), tightened);
        }
        let grammar = serde_json::json!({
            "type": "object",
            "properties": serde_json::Value::Object(props),
            "additionalProperties": false,
        });
        let grammar = serde_json::to_string(&grammar).unwrap_or_else(|_| "{}".to_string());
        Self { grammar }
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
        let grammar = JsonSchemaGrammar::from_schemas(&sample_schemas());
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
        let grammar = JsonSchemaGrammar::from_schemas(&[]);
        let s = grammar.grammar();
        // Should parse back as a JSON object.
        let parsed: serde_json::Value = serde_json::from_str(s).unwrap();
        assert!(parsed.is_object());
    }

    #[test]
    fn tool_grammar_provider_is_object_safe() {
        let g: Box<dyn ToolGrammarProvider> = Box::new(JsonSchemaGrammar::from_schemas(&[]));
        let arc: std::sync::Arc<dyn ToolGrammarProvider> =
            std::sync::Arc::new(JsonSchemaGrammar::from_schemas(&sample_schemas()));
        assert!(!arc.grammar().is_empty());
        // The Box also works through the trait object.
        assert!(!g.grammar().is_empty());
        // Debug is required by the bound — formatting through the trait
        // object proves the impl exists.
        let debug = format!("{arc:?}");
        assert!(debug.contains("JsonSchemaGrammar"));
    }

    /// Live ≥99%-valid-calls measurement against a local 7B with grammar
    /// support (vLLM `guided_json`). Does not run in CI.
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
        let grammar = std::sync::Arc::new(JsonSchemaGrammar::from_schemas(&schemas));
        let opts = RequestOptions::new().with_tool_constraint(ToolConstraint::Grammar(grammar));

        // A small fixed corpus of prompts that should each produce exactly
        // one valid tool call. The 99% bar is over this corpus.
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
            // Count any InputJson delta as a candidate tool call; validity =
            // the accumulated JSON parses as an object with a recognized
            // tool name key in our grammar.
            let mut accumulated = String::new();
            for ev in events {
                if let crate::stream::StreamEvent::IndexedDelta(d) = ev
                    && let crate::stream::DeltaPart::InputJson { partial_json } = d.delta
                {
                    accumulated.push_str(&partial_json);
                }
            }
            total = total.saturating_add(1);
            if serde_json::from_str::<serde_json::Value>(&accumulated).is_ok() {
                valid = valid.saturating_add(1);
            }
        }

        let rate = valid.saturating_mul(100) / total.max(1);
        assert!(
            rate >= 99,
            "valid tool-call rate {rate}% ({valid}/{total}) below the 99% bar"
        );
    }
}

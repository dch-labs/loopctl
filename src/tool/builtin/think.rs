//! The `think` scratchpad tool.
//!
//! Small models plan poorly inside a single forward pass. Registering
//! [`ThinkTool`] gives the model a zero-cost place to reason in the
//! conversation before acting — enumerate options, restate the goal,
//! check a plan against constraints — without changing anything about
//! the loop itself.

use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;

use crate::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolSchema};

/// The acknowledgement every accepted thought receives.
///
/// Constant and tiny on purpose: the thought is already part of the
/// conversation as the call's input, so echoing it back would double
/// its token cost for no gain.
const ACKNOWLEDGEMENT: &str = "ok";

/// A no-side-effect scratchpad the model can write its reasoning into.
///
/// Calling it changes nothing in the world and returns a fixed
/// acknowledgement; the value is the act of writing — the thought
/// becomes part of the conversation the model conditions on for its
/// next action. This is the small-model planning aid: enumerate
/// options, restate the goal, check a plan against constraints before
/// acting.
///
/// [`read_only`](Tool::is_read_only) and
/// [`concurrency_safe`](Tool::is_concurrency_safe) are true by
/// definition — there is no effect to guard.
///
/// # Example
///
/// ```
/// use loopctl::tool::ToolRegistry;
/// use loopctl::tool::builtin::ThinkTool;
///
/// let mut registry = ToolRegistry::new();
/// registry.register(ThinkTool::new());
/// assert!(registry.contains("think"));
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ThinkTool;

impl ThinkTool {
    /// A scratchpad tool advertising the `think` schema.
    ///
    /// The type carries no configuration; all instances behave
    /// identically and are cheap to construct per session.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Tool for ThinkTool {
    fn name(&self) -> &'static str {
        "think"
    }

    fn description(&self) -> &'static str {
        "A scratchpad for your own reasoning. Use it to plan before acting: \
         restate the goal, list the options, check the plan against \
         constraints, decide. Writing a thought changes nothing by itself — \
         you will still need to call the real tool afterwards. Prefer \
         thinking before non-trivial actions."
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            tool: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your reasoning: goal, options, checks, decision"
                    }
                },
                "required": ["thought"],
                "additionalProperties": false
            }),
        }
    }

    fn call(
        &self,
        input: Value,
        _context: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async move {
            if input.get("thought").and_then(Value::as_str).is_none() {
                return Err(ToolError::InvalidInput(
                    "think requires a string `thought` field".to_string(),
                ));
            }
            tracing::debug!(
                target: "loopctl::metrics",
                metric = "loopctl.think.calls",
                "think tool called"
            );
            Ok(ToolOutput::text(ACKNOWLEDGEMENT))
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn is_concurrency_safe(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn call_returns_fixed_acknowledgement() {
        let tool = ThinkTool::new();
        let output = tool
            .call(
                json!({"thought": "what is the goal?"}),
                &ToolContext::default(),
            )
            .await
            .expect("a well-formed thought is accepted");
        assert_eq!(output.text_content(), "ok", "the ack is constant");
    }

    #[tokio::test]
    async fn wrong_typed_input_is_invalid_input() {
        let tool = ThinkTool::new();
        let error = tool
            .call(json!({"thought": 5}), &ToolContext::default())
            .await
            .expect_err("a non-string thought is rejected");
        match error {
            ToolError::InvalidInput(message) => {
                assert!(message.contains("thought"), "the error names the field");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_thought_field_is_invalid_input() {
        let tool = ThinkTool::new();
        let error = tool
            .call(json!({}), &ToolContext::default())
            .await
            .expect_err("a call without a thought is rejected");
        assert!(matches!(error, ToolError::InvalidInput(_)));
    }

    #[test]
    fn flags_are_true() {
        let tool = ThinkTool::new();
        assert!(tool.is_read_only(), "writing a thought has no effect");
        assert!(
            tool.is_concurrency_safe(),
            "thoughts are independent by construction"
        );
    }

    #[test]
    fn schema_advertises_only_the_thought_field() {
        let schema = ThinkTool::new().schema();
        assert_eq!(schema.tool, "think");
        let input = &schema.input_schema;
        assert_eq!(input["type"], "object");
        assert_eq!(
            input["required"],
            json!(["thought"]),
            "the thought field is required"
        );
        assert_eq!(input["additionalProperties"], false);
        let properties = input["properties"].as_object().expect("properties object");
        assert_eq!(properties.len(), 1, "the schema has exactly one field");
        assert!(properties.contains_key("thought"));
    }
}

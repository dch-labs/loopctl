//! End-to-end tests for `#[derive(Tool)]` against the real trait.
//!
//! Compiled only with the `derive` feature.

#![cfg(feature = "derive")]
#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

use std::collections::HashMap;

use loopctl::Tool as DeriveTool;
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry};
use serde::Deserialize;

fn call(tool: &impl Tool, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
    futures::executor::block_on(tool.call(input, &ToolContext::default()))
}

/// Echo a message back to the caller.
#[derive(DeriveTool, Deserialize)]
#[tool(name = "echo", description = "Echo back the provided message")]
struct EchoInput {
    /// The text to echo.
    message: String,
}

impl EchoInput {
    async fn run(&self, input: EchoInput, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.message))
    }
}

#[test]
fn minimal_derived_tool_round_trips() {
    let tool = EchoInput {
        message: String::new(),
    };
    assert_eq!(tool.name(), "echo");
    assert_eq!(tool.description(), "Echo back the provided message");
    let schema = tool.schema();
    assert_eq!(schema.tool, "echo");
    assert_eq!(
        schema.input_schema["properties"]["message"]["type"],
        "string"
    );
    assert_eq!(
        schema.input_schema["properties"]["message"]["description"],
        "The text to echo."
    );
    assert_eq!(schema.input_schema["required"][0], "message");
    assert_eq!(
        schema.input_schema["additionalProperties"],
        serde_json::json!(false)
    );
    let output = call(&tool, serde_json::json!({"message": "hi"}))
        .expect("valid input deserializes and dispatches");
    assert_eq!(output.text_content(), "hi");
}

/// A renamed field, an optional field, and a defaulted field.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Renamed and defaulted fields")]
struct SearchInput {
    #[tool(name = "file_path")]
    #[serde(rename = "file_path")]
    file_path_holder: String,
    limit: Option<u32>,
    #[serde(default)]
    verbose: bool,
}

impl SearchInput {
    async fn run(&self, input: SearchInput, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(format!(
            "{}:{}:{}",
            input.file_path_holder,
            input.limit.unwrap_or(0),
            input.verbose
        )))
    }
}

#[test]
fn field_attributes_shape_the_schema() {
    let tool = SearchInput {
        file_path_holder: String::new(),
        limit: None,
        verbose: false,
    };
    let schema = tool.schema();
    let props = &schema.input_schema["properties"];
    assert!(
        props.get("file_path").is_some(),
        "the serde-renamed property must appear; got {props}"
    );
    assert!(props.get("file_path_holder").is_none());
    let required = schema.input_schema["required"].as_array().unwrap();
    assert_eq!(required.len(), 1, "only the required field: {required:?}");
    assert_eq!(required[0], "file_path");
    let output = call(
        &tool,
        serde_json::json!({"file_path": "/x", "limit": 5, "verbose": true}),
    )
    .expect("all fields provided");
    assert_eq!(output.text_content(), "/x:5:true");
    let output = call(&tool, serde_json::json!({"file_path": "/x"}))
        .expect("optional and defaulted fields may be omitted");
    assert_eq!(output.text_content(), "/x:0:false");
}

/// Every supported type mapping plus recursion.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Type map fixture")]
struct TypesInput {
    text: String,
    flag: bool,
    count: i64,
    ratio: f64,
    tags: Vec<String>,
    matrix: Vec<Vec<String>>,
    index: HashMap<String, String>,
}

impl TypesInput {
    async fn run(&self, _input: TypesInput, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text("ok"))
    }
}

#[test]
fn type_mapping_matches_the_table() {
    let schema = TypesInput {
        text: String::new(),
        flag: false,
        count: 0,
        ratio: 0.0,
        tags: Vec::new(),
        matrix: Vec::new(),
        index: HashMap::new(),
    }
    .schema();
    let props = &schema.input_schema["properties"];
    assert_eq!(props["text"]["type"], "string");
    assert_eq!(props["flag"]["type"], "boolean");
    assert_eq!(props["count"]["type"], "integer");
    assert_eq!(props["ratio"]["type"], "number");
    assert_eq!(props["tags"]["type"], "array");
    assert_eq!(props["tags"]["items"]["type"], "string");
    assert_eq!(props["matrix"]["items"]["items"]["type"], "string");
    assert_eq!(props["index"]["type"], "object");
    assert_eq!(props["index"]["additionalProperties"]["type"], "string");
    let required = schema.input_schema["required"].as_array().unwrap();
    assert_eq!(required.len(), 7, "all fields required: {required:?}");
}

/// Provided-method overrides emit only when the attribute is present.
#[derive(DeriveTool, Deserialize)]
#[tool(
    description = "Overrides fixture",
    read_only,
    concurrency_safe,
    system_prompt = "be terse"
)]
struct OverriddenInput {
    q: String,
}

impl OverriddenInput {
    async fn run(
        &self,
        _input: OverriddenInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text("ok"))
    }
}

#[test]
fn provided_method_overrides() {
    let tool = OverriddenInput { q: String::new() };
    assert!(tool.is_read_only());
    assert!(tool.is_concurrency_safe());
    assert_eq!(tool.system_prompt().as_deref(), Some("be terse"));
    // The plain fixture has none of the attributes: defaults apply.
    let plain = EchoInput {
        message: String::new(),
    };
    assert!(!plain.is_read_only());
    assert!(!plain.is_concurrency_safe());
    assert!(plain.system_prompt().is_none());
}

/// Bad input surfaces as `InvalidInput` through the generated dispatch.
#[test]
fn bad_input_maps_to_invalid_input() {
    let tool = EchoInput {
        message: String::new(),
    };
    let err = call(&tool, serde_json::json!({"wrong_field": 1}))
        .expect_err("a missing required field must fail");
    assert!(
        matches!(err, ToolError::InvalidInput(_)),
        "serde errors map to InvalidInput, got {err:?}"
    );
}

/// A skipped field leaves the schema and still deserializes.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Skip fixture")]
struct SkipInput {
    query: String,
    #[tool(skip)]
    #[serde(default)]
    cache_buster: u32,
}

impl SkipInput {
    async fn run(&self, input: SkipInput, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.query))
    }
}

#[test]
fn skipped_field_leaves_the_schema() {
    let tool = SkipInput {
        query: String::new(),
        cache_buster: 0,
    };
    let schema = tool.schema();
    assert!(
        schema.input_schema["properties"]
            .get("cache_buster")
            .is_none()
    );
    let output =
        call(&tool, serde_json::json!({"query": "q"})).expect("the skipped field defaults");
    assert_eq!(output.text_content(), "q");
}

/// A renamed handler via `#[tool(handler = "...")]`.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Handler fixture", handler = "execute")]
struct HandlerInput {
    payload: String,
}

impl HandlerInput {
    async fn execute(
        &self,
        input: HandlerInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.payload))
    }
}

#[test]
fn named_handler_dispatches() {
    let tool = HandlerInput {
        payload: String::new(),
    };
    let output = call(&tool, serde_json::json!({"payload": "p"})).expect("the named handler runs");
    assert_eq!(output.text_content(), "p");
}

/// The derived tool is indistinguishable from a manual one to the
/// registry.
#[test]
fn registry_integration() {
    let mut registry = ToolRegistry::new();
    registry.register(EchoInput {
        message: String::new(),
    });
    let schemas = registry.all_schemas();
    let derived = schemas
        .iter()
        .find(|s| s.tool == "echo")
        .expect("the derived tool is registered under its name");
    assert_eq!(
        derived.input_schema["properties"]["message"]["type"],
        "string"
    );
    let output = futures::executor::block_on(registry.get("echo").expect("registered").call(
        serde_json::json!({"message": "via registry"}),
        &ToolContext::default(),
    ))
    .expect("callable through the registry");
    assert_eq!(output.text_content(), "via registry");
}

/// Fallbacks: no `name`, no `description` attributes — the doc
/// comment supplies the description and the identifier supplies the
/// snake_cased name.
#[derive(DeriveTool, Deserialize)]
struct FallbackNamingInput {
    payload: String,
}

impl FallbackNamingInput {
    async fn run(
        &self,
        input: FallbackNamingInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.payload))
    }
}

#[test]
fn name_and_description_fall_back_to_the_struct() {
    let tool = FallbackNamingInput {
        payload: String::new(),
    };
    assert_eq!(
        tool.name(),
        "fallback_naming_input",
        "the identifier snake_cases into the default tool name"
    );
    assert_eq!(
        tool.description(),
        "Fallbacks: no `name`, no `description` attributes — the doc comment supplies the description and the identifier supplies the snake_cased name.",
        "the struct's doc comment is the default description"
    );
    assert_eq!(tool.schema().tool, "fallback_naming_input");
}

/// `#[tool(default)]` explicitly marks a non-`Option` field as
/// not-required while keeping it in `properties`.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Explicit default fixture")]
struct ExplicitDefaultInput {
    query: String,
    #[tool(default)]
    #[serde(default)]
    depth: u32,
}

impl ExplicitDefaultInput {
    async fn run(
        &self,
        input: ExplicitDefaultInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(format!("{}:{}", input.query, input.depth)))
    }
}

#[test]
fn explicit_tool_default_leaves_required_but_keeps_the_property() {
    let tool = ExplicitDefaultInput {
        query: String::new(),
        depth: 0,
    };
    let schema = tool.schema();
    let props = &schema.input_schema["properties"];
    assert!(
        props.get("depth").is_some(),
        "the defaulted field stays advertised: {props}"
    );
    let required = schema.input_schema["required"].as_array().unwrap();
    assert_eq!(
        required.len(),
        1,
        "only the truly required field: {required:?}"
    );
    assert_eq!(required[0], "query");
    let output =
        call(&tool, serde_json::json!({"query": "q"})).expect("the defaulted field may be omitted");
    assert_eq!(output.text_content(), "q:0");
}

/// `allow_extra` omits the closed-world flag.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Open fixture", allow_extra)]
struct OpenInput {
    note: String,
}

impl OpenInput {
    async fn run(&self, input: OpenInput, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.note))
    }
}

#[test]
fn allow_extra_omits_additional_properties() {
    let schema = OpenInput {
        note: String::new(),
    }
    .schema();
    assert!(
        schema.input_schema.get("additionalProperties").is_none(),
        "open tools advertise no closed-world flag: {}",
        schema.input_schema
    );
}

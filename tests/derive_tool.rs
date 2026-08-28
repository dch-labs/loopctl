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

/// `#[serde(rename)]` on a field mirrors into the schema.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Serde rename fixture")]
struct SerdeRenameInput {
    #[serde(rename = "file_path")]
    path: String,
}

impl SerdeRenameInput {
    async fn run(
        &self,
        input: SerdeRenameInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.path))
    }
}

#[test]
fn serde_rename_mirrors_into_the_schema() {
    let tool = SerdeRenameInput {
        path: String::new(),
    };
    let schema = tool.schema();
    let props = &schema.input_schema["properties"];
    assert!(
        props.get("file_path").is_some(),
        "the serde-renamed property appears: {props}"
    );
    assert!(props.get("path").is_none());
    assert_eq!(schema.input_schema["required"][0], "file_path");
    let output = call(&tool, serde_json::json!({"file_path": "/x"}))
        .expect("deserialization matches the schema");
    assert_eq!(output.text_content(), "/x");
}

/// `#[serde(rename_all)]` on the struct applies the strategy to every
/// field's schema name.
#[derive(DeriveTool, Deserialize)]
#[serde(rename_all = "camelCase")]
#[tool(description = "Rename-all fixture")]
struct RenameAllInput {
    file_name: String,
}

impl RenameAllInput {
    async fn run(
        &self,
        input: RenameAllInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.file_name))
    }
}

#[test]
fn serde_rename_all_applies_the_strategy() {
    let tool = RenameAllInput {
        file_name: String::new(),
    };
    let schema = tool.schema();
    let props = &schema.input_schema["properties"];
    assert!(
        props.get("fileName").is_some(),
        "camelCase strategy applied: {props}"
    );
    assert!(props.get("file_name").is_none());
    let output =
        call(&tool, serde_json::json!({"fileName": "f"})).expect("camelCase key deserializes");
    assert_eq!(output.text_content(), "f");
}

/// `#[tool(name)]` wins over `#[serde(rename)]` — the tool attribute
/// is the explicit override.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Override fixture")]
struct OverridePrecedenceInput {
    #[serde(rename = "serde_name")]
    #[tool(name = "tool_name")]
    field: String,
}

impl OverridePrecedenceInput {
    async fn run(
        &self,
        input: OverridePrecedenceInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.field))
    }
}

#[test]
fn tool_name_wins_over_serde_rename() {
    let tool = OverridePrecedenceInput {
        field: String::new(),
    };
    let schema = tool.schema();
    let props = &schema.input_schema["properties"];
    assert!(props.get("tool_name").is_some(), "the tool attr wins");
    assert!(props.get("serde_name").is_none());
    // deserialization uses serde's name, the schema uses the tool's
    let output = call(&tool, serde_json::json!({"serde_name": "v"}))
        .expect("serde still renames the Rust field");
    assert_eq!(output.text_content(), "v");
}

/// The full integer range maps to `integer`.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Integer fixture")]
struct IntegersInput {
    small: i8,
    big: u64,
    huge: i128,
}

impl IntegersInput {
    async fn run(
        &self,
        _input: IntegersInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text("ok"))
    }
}

#[test]
fn wide_integers_map_to_integer() {
    let schema = IntegersInput {
        small: 0,
        big: 0,
        huge: 0,
    }
    .schema();
    let props = &schema.input_schema["properties"];
    assert_eq!(props["small"]["type"], "integer");
    assert_eq!(props["big"]["type"], "integer");
    assert_eq!(props["huge"]["type"], "integer");
}

/// `rename_all` with `PascalCase` (a strategy that exercises the
/// `to_pascal_case` helper).
#[derive(DeriveTool, Deserialize)]
#[serde(rename_all = "PascalCase")]
#[tool(description = "Pascal fixture")]
struct PascalRenameInput {
    file_name: String,
}

impl PascalRenameInput {
    async fn run(
        &self,
        input: PascalRenameInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.file_name))
    }
}

#[test]
fn serde_rename_all_pascal_case_applies() {
    let tool = PascalRenameInput {
        file_name: String::new(),
    };
    let schema = tool.schema();
    let props = &schema.input_schema["properties"];
    assert!(props.get("FileName").is_some(), "PascalCase: {props}");
    assert!(props.get("file_name").is_none());
    let output =
        call(&tool, serde_json::json!({"FileName": "f"})).expect("PascalCase key deserializes");
    assert_eq!(output.text_content(), "f");
}

/// `rename_all` with `SCREAMING_SNAKE_CASE`.
#[derive(DeriveTool, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[tool(description = "Screaming fixture")]
struct ScreamingRenameInput {
    file_name: String,
}

impl ScreamingRenameInput {
    async fn run(
        &self,
        input: ScreamingRenameInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.file_name))
    }
}

#[test]
fn serde_rename_all_screaming_snake_applies() {
    let tool = ScreamingRenameInput {
        file_name: String::new(),
    };
    let schema = tool.schema();
    let props = &schema.input_schema["properties"];
    assert!(props.get("FILE_NAME").is_some(), "SCREAMING: {props}");
    let output =
        call(&tool, serde_json::json!({"FILE_NAME": "f"})).expect("SCREAMING key deserializes");
    assert_eq!(output.text_content(), "f");
}

/// `rename_all` with `kebab-case`.
#[derive(DeriveTool, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[tool(description = "Kebab fixture")]
struct KebabRenameInput {
    file_name: String,
}

impl KebabRenameInput {
    async fn run(
        &self,
        input: KebabRenameInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.file_name))
    }
}

#[test]
fn serde_rename_all_kebab_case_applies() {
    let tool = KebabRenameInput {
        file_name: String::new(),
    };
    let schema = tool.schema();
    let props = &schema.input_schema["properties"];
    assert!(props.get("file-name").is_some(), "kebab: {props}");
    let output =
        call(&tool, serde_json::json!({"file-name": "f"})).expect("kebab-case key deserializes");
    assert_eq!(output.text_content(), "f");
}

/// `skip` on an `Option` field (not just `#[serde(default)]`).
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Skip-option fixture")]
struct SkipOptionInput {
    query: String,
    #[tool(skip)]
    cache: Option<String>,
}

impl SkipOptionInput {
    async fn run(
        &self,
        input: SkipOptionInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.query))
    }
}

#[test]
fn skip_on_option_field_leaves_the_schema() {
    let tool = SkipOptionInput {
        query: String::new(),
        cache: None,
    };
    let schema = tool.schema();
    assert!(schema.input_schema["properties"].get("cache").is_none());
    let output = call(&tool, serde_json::json!({"query": "q"}))
        .expect("the skipped Option defaults to None");
    assert_eq!(output.text_content(), "q");
}

/// `#[tool(description = "…")]` on a field overrides the doc-comment
/// fallback for the property's description.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Field description override fixture")]
struct FieldDescInput {
    /// This doc comment would be the description.
    #[tool(description = "The explicit override wins.")]
    target: String,
}

impl FieldDescInput {
    async fn run(
        &self,
        input: FieldDescInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.target))
    }
}

#[test]
fn field_description_override_wins_over_doc_comment() {
    let schema = FieldDescInput {
        target: String::new(),
    }
    .schema();
    assert_eq!(
        schema.input_schema["properties"]["target"]["description"],
        "The explicit override wins."
    );
}

/// `Cow<'static, str>` maps to `string` — the type-map accepts it.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Cow fixture")]
struct CowStrInput {
    text: std::borrow::Cow<'static, str>,
}

impl CowStrInput {
    async fn run(&self, input: CowStrInput, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.text.into_owned()))
    }
}

#[test]
fn cow_str_maps_to_string() {
    let tool = CowStrInput {
        text: std::borrow::Cow::Borrowed(""),
    };
    assert_eq!(
        tool.schema().input_schema["properties"]["text"]["type"],
        "string"
    );
    let output = call(&tool, serde_json::json!({"text": "borrowed"}))
        .expect("Cow<str> deserializes from a JSON string");
    assert_eq!(output.text_content(), "borrowed");
}

/// `Vec<i64>` exercises array recursion into a non-string scalar.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Integer vec fixture")]
struct IntVecInput {
    numbers: Vec<i64>,
}

impl IntVecInput {
    async fn run(&self, input: IntVecInput, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(
            input.numbers.iter().sum::<i64>().to_string(),
        ))
    }
}

#[test]
fn vec_of_integers_recurses_correctly() {
    let schema = IntVecInput {
        numbers: Vec::new(),
    }
    .schema();
    let props = &schema.input_schema["properties"];
    assert_eq!(props["numbers"]["type"], "array");
    assert_eq!(props["numbers"]["items"]["type"], "integer");
    let output = call(&tool_int_vec(), serde_json::json!({"numbers": [1, 2, 3]}))
        .expect("integer array deserializes");
    assert_eq!(output.text_content(), "6");
}

fn tool_int_vec() -> IntVecInput {
    IntVecInput {
        numbers: Vec::new(),
    }
}

/// `Option<Vec<String>>` — an optional complex type.
#[derive(DeriveTool, Deserialize)]
#[tool(description = "Optional vec fixture")]
struct OptionalVecInput {
    required: String,
    tags: Option<Vec<String>>,
}

impl OptionalVecInput {
    async fn run(
        &self,
        input: OptionalVecInput,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(input.tags.unwrap_or_default().join(",")))
    }
}

#[test]
fn optional_vec_unwraps_for_schema_and_leaves_required() {
    let tool = OptionalVecInput {
        required: String::new(),
        tags: None,
    };
    let schema = tool.schema();
    let props = &schema.input_schema["properties"];
    // Option unwraps to the inner array schema
    assert_eq!(props["tags"]["type"], "array");
    assert_eq!(props["tags"]["items"]["type"], "string");
    // Option fields are never required
    let required = schema.input_schema["required"].as_array().unwrap();
    assert_eq!(required.len(), 1);
    assert_eq!(required[0], "required");
    // Calling without the optional field succeeds
    let output =
        call(&tool, serde_json::json!({"required": "r"})).expect("optional vec defaults to None");
    assert_eq!(output.text_content(), "");
    // Calling with the optional field also succeeds
    let output = call(
        &tool,
        serde_json::json!({"required": "r", "tags": ["a", "b"]}),
    )
    .expect("optional vec deserializes when present");
    assert_eq!(output.text_content(), "a,b");
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

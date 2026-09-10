use loopctl::Tool;
use serde::Deserialize;

/// A known-good derived tool driven through its full surface.
///
/// Exercises the tool-level name and description overrides, the
/// generated schema's required list, and the call dispatch into the
/// run handler — the parts of the derive a compile-only check cannot
/// reach.
#[derive(Tool, Deserialize)]
#[tool(name = "greet", description = "Greets")]
struct GreetInput {
    /// The person the greeting addresses.
    ///
    /// A plain required `String` with no defaulting on either side, so
    /// it appears in the schema's required list under its serde name.
    who: String,
}

impl GreetInput {
    async fn run(
        &self,
        _input: GreetInput,
        _ctx: &loopctl::tool::ToolContext,
    ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
        Ok(loopctl::tool::ToolOutput::text("hi"))
    }
}

/// A struct-level `#[serde(default)]` makes every field optional on the
/// wire.
///
/// Serde fills any missing field from the struct's `Default` impl, so
/// no field is truly required and a skipped field needs no field-level
/// default of its own. The skipped field also carries `#[serde(skip)]`
/// — the one skip combination that agrees with the schema, since both
/// sides omit it.
#[derive(Default, Tool, Deserialize)]
#[serde(default)]
#[tool(name = "container_default", description = "Struct-level default")]
struct ContainerDefaultInput {
    /// A count serde fills from the struct's default when absent.
    ///
    /// Present in the schema as an optional property; its absence at
    /// deserialization is legal and yields `Default::default()`.
    count: u32,
    #[tool(skip)]
    #[serde(skip)]
    internal: String,
}

impl ContainerDefaultInput {
    async fn run(
        &self,
        _input: ContainerDefaultInput,
        _ctx: &loopctl::tool::ToolContext,
    ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
        Ok(loopctl::tool::ToolOutput::text("hi"))
    }
}

/// A skip pair with no container-level backing.
///
/// `#[tool(skip)]` plus a read-side `#[serde(skip)]` on a non-`Option`
/// field: serde fills the field from `Default::default()` without
/// reading the wire, which is exactly the fillability the skip
/// validity check demands — no struct-level or field-level `default`
/// is needed.
#[derive(Default, Tool, Deserialize)]
#[tool(name = "skip_pair", description = "Bare skip pair")]
struct SkipPairInput {
    label: String,
    #[tool(skip)]
    #[serde(skip)]
    secret: String,
}

impl SkipPairInput {
    async fn run(
        &self,
        _input: SkipPairInput,
        _ctx: &loopctl::tool::ToolContext,
    ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
        Ok(loopctl::tool::ToolOutput::text("hi"))
    }
}

/// A write-side serde skip on a schema-visible field.
///
/// `#[serde(skip_serializing)]` changes only what serde emits; the
/// field is still read from the wire, so advertising it in the input
/// schema is truthful and the derive maps it normally.
#[derive(Default, Tool, Deserialize)]
#[tool(name = "write_skip", description = "Write-side skip")]
struct WriteSkipInput {
    /// A field serde never emits but always reads.
    ///
    /// The input schema advertises it as required, which is truthful:
    /// only the serialization side skips it.
    #[serde(skip_serializing)]
    echoed: String,
}

impl WriteSkipInput {
    async fn run(
        &self,
        _input: WriteSkipInput,
        _ctx: &loopctl::tool::ToolContext,
    ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
        Ok(loopctl::tool::ToolOutput::text("hi"))
    }
}

fn main() {
    let greet = GreetInput {
        who: "world".into(),
    };
    assert_eq!(greet.name(), "greet", "the tool-level name override wins");
    assert_eq!(greet.description(), "Greets");
    let schema = greet.schema();
    let required = schema
        .input_schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .expect("the schema always carries a required list");
    assert_eq!(
        serde_json::to_string(required).expect("the required list serializes"),
        r#"["who"]"#,
        "a plain field is required under its serde name"
    );
    let output = futures::executor::block_on(greet.call(
        serde_json::json!({"who": "world"}),
        &loopctl::tool::ToolContext::default(),
    ))
    .expect("the generated dispatch reaches the run handler");
    assert_eq!(output.text_content(), "hi");

    let tool = ContainerDefaultInput {
        count: 1,
        internal: String::new(),
    };
    let schema = tool.schema();
    let required = schema
        .input_schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .expect("the schema always carries a required list");
    assert!(
        required.is_empty(),
        "a struct-level serde default makes every field optional in the schema"
    );

    let skip_pair = SkipPairInput {
        label: "x".into(),
        secret: String::new(),
    };
    let schema = skip_pair.schema();
    let properties = schema
        .input_schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .expect("the schema always carries properties");
    assert!(
        properties.len() == 1 && properties.contains_key("label"),
        "the serde-skip pair stays out of the schema without any default attribute: \
        {properties:?}"
    );
}

use loopctl::Tool;
use serde::Deserialize;

/// A known-good derived tool.
#[derive(Tool, Deserialize)]
#[tool(name = "greet", description = "Greets")]
struct GreetInput {
    /// Who to greet.
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

fn main() {
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
}

/// A struct-level `#[serde(default)]` fills any missing field at
/// deserialization, so no field is truly required and a skipped field
/// needs no field-level default of its own.
#[derive(Default, Tool, Deserialize)]
#[serde(default)]
#[tool(name = "container_default", description = "Struct-level default")]
struct ContainerDefaultInput {
    /// A field serde fills from the struct's default when absent.
    count: u32,
    #[tool(skip)]
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

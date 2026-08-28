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

fn main() {}

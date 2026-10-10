use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
#[tool(description = "")]
struct EmptyDescription {
    a: String,
}

impl EmptyDescription {
    async fn run(
        &self,
        input: EmptyDescription,
        ctx: &loopctl::tool::ToolContext,
    ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
        let _ = (input, ctx);
        todo!()
    }
}

fn main() {}

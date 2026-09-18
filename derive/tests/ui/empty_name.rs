use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
#[tool(name = "")]
struct EmptyName {
    a: String,
}

impl EmptyName {
    async fn run(
        &self,
        input: EmptyName,
        ctx: &loopctl::tool::ToolContext,
    ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
        let _ = (input, ctx);
        todo!()
    }
}

fn main() {}

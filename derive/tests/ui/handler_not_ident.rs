use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
#[tool(handler = "not an ident")]
struct Handled {
    a: String,
}

impl Handled {
    async fn run(
        &self,
        input: Handled,
        ctx: &loopctl::tool::ToolContext,
    ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
        let _ = (input, ctx);
        todo!()
    }
}

fn main() {}

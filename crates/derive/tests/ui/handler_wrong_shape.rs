use loopctl::Tool;
use serde::Deserialize;
use loopctl::tool::{ToolContext, ToolError, ToolOutput};

/// Doc.
#[derive(Tool, Deserialize)]
#[tool(description = "x", handler = "compute")]
struct Shaped {
    a: String,
}

impl Shaped {
    fn compute(
        &self,
        input: Shaped,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let _ = (input, ctx);
        Ok(ToolOutput::text("hi"))
    }
}

fn main() {}

//! `#[serde(deny_unknown_fields)]` alone is consistent: the schema's
//! `additionalProperties: false` and serde's strictness agree.
use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Strict {
    pub a: String,
}


impl Strict {
    async fn run(
        &self,
        input: Strict,
        ctx: &loopctl::tool::ToolContext,
    ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
        let _ = (input, ctx);
        Ok(loopctl::tool::ToolOutput::text("ok"))
    }
}

fn main() {}

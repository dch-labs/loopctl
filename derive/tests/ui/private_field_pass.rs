//! A private field in a pub struct is fine: the generated impl names no
//! fields — deserialization goes through serde and the schema is built
//! statically — so field visibility never reaches the expansion.
use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
pub struct WithPrivate {
    pub a: String,
    hidden: Option<String>,
}


impl WithPrivate {
    async fn run(
        &self,
        input: WithPrivate,
        ctx: &loopctl::tool::ToolContext,
    ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
        let _ = (input, ctx);
        Ok(loopctl::tool::ToolOutput::text("ok"))
    }
}

fn main() {
    let _ = WithPrivate {
        a: String::new(),
        hidden: None,
    };
}

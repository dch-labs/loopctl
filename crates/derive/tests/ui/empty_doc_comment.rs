use loopctl::Tool;
use serde::Deserialize;

///
#[derive(Tool, Deserialize)]
struct EmptyDoc {
    a: String,
}

impl EmptyDoc {
    async fn run(
        &self,
        input: EmptyDoc,
        ctx: &loopctl::tool::ToolContext,
    ) -> Result<loopctl::tool::ToolOutput, loopctl::tool::ToolError> {
        let _ = (input, ctx);
        todo!()
    }
}

fn main() {}

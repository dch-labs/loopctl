use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
#[tool(description = "x")]
struct Bad {
    a: String,
    #[tool(default)]
    b: u32,
}

fn main() {}

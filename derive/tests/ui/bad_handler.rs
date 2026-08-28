use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
#[tool(description = "x", handler = "not-an-ident")]
struct Bad {
    a: String,
}

fn main() {}

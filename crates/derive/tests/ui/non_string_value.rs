use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
#[tool(name = 123)]
struct Numbered {
    a: String,
}

fn main() {}

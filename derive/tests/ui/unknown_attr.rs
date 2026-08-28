use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
#[tool(description = "x", totally_made_up = 1)]
struct Weird {
    a: String,
}

fn main() {}

use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
struct BadSkip {
    a: String,
    #[tool(skip)]
    b: String,
}

fn main() {}

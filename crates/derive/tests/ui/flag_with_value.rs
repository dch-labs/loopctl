use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
struct Skipped {
    a: String,
    #[tool(skip = true)]
    b: Option<String>,
}

fn main() {}

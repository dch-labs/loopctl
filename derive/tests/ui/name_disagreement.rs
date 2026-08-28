use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
#[tool(description = "x")]
struct Bad {
    #[tool(name = "schema_key")]
    rust_field: String,
}

fn main() {}

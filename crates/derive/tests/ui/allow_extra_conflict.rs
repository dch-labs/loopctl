use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
#[tool(allow_extra)]
#[serde(deny_unknown_fields)]
struct Conflicting {
    a: String,
}

fn main() {}

use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
struct Unsupported {
    pair: (String, i32),
}

fn main() {}

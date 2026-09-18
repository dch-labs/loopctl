use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
struct Tuple(String);

fn main() {}

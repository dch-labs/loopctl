use loopctl::Tool;
use serde::Deserialize;

#[derive(Tool, Deserialize)]
struct NoDocs {
    a: String,
}

fn main() {}

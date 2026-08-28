use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
struct CowBytes {
    blob: std::borrow::Cow<'static, [u8]>,
}

fn main() {}

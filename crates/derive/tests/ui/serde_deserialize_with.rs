use loopctl::Tool;
use serde::Deserialize;

/// A struct whose field deserializes through a custom module.
///
/// The module reads a wire shape the type's natural schema cannot
/// promise, so the generated property would describe the wrong format.
#[derive(Tool, Deserialize)]
struct CustomWireInput {
    #[serde(deserialize_with = "parse_duration")]
    duration: String,
}

fn main() {}

use loopctl::Tool;
use serde::Deserialize;

/// A struct whose serde input skips a field without the matching tool
/// attribute.
///
/// Serde fills `internal` without reading the wire object, so a
/// generated schema advertising it would invite input serde never
/// accepts.
#[derive(Tool, Deserialize)]
struct SkippedInput {
    name: String,
    #[serde(skip)]
    internal: String,
}

fn main() {}

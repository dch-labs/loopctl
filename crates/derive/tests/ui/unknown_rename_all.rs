use loopctl::Tool;
use serde::Deserialize;

/// A struct declaring a `rename_all` strategy serde itself rejects.
///
/// The unrecognized spelling must fail the derive rather than fall
/// back to raw field names the deserializer would never look for.
#[derive(Tool, Deserialize)]
#[serde(rename_all = "screaming")]
struct RenamedInput {
    file_name: String,
}

fn main() {}

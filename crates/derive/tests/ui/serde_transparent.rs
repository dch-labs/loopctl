use loopctl::Tool;
use serde::Deserialize;

/// A struct whose serde input is the bare inner value.
///
/// `transparent` makes the wire the wrapped value itself, so no object
/// schema can describe what serde reads.
#[derive(Tool, Deserialize)]
#[serde(transparent)]
struct TransparentInput {
    inner: String,
}

fn main() {}

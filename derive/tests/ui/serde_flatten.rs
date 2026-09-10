use loopctl::Tool;
use serde::Deserialize;
use std::collections::HashMap;

/// A struct whose serde input flattens a map into the surrounding
/// object.
///
/// The flatten key moves the map's entries onto the wire object
/// itself, so no `extra` property ever exists for a schema to
/// advertise.
#[derive(Tool, Deserialize)]
struct FlattenInput {
    name: String,
    #[serde(flatten)]
    extra: HashMap<String, String>,
}

fn main() {}

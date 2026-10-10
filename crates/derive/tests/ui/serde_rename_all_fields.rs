use loopctl::Tool;
use serde::Deserialize;

/// Doc.
#[derive(Tool, Deserialize)]
#[serde(rename_all_fields = "kebab-case")]
struct Renamed {
    fieldName: String,
}

fn main() {}

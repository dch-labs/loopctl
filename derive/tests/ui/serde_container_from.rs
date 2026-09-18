use loopctl::Tool;
use serde::Deserialize;
#[derive(Deserialize)]
#[allow(dead_code)]
struct Wire {
    a: String,
}

/// Doc.
#[derive(Tool, Deserialize)]
#[serde(from = "Wire")]
struct FromWire {
    a: String,
}

impl From<Wire> for FromWire {
    fn from(wire: Wire) -> Self {
        Self { a: wire.a }
    }
}

fn main() {}

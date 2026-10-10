use loopctl::Tool;

/// Doc.
#[derive(Tool)]
struct Borrowed<'a> {
    text: &'a str,
}

fn main() {}

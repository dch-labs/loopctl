//! Interactive REPL CLI — read user input, run a turn, repeat.
//!
//! Demonstrates a minimal read-eval-print loop using [`BareLoop`].
//! Each line of stdin becomes a new user message. Type `quit` or
//! `Ctrl-D` to exit.
//!
//! ```sh
//! cargo run --example repl-cli --features testing
//! ```

#![allow(clippy::unwrap_used)]

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use loopctl::config::LoopConfig;
use loopctl::engine::BareLoop;
use loopctl::engine::loop_core::Loop;
use loopctl::testing::MockApiClient;
use loopctl::tool::ToolRegistry;

#[tokio::main]
async fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Read.
        write!(&mut stdout, "> ").unwrap();
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).unwrap_or(0) == 0 {
            break; // EOF (Ctrl-D)
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        // Eval: build a fresh loop per message with a canned response.
        let client =
            MockApiClient::new("repl-model").with_text_response(&format!("You said: {input}"));

        let mut agent = BareLoop::new(Arc::new(client), ToolRegistry::new(), LoopConfig::default());

        match agent.run(input).await {
            Ok(result) => {
                // Print.
                let output = result.final_output.unwrap_or_default();
                writeln!(&mut stdout, "{output}\n").unwrap();
            }
            Err(e) => {
                writeln!(&mut stdout, "Error: {e}\n").unwrap();
            }
        }
    }

    writeln!(&mut stdout, "Goodbye!").unwrap();
}

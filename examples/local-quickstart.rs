//! The smallest agent that talks to a real model — your local Ollama.
//!
//! ```sh
//! ollama pull qwen3:8b
//! cargo run --example local-quickstart --features ollama
//! ```
//!
//! No API key and no cloud: the client points at `http://localhost:11434`
//! by default, and any model you have pulled works as the first argument.
//! The model decides to call the `now` tool, the loop dispatches the call
//! and feeds the result back, and the run ends with the model's final
//! answer — the whole agent loop in about seventy lines.

#![allow(clippy::expect_used, clippy::doc_markdown)]

use std::sync::Arc;

use loopctl::config::SessionConfig;
use loopctl::engine::BareLoop;
use loopctl::engine::RunConfig;
use loopctl::engine::core::Loop;
use loopctl::provider;
use loopctl::tool::{FnTool, ToolOutput, ToolRegistry};
use serde_json::json;

/// The async function backing the `now` tool.
fn now_fn(
    _input: serde_json::Value,
    _ctx: &loopctl::tool::ToolContext,
) -> std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<ToolOutput, loopctl::tool::ToolError>>
            + Send
            + 'static,
    >,
> {
    Box::pin(async move {
        let seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        Ok(ToolOutput::text(seconds.to_string()))
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Any model your local Ollama serves; `qwen3:8b` is the default.
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "qwen3:8b".to_string());
    let client = provider::ollama(&model)?;

    // 2. One tool the model can call.
    let tools = ToolRegistry::new();
    tools.register(
        FnTool::new(
            "now".into(),
            "Return the current Unix time in seconds.".into(),
            json!({"type": "object", "properties": {}}),
            now_fn,
        )
        .read_only(),
    );

    // 3. Construct and run the loop.
    let mut agent = BareLoop::new(Arc::new(client), tools, SessionConfig::default());
    let cancel_signal = agent.cancel_signal();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            cancel_signal.cancel();
        }
    });

    let result = agent
        .run(
            "What time is it? Call the now tool, then answer in one sentence.",
            &RunConfig::default(),
        )
        .await?;
    println!("{}", result.output.unwrap_or_default());
    Ok(())
}

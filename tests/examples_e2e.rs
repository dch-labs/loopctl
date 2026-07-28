//! Live test: basic end-to-end run (Ollama only).
//!
//! Run: `LOOPCTL_E2E=1 OLLAMA_MODEL=qwen2.5:7b cargo test --features ollama --test examples_e2e -- --nocapture`

#![allow(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone
)]

mod helpers;

#[cfg(feature = "ollama")]
use loopctl::engine::core::Loop;

#[cfg(feature = "ollama")]
#[tokio::test]
async fn chat_example_works_end_to_end() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1") {
        eprintln!("skipping (set LOOPCTL_E2E=1)");
        return;
    }

    let model = std::env::var("OLLAMA_MODEL").expect("set OLLAMA_MODEL");
    let client = std::sync::Arc::new(loopctl::provider::ollama(&model).unwrap());
    let mut agent = loopctl::engine::BareLoop::new(
        client,
        loopctl::tool::ToolRegistry::new(),
        loopctl::config::SessionConfig::default(),
    );
    let result = agent
        .run(
            "Say hello in one sentence.",
            &loopctl::engine::RunConfig::default(),
        )
        .await
        .expect("run should succeed");

    let output = result.output.expect("should have output");
    assert!(!output.is_empty());
    println!("Response: {output}");
}

//! Live test: provider survives consecutive requests (Ollama only).
//!
//! Run: `LOOPCTL_E2E=1 OLLAMA_MODEL=qwen2.5:7b cargo test --features ollama --test provider_survival -- --nocapture`

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
async fn provider_survives_consecutive_requests() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1") {
        eprintln!("skipping (set LOOPCTL_E2E=1)");
        return;
    }

    let model = std::env::var("OLLAMA_MODEL").expect("set OLLAMA_MODEL");
    let client = std::sync::Arc::new(loopctl::provider::ollama(&model).unwrap());
    let session = loopctl::config::SessionConfig::default();
    let run = loopctl::engine::RunConfig::default();
    let n: u32 = std::env::var("LOOPCTL_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let mut ok: u32 = 0;

    for i in 0..n {
        let mut agent = loopctl::engine::BareLoop::new(
            std::sync::Arc::clone(&client),
            loopctl::tool::ToolRegistry::new(),
            session.clone(),
        );
        let prompt = format!("Say exactly: response-{i}");
        let pass = agent.run(&prompt, &run).await.is_ok();
        if pass {
            ok += 1;
        }
        let status = if pass {
            "\x1b[32mPASS:\x1b[0m"
        } else {
            "\x1b[31mFAIL:\x1b[0m"
        };
        println!("[{}/{}] {status} {prompt}", i + 1, n);
    }

    let rate = f64::from(ok) / f64::from(n) * 100.0;
    println!("{ok}/{n} ({rate:.0}%) succeeded");
    assert!(rate >= 95.0, "got {rate:.0}%");
}

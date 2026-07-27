//! Live test: tool-call constrained decoding.
//!
//! Run: `LOOPCTL_E2E=1 OLLAMA_MODEL=qwen2.5:7b cargo test --features ollama,grammar --test constrained_decode -- --nocapture`
//!
//! Override iteration count via `LOOPCTL_N=1000`.
//! Re-run only specific steps via `LOOPCTL_ONLY=3,7,12`.

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
use loopctl::api::ApiClient;
#[cfg(feature = "ollama")]
use loopctl::structured::{RequestOptions, ToolConstraint};

#[cfg(feature = "ollama")]
const NOOP_INDICES: &[usize] = &[21, 22];

#[cfg(feature = "ollama")]
#[tokio::test]
async fn strict_mode_produces_valid_tool_calls() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1") {
        eprintln!("skipping (set LOOPCTL_E2E=1)");
        return;
    }

    let model = std::env::var("OLLAMA_MODEL").expect("set OLLAMA_MODEL");
    let client = loopctl::provider::ollama(&model).unwrap();
    let tools = helpers::test_tools();
    let prompts = helpers::tool_call_prompts();
    let n: u32 = std::env::var("LOOPCTL_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let only: Option<Vec<usize>> = std::env::var("LOOPCTL_ONLY")
        .ok()
        .map(|s| s.split(',').filter_map(|n| n.trim().parse().ok()).collect());

    let mut valid: u32 = 0;
    let mut total: u32 = 0;
    let mut errors: u32 = 0;

    for i in 0..n {
        let prompt_idx = (i as usize) % prompts.len();

        if let Some(ref indices) = only
            && !indices.contains(&prompt_idx)
        {
            continue;
        }

        let prompt = prompts[prompt_idx];
        total += 1;
        let req = loopctl::api::StreamRequest::new(vec![loopctl::message::Message::user(prompt)])
            .with_tools(Some(tools.clone()));
        let opts = RequestOptions::default().with_tool_constraint(ToolConstraint::Strict);
        let stream = client.stream_messages_with_options(req, opts);

        let events = match helpers::collect_events(Box::pin(stream)).await {
            Some(events) => events,
            None => {
                errors += 1;
                println!(
                    "[{}/{}] \x1b[31mERROR:\x1b[0m (#{} {prompt})",
                    i + 1,
                    n,
                    prompt_idx
                );
                continue;
            }
        };

        let has_tool_call = helpers::extract_tool_call(&events)
            .is_some_and(|(name, input)| tools.iter().any(|t| t.tool == name) && input.is_object());

        let is_noop = NOOP_INDICES.contains(&prompt_idx);
        let pass = if is_noop {
            !has_tool_call
        } else {
            has_tool_call
        };

        if pass {
            valid += 1;
        }

        let status = if pass {
            "\x1b[32mPASS:\x1b[0m"
        } else {
            "\x1b[31mFAIL:\x1b[0m"
        };
        println!("[{}/{}] {status} (#{} {prompt})", i + 1, n, prompt_idx);
    }

    let rate = f64::from(valid) / f64::from(total) * 100.0;
    println!("Strict: {valid}/{total} ({rate:.0}%)");
    if errors > 0 {
        println!("\x1b[33m{errors} stream errors (not counted in rate)\x1b[0m");
    }
    assert!(rate >= 90.0, "got {rate:.0}%");
}

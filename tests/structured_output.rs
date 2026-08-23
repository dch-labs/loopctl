//! Live test: structured output schema adherence (Ollama only).
//!
//! Run: `LOOPCTL_E2E=1 OLLAMA_MODEL=qwen2.5:7b cargo test --features ollama --test structured_output -- --nocapture`

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
use loopctl::structured::StructuredOutput;
#[cfg(feature = "ollama")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "ollama")]
#[derive(Debug, Serialize, Deserialize)]
struct PersonInfo {
    name: String,
    age: u32,
    city: String,
}

#[cfg(feature = "ollama")]
impl StructuredOutput for PersonInfo {
    fn name() -> &'static str {
        "person_info"
    }
    fn schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "city": {"type": "string"}
            },
            "required": ["name", "age", "city"]
        })
    }
    fn from_value(v: serde_json::Value) -> Result<Self, loopctl::structured::StructuredError> {
        serde_json::from_value(v).map_err(loopctl::structured::StructuredError::from)
    }
}

#[cfg(feature = "ollama")]
#[tokio::test]
async fn structured_outputs_match_schema() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1") {
        eprintln!("skipping (set LOOPCTL_E2E=1)");
        return;
    }

    let model = std::env::var("OLLAMA_MODEL").expect("set OLLAMA_MODEL");
    let client = loopctl::provider::ollama(&model).unwrap();
    let n: u32 = std::env::var("LOOPCTL_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let prompts = [
        "Tell me about Ada Lovelace as JSON with fields: name, age, city",
        "Tell me about Alan Turing as JSON with fields: name, age, city",
        "Tell me about Grace Hopper as JSON with fields: name, age, city",
        "Tell me about Linus Torvalds as JSON with fields: name, age, city",
        "Tell me about Margaret Hamilton as JSON with fields: name, age, city",
    ];
    let mut valid: u32 = 0;
    let dump_raw = std::env::var("LOOPCTL_RAW").as_deref() == Ok("1");

    for i in 0..n {
        let prompt = prompts[(i as usize) % prompts.len()];
        let messages = vec![loopctl::message::Message::user(prompt)];
        let opts = loopctl::structured::RequestOptions::new()
            .with_response_format(loopctl::structured::ResponseFormat::from_type::<PersonInfo>());
        let request = loopctl::api::StreamRequest {
            messages,
            system: None,
            tools: None,
        };
        let pass = match client.create_message_with_options(&request, opts).await {
            Ok(response) => {
                if dump_raw {
                    println!(
                        "--- raw reply (req {i}) ---\n{}\n---",
                        response.message.text_content()
                    );
                }
                match PersonInfo::from_value(client.extract_structured(&response.message)) {
                    Ok(p) if !p.name.is_empty() && !p.city.is_empty() => true,
                    Ok(p) => {
                        eprintln!("req {i}: parsed but empty fields: {p:?}");
                        false
                    }
                    Err(e) => {
                        eprintln!("req {i}: failed: {e}");
                        false
                    }
                }
            }
            Err(e) => {
                eprintln!("req {i}: failed: {e}");
                false
            }
        };
        if pass {
            valid += 1;
        }
        let status = if pass {
            "\x1b[32mPASS:\x1b[0m"
        } else {
            "\x1b[31mFAIL:\x1b[0m"
        };
        println!("[{}/{}] {status} {prompt}", i + 1, n);
    }

    let rate = f64::from(valid) / f64::from(n) * 100.0;
    println!("{valid}/{n} ({rate:.0}%) valid");
    assert!(rate >= 95.0, "got {rate:.0}%");
}

#[cfg(feature = "ollama")]
#[derive(Debug, Serialize, Deserialize)]
struct RepairPlan {
    summary: String,
    steps: Vec<String>,
}

#[cfg(feature = "ollama")]
impl StructuredOutput for RepairPlan {
    fn name() -> &'static str {
        "repair_plan"
    }
    fn schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["summary", "steps"]
        })
    }
    fn from_value(v: serde_json::Value) -> Result<Self, loopctl::structured::StructuredError> {
        serde_json::from_value(v).map_err(loopctl::structured::StructuredError::from)
    }
}

/// Live scored check of the lenient rescue on a real model, on the
/// non-strict path the parser exists for: plain chat requests (no
/// `response_format`, so the server does not enforce JSON) over rotating
/// repair scenarios, where the model is told to wrap its JSON in a markdown
/// fence behind a prose sentence. Each round reports which path served it —
/// a lenient rescue (the reply was wrapped, the strict parse of the whole
/// text failed, the scanner still extracted an object containing an array),
/// clean JSON (the model disobeyed and answered bare), or a failure. The
/// schema forces the exact shape the scanner used to choke on: an object
/// containing an array. Rounds default to `LOOPCTL_N` (20); the prompt-only
/// adherence bar is 90%, and at least one round must actually exercise the
/// rescue — a model that never wraps across every round passed without the
/// code under test ever running.
#[cfg(feature = "ollama")]
#[tokio::test]
async fn lenient_extraction_rescue_rate_across_prompts() {
    if std::env::var("LOOPCTL_E2E").as_deref() != Ok("1") {
        eprintln!("skipping (set LOOPCTL_E2E=1)");
        return;
    }
    let model = std::env::var("OLLAMA_MODEL").expect("set OLLAMA_MODEL");
    let client = loopctl::provider::ollama(&model).unwrap();
    let n: u32 = std::env::var("LOOPCTL_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let prompts = [
        "The build failed with a missing import. Produce a repair plan.",
        "Tests are flaky on CI. Produce a stabilization plan.",
        "The API returns 500 under load. Produce a debugging plan.",
        "The database query takes 40 seconds. Produce an optimization plan.",
        "A git merge conflict blocks the release. Produce a resolution plan.",
        "A Python script crashes on unicode filenames. Produce a fix plan.",
        "The npm install step fails with peer dependency errors. Produce a repair plan.",
        "A bash backup script silently skips locked files. Produce a hardening plan.",
        "The Rust build warns about unused dependencies. Produce a cleanup plan.",
        "A Go service leaks goroutines after each request. Produce a debugging plan.",
        "The cron job that rotates logs stopped running. Produce a recovery plan.",
        "An SSL certificate expires next week. Produce a renewal plan.",
        "The Kubernetes pod restarts every few minutes. Produce a diagnosis plan.",
        "A Ruby gem audit flags several outdated packages. Produce an upgrade plan.",
        "The TypeScript compiler emits implicit-any errors. Produce a typing plan.",
        "A Java service throws OutOfMemoryError nightly. Produce an investigation plan.",
        "The FTP upload script times out on large files. Produce an improvement plan.",
        "Disk space on the build server keeps filling up. Produce a cleanup plan.",
        "A SQL migration locked the production table. Produce a rollback plan.",
        "The README still describes the old CLI flags. Produce a documentation plan.",
        "A React page re-renders on every keystroke. Produce a performance plan.",
        "The Docker image is 2 GB and slow to pull. Produce a slimming plan.",
        "An API client hits rate limits during peak hours. Produce a backoff plan.",
        "A PowerShell script fails on paths with spaces. Produce a fix plan.",
        "The onboarding guide misses environment setup steps. Produce a writing plan.",
        "A C program segfaults on the second fopen call. Produce a debugging plan.",
        "The C++ build fails with an ambiguous overload error. Produce a repair plan.",
        "A Haskell module fails to compile with an infinite type error. Produce a fix plan.",
        "The C library leaks memory reported by valgrind. Produce a cleanup plan.",
        "A Swift app crashes on launch for older iOS users. Produce a compatibility plan.",
    ];
    let system = "Answer with exactly one short friendly sentence, then the JSON object \
                  with fields summary (a string) and steps (an array of 2-3 short strings) \
                  inside a ```json markdown fence."
        .to_string();
    let mut valid: u32 = 0;
    let mut lenient_rescues: u32 = 0;
    let mut clean_json: u32 = 0;

    for i in 0..n {
        let prompt = prompts[(i as usize) % prompts.len()];
        let request = loopctl::api::StreamRequest {
            messages: vec![loopctl::message::Message::user(prompt)],
            system: Some(system.clone()),
            tools: None,
        };
        let mut fail_note = String::new();
        let round = match client.create_message(&request).await {
            Ok(response) => {
                let raw = response.message.text_content();
                if std::env::var("LOOPCTL_RAW").as_deref() == Ok("1") {
                    println!("--- raw reply (round {}) ---\n{raw}\n---", i + 1);
                }
                let wrapped = serde_json::from_str::<serde_json::Value>(&raw).is_err();
                match RepairPlan::from_value(client.extract_structured(&response.message)) {
                    Ok(plan) if !plan.summary.is_empty() && !plan.steps.is_empty() => {
                        Some((wrapped, plan))
                    }
                    Ok(plan) => {
                        eprintln!("req {i}: extracted but empty fields: {plan:?}");
                        fail_note = "empty fields".into();
                        None
                    }
                    Err(e) => {
                        eprintln!("req {i}: extraction failed: {e}");
                        fail_note = "schema mismatch".into();
                        None
                    }
                }
            }
            Err(e) => {
                eprintln!("req {i}: request failed: {e}");
                fail_note = "request failed".into();
                None
            }
        };
        let (status, note) = match &round {
            Some((true, plan)) => {
                lenient_rescues += 1;
                valid += 1;
                (
                    "\x1b[32mPASS:\x1b[0m",
                    format!("lenient, {} steps", plan.steps.len()),
                )
            }
            Some((false, _)) => {
                clean_json += 1;
                valid += 1;
                ("\x1b[32mPASS:\x1b[0m", "clean json".to_string())
            }
            None => ("\x1b[31mFAIL:\x1b[0m", fail_note.clone()),
        };
        println!("[{}/{}] {status} {note} — {prompt}", i + 1, n);
    }

    let rate = f64::from(valid) / f64::from(n) * 100.0;
    println!(
        "{valid}/{n} ({rate:.0}%) valid · {lenient_rescues} lenient rescues · {clean_json} clean json"
    );
    assert!(
        rate >= 90.0,
        "prompt-only adherence below the 90% bar: {rate:.0}%"
    );
    assert!(
        lenient_rescues > 0,
        "every round returned bare JSON — the lenient path never ran; use a chattier model"
    );
}

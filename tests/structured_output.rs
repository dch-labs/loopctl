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

    for i in 0..n {
        let prompt = prompts[(i as usize) % prompts.len()];
        let messages = vec![loopctl::message::Message::user(prompt)];
        let pass =
            match loopctl::structured::request_structured::<PersonInfo>(&client, messages, None)
                .await
            {
                Ok(p) if !p.name.is_empty() && !p.city.is_empty() => true,
                Ok(p) => {
                    eprintln!("req {i}: parsed but empty fields: {p:?}");
                    false
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

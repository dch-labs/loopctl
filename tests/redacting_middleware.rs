//! Secret-scrubbing contracts for [`RedactingMiddleware`].
//!
//! Pins the curated pattern set (bearer, key-value tokens, AWS keys,
//! PEM blocks, GitHub/GitLab PATs), the high-entropy heuristic's
//! on/off behaviour and false-positive discipline, host extension, the
//! text/multipart walk, and the advisory-only contract (loop semantics
//! untouched).
//!
//! Requires the `redaction` feature.

#![cfg(feature = "redaction")]
#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone
)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use loopctl::message::{ToolContent, ToolContentPart};
use loopctl::middleware::{
    RedactingMiddleware, SecretPattern, SecretPatternSet, ToolDispatchContext, ToolMiddleware,
    ToolPipeline,
};
use loopctl::tool::{PermissionCheck, ToolContext, ToolRegistry};

/// A middleware that short-circuits with a fixed output, mirroring the
/// canned-tool shape the other middleware suites use.
struct FixedOutputMiddleware {
    /// The output every dispatch returns.
    output: ToolContent,
    /// The display hint carried on the result.
    display_hint: Option<loopctl::tool::DisplayHint>,
}

impl ToolMiddleware for FixedOutputMiddleware {
    fn name(&self) -> &'static str {
        "fixed_output"
    }
    fn dispatch<'a>(
        &'a self,
        _ctx: &'a mut ToolDispatchContext,
        _next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = loopctl::tool::ToolDispatchResult> + Send + 'a>> {
        let mut output = loopctl::tool::ToolOutput::text("").with_payload(self.output.clone());
        if let Some(hint) = self.display_hint.clone() {
            output = output.with_hint(hint);
        }
        Box::pin(async move { output.into() })
    }
}

/// A dispatch context for the `probe` tool.
fn probe_ctx() -> ToolDispatchContext {
    ToolDispatchContext {
        tool_name: "probe".to_string(),
        input: serde_json::json!({}),
        call_id: "c1".to_string(),
        turn_number: 0,
        cancel: Arc::new(loopctl::cancel::CancelSignal::new()),
        permission: PermissionCheck::Allow,
        tool_context: ToolContext::default(),
    }
}

/// Run one dispatch through a redacting pipeline over `output` and
/// return the rewritten result.
async fn redact(output: ToolContent) -> loopctl::tool::ToolDispatchResult {
    redact_with(output, SecretPatternSet::default_common(), None).await
}

/// Run one dispatch with an explicit pattern set and display hint.
async fn redact_with(
    output: ToolContent,
    patterns: SecretPatternSet,
    display_hint: Option<loopctl::tool::DisplayHint>,
) -> loopctl::tool::ToolDispatchResult {
    let pipeline = ToolPipeline::builder()
        .with_middleware(RedactingMiddleware::new(patterns))
        .with_middleware(FixedOutputMiddleware {
            output,
            display_hint,
        })
        .with_core(Arc::new(ToolRegistry::new()))
        .build()
        .expect("pipeline builds");
    pipeline.invoke(probe_ctx()).await
}

/// The rendered text of a result's output.
fn text_of(result: &loopctl::tool::ToolDispatchResult) -> String {
    result.output.to_string()
}

#[tokio::test]
async fn bearer_header_is_redacted() {
    let result = redact(ToolContent::from_string(
        "curl -H 'Authorization: Bearer abcdef-1234-5678' https://api.example.com",
    ))
    .await;
    let text = text_of(&result);
    assert_eq!(
        text, "curl -H '[REDACTED:bearer]' https://api.example.com",
        "the whole header value is replaced; surrounding text survives"
    );
}

#[tokio::test]
async fn aws_access_key_is_redacted() {
    let result = redact(ToolContent::from_string(
        "configured with key AKIAIOSFODNN7EXAMPLE",
    ))
    .await;
    assert!(
        text_of(&result).contains("[REDACTED:aws_access_key]"),
        "the AKIA literal fires: {}",
        text_of(&result)
    );
}

#[tokio::test]
async fn github_pat_is_redacted() {
    let pat = format!("ghp_{}", "A".repeat(36));
    let result = redact(ToolContent::from_string(format!("token: {pat}"))).await;
    assert!(
        text_of(&result).contains("[REDACTED:github_pat]"),
        "the gh[pousr]_ prefix family fires: {}",
        text_of(&result)
    );
}

#[tokio::test]
async fn gitlab_pat_is_redacted() {
    let pat = format!("glpat-{}", "x".repeat(20));
    let result = redact(ToolContent::from_string(format!("ci token {pat}"))).await;
    assert!(
        text_of(&result).contains("[REDACTED:gitlab_pat]"),
        "the glpat- literal fires: {}",
        text_of(&result)
    );
}

#[tokio::test]
async fn pem_block_collapses_to_one_placeholder() {
    let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n7 lines of base64\n-----END RSA PRIVATE KEY-----";
    let result = redact(ToolContent::from_string(format!("cert:\n{pem}\nafter"))).await;
    let text = text_of(&result);
    assert_eq!(
        text, "cert:\n[REDACTED:pem_private_key]\nafter",
        "the whole block becomes one placeholder, not one per line"
    );
}

#[tokio::test]
async fn api_key_kv_is_redacted() {
    let result = redact(ToolContent::from_string(
        "export api_key=ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
    ))
    .await;
    let text = text_of(&result);
    assert!(
        text.contains("api_key=[REDACTED:api_key_kv]") || text.contains("[REDACTED:api_key_kv]"),
        "the key-value pattern fires: {text}"
    );
}

#[tokio::test]
async fn entropy_heuristic_catches_an_unknown_shape() {
    // 43 chars of mixed-case base64-ish noise: no literal matches.
    let token = "qX7mZ2vQ9wL4nR8tY3uK5jH7gF6dS2aP1oI9bV3cX2z";
    let result = redact(ToolContent::from_string(format!("blob {token} end"))).await;
    assert!(
        text_of(&result).contains("[REDACTED:high_entropy]"),
        "novel high-entropy tokens are redacted: {}",
        text_of(&result)
    );
}

#[tokio::test]
async fn entropy_heuristic_can_be_disabled() {
    let token = "qX7mZ2vQ9wL4nR8tY3uK5jH7gF6dS2aP1oI9bV3cX2z";
    let result = redact_with(
        ToolContent::from_string(format!("blob {token} end")),
        SecretPatternSet::default_common().with_entropy_heuristic(false),
        None,
    )
    .await;
    assert!(
        text_of(&result).contains(token),
        "with the heuristic off, unknown shapes pass through: {}",
        text_of(&result)
    );
}

#[tokio::test]
async fn host_added_pattern_fires_alongside_curated() {
    let custom = SecretPattern {
        kind: "myco_token",
        pattern: regex::Regex::new(r"MYCO-TOKEN-[A-Z0-9]{8,}").expect("valid literal"),
    };
    let result = redact_with(
        ToolContent::from_string(
            "Authorization: Bearer abcdef-1234-5678 and MYCO-TOKEN-AB12CD34EF56",
        ),
        SecretPatternSet::default_common().with_pattern(custom),
        None,
    )
    .await;
    let text = text_of(&result);
    assert!(
        text.contains("[REDACTED:bearer]") && text.contains("[REDACTED:myco_token]"),
        "curated and host shapes both fire: {text}"
    );
}

#[tokio::test]
async fn multipart_text_parts_scrubbed_image_untouched() {
    let image_payload = "iVBORw0KGgoAAAANSUhEUg".to_string();
    let output = ToolContent::from_multipart(vec![
        ToolContentPart::Text {
            text: "key AKIAIOSFODNN7EXAMPLE here".to_string(),
        },
        ToolContentPart::Text {
            text: "perfectly clean text".to_string(),
        },
        ToolContentPart::Image {
            source: loopctl::message::ImageSource {
                encoding: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: image_payload.clone(),
            },
        },
    ]);
    let result = redact(output).await;
    match result.output {
        ToolContent::Multipart(parts) => {
            match &parts[0] {
                ToolContentPart::Text { text } => assert!(
                    text.contains("[REDACTED:aws_access_key]"),
                    "secret-bearing text part is scrubbed: {text}"
                ),
                ToolContentPart::Image { .. } => panic!("expected Text part 0"),
            }
            match &parts[1] {
                ToolContentPart::Text { text } => {
                    assert_eq!(text, "perfectly clean text", "clean part unchanged");
                }
                ToolContentPart::Image { .. } => panic!("expected Text part 1"),
            }
            match &parts[2] {
                ToolContentPart::Image { source, .. } => assert_eq!(
                    source.data, image_payload,
                    "image data is byte-identical after redaction"
                ),
                ToolContentPart::Text { .. } => panic!("expected Image part 2"),
            }
        }
        ToolContent::Text(t) => panic!("expected Multipart, got Text: {t}"),
    }
}

#[tokio::test]
async fn text_single_string_path_is_scrubbed() {
    let result = redact(ToolContent::from_string("AKIAIOSFODNN7EXAMPLE")).await;
    assert_eq!(
        text_of(&result),
        "[REDACTED:aws_access_key]",
        "the Text arm scrubs in place"
    );
}

#[tokio::test]
async fn commit_sha_is_not_redacted() {
    // 40 hex chars: the hex alphabet caps entropy at 4.0 bits/byte,
    // below the 4.5 heuristic threshold.
    let sha = "0123456789abcdef0123456789abcdef01234567";
    let result = redact(ToolContent::from_string(format!("commit {sha} done"))).await;
    assert_eq!(
        text_of(&result),
        format!("commit {sha} done"),
        "a benign hex SHA survives the heuristic"
    );
}

#[test]
fn clean_output_passes_through_identical() {
    let clean = "the build finished in 3.2s with no warnings";
    let mut scrubbed = clean.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(count, 0, "no substitutions on clean text");
    assert_eq!(scrubbed, clean, "byte-identical pass-through");
}

#[test]
fn scrub_reports_the_redaction_count() {
    let mut dirty = String::from("keys AKIAIOSFODNN7EXAMPLE and AKIAIOSFODNN7EXAMPLF");
    let count = SecretPatternSet::default_common().scrub(&mut dirty);
    assert_eq!(count, 2, "one redaction per match");
    assert_eq!(
        dirty,
        "keys [REDACTED:aws_access_key] and [REDACTED:aws_access_key]"
    );
}

#[tokio::test]
async fn middleware_name_is_redaction() {
    let middleware = RedactingMiddleware::new(SecretPatternSet::default_common());
    assert_eq!(middleware.name(), "redaction");
}

#[tokio::test]
async fn redacted_result_keeps_error_state_and_display_hint() {
    let result = redact_with(
        ToolContent::from_string("AKIAIOSFODNN7EXAMPLE"),
        SecretPatternSet::default_common(),
        Some(loopctl::tool::DisplayHint::Diff),
    )
    .await;
    assert!(!result.is_error, "a redacted output is still a success");
    assert!(
        matches!(result.display_hint, Some(loopctl::tool::DisplayHint::Diff)),
        "the original display hint is preserved"
    );
}

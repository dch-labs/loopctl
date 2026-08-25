//! Secret-scrubbing contracts for [`RedactingMiddleware`].
//!
//! Pins the curated pattern set (bearer, key-value tokens, AWS keys,
//! PEM blocks, GitHub/GitLab PATs), the high-entropy heuristic's
//! on/off behaviour, exact boundaries, and false-positive discipline,
//! host extension, the text/multipart walk, the advisory-only contract
//! (loop semantics untouched), honest redaction counts (an echoed
//! placeholder is not a new redaction), and a randomized property
//! holding completeness, survivor preservation, and idempotence over
//! mixed token soup.
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

#[test]
fn entropy_boundary_is_thirty_two_characters_and_four_point_five_bits() {
    let alphabet = "q7Xk2Lm9Pz4Rt6Vb8Nc1Sd3Gf5HjKWyU6MxE0gT8vC5nB1sA4dF2hJ";
    let tok31: String = alphabet.chars().take(31).collect();
    let mut text = format!("id {tok31} end");
    SecretPatternSet::default_common().scrub(&mut text);
    assert!(
        text.contains(&tok31),
        "a 31-char token stays visible even at high entropy: {text}"
    );

    let tok32: String = alphabet.chars().take(32).collect();
    let mut text = format!("id {tok32} end");
    SecretPatternSet::default_common().scrub(&mut text);
    assert!(
        !text.contains(&tok32),
        "a 32-char high-entropy token is redacted: {text}"
    );

    let sha256_hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    let mut text = format!("commit {sha256_hex}");
    SecretPatternSet::default_common().scrub(&mut text);
    assert!(
        text.contains(sha256_hex),
        "hex tops out at 4.0 bits per byte and stays visible: {text}"
    );

    let b64_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = format!("payload {b64_alphabet}");
    SecretPatternSet::default_common().scrub(&mut text);
    assert!(
        !text.contains(b64_alphabet),
        "a full-diversity base64 token is redacted: {text}"
    );
}

#[test]
fn case_and_separator_variants_and_repeated_secrets_all_redacted() {
    let mut text = String::from(
        "AUTHORIZATION:  BEARER abcdefghijklmnopqrstuvwxyz012345 then \
         token: abcdefghijklmnop0123 and SECRET=\"ABCDEFGHIJKLMNOPQRSTUVWXYZ01\"",
    );
    let count = SecretPatternSet::default_common().scrub(&mut text);
    assert_eq!(count, 3, "case and separator variants all fire: {text}");

    let mut both = String::from(
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345 and \
         Authorization: Bearer zyxwvutsrqponmlkjihgfedcba543210",
    );
    let count = SecretPatternSet::default_common().scrub(&mut both);
    assert_eq!(count, 2, "two secrets on one line each count: {both}");
    assert!(
        !both.contains("abcdefghijkl"),
        "no residue survives: {both}"
    );
}

#[test]
fn echoed_placeholder_is_not_counted_again() {
    let clean = "prior result: [REDACTED:high_entropy] and nothing else";
    let mut text = clean.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut text);
    assert_eq!(
        count, 0,
        "a pre-existing placeholder is not a new redaction"
    );
    assert_eq!(text, clean, "byte-identical pass-through");
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A 62-character mixed alphabet for high-entropy secret bodies.
const TOKEN_ALPHABET: &str = "q7Xk2Lm9Pz4Rt6Vb8Nc1Sd3Gf5HjKWyU6MxE0gT8vC5nB1sA4dF2hJ3gZ5lQ";

/// The 16 hex digits, for low-entropy survivors (4.0 bits per byte).
const HEX_ALPHABET: &str = "0123456789abcdef";

/// Uppercase alphanumerics, the AWS access-key-id character class.
const UPPER_ALNUM: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/// A distinct-chars body of exactly `len` characters — guaranteed
/// high-entropy when `len` is large (all characters distinct).
fn distinct_body(rng: &mut Lcg, len: usize) -> String {
    let mut chars: Vec<char> = TOKEN_ALPHABET.chars().collect();
    let n = chars.len();
    for i in 0..len.min(n) {
        let j = i + (rng.below((n - i) as u64) as usize);
        chars.swap(i, j);
    }
    chars.into_iter().take(len).collect()
}

/// A hex string of exactly `len` digits — always at hex's 4.0-bit
/// ceiling, so it must survive scrubbing.
fn hex_body(rng: &mut Lcg, len: usize) -> String {
    (0..len)
        .map(|_| {
            HEX_ALPHABET
                .chars()
                .nth(rng.below(16) as usize)
                .unwrap_or('0')
        })
        .collect()
}

/// Generate one output: `(text, planted secret bodies, benign survivors)`.
fn gen_output(rng: &mut Lcg) -> (String, Vec<String>, Vec<String>) {
    let mut text = String::from("log line\n");
    let mut secrets = Vec::new();
    let mut survivors = Vec::new();
    let pieces = 3 + rng.below(8);
    for i in 0..pieces {
        let extra = rng.below(10) as usize;
        let body = distinct_body(rng, 32 + extra);
        match rng.below(4) {
            0 => {
                let secret = match i % 6 {
                    0 => format!("Authorization: Bearer {body}"),
                    1 => format!("api_key={body}"),
                    2 => {
                        let tail: String = UPPER_ALNUM
                            .chars()
                            .cycle()
                            .skip(rng.below(36) as usize)
                            .take(16)
                            .collect();
                        format!("AKIA{tail}")
                    }
                    3 => format!("-----BEGIN PRIVATE KEY-----\n{body}\n-----END PRIVATE KEY-----"),
                    4 => format!("ghp_{body}"),
                    _ => format!("glpat-{body}"),
                };
                secrets.push(body);
                text.push_str(&secret);
            }
            1 => {
                let sha = hex_body(rng, 40);
                survivors.push(format!("sha1:{sha}"));
                text.push_str(&format!("sha1:{sha}"));
            }
            _ => {
                let word = match rng.below(3) {
                    0 => "build ok".to_string(),
                    1 => format!("file{}.rs", hex_body(rng, 4)),
                    _ => "warnings: none".to_string(),
                };
                survivors.push(word.clone());
                text.push_str(&word);
            }
        }
        text.push(' ');
    }
    (text, secrets, survivors)
}

#[test]
fn scrub_is_complete_survivor_preserving_and_idempotent() {
    let mut rng = Lcg(0x5EED_C0DE);
    for iter in 0..500 {
        let (text, secrets, survivors) = gen_output(&mut rng);
        let set = SecretPatternSet::default_common();

        let mut once = text.clone();
        let count = set.scrub(&mut once);

        for body in &secrets {
            assert!(
                !once.contains(body.as_str()),
                "iter {iter}: secret body {body} survived: {once}"
            );
        }
        for s in &survivors {
            assert!(
                once.contains(s.as_str()),
                "iter {iter}: benign survivor {s} was eaten: {once}"
            );
        }
        assert!(
            count >= secrets.len(),
            "iter {iter}: {count} redactions for {} planted secrets",
            secrets.len()
        );

        let mut twice = once.clone();
        let second = set.scrub(&mut twice);
        assert_eq!(twice, once, "iter {iter}: scrub is not idempotent: {twice}");
        assert_eq!(second, 0, "iter {iter}: second pass found work");
    }
}

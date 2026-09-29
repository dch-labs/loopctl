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

/// The AWS documentation placeholder access-key id, assembled from
/// pieces so the contiguous credential shape never appears in source
/// — push-protection scanners match the shape, and this suite must
/// carry it only at runtime. `final_char` swaps the last character so
/// a count test can plant two distinct keys.
fn aws_key_fixture(final_char: char) -> String {
    let mut key = ["AKIA", "IOSFODNN7EXAMPL"].concat();
    key.push(final_char);
    key
}

#[tokio::test]
async fn aws_access_key_is_redacted() {
    let result = redact(ToolContent::from_string(format!(
        "configured with key {}",
        aws_key_fixture('E')
    )))
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
            text: format!("key {} here", aws_key_fixture('E')),
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
    let result = redact(ToolContent::from_string(aws_key_fixture('E'))).await;
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
    let mut dirty = format!("keys {} and {}", aws_key_fixture('E'), aws_key_fixture('F'));
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
        ToolContent::from_string(aws_key_fixture('E')),
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

/// A strict UpperCamel identifier of 4-5 short humps — the shape a
/// bare type or constant reference takes, which must survive
/// scrubbing. Capped under the entropy lane's 32-character gate, so
/// the identifier rule is the only lane that ever sees it.
fn camel_body(rng: &mut Lcg) -> String {
    const LOWER: &str = "abcdefghijklmnopqrstuvwxyz";
    const UPPER: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let mut token = String::new();
    for _ in 0..4 + rng.below(2) {
        token.push(UPPER.chars().nth(rng.below(26) as usize).unwrap_or('A'));
        for _ in 0..3 + rng.below(2) {
            token.push(LOWER.chars().nth(rng.below(26) as usize).unwrap_or('a'));
        }
    }
    token
}

/// A lowercase-led body carrying one guaranteed `Xqwz` hump — real
/// base64 shape, which must never survive a bare declaration.
fn hump_body(rng: &mut Lcg) -> String {
    const MIXED: &str = "abcdefghijkmnopqrstuvwxyz023456789";
    let head = 8 + rng.below(6) as usize;
    let tail = 6 + rng.below(6) as usize;
    let mut body: String = (0..head)
        .map(|_| MIXED.chars().nth(rng.below(33) as usize).unwrap_or('a'))
        .collect();
    body.push_str("Xqwz");
    body.extend((0..tail).map(|_| MIXED.chars().nth(rng.below(33) as usize).unwrap_or('a')));
    body
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
        match rng.below(5) {
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
            2 => {
                let line = format!("let secret = {}; ", camel_body(rng));
                survivors.push(line.clone());
                text.push_str(&line);
            }
            3 => {
                let body = hump_body(rng);
                let line = format!("const token = {body}; ");
                secrets.push(body);
                text.push_str(&line);
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

#[test]
fn a_space_free_multiline_listing_passes_through_verbatim() {
    // An approved `ls` came back fully redacted (dch session 11e5eecc):
    // one space-free multi-line listing collapsed into a single
    // candidate whose ordinary mixed-case filenames cleared the entropy
    // threshold, swallowing the listing plus the exit line.
    let listing = "Desktop\nDocuments\nDownloads\nProjects\nShot_2026-09-21_10-34-56.png\nSomeReport_Final_v2.pdf\n[exit 0, 2ms]";
    let mut scrubbed = listing.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "no substitution on an ordinary listing: {scrubbed}"
    );
    assert_eq!(
        scrubbed, listing,
        "newline-bounded tokens never merge into one candidate"
    );
}

#[test]
fn known_secrets_inside_a_newline_listing_still_redact() {
    let listing = format!(
        "Desktop\n{}\nShot_2026-09-21_10-34-56.png\nnote: api_key=ABCDEFGHIJKLMNOPQRSTUVWXYZ123456 trailing words\n[exit 0, 2ms]",
        aws_key_fixture('E')
    );
    let mut scrubbed = listing.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(count, 2, "one redaction per secret line: {scrubbed}");
    assert!(
        scrubbed.contains("[REDACTED:aws_access_key]"),
        "the AWS key line still redacts: {scrubbed}"
    );
    assert!(
        scrubbed.contains("[REDACTED:api_key_kv]"),
        "a key-value secret sharing its line with words still redacts: {scrubbed}"
    );
    assert!(
        scrubbed.contains("Desktop")
            && scrubbed.contains("Shot_2026-09-21_10-34-56.png")
            && scrubbed.contains("[exit 0, 2ms]"),
        "the ordinary lines around the secrets pass through: {scrubbed}"
    );
}

#[test]
fn tabs_and_carriage_returns_bound_tokens_like_spaces() {
    let crlf = "Desktop\r\nDocuments\r\n[exit 0, 2ms]";
    let mut scrubbed = crlf.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(count, 0, "a CRLF listing is ordinary output: {scrubbed}");
    assert_eq!(
        scrubbed, crlf,
        "carriage returns and line feeds bound tokens"
    );

    let token = "qX7mZ2vQ9wL4nR8tY3uK5jH7gF6dS2aP1oI9bV3cX2z";
    let mut tabbed = format!("blob\t{token}\tend");
    let count = SecretPatternSet::default_common().scrub(&mut tabbed);
    assert_eq!(count, 1, "the tab-bounded token is one candidate: {tabbed}");
    assert_eq!(
        tabbed, "blob\t[REDACTED:high_entropy]\tend",
        "tabs bound tokens exactly as spaces do — and survive the rewrite"
    );
}

#[test]
fn space_free_rust_expressions_pass_through_verbatim() {
    let source = "let elapsed = u64::try_from(now.duration_since(began).as_millis()).unwrap_or(0);\nlet composer_height = text_rows.saturating_add(INPUT_VERTICAL_PADDING.saturating_mul(2));";
    let mut scrubbed = source.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(count, 0, "no substitution on ordinary code: {scrubbed}");
    assert_eq!(
        scrubbed, source,
        "a call chain with interior punctuation is not one candidate"
    );
}

#[test]
fn markdown_table_link_cells_pass_through_verbatim() {
    let row = "| [L-7](./roadmap/01-loopctl-0.2.0/tasks/L-7-on-text-delta.md) `on_text_delta` | [T-27](./roadmap/02-dch-v1/tasks/T-27-streaming-text-display.md) (streaming text display) |";
    let mut scrubbed = row.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "no substitution on a documentation link: {scrubbed}"
    );
    assert_eq!(scrubbed, row, "markdown link cells pass through verbatim");
}

#[test]
fn a_double_colon_bearing_core_passes_through_verbatim() {
    let path = "Zq7Xk2Lm9Pz4Rt6Vb8Nc1Sd3Gf5HjK::WyU6MxE0gT8vC5nB1sA4dF2hJ";
    let mut scrubbed = format!("use {path};");
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "no substitution on a path-shaped token: {scrubbed}"
    );
    assert!(
        scrubbed.contains(path),
        "`::` bounds candidates exactly as `(` does: {scrubbed}"
    );
}

#[test]
fn an_unpunctuated_high_entropy_run_still_redacts() {
    let token = "qX7mZ2vQ9wL4nR8tY3uK5jH7gF6dS2aP1oI9bV3cX2z";
    let mut scrubbed = format!("blob {token} end");
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "the run bounding is not a blanket off switch: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "blob [REDACTED:high_entropy] end",
        "a paren-free, colon-free credential run still redacts"
    );
}

#[test]
fn a_shell_path_assignment_passes_through_verbatim() {
    let line = "PATH=$HOME/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/bin:$PATH make ci";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "no substitution on a toolchain PATH line: {scrubbed}"
    );
    assert_eq!(
        scrubbed, line,
        "a PATH assignment is shell syntax, not a credential"
    );
}

#[test]
fn git_status_porcelain_paths_pass_through_verbatim() {
    let line = " M roadmap/03e-loopctl-0.3.4/tasks/L-118-event-hub.md";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(count, 0, "no substitution on a git status path: {scrubbed}");
    assert_eq!(
        scrubbed, line,
        "a repository path is a path, not a credential"
    );
}

#[test]
fn cargo_test_output_paths_pass_through_verbatim() {
    let line =
        "     Running tests/sqlite_memory.rs (target/debug/deps/sqlite_memory-7dcdab93082f5511)";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "no substitution on a cargo test binary path: {scrubbed}"
    );
    assert_eq!(
        scrubbed, line,
        "a target/debug/deps path is a build artifact, not a credential"
    );
}

#[test]
fn a_git_dependency_source_line_passes_through_verbatim() {
    let line = "source = \"git+https://github.com/dch-labs/loopctl?branch=master#d4ee826544deb86be6235eea17b7bae06c332b81\"";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "no substitution on a git dependency source: {scrubbed}"
    );
    assert_eq!(
        scrubbed, line,
        "a git+https source URL with its pinned rev is dependency metadata, not a credential"
    );
}

#[test]
fn a_json_raw_string_literal_passes_through_verbatim() {
    let line = "let raw = r#\"{\"type\":\"thinking\",\"thinking\":\"placeholder body for the round trip test\",\"duration_ms\":125}\"#;";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "no substitution on a JSON raw string literal: {scrubbed}"
    );
    assert_eq!(
        scrubbed, line,
        "a JSON literal in source code is code, not a credential"
    );
}

#[test]
fn a_let_declared_token_literal_passes_verbatim() {
    let line = "let token = \"placeholder0123456789\";";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "no substitution on a variable declaration: {scrubbed}"
    );
    assert_eq!(
        scrubbed, line,
        "a let-declared token with an ordinary literal is code, not a dump"
    );
}

#[test]
fn a_mut_declaration_passes_verbatim() {
    let line = "let mut secret = \"placeholder0123456789\".to_string();";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(count, 0, "no substitution on a mut declaration: {scrubbed}");
    assert_eq!(
        scrubbed, line,
        "a let-mut-declared secret with an ordinary literal is code, not a dump"
    );
}

#[test]
fn a_type_annotated_declaration_passes_verbatim() {
    let line = "let secret: SecretStoreHandle012345 = Default::default();";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "a declaration whose type annotation looks like a value is still a declaration: {scrubbed}"
    );
    assert_eq!(
        scrubbed, line,
        "the declaration arm spans the annotation, so the plain arm cannot fire on it"
    );
}

#[test]
fn a_compound_named_declaration_passes_verbatim() {
    let source = "let entropy_token = \"placeholder0123456789\";\nlet client_secret = \"placeholder9876543210\";";
    let mut scrubbed = source.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "no substitution on compound-named declarations: {scrubbed}"
    );
    assert_eq!(
        scrubbed, source,
        "a declaration whose name merely contains token or secret is still a declaration"
    );
}

#[test]
fn an_env_style_compound_key_dump_still_redacts() {
    let line = "OPENAI_API_KEY=placeholder0123456789";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "an underscore-joined uppercase env name is a dump, not a declaration: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "OPENAI_[REDACTED:api_key_kv]",
        "the plain arm still matches keys inside compound env names"
    );
}

#[test]
fn an_exported_quoted_kv_still_redacts() {
    let line = "export TOKEN=\"placeholder0123456789\"";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "export is an env-dump keyword, not a declaration keyword: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "export [REDACTED:api_key_kv]",
        "a quoted exported token still redacts"
    );
}

#[test]
fn a_slash_bearing_credential_run_still_redacts() {
    let token = "qX7mZ/vQ9wL4nR8tY3uK5/jH7gF6dS2aP1oI/9bV3cX2z";
    let mut scrubbed = format!("blob {token} end");
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "slashes alone do not make a mixed-case run a path: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "blob [REDACTED:high_entropy] end",
        "a slash-bearing high-entropy credential run still redacts"
    );
}

#[test]
fn a_dense_segment_inside_a_path_shaped_run_masks_alone() {
    let secret = ["J8kL2mN4pQ6rS8tUv1Wx", "9zQ4wE7rT5yU6iO3pA8b"].concat();
    let url = format!("https://host/repos/tokens/{secret}");
    let mut scrubbed = url.clone();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "a secret riding inside a path-shaped run masks as its own segment: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "https://host/repos/tokens/[REDACTED:high_entropy]",
        "the path structure and word segments survive while only the dense segment masks"
    );
}

#[test]
fn a_bare_credential_shaped_declaration_value_masks_in_place() {
    let line = "const api_key = Ab3dEf5gHi7jKl9mNo1pQr2s;";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "a credential-shaped bare value in a declaration masks in place: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "const api_key = [REDACTED:api_key_kv];",
        "the declaration keyword and identifier survive; only the secret-shaped initializer masks"
    );
}

#[test]
fn a_bare_type_initializer_passes_verbatim() {
    let line = "let token = SecretStoreHandle012345::new();";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "no substitution on a CamelCase bare initializer: {scrubbed}"
    );
    assert_eq!(
        scrubbed, line,
        "a bare initializer that reads as a type or constant reference is code, not a credential"
    );
}

#[test]
fn a_hump_bearing_bare_base64_initializer_masks_in_place() {
    let cases = [
        (
            "const api_key = dGhpc0lzTXlDcmVkaXQ7ab;",
            "const api_key = [REDACTED:api_key_kv];",
        ),
        (
            "const secret = c2VjcmV0X2tleV9oZXJlMTIz;",
            "const secret = [REDACTED:api_key_kv];",
        ),
        (
            "let token = Z2l0aHViX3BhdF90b2tlbl94eXo7;",
            "let token = [REDACTED:api_key_kv];",
        ),
        (
            "const apiKey = dGhpc0lzTXlDcmVkaXQ7ab;",
            "const apiKey = [REDACTED:api_key_kv];",
        ),
    ];
    for (line, expected) in cases {
        let mut scrubbed = line.to_string();
        let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
        assert_eq!(
            count, 1,
            "a hump-bearing bare base64 initializer is a credential, not an identifier: {line} -> {scrubbed}"
        );
        assert_eq!(
            scrubbed, expected,
            "only the initializer masks; the declaration syntax survives: {line}"
        );
    }
}

#[test]
fn a_text_derived_base64_bare_initializer_masks_via_the_key_value_lane() {
    let line = "const api_key = dGhpc0lzQW5vdGhlckxvbmdlclNlY3JldFRvUGx1Z2lu;";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "a 44-character text-derived base64 value sits under the entropy lane's density, so the key-value lane is its only mask: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "const api_key = [REDACTED:api_key_kv];",
        "the value masks while the declaration syntax survives"
    );
}

#[test]
fn a_uniformly_lowercase_bare_initializer_passes_verbatim() {
    let line = "let secret = defaultconfigurationvalue;";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "a uniformly lowercase bare initializer is an identifier, not a credential: {scrubbed}"
    );
    assert_eq!(
        scrubbed, line,
        "a lowercase word-led initializer survives the read verbatim"
    );
}

#[test]
fn a_strict_camel_by_digit_resets_initializer_masks_in_place() {
    let cases = [
        (
            "const api_key = P7XougXTEwqES301BAt8ljr5;",
            "const api_key = [REDACTED:api_key_kv];",
        ),
        (
            "let token = Pq7Ou2Xy5Wz9Cb4Df6G;",
            "let token = [REDACTED:api_key_kv];",
        ),
    ];
    for (line, expected) in cases {
        let mut scrubbed = line.to_string();
        let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
        assert_eq!(
            count, 1,
            "a lone lowercase letter terminated by a digit is credential material: {line} -> {scrubbed}"
        );
        assert_eq!(
            scrubbed, expected,
            "only the initializer masks; the declaration syntax survives: {line}"
        );
    }
}

#[test]
fn a_single_letter_hump_before_digits_masks_like_a_credential() {
    let line = "let secret = Ed25519PrivateKey::new();";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "a single-letter hump before digits is chance camel, not a type name: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "let secret = [REDACTED:api_key_kv]::new();",
        "the initializer masks while the path call and declaration syntax survive"
    );
}

#[test]
fn a_long_crate_name_test_binary_segment_passes_verbatim() {
    let masking = "     Running tests/golden_requests.rs (target/debug/deps/golden_requests-9f3c1b7e5a2d4f68)";
    let mut scrubbed = masking.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "a cargo test-binary artifact is a path whatever the crate-name length: {scrubbed}"
    );
    assert_eq!(
        scrubbed, masking,
        "a long crate name pushes the artifact segment past the length gate — it still passes"
    );

    let passing = "     Running tests/golden_requests.rs (target/debug/deps/golden_requests-77e0e51a2c4b9d31)";
    let mut scrubbed = passing.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "the twin-hash direction keeps passing: {scrubbed}"
    );
}

#[test]
fn an_artifact_shaped_segment_after_a_foreign_parent_still_masks() {
    let line = "fetch failed: https://host/tokens/golden_requests-9f3c1b7e5a2d4f68";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "the artifact-tail exemption is scoped to cargo's own directories: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "fetch failed: https://host/tokens/[REDACTED:high_entropy]",
        "a dense segment after a non-artifact parent stays measured"
    );
}

#[test]
fn a_digit_bearing_segment_run_is_not_path_shaped() {
    let token = ["38gO/rqh", "/l3wyX8KX/", "62sp/JoazycZ9FcQS1fDTK"].concat();
    let mut scrubbed = format!("blob {token} end");
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "short lowercase-or-digit segments do not make a credential run a path: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "blob [REDACTED:high_entropy] end",
        "a slash-bearing credential whose only word-shaped segments carry digits or fall under four letters still redacts"
    );
}

#[test]
fn a_run_is_path_shaped_only_on_four_letter_word_segments() {
    let head = "Xq7mZ2vQ9wL4nR8tY3uK5";
    let tail = "jH7gF6dS2aP1oI9bV3cX2z";

    let words = format!("blob {head}/apps/tools/{tail} end");
    let mut scrubbed = words.clone();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "two pure four-letter word segments keep a slash run a path: {scrubbed}"
    );
    assert_eq!(
        scrubbed, words,
        "a word-segmented slash run passes through verbatim"
    );

    let one_word = format!("blob {head}/apps/{tail} end");
    let mut scrubbed = one_word.clone();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "a single word segment among noise is not a path: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "blob [REDACTED:high_entropy] end",
        "the path guard needs two word segments, not one — the whole run is one candidate"
    );

    let digit_bearing = format!("blob {head}/app2/t00ls/{tail} end");
    let mut scrubbed = digit_bearing.clone();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "a digit inside a segment disqualifies it as a word: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "blob [REDACTED:high_entropy] end",
        "digit-bearing segments leave the whole run measured"
    );

    let short_segments = format!("blob {head}/app/tool/{tail} end");
    let mut scrubbed = short_segments.clone();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "segments under four characters are not words: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "blob [REDACTED:high_entropy] end",
        "three-letter segments leave the whole run measured"
    );
}

#[test]
fn a_dense_quoted_secret_in_a_declaration_masks_its_value() {
    let line = "let token = \"J8kL2mN4pQ6rS8tUv1Wxyz\";";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "a dense quoted literal in a declaration masks its value: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "let token = \"[REDACTED:api_key_kv]\";",
        "only the secret literal is masked — the declaration syntax survives the read"
    );
}

#[test]
fn a_low_entropy_hex_secret_in_a_declaration_masks_its_value() {
    let line = "let secret = \"deadbeefcafe0123456789abcdef0123\";";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "a hex literal sits below the entropy lane's density, so the key-value lane is its only mask: {scrubbed}"
    );
    assert_eq!(
        scrubbed, "let secret = \"[REDACTED:api_key_kv]\";",
        "the value masks while the declaration syntax survives"
    );
}

#[test]
fn a_placeholder_marked_quoted_value_passes_verbatim() {
    let line = "let api_key = \"yourexamplekey0123456789\";";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "no substitution on a placeholder-marked literal: {scrubbed}"
    );
    assert_eq!(
        scrubbed, line,
        "a quoted value that names itself a placeholder is documentation, not a secret"
    );
}

#[test]
fn the_declaration_arm_spans_exactly_48_characters_of_annotation() {
    let mut annotation = String::from("aaaaaa");
    for pair in ["bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj", "kk"] {
        annotation.push_str("::");
        annotation.push_str(pair);
    }
    let spacing = format!(": {annotation}");
    assert_eq!(
        spacing.chars().count(),
        48,
        "the fixture must sit exactly at the declaration arm's annotation tolerance"
    );
    let line = format!("let token{spacing}= \"J8kL2mN4pQ6rS8tUv1Wxyz\";");
    let mut scrubbed = line.clone();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 1,
        "a dense quoted value behind an annotation the arm still spans masks in place: {scrubbed}"
    );
    assert_eq!(
        scrubbed,
        format!("let token{spacing}= \"[REDACTED:api_key_kv]\";"),
        "at exactly 48 characters of annotation the declaration arm still reaches the value"
    );
}

#[test]
fn an_annotation_past_the_tolerance_matches_no_kv_arm() {
    let line = "let token: std::collections::HashMap<String, Vec<Option<u8>>> = \"J8kL2mN4pQ6rS8tUv1Wxyz\";";
    let mut scrubbed = line.to_string();
    let count = SecretPatternSet::default_common().scrub(&mut scrubbed);
    assert_eq!(
        count, 0,
        "past the declaration arm's annotation tolerance no key-value arm fires — the documented cliff: {scrubbed}"
    );
    assert_eq!(
        scrubbed, line,
        "the line passes through; a value behind an over-long annotation leans on the entropy lane alone"
    );
}

.PHONY: check test clippy fmt docs ci lint examples e2e e2e-providers e2e-ollama check-default redaction-minimal

ci: fmt check check-default clippy test docs examples redaction-minimal

check:
	cargo check --all-features

check-default:
	cargo clippy --all-targets -- -D warnings

test:
	cargo test
	cargo test --all-features
	cargo test --doc --all-features

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

fmt:
	cargo fmt --all -- --check

lint:
	cargo fmt --all

docs:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

examples:
	cargo build --examples --all-features

define PROBE_TEXT
fn main() {
    let patterns = loopctl::middleware::redaction::SecretPatternSet::default_common();
    let token = "abcdef12345678901234";
    assert!(token.len() >= 8, "the probe token must be at least 8 bytes for the window scan");
    let mut text = format!("Authorization: Bearer {token}");
    let count = patterns.scrub(&mut text);
    assert!(count > 0, "the curated bearer pattern must compile and redact");
    assert!(text.contains("[REDACTED:"), "the bearer header must be scrubbed, got: {text}");
    // The window scan slices bytes, so keep the probe token ASCII.
    for i in 0..=token.len() - 8 {
        assert!(!text.contains(&token[i..i + 8]), "token material survived at offset {i}: {} in {text}", &token[i..i + 8]);
    }
    let aws_key = "AKIAIOSFODNN7EXAMPLE";
    let mut aws_text = format!("aws_access_key_id = {aws_key}");
    assert!(patterns.scrub(&mut aws_text) > 0, "the curated AWS access-key pattern must compile and redact");
    assert!(!aws_text.contains(aws_key), "the AWS access key survived: {aws_text}");
    let entropy_token = "wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY";
    let mut plain = entropy_token.to_string();
    assert!(patterns.scrub(&mut plain) > 0, "the entropy heuristic must redact a high-entropy token");
    assert!(!plain.contains(entropy_token), "the high-entropy token survived: {plain}");
}
endef

# The probe manifest embeds $(CURDIR) inside a quoted TOML basic string, so a
# checkout path containing spaces stays valid; a quote or backslash in the
# path would still break the manifest. The gate needs a POSIX
# environment (make, mktemp, trap, sed) and the cargo registry — a cold
# machine downloads dependencies on the first run; native Windows shells
# are not supported.
redaction-minimal: export PROBE_MAIN = $(PROBE_TEXT)
redaction-minimal:
	@tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT INT TERM; \
	edition=$$(sed -n 's/^edition = "\([^"]*\)".*/\1/p' Cargo.toml); \
	rust_version=$$(sed -n 's/^rust-version = "\([^"]*\)".*/\1/p' Cargo.toml); \
	[ -n "$$edition" ] && [ -n "$$rust_version" ] || { echo "redaction-minimal: cannot derive edition/rust-version from Cargo.toml" >&2; exit 1; }; \
	mkdir -p "$$tmp/src"; \
	printf '[package]\nname = "redaction-minimal-probe"\nversion = "0.0.0"\nedition = "%s"\nrust-version = "%s"\npublish = false\n\n[dependencies]\nloopctl = { path = "%s", features = ["redaction"] }\n\n[workspace]\n' "$$edition" "$$rust_version" "$(CURDIR)" > "$$tmp/Cargo.toml"; \
	printf '%s' "$$PROBE_MAIN" > "$$tmp/src/main.rs"; \
	CARGO_TARGET_DIR="$(CURDIR)/target/redaction-minimal" cargo run --quiet --manifest-path "$$tmp/Cargo.toml"

e2e: e2e-providers e2e-ollama

e2e-providers:
	LOOPCTL_E2E=1 cargo test --features ollama,openai,anthropic,gemini,grok,deepseek,zai --test provider_e2e -- --nocapture --test-threads=1

e2e-ollama:
	@test -n "$(OLLAMA_MODEL)" || { echo "ERROR: set OLLAMA_MODEL (e.g. make e2e-ollama OLLAMA_MODEL=qwen2.5:7b)"; exit 1; }
	LOOPCTL_E2E=1 cargo test --features ollama,grammar --test constrained_decode -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test examples_e2e -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test provider_survival -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test structured_output -- --nocapture

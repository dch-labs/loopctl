.PHONY: check test clippy fmt docs ci lint examples e2e e2e-providers e2e-ollama check-default redaction-minimal search_tools-minimal digest-canonical derive-consumer darwin-clippy windows-check vector-check vector-e2e

ci: fmt check check-default clippy test docs examples redaction-minimal search_tools-minimal digest-canonical derive-consumer darwin-clippy windows-check vector-check

check:
	cargo check --all-features

check-default:
	cargo clippy --all-targets -- -D warnings

test:
	cargo test
	cargo test --all-features
	cargo test --doc --all-features
	cargo test -p loopctl-derive
	cargo test -p loopctl-sqlite
	cargo test -p loopctl-hnsw

clippy:
	cargo clippy --all-targets --all-features -- -D warnings
	cargo clippy --all-targets -p loopctl-sqlite -- -D warnings
	cargo clippy --all-targets -p loopctl-hnsw -- -D warnings

darwin-clippy:
	cargo clippy --target aarch64-apple-darwin --features fs_tools -- -D warnings

windows-check:
	cargo check --target x86_64-pc-windows-msvc --features shell_tools,fs_tools,search_tools

fmt:
	cargo fmt --all -- --check

lint:
	cargo fmt --all

docs:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p loopctl-sqlite
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p loopctl-hnsw

examples:
	cargo build --examples --all-features
	cargo build -p loopctl-sqlite --examples
	cargo build -p loopctl-hnsw --examples

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

define DIGEST_PROBE_TEXT
fn main() {
    let first = loopctl::tool::permission::GateDecision::args_digest(&serde_json::json!({
        "path": "/tmp/x",
        "limit": 10
    }));
    let second = loopctl::tool::permission::GateDecision::args_digest(&serde_json::json!({
        "limit": 10,
        "path": "/tmp/x"
    }));
    assert_eq!(
        first, "f1ed61cd8ca5e26b",
        "the gate digest must stay canonical under a preserve_order serde_json backend, got {first}"
    );
    assert_eq!(
        first, second,
        "key order must not move the gate digest under a preserve_order backend"
    );
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

# The search-tools feature-graph probe: a consumer enabling only
# search_tools must get the same regex semantics every other enabling
# in this crate configures — non-ASCII case folding and Unicode-aware
# Perl classes. In-crate tests cannot see this (dev-dependencies unify
# regex features into the test build), so the probe builds loopctl as
# a dependency with search_tools alone. Manifest-shape and environment
# constraints mirror redaction-minimal.
define SEARCH_PROBE_TEXT
fn main() {
    let folded = loopctl::tool::builtin::search::content::compile_pattern("\u{0439}", true)
        .expect("the pattern must compile under search_tools alone");
    assert!(folded.is_match("\u{0419}"), "case-insensitive matching must fold non-ASCII letters under search_tools alone");
    let words = loopctl::tool::builtin::search::content::compile_pattern(r"\w+", false)
        .expect("the class pattern must compile under search_tools alone");
    assert!(words.is_match("\u{0441}\u{043b}\u{043e}\u{0432}\u{043e}"), "Perl classes must stay Unicode-aware, not ASCII-only, under search_tools alone");
}
endef

search_tools-minimal: export SEARCH_PROBE_MAIN = $(SEARCH_PROBE_TEXT)
search_tools-minimal:
	@tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT INT TERM; \
	edition=$$(sed -n 's/^edition = "\([^"]*\)".*/\1/p' Cargo.toml); \
	rust_version=$$(sed -n 's/^rust-version = "\([^"]*\)".*/\1/p' Cargo.toml); \
	[ -n "$$edition" ] && [ -n "$$rust_version" ] || { echo "search_tools-minimal: cannot derive edition/rust-version from Cargo.toml" >&2; exit 1; }; \
	mkdir -p "$$tmp/src"; \
	printf '[package]\nname = "search-tools-minimal-probe"\nversion = "0.0.0"\nedition = "%s"\nrust-version = "%s"\npublish = false\n\n[dependencies]\nloopctl = { path = "%s", features = ["search_tools"] }\n\n[workspace]\n' "$$edition" "$$rust_version" "$(CURDIR)" > "$$tmp/Cargo.toml"; \
	printf '%s' "$$SEARCH_PROBE_MAIN" > "$$tmp/src/main.rs"; \
	CARGO_TARGET_DIR="$(CURDIR)/target/search-tools-minimal" cargo run --quiet --manifest-path "$$tmp/Cargo.toml"

digest-canonical: export DIGEST_PROBE_MAIN = $(DIGEST_PROBE_TEXT)
digest-canonical:
	@tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT INT TERM; \
	edition=$$(sed -n 's/^edition = "\([^"]*\)".*/\1/p' Cargo.toml); \
	rust_version=$$(sed -n 's/^rust-version = "\([^"]*\)".*/\1/p' Cargo.toml); \
	[ -n "$$edition" ] && [ -n "$$rust_version" ] || { echo "digest-canonical: cannot derive edition/rust-version from Cargo.toml" >&2; exit 1; }; \
	mkdir -p "$$tmp/src"; \
	printf '[package]\nname = "digest-canonical-probe"\nversion = "0.0.0"\nedition = "%s"\nrust-version = "%s"\npublish = false\n\n[dependencies]\nloopctl = { path = "%s" }\nserde_json = { version = "1", features = ["preserve_order"] }\n\n[workspace]\n' "$$edition" "$$rust_version" "$(CURDIR)" > "$$tmp/Cargo.toml"; \
	printf '%s' "$$DIGEST_PROBE_MAIN" > "$$tmp/src/main.rs"; \
	CARGO_TARGET_DIR="$(CURDIR)/target/digest-canonical" cargo run --quiet --manifest-path "$$tmp/Cargo.toml"

e2e: e2e-providers e2e-ollama

e2e-providers:
	LOOPCTL_E2E=1 cargo test --features ollama,openai,anthropic,gemini,grok,deepseek,zai --test provider_e2e -- --nocapture --test-threads=1

e2e-ollama:
	@test -n "$(OLLAMA_MODEL)" || { echo "ERROR: set OLLAMA_MODEL (e.g. make e2e-ollama OLLAMA_MODEL=qwen2.5:7b)"; exit 1; }
	LOOPCTL_E2E=1 cargo test --features ollama,grammar --test constrained_decode -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test examples_e2e -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test provider_survival -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test structured_output -- --nocapture

derive-consumer:
	RUSTFLAGS="-D warnings" cargo build --manifest-path derive/tests/consumer/Cargo.toml --quiet

vector-check:
	cargo fmt --all --check --manifest-path vector/Cargo.toml
	cargo clippy --manifest-path vector/Cargo.toml --all-targets -- -D warnings
	cargo clippy --manifest-path vector/Cargo.toml --no-default-features --all-targets -- -D warnings
	cargo clippy --manifest-path vector/Cargo.toml --no-default-features --features qdrant --all-targets -- -D warnings
	cargo clippy --manifest-path vector/Cargo.toml --no-default-features --features pgvector --all-targets -- -D warnings
	cargo clippy --manifest-path vector/Cargo.toml --all-targets --all-features -- -D warnings
	cargo check --manifest-path vector/Cargo.toml --all-features
	RUSTDOCFLAGS="-D warnings" cargo doc --manifest-path vector/Cargo.toml --no-deps --all-features
	cargo test --manifest-path vector/Cargo.toml --lib --all-features
	cargo test --manifest-path vector/Cargo.toml --doc --all-features

vector-e2e:
	LOOPCTL_VECTOR_E2E=1 cargo test --manifest-path vector/Cargo.toml --features qdrant,pgvector,testing --test qdrant --test pgvector

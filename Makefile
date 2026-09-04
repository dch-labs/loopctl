.PHONY: check test clippy fmt docs ci lint examples e2e e2e-providers e2e-ollama check-default redaction-minimal

CRATE_EDITION := $(shell sed -n 's/^edition = "\(.*\)"/\1/p' Cargo.toml)
CRATE_RUST_VERSION := $(shell sed -n 's/^rust-version = "\(.*\)"/\1/p' Cargo.toml)

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

redaction-minimal:
	@tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT INT TERM; \
	[ -n "$(CRATE_EDITION)" ] && [ -n "$(CRATE_RUST_VERSION)" ] || { echo "redaction-minimal: cannot derive edition/rust-version from Cargo.toml" >&2; exit 1; }; \
	mkdir -p "$$tmp/src"; \
	printf '[package]\nname = "redaction-minimal-probe"\nversion = "0.0.0"\nedition = "$(CRATE_EDITION)"\nrust-version = "$(CRATE_RUST_VERSION)"\npublish = false\n\n[dependencies]\nloopctl = { path = "$(CURDIR)", features = ["redaction"] }\n\n[workspace]\n' > "$$tmp/Cargo.toml"; \
	printf 'fn main() {\n    let token = "abcdef12345678901234";\n    assert!(token.len() >= 8, "the probe token must be at least 8 bytes for the window scan");\n    let mut text = format!("Authorization: Bearer {token}");\n    let count = loopctl::middleware::redaction::SecretPatternSet::default_common().scrub(&mut text);\n    assert!(count > 0, "the curated bearer pattern must compile and redact");\n    assert!(text.contains("[REDACTED:"), "the bearer header must be scrubbed, got: {text}");\n    for i in 0..=token.len() - 8 {\n        assert!(!text.contains(&token[i..i + 8]), "token material survived: {} in {text}", &token[i..i + 8]);\n    }\n}\n' > "$$tmp/src/main.rs"; \
	CARGO_TARGET_DIR="$(CURDIR)/target" cargo run --quiet --manifest-path "$$tmp/Cargo.toml"

e2e: e2e-providers e2e-ollama

e2e-providers:
	LOOPCTL_E2E=1 cargo test --features ollama,openai,anthropic,gemini,grok,deepseek,zai --test provider_e2e -- --nocapture --test-threads=1

e2e-ollama:
	@test -n "$(OLLAMA_MODEL)" || { echo "ERROR: set OLLAMA_MODEL (e.g. make e2e-ollama OLLAMA_MODEL=qwen2.5:7b)"; exit 1; }
	LOOPCTL_E2E=1 cargo test --features ollama,grammar --test constrained_decode -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test examples_e2e -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test provider_survival -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test structured_output -- --nocapture

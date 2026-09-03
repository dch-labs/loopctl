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

redaction-minimal:
	@tmp=$$(mktemp -d); \
	mkdir -p $$tmp/src; \
	printf '[package]\nname = "redaction-minimal-probe"\nversion = "0.0.0"\nedition = "2024"\nrust-version = "1.98"\npublish = false\n\n[dependencies]\nloopctl = { path = "$(CURDIR)", features = ["redaction"] }\n\n[workspace]\n' > $$tmp/Cargo.toml; \
	printf 'fn main() {\n    let token = "abcdef12345678901234";\n    let mut text = format!("Authorization: Bearer {token}");\n    let count = loopctl::middleware::redaction::SecretPatternSet::default_common().scrub(&mut text);\n    assert!(count > 0, "the curated bearer pattern must compile and redact");\n    assert!(text.contains("[REDACTED:"), "the bearer header must be scrubbed, got: {text}");\n    for i in 0..=token.len() - 8 {\n        assert!(!text.contains(&token[i..i + 8]), "token material survived: {} in {text}", &token[i..i + 8]);\n    }\n}\n' > $$tmp/src/main.rs; \
	CARGO_TARGET_DIR=$(CURDIR)/target cargo run --quiet --manifest-path $$tmp/Cargo.toml; \
	status=$$?; \
	rm -rf $$tmp; \
	exit $$status

e2e: e2e-providers e2e-ollama

e2e-providers:
	LOOPCTL_E2E=1 cargo test --features ollama,openai,anthropic,gemini,grok,deepseek,zai --test provider_e2e -- --nocapture --test-threads=1

e2e-ollama:
	@test -n "$(OLLAMA_MODEL)" || { echo "ERROR: set OLLAMA_MODEL (e.g. make e2e-ollama OLLAMA_MODEL=qwen2.5:7b)"; exit 1; }
	LOOPCTL_E2E=1 cargo test --features ollama,grammar --test constrained_decode -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test examples_e2e -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test provider_survival -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test structured_output -- --nocapture

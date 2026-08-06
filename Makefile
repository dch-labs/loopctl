.PHONY: check test clippy fmt docs ci lint examples e2e e2e-providers e2e-ollama check-default

ci: fmt check check-default clippy test docs examples

check:
	cargo check --all-features

check-default:
	cargo clippy --all-targets -- -D warnings

test:
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

e2e: e2e-providers e2e-ollama

e2e-providers:
	LOOPCTL_E2E=1 cargo test --features ollama,openai,anthropic,gemini,grok,deepseek,zai --test provider_e2e -- --nocapture --test-threads=1

e2e-ollama:
	@test -n "$(OLLAMA_MODEL)" || { echo "ERROR: set OLLAMA_MODEL (e.g. make e2e-ollama OLLAMA_MODEL=qwen2.5:7b)"; exit 1; }
	LOOPCTL_E2E=1 cargo test --features ollama,grammar --test constrained_decode -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test examples_e2e -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test provider_survival -- --nocapture
	LOOPCTL_E2E=1 cargo test --features ollama --test structured_output -- --nocapture

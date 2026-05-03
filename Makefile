.PHONY: check test clippy fmt docs ci

ci: check test clippy fmt docs

check:
	cargo check --all-features

test:
	cargo test --all-features

clippy:
	cargo clippy --all-features -- -D warnings

fmt:
	cargo fmt --all -- --check

docs:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

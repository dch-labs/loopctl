.PHONY: check test clippy fmt docs ci lint examples

ci: fmt check clippy test docs examples

check:
	cargo check --all-features

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

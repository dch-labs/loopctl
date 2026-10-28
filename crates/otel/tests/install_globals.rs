//! `install`'s all-or-nothing global contract, alone in its process.
//!
//! `tracing` globals are set-once per process, so the only honest way
//! to pin `install`'s failure path is a binary where this test is the
//! only global-subscriber owner: the first `install` claims the
//! globals, the second must refuse *without having touched anything* —
//! the first installation's providers stay alive and flushable through
//! their guard, rather than having been silently replaced mid-call by
//! a provider nobody owns.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::clone_on_ref_ptr,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::panic
)]

use loopctl_otel::{OtelConfig, OtelInstallError};

/// A dead-loopback endpoint: the exporter builds fine, nothing listens.
const DEAD_ENDPOINT: &str = "http://127.0.0.1:43191";

#[test]
fn a_refused_second_install_leaves_the_first_running() {
    let config = OtelConfig {
        endpoint: DEAD_ENDPOINT.to_string(),
        service_name: "install-contract".to_string(),
        ..OtelConfig::default()
    };
    let first = loopctl_otel::install(&config).expect("the first install claims the globals");

    let second = loopctl_otel::install(&config);
    assert!(
        matches!(second, Err(OtelInstallError::AlreadyInstalled)),
        "the second install must refuse explicitly, got {second:?}"
    );
    drop(second);

    // The refusal must have been harmless: the first installation's
    // tracer provider — the one its guard owns — still flushes. If the
    // rejected call had set the global tracer provider before failing,
    // the first install's batch worker would have been orphaned and
    // this flush would land on a provider its guard no longer holds.
    first
        .force_flush()
        .expect("the first install still flushes");

    // Teardown of the surviving installation completes cleanly.
    drop(first);
}

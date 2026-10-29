//! Pins the guard's drop path as the real shutdown entry point.
//!
//! A binary of its own because the process-wide meter slot and the
//! tracing global are each set once per process — the `install_globals`
//! pattern: this test installs, exports, drops, and asserts, with no
//! other test in the process touching either global.

#![allow(clippy::expect_used, clippy::missing_panics_doc, clippy::unwrap_used)]

use loopctl_otel::OtelConfig;
use loopctl_otel::OtelInstallError;
use loopctl_otel::OtelMetricsMode;
use loopctl_otel::export_metrics;

#[test]
fn dropping_the_guard_ends_the_pipeline_s_exports() {
    let config = OtelConfig {
        metrics: Some(OtelMetricsMode::Pipeline),
        endpoint: "http://127.0.0.1:1".to_string(),
        ..OtelConfig::default()
    };
    let registry = loopctl::metrics::MetricsRegistry::default();
    let guard = loopctl_otel::install(&config).expect("this process has no prior install");
    assert!(
        export_metrics(None, &registry).is_ok(),
        "the installed pipeline accepts exports"
    );
    drop(guard);
    assert!(
        matches!(
            export_metrics(None, &registry),
            Err(OtelInstallError::PipelineShutDown)
        ),
        "dropping the guard marks the pipeline it owns shut down"
    );
}

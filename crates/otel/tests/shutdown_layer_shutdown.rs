//! Pins the composable layer's own shutdown as the real entry point.
//!
//! A binary of its own because the process-wide meter slot is set once
//! per process — the `install_globals` pattern: this test builds the
//! layer, exports, shuts the layer down, and asserts.

#![allow(clippy::expect_used, clippy::missing_panics_doc, clippy::unwrap_used)]

use loopctl_otel::OtelConfig;
use loopctl_otel::OtelInstallError;
use loopctl_otel::OtelMetricsMode;
use loopctl_otel::export_metrics;

#[test]
fn the_layer_s_shutdown_ends_the_pipeline_s_exports() {
    let config = OtelConfig {
        metrics: Some(OtelMetricsMode::Pipeline),
        endpoint: "http://127.0.0.1:1".to_string(),
        ..OtelConfig::default()
    };
    let registry = loopctl::metrics::MetricsRegistry::default();
    let layer = loopctl_otel::otel_layer(&config).expect("the layer builds");
    assert!(
        export_metrics(None, &registry).is_ok(),
        "the layer's install accepts exports"
    );
    // The flush this shutdown runs targets an unreachable endpoint and
    // fails — that failure is the flush's own concern; the mark this
    // test pins happens before the flush either way.
    let _flush_result = layer.shutdown();
    assert!(
        matches!(
            export_metrics(None, &registry),
            Err(OtelInstallError::PipelineShutDown)
        ),
        "the layer's shutdown marks the pipeline it owns shut down"
    );
}

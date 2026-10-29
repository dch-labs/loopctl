//! Pins the set-once meter race through the real constructors.
//!
//! A binary of its own because the process-wide meter slot is set once
//! per process — the `install_globals` pattern: two `otel_layer`
//! pipelines are built the way a host builds them, the loser is torn
//! down first, and only the winner's teardown may end the exports.

#![allow(clippy::expect_used, clippy::missing_panics_doc, clippy::unwrap_used)]

use loopctl_otel::OtelConfig;
use loopctl_otel::OtelInstallError;
use loopctl_otel::OtelMetricsMode;
use loopctl_otel::export_metrics;

#[test]
fn the_second_pipeline_s_shutdown_cannot_end_the_first_s_exports() {
    let config = || OtelConfig {
        metrics: Some(OtelMetricsMode::Pipeline),
        endpoint: "http://127.0.0.1:1".to_string(),
        ..OtelConfig::default()
    };
    let registry = loopctl::metrics::MetricsRegistry::default();
    let winner = loopctl_otel::otel_layer(&config()).expect("the first pipeline builds");
    let loser = loopctl_otel::otel_layer(&config()).expect("the second pipeline builds");
    assert!(
        export_metrics(None, &registry).is_ok(),
        "the first install owns the process-wide slot and exports"
    );
    // The losing pipeline's shutdown marks only its own orphaned handle;
    // its flush targets an unreachable endpoint, whose failure is the
    // flush's own concern.
    let _flush_result = loser.shutdown();
    assert!(
        export_metrics(None, &registry).is_ok(),
        "the winner's exports survive the loser's teardown — a pipeline \
         whose meter lost the set-once race cannot end them"
    );
    let _flush_result = winner.shutdown();
    assert!(
        matches!(
            export_metrics(None, &registry),
            Err(OtelInstallError::PipelineShutDown)
        ),
        "only the owning pipeline's shutdown ends the exports"
    );
}

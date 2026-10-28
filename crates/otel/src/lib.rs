//! # loopctl-otel
//!
//! `OpenTelemetry` (`OTLP`) export for [loopctl](https://loopctl.rs) agent
//! loops: the engine span hierarchy (`agent.run`/`agent.turn`/`agent.tool`,
//! fields already emitted in `gen_ai.*` shape) renders as a proper trace in
//! Jaeger, Tempo, or Honeycomb — and the run's metrics rollup rides along.
//!
//! ## What this crate does
//!
//! * **Bridges, does not translate.** loopctl's emitters carry spec-shaped
//!   attribute names ([`mapping::otel_attribute`] is the identity proof);
//!   this crate only renames spans per the `GenAI` conventions
//!   (`invoke_agent`, `execute_tool <tool>`), derives span status from the
//!   recorded `outcome`/`error.type` fields, adds the coarse
//!   [`mapping::map_system`] provider, and transports everything over `OTLP`.
//! * **Installs in one call** ([`install`]) for hosts without their own
//!   subscriber, or **composes** ([`otel_layer`]) into hosts that have one.
//! * **Carries metrics two ways**: the closing `invoke_agent` span is
//!   stamped with the [`AgentMetrics`] rollup (plus, in
//!   [`OtelMetricsMode::SpanAttributes`], the subsystem registry), and
//!   [`export_metrics`] replays a [`RegistrySnapshot`] into an `OTLP` metrics
//!   pipeline for Prometheus/Grafana stacks.
//!
//! ## Quick start
//!
//! ```no_run
//! use loopctl_otel::{OtelConfig, OtelMetricsMode};
//!
//! let guard = loopctl_otel::install(&OtelConfig {
//!     service_name: "my-agent".into(),
//!     metrics: Some(OtelMetricsMode::SpanAttributes),
//!     ..OtelConfig::default()
//! })
//! .expect("`OTLP` export installed");
//! // …run agent loops; spans flow to the collector…
//! drop(guard); // flushes on the way out
//! ```
//!
//! [`AgentMetrics`]: loopctl::metrics::AgentMetrics
//! [`RegistrySnapshot`]: loopctl::metrics::RegistrySnapshot

#![warn(missing_docs, missing_debug_implementations)]

pub mod layer;
pub mod mapping;
pub mod metrics;

use std::sync::Arc;
use std::time::Duration;

use loopctl::metrics::{AgentMetrics, MetricsObserver, MetricsRegistry};
use opentelemetry::metrics::MeterProvider as _;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider};
use tracing_subscriber::Registry;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::layer::SubscriberExt;

pub use layer::{OtelTracingLayer, SpecExporter};

/// How run/subsystem metrics reach the collector.
///
/// Chosen per [`OtelConfig::metrics`]; leaving the option `None` omits
/// metrics entirely and exports traces only. The two modes compose:
/// `Pipeline` keeps stamping while it also runs the meter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OtelMetricsMode {
    /// Stamp the metrics onto spans instead of a metric pipeline.
    ///
    /// The closing `invoke_agent` span carries the
    /// [`AgentMetrics`] rollup and, from
    /// the configured
    /// [`MetricsRegistry`], the
    /// subsystem snapshot as span attributes. One pipeline, zero extra
    /// threads.
    SpanAttributes,

    /// Additionally run an `OTLP` metrics pipeline: [`export_metrics`]
    /// exports registry counters/gauges/histograms on demand (a ticker or
    /// at run end). The run rollup stays attributes-only.
    #[default]
    Pipeline,
}

/// Installation knobs.
///
/// Every default aims at the local-collector development path: `OTLP`
/// over HTTP to the standard collector port, full sampling, traces
/// only. Stamping and metrics opt in by handing over the session's
/// [`loopctl::metrics::MetricsObserver`] and
/// [`loopctl::metrics::MetricsRegistry`].
#[derive(Clone)]
pub struct OtelConfig {
    /// `OTLP` endpoint. Default `http://localhost:4318` (`OTLP`/HTTP, the
    /// Jaeger/Tempo/Honeycomb collector default).
    pub endpoint: String,

    /// gRPC transport; requires the crate's `grpc` feature. Default false.
    ///
    /// Requesting gRPC without the feature is an explicit
    /// [`OtelInstallError::Exporter`] failure, never a silent fallback
    /// to HTTP.
    pub use_grpc: bool,

    /// `OTel` resource `service.name`. Default `"loopctl"`.
    ///
    /// The identity collectors group traces under; hosts running several
    /// distinct agent services override it per install.
    pub service_name: String,

    /// Root-span sampler ratio, `0.0..=1.0` (child spans follow their
    /// parent). Default `1.0`.
    pub sample_ratio: f64,

    /// Metrics handling; `None` exports traces only.
    ///
    /// [`OtelMetricsMode::Pipeline`] additionally builds an
    /// `SdkMeterProvider` and installs the process-wide meter that
    /// [`export_metrics`] replays registry snapshots through.
    pub metrics: Option<OtelMetricsMode>,

    /// Session rollup source for run-span stamping. Provide the session
    /// [`MetricsObserver`] to enable it; `None` stamps nothing.
    pub observer: Option<Arc<MetricsObserver>>,

    /// Subsystem store for `SpanAttributes`-mode stamping and
    /// [`export_metrics`]; the same `Arc` handed to the metrics layer.
    pub registry: Option<Arc<MetricsRegistry>>,
}

impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:4318".to_string(),
            use_grpc: false,
            service_name: "loopctl".to_string(),
            sample_ratio: 1.0,
            metrics: None,
            observer: None,
            registry: None,
        }
    }
}

impl std::fmt::Debug for OtelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtelConfig")
            .field("endpoint", &self.endpoint)
            .field("use_grpc", &self.use_grpc)
            .field("service_name", &self.service_name)
            .field("sample_ratio", &self.sample_ratio)
            .field("metrics", &self.metrics)
            .field(
                "observer",
                &self.observer.as_ref().map(|_| "MetricsObserver"),
            )
            .field(
                "registry",
                &self.registry.as_ref().map(|_| "MetricsRegistry"),
            )
            .finish()
    }
}

/// Why installation failed.
///
/// Construction is once-per-process territory — `OTel` globals and the
/// global subscriber are both set-once — so every variant surfaces at
/// [`install`] or [`otel_layer`], never mid-run.
#[derive(Debug, thiserror::Error)]
pub enum OtelInstallError {
    /// A global tracer provider / default subscriber is already installed;
    /// `OTel` globals are once-per-process.
    #[error("an OTel provider or global subscriber is already installed")]
    AlreadyInstalled,

    /// The `OTLP` exporter could not be constructed (bad endpoint, missing
    /// transport feature, …).
    #[error("OTLP exporter construction failed: {0}")]
    Exporter(String),

    /// A flush/shutdown step failed at teardown; spans may have been lost.
    ///
    /// Raised by [`ShutdownGuard::force_flush`] and
    /// [`OtelTracingLayer::shutdown`]; the providers are shut down
    /// afterwards either way, so the loss is bounded to what the failed
    /// flush still held.
    #[error("OTel teardown failed: {0}")]
    Flush(String),

    /// [`export_metrics`] was called with no metrics pipeline installed.
    ///
    /// Either [`OtelConfig::metrics`] was `None` or no install ran in
    /// this process — the meter handle is set-once, first install wins.
    #[error("no OTLP metrics pipeline is installed (set OtelConfig::metrics)")]
    NoMetricsPipeline,
}

/// Flushes pending spans (and metrics) on drop.
///
/// Returned by [`install`]; dropping performs a best-effort shutdown
/// (failures print to stderr, since `Drop` cannot propagate them).
/// [`ShutdownGuard::force_flush`] is the checked flush for hosts that
/// want errors before the last batch leaves the process.
#[derive(Debug)]
pub struct ShutdownGuard {
    /// The tracer provider this guard owns; `None` only for guards
    /// built without a trace pipeline.
    ///
    /// Drop shuts it down (flushing the batch worker first), which is
    /// the only path that ever stops the exporter threads `install`
    /// started.
    tracer_provider: Option<SdkTracerProvider>,

    /// The meter provider, present only in
    /// [`OtelMetricsMode::Pipeline`]; shut down after the tracer so a
    /// final metric flush lands before the process moves on.
    meter_provider: Option<SdkMeterProvider>,
}

impl ShutdownGuard {
    /// Force an immediate flush without tearing down (cheap enough for a
    /// periodic ticker).
    ///
    /// # Errors
    ///
    /// [`OtelInstallError::Flush`] when a provider's flush fails.
    pub fn force_flush(&self) -> Result<(), OtelInstallError> {
        if let Some(provider) = &self.tracer_provider {
            provider
                .force_flush()
                .map_err(|err| OtelInstallError::Flush(err.to_string()))?;
        }
        if let Some(meter) = &self.meter_provider {
            meter
                .force_flush()
                .map_err(|err| OtelInstallError::Flush(err.to_string()))?;
        }
        Ok(())
    }
}

impl Drop for ShutdownGuard {
    fn drop(&mut self) {
        if let Some(provider) = &self.tracer_provider
            && let Err(err) = provider.shutdown()
        {
            eprintln!("loopctl-otel: tracer shutdown failed: {err}");
        }
        if let Some(meter) = &self.meter_provider
            && let Err(err) = meter.shutdown()
        {
            eprintln!("loopctl-otel: meter shutdown failed: {err}");
        }
    }
}

/// Install `OTLP` export into the **global default subscriber**: registry +
/// `OTel` layer + a `loopctl=debug` target filter (spans *and* engine metric
/// events must pass the filter or nothing exports).
///
/// Convenience for hosts with no subscriber of their own; hosts that own
/// one compose [`otel_layer`] instead. Returns a [`ShutdownGuard`] that
/// flushes on drop.
///
/// Installation is all-or-nothing: the single fallible step — claiming
/// the global subscriber — runs **before** any `OTel` global is
/// touched, so a rejected second install leaves the first
/// installation's providers and guard exactly as they were. Errors if
/// a global subscriber is already installed — `OTel` globals are
/// once-per-process.
///
/// # Errors
///
/// [`OtelInstallError::AlreadyInstalled`] when the global subscriber is
/// taken; [`OtelInstallError::Exporter`] when the `OTLP` exporter cannot be built.
pub fn install(config: &OtelConfig) -> Result<ShutdownGuard, OtelInstallError> {
    let stamper = stamper_for(config);
    let (tracer_provider, meter_provider) = build_providers(config)?;
    let layer = compose_layer(&tracer_provider, stamper);
    let subscriber = Registry::default()
        .with(layer)
        .with(Targets::new().with_target("loopctl", LevelFilter::DEBUG));
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|_| OtelInstallError::AlreadyInstalled)?;
    opentelemetry::global::set_tracer_provider(tracer_provider.clone());
    if let Some(meter) = &meter_provider {
        opentelemetry::global::set_meter_provider(meter.clone());
        metrics::install_meter(meter.meter("loopctl"));
    }
    Ok(ShutdownGuard {
        tracer_provider: Some(tracer_provider),
        meter_provider,
    })
}

/// Build the `OTel` layer for hosts that own their subscriber.
///
/// Compose it exactly like any other `tracing_subscriber::Layer`
/// (`Registry::default().with(otel_layer(&config)?)`). The layer installs
/// its own provider pair; call [`OtelTracingLayer::shutdown`] for graceful
/// teardown (a [`ShutdownGuard`] is not needed — the providers belong to
/// the layer).
///
/// # Errors
///
/// As [`install`] for the exporter-construction cases.
pub fn otel_layer(config: &OtelConfig) -> Result<OtelTracingLayer, OtelInstallError> {
    let stamper = stamper_for(config);
    let (tracer_provider, meter_provider) = build_providers(config)?;
    if let Some(meter) = &meter_provider {
        metrics::install_meter(meter.meter("loopctl"));
    }
    let composed = OtelTracingLayer::new(
        compose_layer_inner(&tracer_provider),
        tracer_provider,
        meter_provider,
    );
    Ok(match stamper {
        Some(stamper) => composed.with_stamper(stamper),
        None => composed,
    })
}

/// Export the subsystem registry (and optionally the run rollup) to the
/// installed `OTLP` metrics pipeline.
///
/// Call from a ticker or at run end; `Pipeline` mode. Counters export as
/// deltas since the previous call, gauges as last value, histograms
/// reconstructed from the registry's bucket arrays. Returns
/// [`OtelInstallError::NoMetricsPipeline`] when
/// [`OtelConfig::metrics`] was `None`.
///
/// # Errors
///
/// [`OtelInstallError::NoMetricsPipeline`] when nothing is installed.
pub fn export_metrics(
    agent: Option<&AgentMetrics>,
    registry: &MetricsRegistry,
) -> Result<(), OtelInstallError> {
    let meter = metrics::active_meter().ok_or(OtelInstallError::NoMetricsPipeline)?;
    metrics::record_registry(&meter, &registry.snapshot());
    if let Some(agent) = agent {
        for attr in metrics::run_attributes(agent) {
            let name = attr.key.as_str().to_string();
            let value = match attr.value {
                opentelemetry::Value::I64(n) => metrics::wide_i64_to_f64(n),
                opentelemetry::Value::F64(f) => f,
                opentelemetry::Value::Bool(b) => f64::from(b),
                _ => continue,
            };
            meter.with_gauge(&name, value);
        }
    }
    Ok(())
}

/// Build the tracer (+ optional meter) provider pair from config.
///
/// # Errors
///
/// [`OtelInstallError::Exporter`] when either OTLP exporter cannot be
/// constructed.
fn build_providers(
    config: &OtelConfig,
) -> Result<(SdkTracerProvider, Option<SdkMeterProvider>), OtelInstallError> {
    let resource = Resource::builder()
        .with_service_name(config.service_name.clone())
        .build();
    let exporter = build_span_exporter(config)?;
    let sampler = if config.sample_ratio >= 1.0 {
        Sampler::AlwaysOn
    } else {
        Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
            config.sample_ratio.clamp(0.0, 1.0),
        )))
    };
    let tracer_provider = SdkTracerProvider::builder()
        .with_batch_exporter(SpecExporter::new(exporter))
        .with_sampler(sampler)
        .with_resource(resource)
        .build();
    let meter_provider = match config.metrics {
        Some(OtelMetricsMode::Pipeline) => {
            let exporter = build_metric_exporter(config)?;
            let reader = PeriodicReader::builder(exporter).build();
            Some(
                SdkMeterProvider::builder()
                    .with_reader(reader)
                    .with_resource(
                        Resource::builder()
                            .with_service_name(config.service_name.clone())
                            .build(),
                    )
                    .build(),
            )
        }
        Some(OtelMetricsMode::SpanAttributes) | None => None,
    };
    Ok((tracer_provider, meter_provider))
}

/// Construct the span exporter for the configured transport.
///
/// # Errors
///
/// [`OtelInstallError::Exporter`] when the OTLP builder rejects the
/// endpoint or the `grpc` feature is missing while requested.
fn build_span_exporter(
    config: &OtelConfig,
) -> Result<opentelemetry_otlp::SpanExporter, OtelInstallError> {
    use opentelemetry_otlp::WithExportConfig as _;
    let endpoint = config.endpoint.clone();
    #[cfg(feature = "grpc")]
    if config.use_grpc {
        return opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .map_err(|err| OtelInstallError::Exporter(err.to_string()));
    }
    if config.use_grpc && !cfg!(feature = "grpc") {
        return Err(OtelInstallError::Exporter(
            "gRPC transport requires the `grpc` feature".to_string(),
        ));
    }
    opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(10))
        .build()
        .map_err(|err| OtelInstallError::Exporter(err.to_string()))
}

/// Construct the metric exporter for the configured transport — the
/// meter-side twin of [`build_span_exporter`].
///
/// The metric pipeline honors the same endpoint and transport
/// selection the span pipeline does, so a configured collector
/// receives both signal kinds; before this mirroring the metric
/// exporter silently targeted its own default endpoint over HTTP
/// whatever `OtelConfig::endpoint` and `OtelConfig::use_grpc` said.
///
/// # Errors
///
/// [`OtelInstallError::Exporter`] when the OTLP builder rejects the
/// endpoint or the `grpc` feature is missing while requested.
fn build_metric_exporter(
    config: &OtelConfig,
) -> Result<opentelemetry_otlp::MetricExporter, OtelInstallError> {
    use opentelemetry_otlp::WithExportConfig as _;
    let endpoint = config.endpoint.clone();
    #[cfg(feature = "grpc")]
    if config.use_grpc {
        return opentelemetry_otlp::MetricExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .with_timeout(Duration::from_secs(10))
            .build()
            .map_err(|err| OtelInstallError::Exporter(err.to_string()));
    }
    if config.use_grpc && !cfg!(feature = "grpc") {
        return Err(OtelInstallError::Exporter(
            "gRPC transport requires the `grpc` feature".to_string(),
        ));
    }
    opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(10))
        .build()
        .map_err(|err| OtelInstallError::Exporter(err.to_string()))
}

/// The close-time stamp sources, when configured.
///
/// Built whenever an observer **or** a registry is configured: the
/// registry-only shape (a `SpanAttributes` host that stamps subsystem
/// series but tracks no session rollup) stamps exactly its series, so
/// valid series never wait on an observer nobody configured. The
/// registry handle and the `SpanAttributes` flag ride along exactly as
/// configured. The stamper feeds
/// [`OtelTracingLayer::with_stamper`](layer::OtelTracingLayer::with_stamper),
/// which writes the rollup at the run-end record — not the exporter,
/// whose later read of a sustained observer would fold every
/// subsequent run's numbers into this span's stamp.
fn stamper_for(config: &OtelConfig) -> Option<layer::Stamper> {
    if config.observer.is_none() && config.registry.is_none() {
        return None;
    }
    Some(layer::Stamper {
        observer: config.observer.clone(),
        registry: config.registry.clone(),
        span_attributes: matches!(config.metrics, Some(OtelMetricsMode::SpanAttributes)),
    })
}

/// The stock bridge layer: the `tracing-opentelemetry` layer bound to
/// `provider`'s tracer.
fn compose_layer_inner(
    provider: &SdkTracerProvider,
) -> impl tracing_subscriber::Layer<Registry> + Send + Sync {
    tracing_opentelemetry::layer().with_tracer(provider.tracer("loopctl-otel"))
}

/// Bind the stock bridge into the shipped layer shape.
///
/// [`compose_layer_inner`] builds the bridge; this pins it to the
/// provider and attaches the run-end stamper when one is configured —
/// the composition [`install`] serves its global subscriber
/// (`otel_layer` builds the same shape inline for its composable
/// return).
fn compose_layer(
    provider: &SdkTracerProvider,
    stamper: Option<layer::Stamper>,
) -> OtelTracingLayer {
    let composed = OtelTracingLayer::new(compose_layer_inner(provider), provider.clone(), None);
    match stamper {
        Some(stamper) => composed.with_stamper(stamper),
        None => composed,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::missing_panics_doc, clippy::expect_used)]

    use super::*;

    /// A registry-only `SpanAttributes` config still gets its stamper:
    /// stamping series onto run spans must not wait on an observer the
    /// host never configured.
    #[test]
    fn a_registry_only_config_still_builds_a_stamper() {
        let config = OtelConfig {
            metrics: Some(OtelMetricsMode::SpanAttributes),
            registry: Some(Arc::new(MetricsRegistry::default())),
            ..OtelConfig::default()
        };
        let stamper = stamper_for(&config).expect("registry-only configs stamp");
        assert!(stamper.observer.is_none(), "no observer was configured");
        assert!(stamper.span_attributes);

        let bare = stamper_for(&OtelConfig::default());
        assert!(bare.is_none(), "nothing to stamp, no stamper");
    }

    /// The metric pipeline honors the configured endpoint: an endpoint
    /// no HTTP client can route must fail the pipeline build, exactly
    /// as it fails the span exporter build.
    #[test]
    fn a_pipeline_with_an_unroutable_endpoint_fails_to_build() {
        let config = OtelConfig {
            endpoint: "::::not-a-url".to_string(),
            metrics: Some(OtelMetricsMode::Pipeline),
            ..OtelConfig::default()
        };
        assert!(
            build_providers(&config).is_err(),
            "the metric exporter must reject an unroutable endpoint instead of \
             silently building against its own default"
        );
    }
}

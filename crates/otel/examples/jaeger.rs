//! Manual smoke against a real collector — no test automation on purpose.
//!
//! ```text
//! docker run -d --name jaeger -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one:latest
//! cargo run -p loopctl-otel --example jaeger
//! # then open http://localhost:16686 and look for service "loopctl-example"
//! ```
//!
//! Expect one `invoke_agent` trace per run with an `agent.turn` and an
//! `execute_tool echo` child, the `gen_ai.*` attributes, the run
//! rollup (`loopctl.run.*`) stamped on the root span, and the
//! subsystem registry series (`loopctl.tools.calls.*` and kin) riding
//! the same stamp — the example composes a `MetricsLayer` over the
//! registry so the engine's metric events actually reach it.
//!
//! The endpoint is built through [`loopctl_otel::traces_endpoint`] rather than
//! passed bare. OTLP/HTTP delivers each signal at its own path, and the SDK
//! appends that path only for an endpoint it resolves from the environment — a
//! supplied one is used verbatim — so a bare `http://localhost:4318` posts to
//! `/` and the collector answers `404`. Spans leave in a batch, so that failure
//! surfaces only at the closing flush: the least useful moment to learn it.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, core::Loop};
use loopctl::metrics::{MetricsObserver, MetricsRegistry};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use loopctl_otel::SpecExporter;
use loopctl_otel::layer::{OtelTracingLayer, Stamper};

struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }
    fn description(&self) -> &'static str {
        "Echoes a fixed output"
    }
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name().to_string(),
            self.description().to_string(),
            serde_json::json!({"type": "object"}),
        )
    }
    fn call(
        &self,
        _input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        Box::pin(async { Ok(ToolOutput::text("done")) })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig as _;
    use tracing_subscriber::layer::SubscriberExt as _;

    let observer = Arc::new(MetricsObserver::new());
    let registry = Arc::new(MetricsRegistry::default());
    // Composed by hand rather than through `install`: the example's
    // registry only fills if a `MetricsLayer` rides the same subscriber
    // as the OTel layer, and the one-call `install` owns its subscriber
    // internally. The provider is held here so the end-of-main flush can
    // ship the last batch before the process exits.
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(loopctl_otel::traces_endpoint("http://localhost:4318"))
        .with_timeout(std::time::Duration::from_secs(10))
        .build()?;
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(SpecExporter::new(exporter))
        .with_resource(
            opentelemetry_sdk::Resource::builder()
                .with_service_name("loopctl-example")
                .build(),
        )
        .build();
    let otel = OtelTracingLayer::new(
        tracing_opentelemetry::layer().with_tracer(provider.tracer("loopctl-example")),
        provider.clone(),
        None,
    )
    .with_stamper(Stamper {
        observer: Some(Arc::<MetricsObserver>::clone(&observer)),
        registry: Some(Arc::<MetricsRegistry>::clone(&registry)),
        span_attributes: true,
    });
    let subscriber = tracing_subscriber::registry()
        .with(otel)
        .with(loopctl::metrics::MetricsLayer::new(
            Arc::<MetricsRegistry>::clone(&registry),
        ))
        .with(
            tracing_subscriber::filter::Targets::new()
                .with_target("loopctl", tracing::level_filters::LevelFilter::DEBUG),
        );
    tracing::subscriber::set_global_default(subscriber)?;

    let client = Arc::new(MockApiClient::new("claude-test").with_responses(vec![
        MockResponse {
            text: "calling echo".to_string(),
            tool_call: Some(MockToolCall {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            }),
            stop_reason: "tool_use".to_string(),
        },
        MockResponse {
            text: "finished".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        },
    ]));
    let tools = ToolRegistry::new();
    tools.register(EchoTool);
    let mut engine =
        BareLoop::new(client, tools, SessionConfig::default())
            .with_observer(Arc::<MetricsObserver>::clone(&observer));

    engine.run("hello jaeger", &RunConfig::default()).await?;
    // SpanAttributes mode needs no explicit metrics export: the registry
    // snapshot rides the run span's stamp (`loopctl.*` attributes on the
    // `invoke_agent` root). A `Pipeline` config is the one that exports
    // through the meter provider instead.

    provider.force_flush()?;

    println!("exported one trace; check http://localhost:16686");
    Ok(())
}

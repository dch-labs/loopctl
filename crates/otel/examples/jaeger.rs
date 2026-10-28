//! Manual smoke against a real collector — no test automation on purpose.
//!
//! ```text
//! docker run -d --name jaeger -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one:latest
//! cargo run -p loopctl-otel --example jaeger
//! # then open http://localhost:16686 and look for service "loopctl-example"
//! ```
//!
//! Expect one `invoke_agent` trace per run with an `agent.turn` and an
//! `execute_tool echo` child, the `gen_ai.*` attributes, and the run
//! rollup (`loopctl.run.*`) stamped on the root span.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, core::Loop};
use loopctl::metrics::{MetricsObserver, MetricsRegistry};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use loopctl_otel::{OtelConfig, OtelMetricsMode};

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
    let observer = Arc::new(MetricsObserver::new());
    let registry = Arc::new(MetricsRegistry::default());
    let _guard = loopctl_otel::install(&OtelConfig {
        service_name: "loopctl-example".to_string(),
        metrics: Some(OtelMetricsMode::SpanAttributes),
        observer: Some(Arc::<MetricsObserver>::clone(&observer)),
        registry: Some(Arc::<MetricsRegistry>::clone(&registry)),
        ..OtelConfig::default()
    })?;

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

    println!("exported one trace; check http://localhost:16686");
    Ok(())
}

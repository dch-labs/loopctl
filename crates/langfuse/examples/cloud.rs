//! Manual smoke against Langfuse Cloud — no test automation on purpose.
//!
//! ```text
//! export LANGFUSE_PUBLIC_KEY=pk-lf-...
//! export LANGFUSE_SECRET_KEY=sk-lf-...
//! export LANGFUSE_HOST=https://cloud.langfuse.com
//! cargo run -p loopctl-langfuse --example cloud
//! # then open the Langfuse UI and look for the "agent.run" trace
//! ```
//!
//! Expect one trace per run named `agent.run`, tagged `loopctl`, grouped
//! under the session id, with `agent.turn` / `agent.tool` observations
//! and the auto-score set (tokens, turns, latency p90, outcome).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, core::Loop};
use loopctl::metrics::MetricsObserver;
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use loopctl_langfuse::{LangfuseConfig, LangfuseLayer, install};
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;

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

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = LangfuseConfig::from_env()?;
    let observer = Arc::new(MetricsObserver::new());

    // The guard flushes on drop; keep it alive until the run is done.
    let (layer, guard) = install(config)?;
    let source = Arc::<MetricsObserver>::clone(&observer);
    let layer =
        LangfuseLayer::with_metrics_source(layer, Arc::new(move || Some(source.snapshot())));
    let subscriber = Registry::default().with(layer);

    let client = Arc::new(MockApiClient::new("test-model").with_responses(vec![
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
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    let mut engine = BareLoop::new(client, registry, SessionConfig::default())
        .with_observer(Arc::<MetricsObserver>::clone(&observer));

    let result = {
        let _guard = tracing::subscriber::set_default(subscriber);
        engine
            .run("langfuse cloud smoke", &RunConfig::default())
            .await
    };
    println!("run ok: {}", result.is_ok());

    // Drain cooperatively — `flush` is the drop-time wait's twin that
    // also runs on a runtime thread; the following drop returns
    // immediately because nothing is pending anymore.
    let drained = guard.flush(std::time::Duration::from_secs(5)).await;
    drop(guard);
    println!("flushed (drained: {drained}); check the Langfuse UI for the agent.run trace");
    Ok(())
}

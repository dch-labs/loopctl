//! One loop, two runs, one sustained observer — and what each run stamps.
//!
//! The rollup a closing `invoke_agent` span carries is read from the
//! [`MetricsObserver`] at that run's own outcome record, so a host that keeps
//! one observer across a session's runs expects each span to hold its own
//! run's figures. This binary drives the shape that claim is hardest to keep:
//! a run that fails at the manager-reset boundary, after the loop's spans
//! have opened but before any turn of its own. The recorder is the same
//! in-process stand-in the export suite uses, so the assertions read the
//! spans a collector would receive.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::clone_on_ref_ptr,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::panic,
    clippy::too_many_lines
)]

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use loopctl::capabilities::Detectable as _;
use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, core::Loop};
use loopctl::managers::LoopManagers;
use loopctl::metrics::MetricsObserver;
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use loopctl_otel::SpecExporter;
use loopctl_otel::layer::{OtelTracingLayer, Stamper};
use opentelemetry::Value;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{Registry, filter::LevelFilter};

/// Captured spans, appended by the recorder on every export.
static EXPORTED: Mutex<Vec<SpanData>> = Mutex::new(Vec::new());

/// In-process stand-in for the OTLP transport.
#[derive(Debug)]
struct RecordingExporter;

impl SpanExporter for RecordingExporter {
    fn export(&self, batch: Vec<SpanData>) -> impl Future<Output = OTelSdkResult> + Send {
        let mut store = EXPORTED.lock().unwrap();
        store.extend(batch);
        std::future::ready(Ok(()))
    }
}

/// A tool that echoes a fixed successful output.
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

/// One scripted tool-call turn plus the terminal reply that closes the run.
fn tool_turn_then_terminal() -> Vec<MockResponse> {
    vec![
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
    ]
}

/// Read a stamped attribute off an exported span.
fn attr<'a>(span: &'a SpanData, key: &str) -> Option<&'a Value> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| &kv.value)
}

/// Poison the convergence detector the loop holds, the way the engine's own
/// span suite does: panic while the guard is held.
fn poison_detector(engine: &BareLoop<MockApiClient>) {
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = engine
            .managers()
            .detection()
            .convergence_detector()
            .lock()
            .expect("lock before poisoning");
        panic!("poison the convergence detector");
    }));
    assert!(
        poisoned.is_err(),
        "the panic inside the guard is what poisons the lock"
    );
}

#[tokio::test]
async fn a_run_failing_at_manager_reset_stamps_its_own_rollup() {
    let observer = Arc::new(MetricsObserver::new());

    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(SpecExporter::new(RecordingExporter))
        .build();
    let otel = OtelTracingLayer::new(
        tracing_opentelemetry::layer().with_tracer(provider.tracer("test")),
        provider.clone(),
        None,
    )
    .with_stamper(Stamper {
        observer: Some(Arc::<MetricsObserver>::clone(&observer)),
        registry: None,
        span_attributes: false,
    });
    let subscriber = Registry::default().with(otel).with(
        tracing_subscriber::filter::Targets::new().with_target("loopctl", LevelFilter::DEBUG),
    );

    let client =
        Arc::new(MockApiClient::new("test-model").with_responses(tool_turn_then_terminal()));
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    let mut engine = BareLoop::new_with_managers(
        client,
        registry,
        SessionConfig::default(),
        LoopManagers::new(),
    )
    .with_observer(Arc::<MetricsObserver>::clone(&observer));

    {
        let _guard = tracing::subscriber::set_default(subscriber);

        let first = engine
            .run("the run that completes", &RunConfig::default())
            .await;
        assert!(first.is_ok(), "the first run completes: {first:?}");

        poison_detector(&engine);
        let second = engine
            .run(
                "the run that fails at the reset",
                &RunConfig::default().with_reset_managers(true),
            )
            .await;
        assert!(
            matches!(second, Err(loopctl::error::LoopError::LockPoisoned { .. })),
            "a poisoned manager reset fails the run with its own error: {second:?}"
        );
    }
    provider.force_flush().unwrap();

    let spans = EXPORTED.lock().unwrap().clone();
    let runs: Vec<&SpanData> = spans
        .iter()
        .filter(|span| span.name == "invoke_agent")
        .collect();
    assert_eq!(
        runs.len(),
        2,
        "one run span per run, the settled failure included: {:?}",
        spans.iter().map(|s| s.name.as_ref()).collect::<Vec<_>>()
    );

    let failed = runs
        .iter()
        .copied()
        .find(|span| attr(span, "error.type").is_some())
        .expect("the failed run settles its span with the error class it returned on");
    let completed = runs
        .iter()
        .copied()
        .find(|span| attr(span, "error.type").is_none())
        .expect("the completed run settles its span clean");

    assert_eq!(
        attr(completed, "loopctl.run.turns"),
        Some(&Value::I64(2)),
        "the completed run stamps its own two turns — the rollup still works"
    );
    assert!(
        attr(completed, "loopctl.tool.echo.calls").is_some(),
        "the completed run stamps its tool rollup"
    );

    assert_eq!(
        attr(failed, "loopctl.run.turns"),
        Some(&Value::I64(0)),
        "the failed run stamps no one else's turn count"
    );
    assert_eq!(
        attr(failed, "loopctl.run.success"),
        Some(&Value::Bool(false)),
        "the failed run's rollup does not report a success"
    );
    assert_eq!(
        attr(failed, "gen_ai.usage.input_tokens"),
        Some(&Value::I64(0)),
        "a run that never reached the provider stamps no tokens"
    );
    assert!(
        attr(failed, "loopctl.tool.echo.calls").is_none(),
        "the previous run's per-tool rollup cannot ride the failed run's span"
    );
}

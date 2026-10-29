//! What a collector would receive for one loopctl run — minus the network.
//!
//! The `OTLP` exporter is replaced with an in-process recorder behind the
//! same [`SpecExporter`] transform the shipped pipeline uses, and the
//! composed layer carries the same [`Stamper`] — so the test asserts the
//! exact spans Jaeger/Tempo would see: `GenAI`-shaped names, kinds,
//! statuses, the identity-copied `gen_ai.*` attributes, the run-end
//! rollup stamp, and the error legs of the status transform. One test
//! function, one deterministic sequence — tracing caches per-callsite
//! interest process-wide, exactly as the engine's own span suite
//! documents.

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

use loopctl::config::SessionConfig;
use loopctl::engine::{BareLoop, RunConfig, core::Loop};
use loopctl::metrics::{MetricsLayer, MetricsObserver, MetricsRegistry};
use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
use loopctl::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSchema};
use loopctl_otel::SpecExporter;
use loopctl_otel::layer::{OtelTracingLayer, Stamper};
use opentelemetry::Value;
use opentelemetry::trace::{Status, TracerProvider as _};
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
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

/// A tool that always fails — the `error.type` → `OTel` status leg.
struct FailingTool;

impl Tool for FailingTool {
    fn name(&self) -> &'static str {
        "boom"
    }
    fn description(&self) -> &'static str {
        "Always fails"
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
        Box::pin(async { Err(ToolError::Execution("kaboom".to_string())) })
    }
}

/// One scripted tool-call turn plus the terminal reply it enables.
fn tool_turn_then_terminal() -> Vec<MockResponse> {
    tool_turn_then_terminal_for("echo")
}

/// As [`tool_turn_then_terminal`], naming the tool the model calls.
fn tool_turn_then_terminal_for(tool: &str) -> Vec<MockResponse> {
    vec![
        MockResponse {
            text: format!("calling {tool}"),
            tool_call: Some(MockToolCall {
                id: "call_1".to_string(),
                name: tool.to_string(),
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

/// A single terminal reply — one settled turn, no tool phase.
fn terminal() -> MockResponse {
    MockResponse {
        text: "done".to_string(),
        tool_call: None,
        stop_reason: "end_turn".to_string(),
    }
}

fn attr<'a>(span: &'a SpanData, key: &str) -> Option<&'a Value> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| &kv.value)
}

fn find_span<'a>(spans: &'a [SpanData], name: &str) -> Option<&'a SpanData> {
    spans.iter().find(|span| span.name == name)
}

#[tokio::test(flavor = "current_thread")]
async fn the_collector_sees_genai_shaped_spans() {
    let observer = Arc::new(MetricsObserver::new());
    let store = Arc::new(MetricsRegistry::default());

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
        registry: Some(Arc::<MetricsRegistry>::clone(&store)),
        span_attributes: true,
    });
    let subscriber = Registry::default()
        .with(otel)
        .with(MetricsLayer::new(Arc::<MetricsRegistry>::clone(&store)))
        .with(
            tracing_subscriber::filter::Targets::new().with_target("loopctl", LevelFilter::DEBUG),
        );

    let client =
        Arc::new(MockApiClient::new("test-model").with_responses(tool_turn_then_terminal()));
    let registry = ToolRegistry::new();
    registry.register(EchoTool);
    registry.register(FailingTool);
    let mut engine = BareLoop::new(client, registry, SessionConfig::default())
        .with_observer(Arc::<MetricsObserver>::clone(&observer));

    {
        let _guard = tracing::subscriber::set_default(subscriber);
        let result = engine.run("otel e2e", &RunConfig::default()).await;
        assert!(result.is_ok(), "run must succeed: {:?}", result.err());
    }
    provider.force_flush().unwrap();

    let spans = EXPORTED.lock().unwrap().clone();

    // Span names follow the GenAI conventions, kinds are INTERNAL.
    let run = find_span(&spans, "invoke_agent").expect("invoke_agent span");
    assert_eq!(run.span_kind, opentelemetry::trace::SpanKind::Internal);
    let tool = find_span(&spans, "execute_tool echo").expect("execute_tool span");
    assert_eq!(tool.span_kind, opentelemetry::trace::SpanKind::Internal);
    assert!(
        find_span(&spans, "agent.turn").is_some(),
        "turn span keeps its loopctl-defined name, got: {:?}",
        spans.iter().map(|s| s.name.as_ref()).collect::<Vec<_>>()
    );

    // Identity mapping: the engine's fields arrive verbatim.
    assert_eq!(
        attr(tool, "gen_ai.tool.name"),
        Some(&Value::String("echo".into()))
    );
    assert_eq!(
        attr(run, "gen_ai.request.model"),
        Some(&Value::String("test-model".into()))
    );
    assert!(
        attr(run, "gen_ai.conversation.id").is_some(),
        "conversation id must ride"
    );

    // Outcome → status: a clean run closes Ok.
    assert!(
        matches!(run.status, Status::Ok),
        "run status: {:?}",
        run.status
    );
    assert!(
        matches!(tool.status, Status::Ok),
        "tool status: {:?}",
        tool.status
    );

    // Derived coarse system from the model id (unknown model → unknown).
    assert_eq!(
        attr(run, "gen_ai.system"),
        Some(&Value::String("unknown".into()))
    );

    // Close-time stamp: the run rollup and registry diagnostics.
    assert!(
        attr(run, "loopctl.run.turns").is_some(),
        "rollup must be stamped"
    );
    assert!(
        attr(run, "gen_ai.usage.input_tokens").is_some(),
        "tokens must be stamped"
    );
    assert!(
        attr(run, "loopctl.telemetry.unknown_metric").is_some(),
        "registry must ride"
    );

    // Metric-target events were NOT mirrored as span events.
    for span in spans.iter().filter(|s| s.name == "invoke_agent") {
        for event in span.events.clone() {
            let target = event
                .attributes
                .iter()
                .find(|kv| kv.key.as_str() == "target");
            if let Some(kv) = target {
                assert_ne!(
                    kv.value,
                    Value::String("loopctl::metrics".into()),
                    "metrics event leaked into span events"
                );
            }
        }
    }

    // Scenario 2: a failing tool closes its span with an error status —
    // the `outcome`/`error.type` → `OTel` status leg of the transform,
    // which nothing above exercises (every span so far settled Ok).
    EXPORTED.lock().unwrap().clear();
    let failing_subscriber = Registry::default()
        .with(
            OtelTracingLayer::new(
                tracing_opentelemetry::layer().with_tracer(provider.tracer("test")),
                provider.clone(),
                None,
            )
            .with_stamper(Stamper {
                observer: Some(Arc::<MetricsObserver>::clone(&observer)),
                registry: Some(Arc::<MetricsRegistry>::clone(&store)),
                span_attributes: true,
            }),
        )
        .with(MetricsLayer::new(Arc::<MetricsRegistry>::clone(&store)))
        .with(
            tracing_subscriber::filter::Targets::new().with_target("loopctl", LevelFilter::DEBUG),
        );
    let failing_client = Arc::new(
        MockApiClient::new("test-model").with_responses(tool_turn_then_terminal_for("boom")),
    );
    let failing_registry = ToolRegistry::new();
    failing_registry.register(FailingTool);
    let mut failing_engine =
        BareLoop::new(failing_client, failing_registry, SessionConfig::default())
            .with_observer(Arc::<MetricsObserver>::clone(&observer));
    {
        let _guard = tracing::subscriber::set_default(failing_subscriber);
        let result = failing_engine
            .run("failing tool", &RunConfig::default())
            .await;
        assert!(result.is_ok(), "a soft tool error does not fail the run");
    }
    provider.force_flush().unwrap();
    let failed_spans = EXPORTED.lock().unwrap().clone();
    let failed_tool =
        find_span(&failed_spans, "execute_tool boom").expect("the failing tool's span exports");
    assert!(
        matches!(failed_tool.status, Status::Error { .. }),
        "an errored tool span closes with OTel status Error, got {:?}",
        failed_tool.status
    );
    assert_eq!(
        attr(failed_tool, "outcome"),
        Some(&Value::String("error".into())),
        "the outcome attribute rides alongside the status"
    );

    // Scenario 3: two runs on one observer, one flush — each run span
    // must carry its OWN run's rollup. The stamp is written at the
    // run-end record (inside the span's scope, after that run's
    // observer callbacks); an export-time stamp would read the observer
    // at flush time and print the newest run's numbers on both. The
    // second run takes a tool turn plus its terminal reply (two turns)
    // against the first run's single turn, so the two readings differ
    // and the assertion can tell them apart.
    EXPORTED.lock().unwrap().clear();
    let multi_observer = Arc::new(MetricsObserver::new());
    let multi_subscriber = Registry::default()
        .with(
            OtelTracingLayer::new(
                tracing_opentelemetry::layer().with_tracer(provider.tracer("test")),
                provider.clone(),
                None,
            )
            .with_stamper(Stamper {
                observer: Some(Arc::<MetricsObserver>::clone(&multi_observer)),
                registry: Some(Arc::<MetricsRegistry>::clone(&store)),
                span_attributes: true,
            }),
        )
        .with(MetricsLayer::new(Arc::<MetricsRegistry>::clone(&store)))
        .with(
            tracing_subscriber::filter::Targets::new().with_target("loopctl", LevelFilter::DEBUG),
        );
    let multi_responses = vec![terminal()]
        .into_iter()
        .chain(tool_turn_then_terminal())
        .collect();
    let multi_client = Arc::new(MockApiClient::new("test-model").with_responses(multi_responses));
    let multi_registry = ToolRegistry::new();
    multi_registry.register(EchoTool);
    let mut multi_engine = BareLoop::new(multi_client, multi_registry, SessionConfig::default())
        .with_observer(Arc::<MetricsObserver>::clone(&multi_observer));
    {
        let _guard = tracing::subscriber::set_default(multi_subscriber);
        multi_engine
            .run("first run", &RunConfig::default())
            .await
            .expect("run one completes");
        multi_engine
            .run("second run", &RunConfig::default())
            .await
            .expect("run two completes");
    }
    provider.force_flush().unwrap();
    let mut stamps: Vec<i64> = EXPORTED
        .lock()
        .unwrap()
        .iter()
        .filter(|span| span.name == "invoke_agent")
        .filter_map(|span| match attr(span, "loopctl.run.turns") {
            Some(Value::I64(turns)) => Some(*turns),
            _ => None,
        })
        .collect();
    stamps.sort_unstable();
    assert_eq!(
        stamps,
        vec![1, 2],
        "each run span carries its own run's rollup — the flush-time session \
         total would read 2 on both"
    );

    // Scenario 4: a registry-only stamper — no observer configured —
    // still stamps its series onto the run span, and stamps no rollup.
    // Stamp readers that gate on an observer would omit every series a
    // `SpanAttributes` host actually collected.
    EXPORTED.lock().unwrap().clear();
    let bare_store = Arc::new(MetricsRegistry::default());
    let bare_subscriber = Registry::default()
        .with(
            OtelTracingLayer::new(
                tracing_opentelemetry::layer().with_tracer(provider.tracer("test")),
                provider.clone(),
                None,
            )
            .with_stamper(Stamper {
                observer: None,
                registry: Some(Arc::<MetricsRegistry>::clone(&bare_store)),
                span_attributes: true,
            }),
        )
        .with(MetricsLayer::new(Arc::<MetricsRegistry>::clone(
            &bare_store,
        )))
        .with(
            tracing_subscriber::filter::Targets::new().with_target("loopctl", LevelFilter::DEBUG),
        );
    let bare_client =
        Arc::new(MockApiClient::new("test-model").with_responses(tool_turn_then_terminal()));
    let bare_registry = ToolRegistry::new();
    bare_registry.register(EchoTool);
    let mut bare_engine = BareLoop::new(bare_client, bare_registry, SessionConfig::default());
    {
        let _guard = tracing::subscriber::set_default(bare_subscriber);
        bare_engine
            .run("registry only", &RunConfig::default())
            .await
            .expect("run completes");
    }
    provider.force_flush().unwrap();
    let bare_spans = EXPORTED.lock().unwrap().clone();
    let bare_run = find_span(&bare_spans, "invoke_agent").expect("the registry-only run span");
    let stamped_calls = bare_run
        .attributes
        .iter()
        .find(|kv| kv.key.as_str().starts_with("loopctl.tools.calls"))
        .expect("the registry's series ride the run span without an observer");
    assert_eq!(
        stamped_calls.value,
        opentelemetry::Value::I64(1),
        "the fixture's single echo call stamps as one call — presence alone \
         would pass on a zero or a doubled count: {:?}",
        stamped_calls.value
    );
    assert!(
        attr(bare_run, "loopctl.run.turns").is_none(),
        "no observer, no rollup — the stamp claims nothing it cannot read"
    );
}

#[test]
fn pipeline_exports_registry_series_to_the_meter() {
    use loopctl::metrics::{MetricsRegistry, lookup};
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
    use std::collections::BTreeMap;

    let exporter = InMemoryMetricExporter::default();
    let reader = PeriodicReader::builder(exporter.clone()).build();
    let provider = SdkMeterProvider::builder().with_reader(reader).build();
    loopctl_otel::metrics::install_meter(provider.meter("test"));

    let store = MetricsRegistry::default();
    let calls = lookup("loopctl.tools.calls").expect("declared counter");
    let window = lookup("loopctl.context.window").expect("declared gauge");
    let calibration = lookup("loopctl.estimate.calibration").expect("declared histogram");
    store.counter_add(calls, BTreeMap::from([("tool", "bash".to_string())]), 3);
    store.gauge_set(
        window,
        BTreeMap::from([("source", "preset".to_string())]),
        200_000.0,
    );
    store.histogram_record(calibration, BTreeMap::new(), 0.12);
    store.histogram_record(calibration, BTreeMap::new(), 0.9);

    // A run rollup with a total past `i32::MAX` rides the same export:
    // the gauge it lands on must keep the magnitude, not cap at the
    // `i32` ceiling where distinct large totals are indistinguishable.
    let wide_run = loopctl::metrics::AgentMetrics {
        total_turns: 5_000_000_000,
        ..loopctl::metrics::AgentMetrics::default()
    };

    loopctl_otel::export_metrics(Some(&wide_run), &store).expect("pipeline installed");
    provider.force_flush().expect("flush");

    let resource_metrics = exporter.get_finished_metrics().expect("exported");
    let mut names: Vec<String> = Vec::new();
    let mut counter_total = None;
    let mut histogram_count = None;
    let mut wide_turns = None;
    for rm in &resource_metrics {
        for sm in rm.scope_metrics() {
            for metric in sm.metrics() {
                names.push(metric.name().to_string());
                match (metric.name(), metric.data()) {
                    ("loopctl.tools.calls", AggregatedMetrics::U64(MetricData::Sum(sum))) => {
                        counter_total = sum
                            .data_points()
                            .next()
                            .map(opentelemetry_sdk::metrics::data::SumDataPoint::value);
                    }
                    (
                        "loopctl.estimate.calibration",
                        AggregatedMetrics::F64(MetricData::Histogram(histogram)),
                    ) => {
                        histogram_count = histogram
                            .data_points()
                            .next()
                            .map(opentelemetry_sdk::metrics::data::HistogramDataPoint::count);
                    }
                    ("loopctl.run.turns", AggregatedMetrics::F64(MetricData::Gauge(gauge))) => {
                        wide_turns = gauge
                            .data_points()
                            .next()
                            .map(opentelemetry_sdk::metrics::data::GaugeDataPoint::value);
                    }
                    _ => {}
                }
            }
        }
    }
    for want in [
        "loopctl.tools.calls",
        "loopctl.context.window",
        "loopctl.estimate.calibration",
        "loopctl.telemetry.unknown_metric",
        "loopctl.telemetry.missing_labels",
    ] {
        assert!(
            names.iter().any(|n| n == want),
            "missing {want} in {names:?}"
        );
    }
    assert_eq!(counter_total, Some(3), "counter total");
    assert_eq!(histogram_count, Some(2), "histogram sample count");
    let wide_turns = wide_turns.unwrap_or_default();
    assert!(
        (wide_turns - 5_000_000_000.0).abs() < 1.0,
        "a run counter past i32::MAX exports at its true magnitude: {wide_turns}"
    );
}

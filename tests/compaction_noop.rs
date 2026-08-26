//! Compaction contracts for default-constructed and profile-configured loops.
//!
//! Pins three invariants: no request is ever sent with a conversation whose
//! estimated size exceeds the configured context window, a default-constructed
//! loop has compaction machinery behind its threshold (observer-visible
//! compaction, not silent estimate resets), and the small-model profile keeps
//! the same promise. Also covers the pre-compact-hook veto path (a measured
//! estimate, not a hard-coded zero) and host-installed context managers
//! surviving the constructor's default seeding.
//!
//! Requires the `testing` feature; the hook-veto test also requires `hooks`.

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone
)]

#[cfg(feature = "testing")]
mod scenarios {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use futures::Stream;
    use loopctl::api::error::ApiError;
    use loopctl::api::{ApiClient, NonStreamingResponse, StreamRequest};
    use loopctl::compact::{
        CompactReason, CompactionContext, CompactionOutcome, ContextCompactor, ContextManager,
        HeuristicTokenCounter, TokenCounter,
    };
    use loopctl::config::SessionConfig;
    use loopctl::contributor::{ContextContributor, ContributorContext};
    use loopctl::engine::core::Loop;
    use loopctl::engine::{BareLoop, RunConfig};
    use loopctl::error::LoopError;
    use loopctl::message::Message;
    use loopctl::observer::{CompactedContext, LoopObserver, TurnStartContext};
    use loopctl::stream::StreamEvent;
    use loopctl::testing::{MockApiClient, MockResponse, MockToolCall};
    use loopctl::tool::{Tool, ToolContext, ToolOutput, ToolRegistry, ToolSchema};

    #[derive(Clone)]
    struct RecordingClient {
        inner: MockApiClient,
        request_tokens: Arc<Mutex<Vec<u64>>>,
    }

    impl RecordingClient {
        fn wrap(inner: MockApiClient) -> Self {
            Self {
                inner,
                request_tokens: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn record(&self, request: &StreamRequest) {
            let mut tokens = HeuristicTokenCounter.count(&request.messages);
            if let Some(system) = &request.system {
                tokens += HeuristicTokenCounter.count(&[Message::user(system.clone())]);
            }
            if let Some(tools) = &request.tools {
                let rendered = serde_json::to_string(tools).unwrap_or_default();
                tokens += HeuristicTokenCounter.count(&[Message::user(rendered)]);
            }
            self.request_tokens
                .lock()
                .expect("request log lock")
                .push(tokens);
        }

        fn served_request_tokens(&self) -> Vec<u64> {
            self.request_tokens
                .lock()
                .expect("request log lock")
                .clone()
        }
    }

    impl ApiClient for RecordingClient {
        fn model(&self) -> String {
            self.inner.model()
        }

        fn stream_messages(
            &self,
            request: &StreamRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
            self.record(request);
            self.inner.stream_messages(request)
        }

        fn create_message(
            &self,
            request: &StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>
        {
            self.record(request);
            self.inner.create_message(request)
        }
    }

    struct CompactionCounter {
        events: AtomicUsize,
    }

    /// A contributor injecting a fixed chunk of transient context.
    struct ChunkyContributor {
        chars: usize,
    }

    impl ContextContributor for ChunkyContributor {
        fn contribute(&self, _ctx: &ContributorContext<'_>) -> Option<Message> {
            Some(Message::user("m".repeat(self.chars)))
        }
    }

    /// A contributor that counts its consultations.
    struct CountingContributor {
        consultations: Arc<AtomicUsize>,
    }

    impl ContextContributor for CountingContributor {
        fn contribute(&self, _ctx: &ContributorContext<'_>) -> Option<Message> {
            self.consultations.fetch_add(1, Ordering::SeqCst);
            Some(Message::user("m".repeat(400)))
        }
    }

    /// A contributor injecting its chunk only while armed — used to
    /// overload exactly one run of a multi-run sequence.
    struct ArmedContributor {
        armed: Arc<AtomicBool>,
        chars: usize,
    }

    impl ContextContributor for ArmedContributor {
        fn contribute(&self, _ctx: &ContributorContext<'_>) -> Option<Message> {
            if self.armed.load(Ordering::SeqCst) {
                Some(Message::user("m".repeat(self.chars)))
            } else {
                None
            }
        }
    }

    /// An observer counting turn starts.
    struct TurnStartCounter {
        starts: Arc<AtomicUsize>,
    }

    impl LoopObserver for TurnStartCounter {
        fn name(&self) -> &'static str {
            "TurnStartCounter"
        }

        fn on_turn_start(&self, _ctx: &TurnStartContext) {
            self.starts.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl LoopObserver for CompactionCounter {
        fn name(&self) -> &'static str {
            "CompactionCounter"
        }

        fn on_compaction(&self, _ctx: &CompactedContext) {
            self.events.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }

        fn description(&self) -> &'static str {
            "Returns its input inside a fixed-size result"
        }

        fn schema(&self) -> ToolSchema {
            ToolSchema {
                tool: self.name().to_string(),
                description: self.description().to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        fn call(
            &self,
            input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, loopctl::tool::ToolError>> + Send + '_>>
        {
            Box::pin(async move {
                let fill = input
                    .get("fill")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|fill| usize::try_from(fill).ok())
                    .unwrap_or(200);
                let payload = format!("echo: {input} {}", "r".repeat(fill));
                Ok(ToolOutput::text(payload))
            })
        }
    }

    /// One scripted assistant turn: `text_chars` of text plus an echo call.
    ///
    /// The step index varies both the text and the tool input so loop and
    /// convergence detection see distinct responses across turns.
    fn tool_turn_response(step: usize, text_chars: usize) -> MockResponse {
        MockResponse {
            text: format!("step {step} {}", "w".repeat(text_chars)),
            tool_call: Some(MockToolCall {
                id: format!("call_{step}"),
                name: "echo".to_string(),
                input: serde_json::json!({"step": step}),
            }),
            stop_reason: "tool_use".to_string(),
        }
    }

    /// A tool turn whose echo result is `fill` characters, so a single
    /// dispatch can grow the history by a controlled amount.
    fn tool_turn_response_with_fill(step: usize, text_chars: usize, fill: usize) -> MockResponse {
        MockResponse {
            text: format!("step {step} {}", "w".repeat(text_chars)),
            tool_call: Some(MockToolCall {
                id: format!("call_{step}"),
                name: "echo".to_string(),
                input: serde_json::json!({"step": step, "fill": fill}),
            }),
            stop_reason: "tool_use".to_string(),
        }
    }

    fn final_response() -> MockResponse {
        MockResponse {
            text: "all done".to_string(),
            tool_call: None,
            stop_reason: "end_turn".to_string(),
        }
    }

    /// A conversation of `turns` echo turns that grows past a small window,
    /// ending with a text-only response so the run completes.
    fn growing_conversation_script(turns: usize) -> Vec<MockResponse> {
        (0..turns)
            .map(|step| tool_turn_response(step, 40))
            .chain(std::iter::once(final_response()))
            .collect()
    }

    fn registry_with_echo() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool);
        registry
    }

    #[tokio::test]
    async fn over_limit_context_is_never_sent_to_the_provider() {
        let script = vec![
            final_response(),
            tool_turn_response(0, 40),
            tool_turn_response(1, 40),
            tool_turn_response(2, 40),
            tool_turn_response(3, 40),
            tool_turn_response(4, 40),
        ];
        let client = RecordingClient::wrap(MockApiClient::new("test-model").with_responses(script));

        let config = SessionConfig::default()
            .with_context_window(200)
            .with_compact_threshold(50);
        let client_handle = client.clone();
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);

        let first = agent.run(&"x".repeat(200), &RunConfig::default()).await;
        assert!(first.is_ok(), "the first run stays under the threshold");

        let _second = agent.run(&"y".repeat(150), &RunConfig::default()).await;

        let served = client_handle.served_request_tokens();
        assert!(
            !served.is_empty(),
            "the loop must have served at least one request"
        );
        for tokens in &served {
            assert!(
                *tokens <= 260,
                "no request may exceed the 260-token window; served estimates {served:?}"
            );
        }
    }

    #[tokio::test]
    async fn first_request_of_an_over_window_run_is_never_sent() {
        let script = vec![
            final_response(),
            tool_turn_response(0, 40),
            tool_turn_response(1, 40),
        ];
        let client = RecordingClient::wrap(MockApiClient::new("test-model").with_responses(script));

        let config = SessionConfig::default()
            .with_context_window(260)
            .with_compact_threshold(80);
        let client_handle = client.clone();
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);

        let first = agent.run(&"x".repeat(500), &RunConfig::default()).await;
        assert!(
            first.is_ok(),
            "the first run starts under the threshold and completes: {first:?}"
        );

        let _second = agent.run(&"y".repeat(200), &RunConfig::default()).await;

        let served = client_handle.served_request_tokens();
        for tokens in &served {
            assert!(
                *tokens <= 200,
                "committed history over the window must trigger compaction (or a typed \
                 failure) before the run's first request; served estimates {served:?}"
            );
        }
    }

    #[tokio::test]
    async fn tool_result_growth_alone_crosses_the_threshold() {
        let script = vec![tool_turn_response_with_fill(0, 20, 8_000), final_response()];
        let client = RecordingClient::wrap(MockApiClient::new("test-model").with_responses(script));

        let config = SessionConfig::default()
            .with_context_window(2_000)
            .with_compact_threshold(80);
        let client_handle = client.clone();
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);

        let result = agent.run("keep echoing", &RunConfig::default()).await;

        match result {
            Err(LoopError::ContextExceeded { .. }) => {}
            other => panic!(
                "tool-result growth alone must trip the compaction check and fail to \
                 shrink the three-message history; got {other:?}"
            ),
        }
        let served = client_handle.served_request_tokens();
        assert_eq!(
            served.len(),
            1,
            "the follow-up request must wait for the compaction check to see the \
             tool-result growth; served {served:?}"
        );
        for tokens in &served {
            assert!(
                *tokens <= 2_000,
                "no request may exceed the 2_000-token window; served estimates {served:?}"
            );
        }
    }

    #[tokio::test]
    async fn run_start_compacts_at_the_threshold() {
        // The policy twin of the invariant above: committed history left
        // over the threshold (under the window) by a prior run compacts
        // before the next run's first request — observer-visible, and
        // the new run's first request carries the compacted history,
        // not the grown one.
        let script = vec![
            tool_turn_response_with_fill(0, 40, 320),
            tool_turn_response_with_fill(1, 40, 320),
            tool_turn_response_with_fill(2, 40, 320),
            tool_turn_response_with_fill(3, 40, 320),
            final_response(),
            final_response(),
        ];
        let client = RecordingClient::wrap(MockApiClient::new("test-model").with_responses(script));
        let observer = Arc::new(CompactionCounter {
            events: AtomicUsize::new(0),
        });

        let config = SessionConfig::default()
            .with_context_window(700)
            .with_compact_threshold(80);
        let client_handle = client.clone();
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);
        agent.register_observer(Arc::clone(&observer) as Arc<dyn LoopObserver>);

        let first = agent.run("grow the history", &RunConfig::default()).await;
        assert!(
            first.is_ok(),
            "run 1 grows the history and completes: {first:?}"
        );
        let events_after_run1 = observer.events.load(Ordering::SeqCst);
        let served_after_run1 = client_handle.served_request_tokens().len();
        let last_of_run1 = client_handle
            .served_request_tokens()
            .get(served_after_run1.saturating_sub(1))
            .copied()
            .unwrap_or_default();

        let second = agent.run(&"y".repeat(300), &RunConfig::default()).await;
        assert!(
            second.is_ok(),
            "run 2 compacts at start and completes: {second:?}"
        );

        let served = client_handle.served_request_tokens();
        assert!(
            observer.events.load(Ordering::SeqCst) > events_after_run1,
            "the run-start estimate feeds the trigger — compaction fires \
             before the new run's first request"
        );
        let first_of_run2 = served.get(served_after_run1).copied().unwrap_or_default();
        assert!(
            first_of_run2 < last_of_run1,
            "the new run's first request carries the compacted history, not \
             the grown one: first of run 2 {first_of_run2}, last of run 1 \
             {last_of_run1} (served {served:?})"
        );
        for tokens in &served {
            assert!(
                *tokens <= 700,
                "no request may exceed the 700-token window; served \
                 estimates {served:?}"
            );
        }
    }

    #[tokio::test]
    async fn served_requests_count_the_whole_payload() {
        // The invariant the history-scoped checks never pinned: the
        // provider receives transients + history + system prompt +
        // tool schemas, and no served request may exceed the window
        // once the whole payload is counted.
        let script = vec![
            tool_turn_response(0, 40),
            tool_turn_response(1, 40),
            final_response(),
        ];
        let client = RecordingClient::wrap(MockApiClient::new("test-model").with_responses(script));

        let config = SessionConfig::default()
            .with_context_window(250)
            .with_compact_threshold(80)
            .with_system_prompt("s".repeat(200));
        let client_handle = client.clone();
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);
        agent.add_contributor(Box::new(ChunkyContributor { chars: 300 }));

        let _outcome = agent.run("count everything", &RunConfig::default()).await;

        let served = client_handle.served_request_tokens();
        assert!(
            !served.is_empty(),
            "the run must attempt at least one request"
        );
        for tokens in &served {
            assert!(
                *tokens <= 250,
                "no served request may exceed the window once history, \
                 system prompt, tool schemas, and transients are all \
                 counted; served {served:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_transient_overload_defers_to_compaction_and_the_retry_proceeds() {
        // The deferral contract: a turn whose transients push the
        // payload over the threshold defers to compaction before its
        // request — the deferred attempt fires no turn events, the
        // retried turn re-consults the contributor against the
        // compacted history, and its request is then served under the
        // window.
        let script = vec![
            tool_turn_response(0, 40),
            tool_turn_response(1, 40),
            tool_turn_response(2, 40),
            final_response(),
        ];
        let client = RecordingClient::wrap(MockApiClient::new("test-model").with_responses(script));
        let consultations = Arc::new(AtomicUsize::new(0));
        let starts = Arc::new(AtomicUsize::new(0));

        let config = SessionConfig::default()
            .with_context_window(400)
            .with_compact_threshold(80);
        let client_handle = client.clone();
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);
        agent.add_contributor(Box::new(CountingContributor {
            consultations: Arc::clone(&consultations),
        }));
        agent.register_observer(Arc::new(TurnStartCounter {
            starts: Arc::clone(&starts),
        }) as Arc<dyn LoopObserver>);

        let result = agent.run("defer then retry", &RunConfig::default()).await;
        assert!(
            result.is_ok(),
            "the deferring run compacts and completes: {result:?}"
        );

        let served = client_handle.served_request_tokens();
        assert!(
            !served.is_empty(),
            "the deferred turn's retry must reach the provider"
        );
        assert!(
            consultations.load(Ordering::SeqCst) > starts.load(Ordering::SeqCst),
            "at least one consultation was consumed by a deferral — every \
             served turn consults once, a deferred attempt consults \
             without a turn starting"
        );
        assert_eq!(
            starts.load(Ordering::SeqCst),
            served.len(),
            "a deferred attempt fires no turn events — one turn start per \
             served request"
        );
        for tokens in &served {
            assert!(
                *tokens <= 400,
                "every served request carries the whole payload under the \
                 window; served {served:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_reserved_overflow_reports_payload_comparable_numbers() {
        // The reserved fit check fires when the history alone fits the
        // window but history + reserve does not — the reported numbers
        // must read as an overflow. A history-only `tokens_used`
        // against the full window reads as a pass (170 under a limit
        // of 300) and hides the reserve that forced the failure.
        struct KeepEverythingCompactor;

        impl ContextCompactor for KeepEverythingCompactor {
            fn compact(
                &self,
                messages: Vec<Message>,
                _target_tokens: u64,
                context: CompactionContext,
            ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
                Box::pin(async move {
                    let tokens_after = context.counter.count(&messages);
                    CompactionOutcome {
                        tokens_saved: 0,
                        messages,
                        tokens_after,
                        success: true,
                        error: None,
                    }
                })
            }
        }

        let manager =
            ContextManager::new(Arc::new(KeepEverythingCompactor)).with_context_window(300);
        let messages = vec![Message::user("x".repeat(680))];
        let overflow = manager
            .compact_with_reason(
                messages,
                0,
                CompactReason::ThresholdExceeded,
                None,
                Vec::new(),
                143,
            )
            .await
            .expect_err(
                "170 history tokens against a window of 300 minus a \
                         143-token reserve: the fit check fires",
            );

        assert!(
            overflow.tokens_used >= overflow.context_window,
            "the overflow reads as an overflow — the payload including the \
             reserve against the window: used {}, window {}",
            overflow.tokens_used,
            overflow.context_window
        );
        assert!(
            overflow.trigger == CompactReason::ThresholdExceeded,
            "the overflow names the reason that triggered the failed pass, \
             got {:?}",
            overflow.trigger
        );
    }

    #[tokio::test]
    async fn growing_transients_never_serve_over_the_window() {
        // Adversarial sequence: a contributor whose output grows with
        // every consultation (including the extra consultations
        // consumed by deferrals), so the transient budget compounds.
        // Whatever the compaction loop does — fit, re-defer, or fail —
        // no served request may exceed the window, and the run must
        // terminate rather than loop.
        struct IncreasingContributor {
            consultations: Arc<AtomicUsize>,
        }

        impl ContextContributor for IncreasingContributor {
            fn contribute(&self, _ctx: &ContributorContext<'_>) -> Option<Message> {
                let n = self.consultations.fetch_add(1, Ordering::SeqCst) + 1;
                Some(Message::user("g".repeat(80 * n)))
            }
        }

        let script = vec![
            tool_turn_response(0, 40),
            tool_turn_response(1, 40),
            tool_turn_response(2, 40),
            tool_turn_response(3, 40),
            final_response(),
        ];
        let client = RecordingClient::wrap(MockApiClient::new("test-model").with_responses(script));

        let config = SessionConfig::default()
            .with_context_window(350)
            .with_compact_threshold(80);
        let client_handle = client.clone();
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);
        agent.add_contributor(Box::new(IncreasingContributor {
            consultations: Arc::new(AtomicUsize::new(0)),
        }));

        let result = agent
            .run("grow against the window", &RunConfig::default())
            .await;
        assert!(
            matches!(result, Ok(_) | Err(LoopError::ContextExceeded { .. })),
            "the run terminates — fitting or failing honestly: {result:?}"
        );

        let served = client_handle.served_request_tokens();
        for tokens in &served {
            assert!(
                *tokens <= 350,
                "no served request exceeds the window however the \
                 transients grow; served {served:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_deferred_compaction_reserves_the_transient_budget() {
        // The deferred turn's transients ride the retry too: the
        // compaction target and the fit check reserve room for them,
        // so the retried full payload fits. Without the reserve, the
        // first pass compacts to "fits the window" while ignoring the
        // transients, the retry defers again, and the idempotent
        // second pass dead-ends on ContextExceeded.
        struct TargetSheddingCompactor {
            targets: Arc<Mutex<Vec<u64>>>,
        }

        impl ContextCompactor for TargetSheddingCompactor {
            fn compact(
                &self,
                messages: Vec<Message>,
                target_tokens: u64,
                context: CompactionContext,
            ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
                let targets = Arc::clone(&self.targets);
                Box::pin(async move {
                    targets.lock().expect("targets lock").push(target_tokens);
                    let mut kept = messages;
                    while kept.len() > 2 {
                        if context.counter.count(&kept) <= target_tokens {
                            break;
                        }
                        // Shed after the unconditionally-preserved first
                        // message.
                        kept.remove(1);
                    }
                    let tokens_after = context.counter.count(&kept);
                    CompactionOutcome {
                        tokens_saved: context.tokens_before.saturating_sub(tokens_after),
                        messages: kept,
                        tokens_after,
                        success: true,
                        error: None,
                    }
                })
            }
        }

        let script = vec![
            tool_turn_response(0, 40),
            tool_turn_response(1, 40),
            tool_turn_response(2, 40),
            final_response(),
        ];
        let client = RecordingClient::wrap(MockApiClient::new("test-model").with_responses(script));
        let targets = Arc::new(Mutex::new(Vec::new()));

        let config = SessionConfig::default()
            .with_context_window(300)
            .with_compact_threshold(80);
        let client_handle = client.clone();
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);
        agent.set_context_manager(Arc::new(ContextManager::new(Arc::new(
            TargetSheddingCompactor {
                targets: Arc::clone(&targets),
            },
        ))));
        agent.add_contributor(Box::new(ChunkyContributor { chars: 400 }));

        let result = agent.run("reserve the budget", &RunConfig::default()).await;
        assert!(
            result.is_ok(),
            "the compaction reserved the transient budget and the retried \
             full payload fits: {result:?}"
        );

        let served = client_handle.served_request_tokens();
        assert!(!served.is_empty(), "the retried turn reaches the provider");
        for tokens in &served {
            assert!(
                *tokens <= 300,
                "every served request carries the whole payload under the \
                 window; served {served:?}"
            );
        }
        assert!(
            !targets.lock().expect("targets lock").is_empty(),
            "the deferred turn's compaction ran"
        );
    }

    #[cfg(feature = "hooks")]
    #[tokio::test]
    async fn a_stale_deferred_budget_reserves_nothing_in_the_next_run() {
        // A run that dies with an unconsumed deferred budget — its
        // compaction was vetoed after the deferral set the budget —
        // must not shrink the next run's compaction target: the budget
        // is cleared at run start. The stale chunk here dwarfs the
        // window, so a surviving budget zeroes the target and the fit
        // limit and the next run dead-ends on ContextExceeded.
        use loopctl::hooks::context::{CompactResult, PreCompactContext};
        use loopctl::hooks::{Hook, HookExecutor};

        struct VetoFirstArmedCompaction {
            armed: Arc<AtomicBool>,
            vetoed: AtomicBool,
        }

        impl Hook for VetoFirstArmedCompaction {
            fn name(&self) -> &str {
                "veto_first_armed_compaction"
            }

            fn on_pre_compact(&self, _ctx: &PreCompactContext) -> Option<CompactResult> {
                if self.armed.load(Ordering::SeqCst) && !self.vetoed.swap(true, Ordering::SeqCst) {
                    Some(CompactResult::abort("not now"))
                } else {
                    None
                }
            }
        }

        let script = vec![
            tool_turn_response_with_fill(0, 40, 320),
            tool_turn_response_with_fill(1, 40, 320),
            tool_turn_response_with_fill(2, 40, 320),
            tool_turn_response_with_fill(3, 40, 320),
            final_response(),
            tool_turn_response(0, 40),
            final_response(),
        ];
        let client = RecordingClient::wrap(MockApiClient::new("test-model").with_responses(script));

        let config = SessionConfig::default()
            .with_context_window(700)
            .with_compact_threshold(80);
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);
        let mut executor = HookExecutor::new();
        let armed = Arc::new(AtomicBool::new(false));
        executor.register(Arc::new(VetoFirstArmedCompaction {
            armed: Arc::clone(&armed),
            vetoed: AtomicBool::new(false),
        }));
        agent.set_hook_executor(Arc::new(executor));
        agent.add_contributor(Box::new(ArmedContributor {
            armed: Arc::clone(&armed),
            chars: 12_000,
        }));

        let first = agent.run("grow the history", &RunConfig::default()).await;
        assert!(
            first.is_ok(),
            "run 1 grows the committed history and completes: {first:?}"
        );

        armed.store(true, Ordering::SeqCst);
        let second = agent.run("overload now", &RunConfig::default()).await;
        assert!(
            matches!(second, Err(LoopError::ContextExceeded { .. })),
            "run 2 defers on the transient overload and dies at the vetoed \
             compaction with the budget unconsumed: {second:?}"
        );

        armed.store(false, Ordering::SeqCst);
        let third = agent.run(&"y".repeat(300), &RunConfig::default()).await;
        assert!(
            third.is_ok(),
            "run 3 compacts at start against the unreserved target — the \
             stale budget was cleared at run start: {third:?}"
        );
    }

    #[tokio::test]
    async fn default_loop_compacts_at_the_threshold() {
        let client = RecordingClient::wrap(
            MockApiClient::new("test-model").with_responses(growing_conversation_script(12)),
        );
        let observer = Arc::new(CompactionCounter {
            events: AtomicUsize::new(0),
        });

        let config = SessionConfig::default()
            .with_context_window(600)
            .with_compact_threshold(80);
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);
        agent.register_observer(Arc::clone(&observer) as Arc<dyn LoopObserver>);

        let result = agent.run("keep echoing", &RunConfig::default()).await;

        assert!(
            result.is_ok(),
            "a default-configured loop must survive crossing the threshold: {result:?}"
        );
        assert!(
            observer.events.load(Ordering::SeqCst) >= 1,
            "a default-constructed loop must compact at the threshold (on_compaction fired {} times)",
            observer.events.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn small_model_profile_compacts_at_the_threshold() {
        let client = RecordingClient::wrap(
            MockApiClient::new("test-model").with_responses(growing_conversation_script(12)),
        );
        let observer = Arc::new(CompactionCounter {
            events: AtomicUsize::new(0),
        });

        let config =
            loopctl::presets::ConstrainedProfile::session_config().with_context_window(600);
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);
        loopctl::presets::ConstrainedProfile::apply(&mut agent).expect("profile applies");
        agent.register_observer(Arc::clone(&observer) as Arc<dyn LoopObserver>);

        let result = agent.run("keep echoing", &RunConfig::default()).await;

        assert!(
            result.is_ok(),
            "a profile-configured loop must survive crossing the threshold: {result:?}"
        );
        assert!(
            observer.events.load(Ordering::SeqCst) >= 1,
            "the small-model profile must compact at the threshold (on_compaction fired {} times)",
            observer.events.load(Ordering::SeqCst)
        );
    }

    #[cfg(feature = "hooks")]
    #[tokio::test]
    async fn pre_compact_hook_veto_reports_measured_estimate() {
        use loopctl::hooks::context::{CompactResult, PreCompactContext};
        use loopctl::hooks::{Hook, HookExecutor};

        struct VetoCompactHook;

        impl Hook for VetoCompactHook {
            fn name(&self) -> &str {
                "veto_compact"
            }

            fn on_pre_compact(&self, _ctx: &PreCompactContext) -> Option<CompactResult> {
                Some(CompactResult::abort("not now"))
            }
        }

        let client = RecordingClient::wrap(
            MockApiClient::new("test-model").with_responses(growing_conversation_script(12)),
        );

        let config = SessionConfig::default()
            .with_context_window(200)
            .with_compact_threshold(50);
        let client_handle = client.clone();
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);
        let mut executor = HookExecutor::new();
        executor.register(Arc::new(VetoCompactHook));
        agent.set_hook_executor(Arc::new(executor));

        let result = agent.run("keep echoing", &RunConfig::default()).await;

        match result {
            Err(LoopError::ContextExceeded { used, .. }) => {
                assert!(
                    used > 0,
                    "the vetoed pass must report a measured estimate, got {used}"
                );
            }
            other => panic!(
                "a vetoed compaction over the threshold must fail with ContextExceeded, got {other:?}"
            ),
        }
        for tokens in client_handle.served_request_tokens() {
            assert!(
                tokens <= 200,
                "the vetoed pass must not keep sending over-window requests (window 200)"
            );
        }
    }

    #[tokio::test]
    async fn host_installed_context_manager_is_unaffected() {
        struct ShrinkingCompactor {
            ran: Arc<AtomicBool>,
        }

        impl ContextCompactor for ShrinkingCompactor {
            fn compact(
                &self,
                messages: Vec<Message>,
                _target_tokens: u64,
                context: CompactionContext,
            ) -> Pin<Box<dyn Future<Output = CompactionOutcome> + Send + '_>> {
                let ran = Arc::clone(&self.ran);
                Box::pin(async move {
                    ran.store(true, Ordering::SeqCst);
                    let kept: Vec<Message> = messages.last().cloned().into_iter().collect();
                    let tokens_after = context.counter.count(&kept);
                    CompactionOutcome {
                        tokens_saved: context.tokens_before.saturating_sub(tokens_after),
                        messages: kept,
                        tokens_after,
                        success: true,
                        error: None,
                    }
                })
            }
        }

        let script = growing_conversation_script(12);
        let client = RecordingClient::wrap(MockApiClient::new("test-model").with_responses(script));

        let ran = Arc::new(AtomicBool::new(false));
        let manager = ContextManager::new(Arc::new(ShrinkingCompactor {
            ran: Arc::clone(&ran),
        }));

        let config = SessionConfig::default()
            .with_context_window(200)
            .with_compact_threshold(50);
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);
        agent.set_context_manager(Arc::new(manager));

        let result = agent.run("keep echoing", &RunConfig::default()).await;

        assert!(
            result.is_ok(),
            "a run with a host-installed manager must complete: {result:?}"
        );
        assert!(
            ran.load(Ordering::SeqCst),
            "the host-installed compactor must serve the compaction, not the constructor default"
        );
        let conversation = agent.conversation();
        assert!(
            conversation.len() < 10,
            "the host compactor's aggressive shape must survive; got {} messages",
            conversation.len()
        );
        assert!(
            !conversation.iter().any(|m| {
                m.parts.iter().any(|p| {
                    matches!(
                        p,
                        loopctl::message::MessagePart::Text { text } if text.contains("keep echoing")
                    )
                })
            }),
            "the default truncator always preserves the first message; its absence proves the host compactor served the pass"
        );
    }
}

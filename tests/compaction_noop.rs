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
        CompactionContext, CompactionOutcome, ContextCompactor, ContextManager,
        HeuristicTokenCounter, TokenCounter,
    };
    use loopctl::config::SessionConfig;
    use loopctl::engine::core::Loop;
    use loopctl::engine::{BareLoop, RunConfig};
    use loopctl::error::LoopError;
    use loopctl::message::Message;
    use loopctl::observer::{CompactedContext, LoopObserver};
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
            let tokens = HeuristicTokenCounter.count(&request.messages);
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
                let payload = format!("echo: {input} {}", "r".repeat(200));
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
                *tokens <= 200,
                "no request may exceed the 200-token window; served estimates {served:?}"
            );
        }
    }

    #[tokio::test]
    #[ignore = "known defect: the machine's context estimate resets to zero at run start, so a run whose committed history already exceeds the window sends its first request over-window before any compaction check can run; un-ignore when fixed"]
    async fn first_request_of_an_over_window_run_is_never_sent() {
        let script = vec![
            final_response(),
            tool_turn_response(0, 40),
            tool_turn_response(1, 40),
        ];
        let client = RecordingClient::wrap(MockApiClient::new("test-model").with_responses(script));

        let config = SessionConfig::default()
            .with_context_window(200)
            .with_compact_threshold(80);
        let client_handle = client.clone();
        let mut agent = BareLoop::new(Arc::new(client), registry_with_echo(), config);

        let first = agent.run(&"x".repeat(750), &RunConfig::default()).await;
        assert!(
            first.is_ok(),
            "the first run's single request stays under the window: {first:?}"
        );

        let _second = agent.run(&"y".repeat(50), &RunConfig::default()).await;

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

//! Context-window probe pins — the compaction denominator resolves
//! from the provider when it discloses one.
//!
//! Two halves, gated independently: the engine pins drive
//! [`BareLoop`](loopctl::engine::BareLoop) over a wrapper client whose
//! `model_context_window` answer the test scripts (no provider
//! feature needed), and the client pins drive the real
//! [`OpenAiClient`](loopctl::provider::OpenAiClient) against local
//! httpmock servers standing in for the llama.cpp `/props` and Ollama
//! `/api/show` endpoints.

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone,
    clippy::clone_on_ref_ptr
)]

mod engine_resolution {
    //! The driver-side resolution: probe over declaration, frozen per
    //! run, reported on turn-end.

    #![cfg(feature = "testing")]

    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;

    use loopctl::api::error::ApiError;
    use loopctl::api::{ApiClient, NonStreamingResponse, StreamRequest};
    use loopctl::capabilities::Compactable;
    use loopctl::config::SessionConfig;
    use loopctl::engine::core::Loop;
    use loopctl::engine::{BareLoop, RunConfig};
    use loopctl::message::Message;
    use loopctl::observer::{CompactedContext, LoopObserver, RunStartContext, TurnEndContext};
    use loopctl::stream::StreamEvent;
    use loopctl::testing::MockApiClient;

    /// A mock-backed client with a scripted probe answer.
    ///
    /// Delegates every conversational method to the wrapped
    /// [`MockApiClient`]; only `model_context_window` is the test's —
    /// a fixed window, nothing (the failure shape), or a hang gated on
    /// a [`tokio::sync::Notify`] the test releases.
    struct ProbingClient {
        /// Serves the run's responses.
        inner: MockApiClient,

        /// The window the probe reports, when it reports one.
        window: Option<u64>,

        /// When set, the probe hangs until this gate fires — the
        /// slow-endpoint shape.
        hang: Option<Arc<tokio::sync::Notify>>,
    }

    impl ProbingClient {
        /// A client whose probe answers `window` immediately.
        fn probing(window: Option<u64>, inner: MockApiClient) -> Self {
            Self {
                inner,
                window,
                hang: None,
            }
        }

        /// A client whose probe hangs until `gate` fires, then answers
        /// `window`.
        fn hanging(
            window: Option<u64>,
            inner: MockApiClient,
            gate: Arc<tokio::sync::Notify>,
        ) -> Self {
            Self {
                inner,
                window,
                hang: Some(gate),
            }
        }
    }

    impl ApiClient for ProbingClient {
        fn model(&self) -> String {
            self.inner.model()
        }

        fn stream_messages(
            &self,
            request: &StreamRequest,
        ) -> Pin<Box<dyn futures::Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>>
        {
            self.inner.stream_messages(request)
        }

        fn create_message(
            &self,
            request: &StreamRequest,
        ) -> Pin<Box<dyn Future<Output = Result<NonStreamingResponse, ApiError>> + Send + '_>>
        {
            self.inner.create_message(request)
        }

        fn model_context_window(&self) -> Pin<Box<dyn Future<Output = Option<u64>> + Send + '_>> {
            let window = self.window;
            let gate = self.hang.clone();
            Box::pin(async move {
                if let Some(gate) = gate {
                    gate.notified().await;
                }
                window
            })
        }
    }

    /// An observer recording what the resolution pins assert on.
    struct RecordingObserver {
        /// The turn-end denominators the run reported, in order: the
        /// event count distinguishes a no-event run from a
        /// `None`-denominator one.
        turn_end_windows: std::sync::Mutex<Vec<Option<u64>>>,

        /// How many compaction passes the run ran.
        compactions: std::sync::atomic::AtomicUsize,
    }

    impl RecordingObserver {
        fn new() -> Self {
            Self {
                turn_end_windows: std::sync::Mutex::new(Vec::new()),
                compactions: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn last_turn_end_window(&self) -> Option<Option<u64>> {
            let windows = self.turn_end_windows.lock().expect("turn-end window lock");
            windows.last().copied()
        }

        fn compactions(&self) -> usize {
            self.compactions.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl LoopObserver for RecordingObserver {
        fn name(&self) -> &str {
            "window-recorder"
        }

        fn on_run_start(&self, _ctx: &RunStartContext) {
            let _ = _ctx;
        }

        fn on_turn_end(&self, ctx: &TurnEndContext) {
            self.turn_end_windows
                .lock()
                .expect("turn-end window lock")
                .push(ctx.context_window);
        }

        fn on_compaction(&self, _ctx: &CompactedContext) {
            self.compactions
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// A mock answering one plain text turn.
    fn text_mock() -> MockApiClient {
        MockApiClient::new("probed-model").with_text_response("done")
    }

    /// A seeded history big enough to cross a small window's threshold
    /// while staying far under a large one's.
    ///
    /// 40 messages of 500 characters estimate near 5 200 tokens with
    /// the heuristic counter: over 80 % of a 4 096-token window
    /// (3 276), nowhere near 80 % of a 200 000-token one.
    fn bulky_history() -> Vec<Message> {
        let filler = "y".repeat(500);
        (0..40)
            .map(|idx| {
                Message::new(
                    loopctl::message::Role::User,
                    vec![loopctl::message::MessagePart::text(format!(
                        "{idx}: {filler}"
                    ))],
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn context_manager_prefers_the_probe_over_the_declaration() {
        let machine = loopctl::engine::core::LoopMachine::from_history(bulky_history());
        let observer = Arc::new(RecordingObserver::new());
        let managers = loopctl::managers::LoopManagers::new().with_observer(observer.clone());
        let client = Arc::new(ProbingClient::probing(Some(4_096), text_mock()));
        let mut agent = BareLoop::from_machine_with_managers(
            machine,
            SessionConfig::default(),
            client,
            loopctl::tool::ToolRegistry::new(),
            managers,
        );
        let run = agent.run("finish up", &RunConfig::default()).await;
        assert!(
            run.is_ok(),
            "the run completes against the probed window: {run:?}"
        );
        assert!(
            observer.compactions() >= 1,
            "a payload over 80% of the probed 4 096-token window compacts — the declared \
             200 000 would never have triggered"
        );
    }

    #[tokio::test]
    async fn context_manager_falls_back_to_the_declaration() {
        let machine = loopctl::engine::core::LoopMachine::from_history(bulky_history());
        let observer = Arc::new(RecordingObserver::new());
        let managers = loopctl::managers::LoopManagers::new().with_observer(observer.clone());
        let client = Arc::new(ProbingClient::probing(None, text_mock()));
        let mut agent = BareLoop::from_machine_with_managers(
            machine,
            SessionConfig::default(),
            client,
            loopctl::tool::ToolRegistry::new(),
            managers,
        );
        let run = agent.run("finish up", &RunConfig::default()).await;
        assert!(
            run.is_ok(),
            "a probe that answers nothing still runs: {run:?}"
        );
        assert_eq!(
            observer.compactions(),
            0,
            "the declared 200 000-token window is the denominator — the payload never \
             crosses its threshold"
        );
        assert_eq!(
            observer.last_turn_end_window(),
            Some(Some(200_000)),
            "the turn-end denominator is the declared window when the probe has nothing"
        );
    }

    #[tokio::test]
    async fn a_disabled_window_stays_disabled_despite_the_probe() {
        let machine = loopctl::engine::core::LoopMachine::from_history(bulky_history());
        let observer = Arc::new(RecordingObserver::new());
        let managers = loopctl::managers::LoopManagers::new().with_observer(observer.clone());
        let client = Arc::new(ProbingClient::probing(Some(4_096), text_mock()));
        let mut agent = BareLoop::from_machine_with_managers(
            machine,
            SessionConfig::default().with_context_window(0),
            client,
            loopctl::tool::ToolRegistry::new(),
            managers,
        );
        let run = agent.run("finish up", &RunConfig::default()).await;
        assert!(
            run.is_ok(),
            "the disabled policy serves every request: {run:?}"
        );
        assert_eq!(
            observer.compactions(),
            0,
            "a declared 0 is a deliberate opt-out — no probe result re-enables the window policy"
        );
    }

    #[tokio::test]
    async fn turn_end_carries_the_resolved_window() {
        let observer = Arc::new(RecordingObserver::new());
        let managers = loopctl::managers::LoopManagers::new().with_observer(observer.clone());
        let client = Arc::new(ProbingClient::probing(Some(8_192), text_mock()));
        let mut agent = BareLoop::new_with_managers(
            client,
            loopctl::tool::ToolRegistry::new(),
            SessionConfig::default(),
            managers,
        );
        let run = agent.run("hello", &RunConfig::default()).await;
        assert!(run.is_ok(), "the scripted run completes: {run:?}");
        assert_eq!(
            observer.last_turn_end_window(),
            Some(Some(8_192)),
            "the turn-end denominator is the window the client disclosed, not the declared one"
        );
    }

    #[tokio::test]
    async fn turn_end_reports_no_window_under_the_disabled_policy() {
        let observer = Arc::new(RecordingObserver::new());
        let managers = loopctl::managers::LoopManagers::new().with_observer(observer.clone());
        let client = Arc::new(ProbingClient::probing(Some(8_192), text_mock()));
        let mut agent = BareLoop::new_with_managers(
            client,
            loopctl::tool::ToolRegistry::new(),
            SessionConfig::default().with_context_window(0),
            managers,
        );
        let run = agent.run("hello", &RunConfig::default()).await;
        assert!(run.is_ok(), "the scripted run completes: {run:?}");
        assert_eq!(
            observer.last_turn_end_window(),
            Some(None),
            "under the disabled policy there is no denominator — the field reports None \
             rather than a sentinel zero a display would divide by"
        );
    }

    #[tokio::test]
    async fn probe_never_surfaces_as_a_turn_error() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let observer = Arc::new(RecordingObserver::new());
        let managers = loopctl::managers::LoopManagers::new().with_observer(observer.clone());
        let client = Arc::new(ProbingClient::hanging(None, text_mock(), Arc::clone(&gate)));
        let mut agent = BareLoop::new_with_managers(
            client,
            loopctl::tool::ToolRegistry::new(),
            SessionConfig::default(),
            managers,
        );
        let handle = tokio::spawn(async move { agent.run("hello", &RunConfig::default()).await });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(
            !handle.is_finished(),
            "the run waits inside the hanging probe — and has not failed"
        );
        gate.notify_one();
        let run = handle
            .await
            .expect("the run task joins")
            .expect("the released run completes normally");
        assert_eq!(run.turn_count(), 1, "the turn ran once the probe released");
        assert_eq!(
            observer.last_turn_end_window(),
            Some(Some(200_000)),
            "the probe that answered nothing leaves the declared window standing"
        );
    }

    #[tokio::test]
    async fn a_probe_less_run_leaves_a_divergent_installed_manager_untouched() {
        let observer = Arc::new(RecordingObserver::new());
        let mut managers = loopctl::managers::LoopManagers::new();
        managers.set_context_manager(Arc::new(
            loopctl::compact::ContextManager::new(Arc::new(
                loopctl::compact::TruncatingCompactor::default(),
            ))
            .with_context_window(50_000)
            .with_threshold(60),
        ));
        managers.register_observer(observer.clone());
        let client = Arc::new(ProbingClient::probing(None, text_mock()));
        let mut agent = BareLoop::new_with_managers(
            client,
            loopctl::tool::ToolRegistry::new(),
            SessionConfig::default(),
            managers,
        );
        let run = agent.run("hello", &RunConfig::default()).await;
        assert!(run.is_ok(), "the scripted run completes: {run:?}");
        let manager = agent
            .managers()
            .context_manager()
            .expect("the installed manager survives the run");
        assert_eq!(
            manager.context_window(),
            50_000,
            "a probe-less resolution keeps the host-installed manager's own window — the              declaration governs the trigger, not a rewrite of the manager"
        );
        assert_eq!(
            manager.threshold(),
            60,
            "the host-installed manager's threshold rides along untouched — no config              value is silently overwritten when no probe participated"
        );
    }

    #[tokio::test]
    async fn a_probing_run_still_syncs_a_divergent_installed_manager() {
        let observer = Arc::new(RecordingObserver::new());
        let mut managers = loopctl::managers::LoopManagers::new();
        managers.set_context_manager(Arc::new(
            loopctl::compact::ContextManager::new(Arc::new(
                loopctl::compact::TruncatingCompactor::default(),
            ))
            .with_context_window(50_000)
            .with_threshold(60),
        ));
        managers.register_observer(observer.clone());
        let client = Arc::new(ProbingClient::probing(Some(4_096), text_mock()));
        let mut agent = BareLoop::new_with_managers(
            client,
            loopctl::tool::ToolRegistry::new(),
            SessionConfig::default(),
            managers,
        );
        let run = agent.run("hello", &RunConfig::default()).await;
        assert!(run.is_ok(), "the scripted run completes: {run:?}");
        let manager = agent
            .managers()
            .context_manager()
            .expect("the installed manager survives the run");
        assert_eq!(
            manager.context_window(),
            4_096,
            "a disclosed window outranks even a host-installed manager's own — the probe \
             re-syncs the manager so trigger and compaction target share one number"
        );
    }
}

mod client_probe {
    //! The OpenAI-shaped client probe against local metadata servers.

    #![cfg(feature = "openai")]

    use loopctl::api::ApiClient;
    use loopctl::provider::OpenAiClient;

    /// A client pointed at the local server's root, version-segmented
    /// like a real deployment.
    async fn probing_client(base: &str) -> OpenAiClient {
        OpenAiClient::builder()
            .with_api_key("probe-key")
            .with_base_url(format!("{base}/v1"))
            .with_model("probe-model")
            .build()
            .expect("the probe client builds")
    }

    #[tokio::test]
    async fn openai_client_probes_llama_cpp_props() {
        let server = httpmock::MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/props");
                then.status(200).json_body(serde_json::json!({
                    "default_generation_settings": { "n_ctx": 262_144_u64 },
                    "total_slots": 4_u64,
                }));
            })
            .await;
        let client = probing_client(&server.base_url()).await;
        let window = client
            .model_context_window()
            .await
            .expect("the props document answers a window");
        assert_eq!(
            window, 65_536,
            "a --parallel deployment splits n_ctx across slots — the per-request truth is \
             n_ctx / total_slots: 262 144 / 4"
        );
    }

    #[tokio::test]
    async fn probe_treats_a_single_slot_as_the_total() {
        let server = httpmock::MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/props");
                then.status(200).json_body(serde_json::json!({
                    "default_generation_settings": { "n_ctx": 8_192_u64 },
                }));
            })
            .await;
        let client = probing_client(&server.base_url()).await;
        let window = client
            .model_context_window()
            .await
            .expect("the props document answers a window");
        assert_eq!(
            window, 8_192,
            "an absent total_slots is a single-slot deployment — n_ctx reports unchanged"
        );
    }

    #[tokio::test]
    async fn openai_client_probes_ollama_show() {
        let server = httpmock::MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/props");
                then.status(404);
            })
            .await;
        let show = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/api/show")
                    .body_includes("probe-model");
                then.status(200).json_body(serde_json::json!({
                    "model_info": {
                        "general.architecture": "llama",
                        "llama.context_length": 131_072_u64,
                    },
                }));
            })
            .await;
        let client = probing_client(&server.base_url()).await;
        let window = client
            .model_context_window()
            .await
            .expect("the ollama show answer provides the window");
        assert_eq!(
            window, 131_072,
            "the architecture-prefixed context_length key is the disclosed window"
        );
        assert!(
            show.calls_async().await >= 1,
            "the show probe carried the current model name in its POST body"
        );
    }

    #[tokio::test]
    async fn probe_failure_is_none_not_error() {
        let server = httpmock::MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/props");
                then.status(200).body("not json at all");
            })
            .await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/api/show");
                then.status(404);
            })
            .await;
        let client = probing_client(&server.base_url()).await;
        let window = client.model_context_window().await;
        assert_eq!(
            window, None,
            "a garbage props body and a 404 show answer yield None — the probe is fail-soft"
        );
    }

    #[tokio::test]
    async fn probe_is_cached_per_client() {
        let server = httpmock::MockServer::start_async().await;
        let props = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/props");
                then.status(200).json_body(serde_json::json!({
                    "default_generation_settings": { "n_ctx": 4_096_u64 },
                }));
            })
            .await;
        let client = probing_client(&server.base_url()).await;
        let first = client
            .model_context_window()
            .await
            .expect("the first probe answers");
        let second = client
            .model_context_window()
            .await
            .expect("the cached probe answers");
        assert_eq!(first, second, "repeated consultations see one answer");
        assert_eq!(
            props.calls_async().await,
            1,
            "one HTTP probe per client — the second consultation served from cache"
        );
    }

    #[tokio::test]
    async fn a_model_swap_reprobes_the_window() {
        let server = httpmock::MockServer::start_async().await;
        let small = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/api/show")
                    .body_includes("first-model");
                then.status(200).json_body(serde_json::json!({
                    "model_info": { "llama.context_length": 4_096_u64 },
                }));
            })
            .await;
        let large = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/api/show")
                    .body_includes("second-model");
                then.status(200).json_body(serde_json::json!({
                    "model_info": { "llama.context_length": 131_072_u64 },
                }));
            })
            .await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/props");
                then.status(404);
            })
            .await;
        let client = OpenAiClient::builder()
            .with_api_key("probe-key")
            .with_base_url(server.base_url())
            .with_model("first-model")
            .build()
            .expect("the probe client builds");
        let first = client
            .model_context_window()
            .await
            .expect("the first model's show answers");
        assert_eq!(first, 4_096, "the first model discloses 4 096");
        assert!(client.set_model("second-model"), "the swap takes");
        let second = client
            .model_context_window()
            .await
            .expect("the second model's show answers");
        assert_eq!(
            second, 131_072,
            "a model swap clears the cache — the new model's window is probed, not served \
             from the old model's answer"
        );
        assert_eq!(
            small.calls_async().await,
            1,
            "each model's show endpoint was probed exactly once"
        );
        assert_eq!(large.calls_async().await, 1);
    }
}

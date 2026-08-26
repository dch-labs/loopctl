//! Fallback model switching: the tripped breaker routes requests.
//!
//! Run: `cargo test --all-features --test fallback_switch -- --nocapture`
//!
//! Requires the `streaming` feature: the scripted model responses drive
//! the streaming engine path, and without it every run fails before the
//! fallback machinery is consulted.

#![cfg(feature = "streaming")]
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::missing_panics_doc
)]

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::Stream;
use loopctl::api::error::ApiError;
use loopctl::api::{ApiClient, StreamRequest};
use loopctl::config::SessionConfig;
use loopctl::engine::core::Loop;
use loopctl::engine::{BareLoop, RunConfig};
use loopctl::fallback::FallbackManager;
use loopctl::managers::LoopManagers;
use loopctl::stream::{
    DeltaPart, IndexedDelta, MessageDelta, MessageDeltaPayload, MessageMetadata, MessageStart,
    PartStart, StreamEvent, Usage,
};
use loopctl::tool::ToolRegistry;

/// One scripted turn outcome.
enum Step {
    /// Permanent auth error: not retried, so one request equals one turn.
    AuthFail,
    /// Rate limit: escalates on the first 429 (the breaker's retrip kind).
    RateLimit,
    /// A successful single-text response.
    Text(String),
}

/// A client that records the per-request model override and replays a script.
struct ScriptedClient {
    requests: Arc<Mutex<Vec<Option<String>>>>,
    script: Mutex<Vec<Step>>,
}

impl ScriptedClient {
    fn new(script: Vec<Step>) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            script: Mutex::new(script),
        }
    }
}

impl ApiClient for ScriptedClient {
    fn model(&self) -> String {
        "primary-model".to_string()
    }

    fn stream_messages(
        &self,
        _request: &StreamRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        Box::pin(futures::stream::empty())
    }

    fn create_message(
        &self,
        _request: &StreamRequest,
    ) -> Pin<
        Box<dyn Future<Output = Result<loopctl::api::NonStreamingResponse, ApiError>> + Send + '_>,
    > {
        Box::pin(async { Err(ApiError::api("these tests drive the streaming path")) })
    }

    fn stream_messages_with_options(
        &self,
        _request: &StreamRequest,
        options: loopctl::structured::RequestOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        self.requests.lock().unwrap().push(options.model);
        let step = self.script.lock().unwrap().remove(0);
        let events = match step {
            Step::AuthFail => {
                return Box::pin(futures::stream::iter(vec![Err(
                    ApiError::auth_invalid_key("scripted permanent failure"),
                )]));
            }
            Step::RateLimit => {
                return Box::pin(futures::stream::iter(vec![Err(ApiError::RateLimit {
                    retry_after: None,
                    message: "scripted rate limit".into(),
                })]));
            }
            Step::Text(text) => text_events(&text),
        };
        Box::pin(futures::stream::iter(events))
    }
}

fn text_events(text: &str) -> Vec<Result<StreamEvent, ApiError>> {
    vec![
        Ok(StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg_1".into(),
                role: "assistant".into(),
                model: "served".into(),
            },
        })),
        Ok(StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(loopctl::message::MessagePart::text("")),
        })),
        Ok(StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text { text: text.into() },
        })),
        Ok(StreamEvent::PartStop { index: Some(0) }),
        Ok(StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("end_turn".into()),
            },
            usage: Some(Usage::new(1, 1)),
        })),
        Ok(StreamEvent::MessageStop),
    ]
}

fn make_agent(client: ScriptedClient, manager: FallbackManager) -> BareLoop<ScriptedClient> {
    let managers = LoopManagers::new().with_fallback(manager);
    BareLoop::new_with_managers(
        Arc::new(client),
        ToolRegistry::new(),
        SessionConfig::default(),
        managers,
    )
}

#[tokio::test]
async fn tripped_breaker_routes_subsequent_requests_to_fallback_model() {
    let manager = FallbackManager::new(1, 1);
    manager.set_original_model("primary-model".to_string());
    manager.set_fallback_model("fallback-model");

    let client = ScriptedClient::new(vec![Step::AuthFail, Step::Text("recovered".into())]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);

    let first = agent.run("q", &RunConfig::default()).await;
    assert!(
        first.is_err(),
        "the scripted permanent failure fails the run"
    );

    let second = agent.run("q", &RunConfig::default()).await;
    assert!(
        second.is_ok(),
        "the fallback model serves the next run: {second:?}"
    );

    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![
            Some("primary-model".to_string()),
            Some("fallback-model".to_string())
        ],
        "the request after the trip must carry the fallback model override"
    );
    let message = second.unwrap().turns.last().unwrap().output.clone();
    assert!(
        message.contains("recovered"),
        "the fallback-served response text must survive: {message:?}"
    );
}

#[tokio::test]
async fn recovery_probe_returns_to_primary() {
    let config = loopctl::fallback::FallbackConfig {
        trip_threshold: 1,
        recovery_timeout: std::time::Duration::from_millis(50),
        ..loopctl::fallback::FallbackConfig::default()
    };
    let manager = FallbackManager::new_with_config(config);
    manager.set_original_model("primary-model".to_string());
    manager.set_fallback_model("fallback-model");

    let client = ScriptedClient::new(vec![
        Step::AuthFail,
        Step::Text("on fallback".into()),
        Step::Text("back on primary".into()),
    ]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);

    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(
        agent.run("q", &RunConfig::default()).await.is_ok(),
        "inside the cooldown the fallback model serves"
    );
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;
    assert!(
        agent.run("q", &RunConfig::default()).await.is_ok(),
        "after the cooldown the probe returns to the primary"
    );

    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![
            Some("primary-model".to_string()),
            Some("fallback-model".to_string()),
            Some("primary-model".to_string()),
        ],
        "the trip serves the fallback model; the first request after the cooldown is the primary again"
    );
}

#[tokio::test]
async fn failure_while_on_fallback_advances_the_chain() {
    let manager = FallbackManager::new(1, 1);
    manager.set_original_model("primary-model".to_string());
    manager.set_fallback_model("fallback-1");
    manager.add_fallback_model("fallback-2");

    // fallback-1 needs max_fail_count (2) failing turns before the chain
    // advances, so the script fails twice on it before fallback-2 serves.
    let client = ScriptedClient::new(vec![
        Step::AuthFail,
        Step::AuthFail,
        Step::AuthFail,
        Step::Text("served by fallback-2".into()),
    ]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);

    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(agent.run("q", &RunConfig::default()).await.is_ok());

    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![
            Some("primary-model".to_string()),
            Some("fallback-1".to_string()),
            Some("fallback-1".to_string()),
            Some("fallback-2".to_string()),
        ],
        "two failures while on fallback-1 must advance the chain to fallback-2"
    );
}

#[tokio::test]
async fn model_switch_observer_fires_on_trip_and_recovery() {
    use loopctl::observer::{LoopObserver, ModelSwitchedContext};
    use std::sync::Mutex as StdMutex;

    struct SwitchRecorder {
        switches: Arc<StdMutex<Vec<(String, String)>>>,
    }

    impl LoopObserver for SwitchRecorder {
        fn name(&self) -> &'static str {
            "switch-recorder"
        }
        fn on_model_switched(&self, ctx: &ModelSwitchedContext) {
            self.switches
                .lock()
                .unwrap()
                .push((ctx.from.clone(), ctx.to.clone()));
        }
    }

    let config = loopctl::fallback::FallbackConfig {
        trip_threshold: 1,
        recovery_timeout: std::time::Duration::from_millis(50),
        ..loopctl::fallback::FallbackConfig::default()
    };
    let manager = FallbackManager::new_with_config(config);
    manager.set_original_model("primary-model".to_string());
    manager.set_fallback_model("fallback-model");

    let client = ScriptedClient::new(vec![
        Step::AuthFail,
        Step::Text("on fallback".into()),
        Step::Text("back on primary".into()),
    ]);
    let mut agent = make_agent(client, manager);

    let switches = Arc::new(StdMutex::new(Vec::new()));
    agent.register_observer(Arc::new(SwitchRecorder {
        switches: Arc::clone(&switches),
    }));

    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(agent.run("q", &RunConfig::default()).await.is_ok());
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;
    assert!(agent.run("q", &RunConfig::default()).await.is_ok());

    let switches = switches.lock().unwrap().clone();
    assert_eq!(
        switches,
        vec![
            ("primary-model".to_string(), "fallback-model".to_string()),
            ("fallback-model".to_string(), "primary-model".to_string()),
        ],
        "one switch signal per model change: trip to fallback, then the recovery probe back"
    );
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn host_model_override_yields_to_the_configured_chain() {
    let config = loopctl::fallback::FallbackConfig {
        trip_threshold: 1,
        ..loopctl::fallback::FallbackConfig::default()
    };
    let manager = FallbackManager::new_with_config(config);
    manager.set_original_model("primary-model".to_string());
    manager.set_fallback_model("fallback-1");

    let client = ScriptedClient::new(vec![
        Step::Text("served by the primary".into()),
        Step::AuthFail,
        Step::Text("served by the fallback".into()),
    ]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);
    agent.set_request_options(loopctl::structured::RequestOptions::new().with_model("host-pick"));

    assert!(agent.run("q", &RunConfig::default()).await.is_ok());
    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(agent.run("q", &RunConfig::default()).await.is_ok());

    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![
            Some("primary-model".to_string()),
            Some("primary-model".to_string()),
            Some("fallback-1".to_string()),
        ],
        "a configured manager's resolution is exclusive: the host override never reaches the wire"
    );
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn host_model_override_stands_without_a_manager() {
    let client = ScriptedClient::new(vec![Step::Text("served by the host pick".into())]);
    let requests = Arc::clone(&client.requests);
    let mut agent = BareLoop::new(
        Arc::new(client),
        ToolRegistry::new(),
        SessionConfig::default(),
    );
    agent.set_request_options(loopctl::structured::RequestOptions::new().with_model("host-pick"));

    assert!(agent.run("q", &RunConfig::default()).await.is_ok());
    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![Some("host-pick".to_string())],
        "without a configured manager the host's per-request model is honored"
    );
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn stream_observers_report_the_model_that_served_the_turn() {
    use loopctl::observer::{LoopObserver, StreamContext, StreamFailureContext};
    use std::sync::Mutex as StdMutex;

    struct StreamRecorder {
        successes: Arc<StdMutex<Vec<String>>>,
        failures: Arc<StdMutex<Vec<String>>>,
    }

    impl LoopObserver for StreamRecorder {
        fn name(&self) -> &'static str {
            "stream-recorder"
        }
        fn on_stream_success(&self, ctx: &StreamContext) {
            self.successes.lock().unwrap().push(ctx.model.clone());
        }
        fn on_stream_failure(&self, ctx: &StreamFailureContext) {
            self.failures.lock().unwrap().push(ctx.model.clone());
        }
    }

    // Default (1-minute) recovery timeout keeps the whole test inside
    // one fallback stint, so every turn after the trip is served by
    // fallback-1.
    let config = loopctl::fallback::FallbackConfig {
        trip_threshold: 1,
        ..loopctl::fallback::FallbackConfig::default()
    };
    let manager = FallbackManager::new_with_config(config);
    manager.set_original_model("primary-model".to_string());
    manager.set_fallback_model("fallback-1");

    let client = ScriptedClient::new(vec![
        Step::AuthFail,
        Step::Text("served by the fallback".into()),
        Step::AuthFail,
    ]);
    let mut agent = make_agent(client, manager);

    let successes = Arc::new(StdMutex::new(Vec::new()));
    let failures = Arc::new(StdMutex::new(Vec::new()));
    agent.register_observer(Arc::new(StreamRecorder {
        successes: Arc::clone(&successes),
        failures: Arc::clone(&failures),
    }));

    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(agent.run("q", &RunConfig::default()).await.is_ok());
    assert!(agent.run("q", &RunConfig::default()).await.is_err());

    assert_eq!(
        failures.lock().unwrap().clone(),
        vec!["primary-model".to_string(), "fallback-1".to_string(),],
        "failure contexts name the model that was serving, not the client's static name"
    );
    assert_eq!(
        successes.lock().unwrap().clone(),
        vec!["fallback-1".to_string()],
        "success contexts name the fallback that served the turn"
    );
}

#[tokio::test]
async fn exhausted_chain_fails_the_turn_instead_of_serving_the_primary() {
    let manager = FallbackManager::new(1, 2);
    manager.set_original_model("primary-model".to_string());
    manager.set_fallback_model("fallback-1");
    manager.add_fallback_model("fallback-2");

    // Trip on the primary, then two failing turns each on fallback-1 and
    // fallback-2 to exhaust them.
    let client = ScriptedClient::new(vec![
        Step::AuthFail,
        Step::AuthFail,
        Step::AuthFail,
        Step::AuthFail,
        Step::AuthFail,
        Step::Text("must not be served".into()),
    ]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);

    for _ in 0..5 {
        assert!(agent.run("q", &RunConfig::default()).await.is_err());
    }
    let sixth = agent.run("q", &RunConfig::default()).await;
    assert!(
        matches!(sixth, Err(loopctl::error::LoopError::FallbackExhausted)),
        "an exhausted chain must fail the turn with a typed error, got {sixth:?}"
    );
    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models.len(),
        5,
        "no request may be sent after the chain is exhausted: {models:?}"
    );
    assert_eq!(
        models.last(),
        Some(&Some("fallback-2".to_string())),
        "the last served request was the final chain entry"
    );
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn exhausted_chain_recovers_to_primary_after_cooldown() {
    let config = loopctl::fallback::FallbackConfig {
        trip_threshold: 1,
        recovery_timeout: std::time::Duration::from_millis(500),
        recovery_successes_needed: 1,
        ..loopctl::fallback::FallbackConfig::default()
    };
    let manager = FallbackManager::new_with_config(config);
    manager.set_original_model("primary-model".to_string());
    manager.set_fallback_model("fallback-1");

    let client = ScriptedClient::new(vec![
        Step::AuthFail,
        Step::AuthFail,
        Step::AuthFail,
        Step::Text("primary is back".into()),
    ]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);

    // Trip the primary, then two failures on fallback-1 (the default
    // `max_fail_count` of 2) to take it out of rotation.
    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(agent.run("q", &RunConfig::default()).await.is_err());

    // Within the cooldown the exhausted chain fails the turn, typed,
    // without sending a request.
    let third = agent.run("q", &RunConfig::default()).await;
    assert!(
        matches!(third, Err(loopctl::error::LoopError::FallbackExhausted)),
        "within the cooldown an exhausted chain fails the turn: {third:?}"
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        3,
        "no request may be sent while exhausted inside the cooldown"
    );

    tokio::time::sleep(std::time::Duration::from_millis(550)).await;

    let fourth = agent.run("q", &RunConfig::default()).await;
    assert!(
        fourth.is_ok(),
        "after the cooldown the exhausted chain still gets its primary probe"
    );
    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![
            Some("primary-model".to_string()),
            Some("fallback-1".to_string()),
            Some("fallback-1".to_string()),
            Some("primary-model".to_string()),
        ],
        "trip → fallback fails twice (exhausting the chain) → primary probe after cooldown"
    );
}

#[tokio::test]
#[cfg(feature = "streaming")]
async fn probe_failure_retrips_to_the_fallback_model() {
    let config = loopctl::fallback::FallbackConfig {
        trip_threshold: 1,
        recovery_timeout: std::time::Duration::from_millis(50),
        ..loopctl::fallback::FallbackConfig::default()
    };
    let manager = FallbackManager::new_with_config(config);
    manager.set_original_model("primary-model".to_string());
    manager.set_fallback_model("fallback-model");

    // The retrip arm fires on a rate-limit escalation, which only the
    // escalation-armed handler produces (fallback_after_retries = 0) — the
    // pairing the rate-limit escalation path deploys with the breaker.
    let handler = loopctl::stream::handler::StreamHandler::new().with_rate_limit_config(
        loopctl::stream::handler::RateLimitConfig {
            fallback_after_retries: 0,
            default_delay: std::time::Duration::from_millis(1),
            max_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        },
    );
    let managers = LoopManagers::new()
        .with_fallback(manager)
        .with_stream_handler(handler);

    let client = ScriptedClient::new(vec![
        Step::AuthFail,
        Step::Text("on fallback".into()),
        Step::RateLimit,
        Step::Text("on fallback again".into()),
    ]);
    let requests = Arc::clone(&client.requests);
    let mut agent = BareLoop::new_with_managers(
        Arc::new(client),
        ToolRegistry::new(),
        SessionConfig::default(),
        managers,
    );

    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(agent.run("q", &RunConfig::default()).await.is_ok());
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;
    assert!(
        agent.run("q", &RunConfig::default()).await.is_err(),
        "the rate-limited probe fails the turn"
    );

    assert!(
        agent.run("q", &RunConfig::default()).await.is_ok(),
        "the re-tripped breaker serves the fallback model again"
    );

    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![
            Some("primary-model".to_string()),
            Some("fallback-model".to_string()),
            Some("primary-model".to_string()),
            Some("fallback-model".to_string()),
        ],
        "trip → fallback → probe (primary, rate-limited) → re-trip to fallback"
    );
}

#[test]
fn scripted_client_text_events_shape_is_valid() {
    let events = text_events("x");
    assert!(matches!(events.last(), Some(Ok(StreamEvent::MessageStop))));
    assert!(matches!(
        events.first(),
        Some(Ok(StreamEvent::MessageStart(_)))
    ));
}

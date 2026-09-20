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
    model: Mutex<String>,
}

impl ScriptedClient {
    fn new(script: Vec<Step>) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            script: Mutex::new(script),
            model: Mutex::new("primary-model".to_string()),
        }
    }
}

impl ApiClient for ScriptedClient {
    fn model(&self) -> String {
        self.model.lock().unwrap().clone()
    }

    fn set_model(&self, model: &str) -> bool {
        *self.model.lock().unwrap() = model.to_string();
        true
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
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-model").unwrap();

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
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-model").unwrap();

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
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-1").unwrap();
    manager.add_fallback_model("fallback-2").unwrap();

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
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-model").unwrap();

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
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-1").unwrap();

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
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-1").unwrap();

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
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-1").unwrap();
    manager.add_fallback_model("fallback-2").unwrap();

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
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-1").unwrap();

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
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-model").unwrap();

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

#[tokio::test]
async fn transient_probe_failure_retrips_to_the_fallback_model() {
    let config = loopctl::fallback::FallbackConfig {
        trip_threshold: 1,
        recovery_timeout: std::time::Duration::from_millis(50),
        ..loopctl::fallback::FallbackConfig::default()
    };
    let manager = FallbackManager::new_with_config(config);
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-model").unwrap();

    let client = ScriptedClient::new(vec![
        Step::AuthFail,
        Step::Text("on fallback".into()),
        Step::AuthFail,
        Step::Text("on fallback again".into()),
    ]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);

    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(agent.run("q", &RunConfig::default()).await.is_ok());
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;
    assert!(
        agent.run("q", &RunConfig::default()).await.is_err(),
        "the transient-failing probe fails the run"
    );
    assert!(
        agent.run("q", &RunConfig::default()).await.is_ok(),
        "the re-tripped breaker serves the fallback again instead of locking out"
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
        "trip → fallback → transient-failing probe re-trips → fallback again"
    );
}

#[tokio::test]
async fn tripped_empty_chain_fails_the_turn_instead_of_serving_the_primary() {
    let config = loopctl::fallback::FallbackConfig {
        trip_threshold: 1,
        ..loopctl::fallback::FallbackConfig::default()
    };
    let manager = FallbackManager::new_with_config(config);
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();

    let client = ScriptedClient::new(vec![Step::AuthFail, Step::Text("never".into())]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);

    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    let second = agent.run("q", &RunConfig::default()).await;
    assert!(
        matches!(second, Err(loopctl::error::LoopError::FallbackExhausted)),
        "a tripped breaker with no fallback chain refuses to serve the known-bad primary: {second:?}"
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        1,
        "no request may be sent to the primary after the trip"
    );
}

#[tokio::test]
async fn reset_managers_returns_a_tripped_breaker_to_the_primary() {
    let manager = FallbackManager::new(1, 1);
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-model").unwrap();

    let client = ScriptedClient::new(vec![Step::AuthFail, Step::AuthFail]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);

    assert!(
        agent.run("q", &RunConfig::default()).await.is_err(),
        "the first run trips the breaker"
    );
    let second = agent
        .run("q", &RunConfig::default().with_reset_managers(true))
        .await;
    assert!(
        second.is_err(),
        "with no scripted responses left the re-probed primary fails again"
    );
    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![
            Some("primary-model".to_string()),
            Some("primary-model".to_string()),
        ],
        "reset_managers clears the trip: the second run routes to the primary, not the fallback"
    );
}

#[tokio::test]
async fn routing_uses_slash_bearing_model_names_verbatim() {
    // A second primary-name family: provider-style names with slashes
    // and dots ride the routing machinery verbatim — no sanitization,
    // no truncation at a separator.
    let manager = FallbackManager::new(1, 1);
    manager
        .set_original_model("org/infra.primary-2".to_string())
        .unwrap();
    manager
        .set_fallback_model("fallback.vendor/model-b")
        .unwrap();
    let client = ScriptedClient::new(vec![Step::AuthFail, Step::Text("served".into())]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);

    assert!(agent.run("q", &RunConfig::default()).await.is_err());
    assert!(agent.run("q", &RunConfig::default()).await.is_ok());
    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![
            Some("org/infra.primary-2".to_string()),
            Some("fallback.vendor/model-b".to_string()),
        ],
        "slash-bearing names route verbatim in both directions"
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

/// Records every model switch an observer sees.
struct SwitchLog {
    switches: Arc<Mutex<Vec<(String, String)>>>,
}

impl loopctl::observer::LoopObserver for SwitchLog {
    fn name(&self) -> &'static str {
        "switch-log"
    }
    fn on_model_switched(&self, ctx: &loopctl::observer::ModelSwitchedContext) {
        self.switches
            .lock()
            .unwrap()
            .push((ctx.from.clone(), ctx.to.clone()));
    }
}

#[tokio::test]
async fn a_programmatic_switch_fires_the_observer_once_across_apply_and_a_turn() {
    // switch_model().apply() fires on_model_switched itself; the next
    // routed turn must not re-fire the same transition — observers
    // would double-count programmatic switches.
    let manager = FallbackManager::new(1, 1);
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    let client = ScriptedClient::new(vec![
        Step::Text("first".into()),
        Step::Text("second".into()),
    ]);
    let mut agent = make_agent(client, manager);
    let switches = Arc::new(Mutex::new(Vec::new()));
    agent.register_observer(Arc::new(SwitchLog {
        switches: Arc::clone(&switches),
    }));

    agent
        .run("q1", &RunConfig::default())
        .await
        .expect("the first turn serves the primary");
    agent
        .switch_model("replacement-model")
        .apply()
        .expect("the programmatic switch applies");
    agent
        .run("q2", &RunConfig::default())
        .await
        .expect("the next turn serves the replacement");

    let switches = switches.lock().unwrap().clone();
    assert_eq!(
        switches,
        vec![("primary-model".to_string(), "replacement-model".to_string())],
        "exactly one switch signal across apply() and the subsequent turn: {switches:?}"
    );
}

#[tokio::test]
async fn the_count_up_to_a_threshold_three_trip_interleaves_with_real_turns() {
    // threshold 3: two failures stay on the primary, the third trips —
    // the count-up ordering through real engine turns, not a
    // threshold-1 fixture.
    let config = loopctl::fallback::FallbackConfig {
        trip_threshold: 3,
        recovery_timeout: std::time::Duration::from_secs(60),
        ..loopctl::fallback::FallbackConfig::default()
    };
    let manager = FallbackManager::new_with_config(config);
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-model").unwrap();

    let client = ScriptedClient::new(vec![
        Step::AuthFail,
        Step::AuthFail,
        Step::AuthFail,
        Step::Text("served".into()),
    ]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);

    assert!(agent.run("q1", &RunConfig::default()).await.is_err());
    assert!(agent.run("q2", &RunConfig::default()).await.is_err());
    assert!(agent.run("q3", &RunConfig::default()).await.is_err());
    assert!(
        agent.run("q4", &RunConfig::default()).await.is_ok(),
        "after the third failure trips the breaker, the fallback serves"
    );

    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![
            Some("primary-model".to_string()),
            Some("primary-model".to_string()),
            Some("primary-model".to_string()),
            Some("fallback-model".to_string()),
        ],
        "two failures below the threshold keep routing to the primary; the third trips"
    );
}

#[tokio::test]
async fn a_two_success_recovery_requires_both_probes() {
    // recovery_successes_needed 2: the first post-cooldown success on
    // the primary is a probe, not a close — the second completes the
    // recovery.
    let config = loopctl::fallback::FallbackConfig {
        trip_threshold: 1,
        recovery_timeout: std::time::Duration::from_millis(50),
        recovery_successes_needed: 2,
        ..loopctl::fallback::FallbackConfig::default()
    };
    let manager = FallbackManager::new_with_config(config);
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-model").unwrap();

    let client = ScriptedClient::new(vec![
        Step::AuthFail,
        Step::Text("on fallback".into()),
        Step::Text("probe one".into()),
        Step::Text("probe two".into()),
    ]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);

    assert!(agent.run("q1", &RunConfig::default()).await.is_err());
    assert!(
        agent.run("q2", &RunConfig::default()).await.is_ok(),
        "inside the cooldown the fallback serves"
    );
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;
    assert!(
        agent.run("q3", &RunConfig::default()).await.is_ok(),
        "the first probe serves on the primary"
    );
    assert!(
        agent.run("q4", &RunConfig::default()).await.is_ok(),
        "the second probe completes the recovery"
    );

    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![
            Some("primary-model".to_string()),
            Some("fallback-model".to_string()),
            Some("primary-model".to_string()),
            Some("primary-model".to_string()),
        ],
        "recovery needs two primary successes: probe one, then the closing probe"
    );
}

/// Records the full options value of every call and serves text on
/// both transports — the matrix cells the streaming-only pins leave.
struct OptionsClient {
    seen: Arc<Mutex<Vec<loopctl::structured::RequestOptions>>>,
    script: Mutex<Vec<Step>>,
}

impl OptionsClient {
    fn new(script: Vec<Step>) -> Self {
        Self {
            seen: Arc::new(Mutex::new(Vec::new())),
            script: Mutex::new(script),
        }
    }

    fn next_step(&self) -> Step {
        self.script.lock().unwrap().remove(0)
    }
}

impl ApiClient for OptionsClient {
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
        Box::pin(async { Err(ApiError::api("these tests drive the options transports")) })
    }

    fn stream_messages_with_options(
        &self,
        _request: &StreamRequest,
        options: loopctl::structured::RequestOptions,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send + 'static>> {
        self.seen.lock().unwrap().push(options);
        match self.next_step() {
            Step::Text(text) => Box::pin(futures::stream::iter(text_events(&text))),
            _ => Box::pin(futures::stream::iter(vec![Err(ApiError::api(
                "script exhausted",
            ))])),
        }
    }

    fn create_message_with_options(
        &self,
        _request: &StreamRequest,
        options: loopctl::structured::RequestOptions,
    ) -> Pin<
        Box<dyn Future<Output = Result<loopctl::api::NonStreamingResponse, ApiError>> + Send + '_>,
    > {
        self.seen.lock().unwrap().push(options);
        let text = match self.next_step() {
            Step::Text(text) => text,
            _ => "script exhausted".to_string(),
        };
        Box::pin(async move {
            Ok(loopctl::api::NonStreamingResponse {
                message: loopctl::message::Message::new(
                    loopctl::message::Role::Assistant,
                    vec![loopctl::message::MessagePart::text(text)],
                ),
                stop_reason: loopctl::stream::StreamStopReason::EndTurn,
                usage: Some(Usage::new(1, 1)),
            })
        })
    }
}

fn options_agent(
    client: OptionsClient,
    manager: Option<FallbackManager>,
) -> BareLoop<OptionsClient> {
    let managers = match manager {
        Some(manager) => LoopManagers::new().with_fallback(manager),
        None => LoopManagers::new(),
    };
    BareLoop::new_with_managers(
        Arc::new(client),
        ToolRegistry::new(),
        SessionConfig::default(),
        managers,
    )
}

#[tokio::test]
async fn host_override_precedence_holds_on_the_non_streaming_transport() {
    // The two precedence cells the streaming pins cover, driven through
    // the non-streaming transport: a configured manager's resolution
    // wins over the host override; without a manager the override
    // stands.
    let manager = FallbackManager::new(1, 1);
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    let client = OptionsClient::new(vec![Step::Text("served".into())]);
    let seen = Arc::clone(&client.seen);
    let mut agent = options_agent(client, Some(manager));
    agent.set_turn_mode(loopctl::engine::TurnMode::NonStreaming);
    agent.set_request_options(loopctl::structured::RequestOptions::new().with_model("host-model"));
    agent
        .run("q", &RunConfig::default())
        .await
        .expect("run completes");
    assert_eq!(
        seen.lock().unwrap()[0].model.as_deref(),
        Some("primary-model"),
        "configured manager: the chain's resolution replaces the host override"
    );

    let client = OptionsClient::new(vec![Step::Text("served".into())]);
    let seen = Arc::clone(&client.seen);
    let mut agent = options_agent(client, None);
    agent.set_request_options(loopctl::structured::RequestOptions::new().with_model("host-model"));
    agent
        .run("q", &RunConfig::default())
        .await
        .expect("run completes");
    assert_eq!(
        seen.lock().unwrap()[0].model.as_deref(),
        Some("host-model"),
        "unconfigured manager: the host override stands"
    );
}

#[tokio::test]
async fn a_whitespace_model_override_is_ignored_through_the_engine() {
    // The builder's trim-empty guard, driven end to end: a
    // whitespace-only override produces a request with no model
    // override at all.
    let client = OptionsClient::new(vec![Step::Text("served".into())]);
    let seen = Arc::clone(&client.seen);
    let mut agent = options_agent(client, None);
    agent.set_turn_mode(loopctl::engine::TurnMode::NonStreaming);
    agent.set_request_options(loopctl::structured::RequestOptions::new().with_model("   "));
    agent
        .run("q", &RunConfig::default())
        .await
        .expect("run completes");
    assert_eq!(
        seen.lock().unwrap()[0].model,
        None,
        "a whitespace override never reaches the wire as a model"
    );
}

#[tokio::test]
async fn combined_options_travel_to_the_wire_as_one_value() {
    // One options value carrying all three knobs: the override yields
    // to the chain while the format and constraint ride unchanged —
    // the combination no test had pinned.
    let manager = FallbackManager::new(1, 1);
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    let client = OptionsClient::new(vec![Step::Text("served".into())]);
    let seen = Arc::clone(&client.seen);
    let mut agent = options_agent(client, Some(manager));
    agent.set_request_options(
        loopctl::structured::RequestOptions::new()
            .with_model("host-model")
            .with_response_format(loopctl::structured::ResponseFormat::new(
                "combined",
                serde_json::json!({"type": "object"}),
            ))
            .with_tool_constraint(loopctl::structured::ToolConstraint::Strict),
    );
    agent
        .run("q", &RunConfig::default())
        .await
        .expect("run completes");
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0].model.as_deref(), Some("primary-model"));
    assert!(
        seen[0].response_format.is_some(),
        "the format rides alongside the routed model"
    );
    assert!(
        matches!(
            seen[0].tool_constraint,
            loopctl::structured::ToolConstraint::Strict
        ),
        "the constraint rides alongside the routed model"
    );
}

#[tokio::test]
async fn the_full_adversarial_cycle_trips_retrips_exhausts_and_recovers() {
    // One scripted lifetime through every breaker transition the
    // engine drives: trip to the fallback, fallback service, a probe
    // that closes to the primary, a second trip, the fallback itself
    // dying to exhaustion, and the post-cooldown probe that returns
    // the primary — the wire model asserted per phase, one on_fallback
    // per genuine activation, the terminal FallbackExhausted, and the
    // return to service.
    struct FallbackCounter {
        activations: Arc<Mutex<usize>>,
    }
    impl loopctl::observer::LoopObserver for FallbackCounter {
        fn name(&self) -> &'static str {
            "fallback-counter"
        }
        fn on_fallback(&self, _ctx: &loopctl::observer::FallbackContext) {
            *self.activations.lock().unwrap() += 1;
        }
    }

    let config = loopctl::fallback::FallbackConfig {
        trip_threshold: 1,
        recovery_timeout: std::time::Duration::from_millis(60),
        recovery_successes_needed: 1,
        max_fail_count: 1,
    };
    let manager = FallbackManager::new_with_config(config);
    manager
        .set_original_model("primary-model".to_string())
        .unwrap();
    manager.set_fallback_model("fallback-model").unwrap();

    let client = ScriptedClient::new(vec![
        Step::AuthFail,           // 1: the primary trips
        Step::Text("one".into()), // 2: the fallback serves
        Step::Text("two".into()), // 3: post-cooldown probe closes to primary
        Step::AuthFail,           // 4: the primary trips again
        Step::AuthFail,           // 5: the fallback dies serving (max_fail_count 1)
        //    — run 6 sends no request at all
        Step::Text("three".into()), // 6: post-exhaustion probe recovers
    ]);
    let requests = Arc::clone(&client.requests);
    let mut agent = make_agent(client, manager);
    let activations = Arc::new(Mutex::new(0usize));
    agent.register_observer(Arc::new(FallbackCounter {
        activations: Arc::clone(&activations),
    }));

    assert!(agent.run("q1", &RunConfig::default()).await.is_err());
    assert!(
        agent.run("q2", &RunConfig::default()).await.is_ok(),
        "the fallback serves after the trip"
    );
    tokio::time::sleep(std::time::Duration::from_millis(70)).await;
    assert!(
        agent.run("q3", &RunConfig::default()).await.is_ok(),
        "the probe returns to the primary"
    );
    assert!(agent.run("q4", &RunConfig::default()).await.is_err());
    assert!(
        agent.run("q5", &RunConfig::default()).await.is_err(),
        "the fallback dies serving the fifth run"
    );
    let sixth = agent.run("q6", &RunConfig::default()).await;
    assert!(
        matches!(sixth, Err(loopctl::error::LoopError::FallbackExhausted)),
        "the exhausted chain fails the next turn typed: {sixth:?}"
    );
    tokio::time::sleep(std::time::Duration::from_millis(70)).await;
    assert!(
        agent.run("q7", &RunConfig::default()).await.is_ok(),
        "the post-exhaustion cooldown probe recovers the primary"
    );

    let models = requests.lock().unwrap().clone();
    assert_eq!(
        models,
        vec![
            Some("primary-model".to_string()),
            Some("fallback-model".to_string()),
            Some("primary-model".to_string()),
            Some("primary-model".to_string()),
            Some("fallback-model".to_string()),
            Some("primary-model".to_string()),
        ],
        "each served phase routes to the model the breaker believed active — \
         the exhausted turn sent none"
    );
    assert_eq!(
        *activations.lock().unwrap(),
        2,
        "one on_fallback per genuine activation: the trip, then the re-trip"
    );
}

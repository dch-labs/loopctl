//! The hook follow-up: a clean run may be extended by one engine-injected turn.
//!
//! These drive a full [`BareLoop`] over the canned
//! [`MockApiClient`](loopctl::testing::MockApiClient) with a hook
//! executor attached, proving the engine-level contract end to end:
//! the consult happens only on a clean end, the injected turn joins
//! the same run record, the one-follow-up bound disarms the consult
//! for the injected turn's own end, and a run without an executor
//! behaves exactly as before the mechanism existed.

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

#[cfg(all(feature = "hooks", feature = "testing"))]
mod followup_tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use loopctl::engine::core::Loop;
    use loopctl::engine::{BareLoop, RunConfig};
    use loopctl::hooks::context::RunEndContext;
    use loopctl::hooks::{Hook, HookExecutor, PostRunAction};
    use loopctl::managers::LoopManagers;
    use loopctl::testing::MockApiClient;
    use loopctl::tool::ToolRegistry;

    /// A hook that offers a follow-up, counting its consults and its
    /// run-end notifications.
    struct OfferingHook {
        name: &'static str,
        prompt: &'static str,
        consults: AtomicUsize,
        run_ends: AtomicUsize,
    }

    impl OfferingHook {
        fn new(name: &'static str, prompt: &'static str) -> Self {
            Self {
                name,
                prompt,
                consults: AtomicUsize::new(0),
                run_ends: AtomicUsize::new(0),
            }
        }
    }

    impl Hook for OfferingHook {
        fn name(&self) -> &str {
            self.name
        }

        fn on_run_end(&self, _ctx: &RunEndContext) {
            self.run_ends.fetch_add(1, Ordering::SeqCst);
        }

        fn on_run_followup(&self, _ctx: &RunEndContext) -> Option<PostRunAction> {
            self.consults.fetch_add(1, Ordering::SeqCst);
            Some(PostRunAction::FollowUp {
                prompt: self.prompt.to_string(),
                origin: format!("hook:{}", self.name),
            })
        }
    }

    /// A hook that declines, counting its consults.
    struct DecliningHook {
        consults: AtomicUsize,
    }

    impl Hook for DecliningHook {
        fn name(&self) -> &str {
            "declining"
        }

        fn on_run_followup(&self, _ctx: &RunEndContext) -> Option<PostRunAction> {
            self.consults.fetch_add(1, Ordering::SeqCst);
            None
        }
    }

    /// Build a loop over a canned client with the given hook executor.
    fn loop_with(
        client: MockApiClient,
        executor: Option<Arc<HookExecutor>>,
    ) -> BareLoop<MockApiClient> {
        let mut managers = LoopManagers::new();
        if let Some(executor) = executor {
            managers = managers.with_hook_executor(executor);
        }
        BareLoop::new_with_managers(
            Arc::new(client),
            ToolRegistry::new(),
            Default::default(),
            managers,
        )
    }

    /// Every request the mock served, on either transport, oldest-last.
    ///
    /// The engine picks its transport from the session config, and the
    /// feature set of the build can flip the default — a test that counts
    /// provider calls must read both ledgers rather than guessing which
    /// one carried the run.
    fn served_requests(client: &MockApiClient) -> Vec<loopctl::api::StreamRequest> {
        let mut all = client.captured_requests();
        all.extend(client.captured_stream_requests());
        all
    }

    /// The last user message a request carried, as plain text.
    fn last_user_text(request: &loopctl::api::StreamRequest) -> String {
        request
            .messages
            .iter()
            .rev()
            .find_map(|message| {
                if matches!(message.role, loopctl::message::Role::User) {
                    Some(
                        message
                            .parts
                            .iter()
                            .filter_map(|part| match part {
                                loopctl::message::MessagePart::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<String>(),
                    )
                } else {
                    None
                }
            })
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn a_followup_hook_extends_a_clean_run_by_one_turn() {
        // The whole mechanism in one shape: the run completes, the hook is
        // offered the end and accepts, and the injected prompt runs as a
        // second turn inside the same run — the provider sees it as the
        // newest user message of the next request, the run record carries
        // the hook's origin, and the run's start/end notifications fired
        // exactly once each, the injected turn joining rather than
        // starting its own run.
        let client = MockApiClient::new("test-model")
            .with_text_response("the first answer")
            .with_text_response("the follow-up answer");
        let handle = client.clone();
        let hook = Arc::new(OfferingHook::new("offerer", "close the list"));
        let mut agent = loop_with(
            client,
            Some(Arc::new(
                HookExecutor::new().with_hook(Arc::clone(&hook) as Arc<dyn Hook>),
            )),
        );

        let run = agent
            .run("the prompt", &RunConfig::default())
            .await
            .expect("the run completes, follow-up included");

        let requests = served_requests(&handle);
        assert_eq!(requests.len(), 2, "the injected turn reached the provider");
        assert_eq!(
            last_user_text(&requests[1]),
            "close the list",
            "the follow-up turn carries the hook's prompt as its input"
        );
        assert_eq!(
            run.output.as_deref(),
            Some("the follow-up answer"),
            "the run's output is the injected turn's answer"
        );
        assert_eq!(
            run.followup_origin.as_deref(),
            Some("hook:offerer"),
            "the run record names the hook whose turn rode inside it"
        );
        assert_eq!(
            hook.consults.load(Ordering::SeqCst),
            1,
            "the hook was consulted exactly once"
        );
        assert_eq!(
            hook.run_ends.load(Ordering::SeqCst),
            1,
            "the run-end notification fired once — the injected turn joins the run"
        );
    }

    #[tokio::test]
    async fn the_injected_turn_is_never_offered_its_own_followup() {
        // The no-loop bound is the engine's, not the policy's: this hook
        // accepts every offer it receives, and the run still extends by
        // exactly one turn — the injected turn's own clean end consults
        // nothing, so a policy that would recurse forever cannot.
        let client = MockApiClient::new("test-model")
            .with_text_response("one")
            .with_text_response("two")
            .with_text_response("three")
            .with_text_response("four");
        let handle = client.clone();
        let hook = Arc::new(OfferingHook::new("greedy", "again"));
        let mut agent = loop_with(
            client,
            Some(Arc::new(
                HookExecutor::new().with_hook(Arc::clone(&hook) as Arc<dyn Hook>),
            )),
        );

        agent
            .run("the prompt", &RunConfig::default())
            .await
            .expect("the run completes after its single follow-up");

        assert_eq!(
            served_requests(&handle).len(),
            2,
            "one follow-up per run, however eagerly the hook accepts"
        );
        assert_eq!(
            hook.consults.load(Ordering::SeqCst),
            1,
            "the injected turn's own end was not offered"
        );
    }

    #[tokio::test]
    async fn a_failing_run_is_not_offered_a_followup() {
        // The consult is a clean-end offer: a run that dies on a provider
        // error ends as it stood, and a hook that would have accepted is
        // never asked.
        let client = MockApiClient::new("test-model")
            .with_text_response("fine")
            .with_error("the provider refused");
        let hook = Arc::new(OfferingHook::new("offerer", "close the list"));
        let mut agent = loop_with(
            client,
            Some(Arc::new(
                HookExecutor::new().with_hook(Arc::clone(&hook) as Arc<dyn Hook>),
            )),
        );

        let second = agent
            .run("second prompt", &RunConfig::default())
            .await
            .expect_err("the scripted run fails");

        let _ = second;
        assert_eq!(
            hook.consults.load(Ordering::SeqCst),
            0,
            "a failed end is not offered a follow-up"
        );
    }

    #[tokio::test]
    async fn the_first_hook_to_accept_wins_and_stops_the_consult() {
        // The executor's consult short-circuits like its pre-tool-use
        // half: a declining hook registered after an accepting one is
        // never asked, and an accepting hook registered after a declining
        // one wins the turn with the decline recorded as a plain pass.
        let client = MockApiClient::new("test-model")
            .with_text_response("one")
            .with_text_response("two");
        let handle = client.clone();
        let later = Arc::new(DecliningHook {
            consults: AtomicUsize::new(0),
        });
        let mut agent = loop_with(
            client,
            Some(Arc::new(
                HookExecutor::new()
                    .with_hook(Arc::new(DecliningHook {
                        consults: AtomicUsize::new(0),
                    }))
                    .with_hook(Arc::new(OfferingHook::new("second", "the prompt")))
                    .with_hook(Arc::clone(&later) as Arc<dyn Hook>),
            )),
        );

        agent
            .run("the prompt", &RunConfig::default())
            .await
            .expect("the run completes after the accepting hook's follow-up");

        assert_eq!(
            served_requests(&handle).len(),
            2,
            "the accepting hook's follow-up ran"
        );
        assert_eq!(
            later.consults.load(Ordering::SeqCst),
            0,
            "hooks after the first acceptance are not consulted"
        );
    }

    #[tokio::test]
    async fn the_injected_turn_continues_the_run_wide_turn_sequence() {
        // The injected turn is the run's next turn, not a restarted run:
        // the record's turn indexes continue (0 then 1), the follow-up
        // prompt is what the second turn was prepared and recorded under
        // — memory retrieval and the turn record see the hook's prompt,
        // not the run's original input — and the pair shares one
        // `max_turns` budget of two.
        let client = MockApiClient::new("test-model")
            .with_text_response("the first answer")
            .with_text_response("the follow-up answer");
        let hook = Arc::new(OfferingHook::new("offerer", "close the list"));
        let mut agent = loop_with(
            client,
            Some(Arc::new(
                HookExecutor::new().with_hook(Arc::clone(&hook) as Arc<dyn Hook>),
            )),
        );

        let run = agent
            .run("the prompt", &RunConfig::default().with_max_turns(2))
            .await
            .expect("both turns fit the shared budget of two");

        let indexes: Vec<usize> = run.turns.iter().map(|turn| turn.index).collect();
        assert_eq!(
            indexes,
            vec![0, 1],
            "the follow-up continues the turn sequence — a restarted \
             counter would record turn 0 twice"
        );
        assert_eq!(
            run.turns[0].input, "the prompt",
            "the first turn is recorded under the run's input"
        );
        assert_eq!(
            run.turns[1].input, "close the list",
            "the injected turn is recorded under the follow-up prompt, \
             never the original input"
        );
        assert_eq!(
            run.output.as_deref(),
            Some("the follow-up answer"),
            "the run's output is the injected turn's answer"
        );
    }

    #[tokio::test]
    async fn an_exhausted_turn_budget_is_not_offered_a_followup() {
        // The injected turn spends from the run's own budget: when the
        // budget is spent, the consult never happens — the run stands
        // as it completed, its answer intact, no origin on the record.
        let client = MockApiClient::new("test-model").with_text_response("the only answer");
        let handle = client.clone();
        let hook = Arc::new(OfferingHook::new("offerer", "close the list"));
        let mut agent = loop_with(
            client,
            Some(Arc::new(
                HookExecutor::new().with_hook(Arc::clone(&hook) as Arc<dyn Hook>),
            )),
        );

        let run = agent
            .run("the prompt", &RunConfig::default().with_max_turns(1))
            .await
            .expect("the run completes on its single turn");

        assert_eq!(
            served_requests(&handle).len(),
            1,
            "no follow-up ran — the budget had no room to offer"
        );
        assert_eq!(
            run.output.as_deref(),
            Some("the only answer"),
            "the clean run's answer stands"
        );
        assert_eq!(
            run.followup_origin, None,
            "no origin — nothing was injected"
        );
        assert_eq!(
            hook.consults.load(Ordering::SeqCst),
            0,
            "the hook was never asked: an offer with no budget behind it \
             is not an offer"
        );
    }

    #[tokio::test]
    async fn a_failed_followup_leaves_no_output_on_the_run_record() {
        // The first pass's answer is provisional the moment a follow-up
        // is accepted: if the injected turn fails, the run record must
        // not present that answer as the run's output — a failed run
        // carries its error, not the superseded text.
        let client = MockApiClient::new("test-model")
            .with_text_response("the first answer")
            .with_errors(vec![
                None,
                Some("the follow-up provider call refused".to_string()),
            ]);
        let hook = Arc::new(OfferingHook::new("offerer", "close the list"));
        let mut agent = loop_with(
            client,
            Some(Arc::new(
                HookExecutor::new().with_hook(Arc::clone(&hook) as Arc<dyn Hook>),
            )),
        );

        let error = agent
            .run("the prompt", &RunConfig::default().with_max_turns(4))
            .await
            .expect_err("the injected turn fails the run");

        assert!(
            error.to_string().contains("refused"),
            "the failure is the follow-up's own error: {error}"
        );
        let run = agent
            .session()
            .runs
            .first()
            .expect("the failed run's record survives for inspection");
        assert_eq!(
            run.output, None,
            "the superseded first answer is gone — a failed run does not \
             present it as its output"
        );
        assert_eq!(
            run.followup_origin.as_deref(),
            Some("hook:offerer"),
            "the record still names the hook whose turn failed the run"
        );
    }

    #[tokio::test]
    async fn a_run_without_an_executor_behaves_as_before() {
        // No executor, no offers: the feature-gated stage is a no-op, and
        // the run's shape — one request, the model's answer as the output,
        // no origin on the record — is exactly the pre-mechanism behavior.
        let client = MockApiClient::new("test-model").with_text_response("the answer");
        let handle = client.clone();
        let mut agent = loop_with(client, None);

        let run = agent
            .run("the prompt", &RunConfig::default())
            .await
            .expect("the run completes on its own");

        assert_eq!(served_requests(&handle).len(), 1);
        assert_eq!(run.output.as_deref(), Some("the answer"));
        assert_eq!(run.followup_origin, None);
    }
}

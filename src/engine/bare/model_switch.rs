//! The [`ModelSwitch`] builder — created by [`BareLoop::switch_model`] to
//! update the model (and optionally the context window) atomically.

use super::{ApiClient, BareLoop, LoopError};
use crate::capabilities::FallbackCapable;
use crate::observer::ModelSwitchedContext;

/// Builder for a model switch on [`BareLoop`].
///
/// Created by [`BareLoop::switch_model`]. Allows updating
/// the context window alongside the model name, then applies
/// all changes atomically via [`apply`](Self::apply).
///
/// The switch resets the fallback circuit breaker (stale failure counts
/// from the old model are meaningless for the new one) and fires
/// [`on_model_switched`](crate::observer::LoopObserver::on_model_switched)
/// to all observers.
pub struct ModelSwitch<'a, C: ApiClient> {
    /// The loop being reconfigured, borrowed mutably for the duration of the
    /// builder.
    ///
    /// Carried by value so [`apply`](Self::apply) can destructure the builder
    /// and operate on the loop directly — the borrow lives until `apply`
    /// consumes the builder, after which the loop is usable again. Holding a
    /// `&mut` rather than an owned handle is what makes the builder
    /// non-`Clone`: two concurrent switches on one loop would race on the
    /// client, session, and fallback state, so the type system forbids it.
    pub(super) loop_: &'a mut BareLoop<C>,

    /// The model name to switch to, exactly as passed to
    /// [`switch_model`](BareLoop::switch_model).
    ///
    /// [`apply`](Self::apply) trims surrounding whitespace and rejects an
    /// empty result with [`LoopError::Config`], so a builder constructed with
    /// `"  "` fails at apply time rather than silently keeping the old model.
    /// The untrimmed string is stored so the validation lives in one place
    /// and the builder stays a pure data carrier until applied.
    pub(super) target_model: String,

    /// Optional new context window, in tokens, applied to the session config
    /// on [`apply`](Self::apply) when set.
    ///
    /// `None` (the default) keeps the existing context window — appropriate
    /// when the new model shares the old one's limit. Set via
    /// [`with_context_window`](Self::with_context_window); omitting it when
    /// switching to a model with a different window leaves the
    /// auto-compactor's threshold stale, which is why the setter is
    /// documented as important rather than cosmetic.
    pub(super) context_window: Option<u64>,
}

impl<C: ApiClient> ModelSwitch<'_, C> {
    /// Set the context window (in tokens) for the new model.
    ///
    /// If omitted, the existing context window is kept. Updating this is
    /// important when switching to a model with a significantly different
    /// context window — otherwise the auto-compactor will use the wrong
    /// threshold.
    #[must_use]
    pub fn with_context_window(mut self, tokens: u64) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// Apply the model switch.
    ///
    /// Performs the following steps in order:
    /// 1. Validates the target model is non-empty.
    /// 2. Preflights the fallback manager: a poisoned breaker aborts
    ///    the switch before any state — the client's model included —
    ///    changes.
    /// 3. Delegates to [`ApiClient::set_model`] on the underlying client.
    ///    When the client returns `false` (runtime switching unsupported),
    ///    the switch is rejected outright — nothing downstream runs, and
    ///    the effective model, the breaker, and the observers are left
    ///    untouched. Proceeding would re-point the fallback manager at a
    ///    model the client never adopted, and the per-request model
    ///    override would then serve it anyway.
    /// 4. Updates the session context window.
    /// 5. Resets the [`FallbackManager`](crate::fallback::FallbackManager)
    ///    circuit breaker to `Primary` and updates the original-model
    ///    tracker to the new model.
    /// 6. Fires [`on_model_switched`](crate::observer::LoopObserver::on_model_switched).
    ///
    /// # Errors
    ///
    /// - [`LoopError::Config`] if the model name is empty/whitespace.
    /// - [`LoopError::Config`] if the client rejects the switch
    ///   (`set_model` returned `false`).
    pub fn apply(self) -> Result<(), LoopError> {
        let Self {
            loop_,
            target_model,
            context_window,
        } = self;

        let trimmed = target_model.trim();
        if trimmed.is_empty() {
            return Err(LoopError::Config(
                "model name must not be empty or whitespace".into(),
            ));
        }

        // Poison preflight: a poisoned breaker must abort the switch
        // before any state — the client's model included — changes.
        loop_.managers.fallback().state()?;
        let from = loop_.client.model();
        if !loop_.client.set_model(trimmed) {
            return Err(LoopError::Config(format!(
                "client rejected the model switch to {trimmed:?} \
                 (set_model returned false); nothing was changed"
            )));
        }

        if let Some(cw) = context_window {
            loop_.session.config.context_window = cw;
        }

        loop_.managers.fallback().reset()?;
        loop_
            .managers
            .fallback()
            .set_original_model(trimmed.to_string())?;
        loop_
            .managers
            .observers()
            .on_model_switched(&ModelSwitchedContext {
                from,
                to: trimmed.to_string(),
            });

        Ok(())
    }
}

//! Canonical translation of terminal [`MachineOutcome`] into [`LoopError`].
//!
//! The driver's `Done` arm and [`Loop::stop_reason`] both need to turn a
//! [`MachineOutcome`] into the [`LoopError`] propagated from `run()`. Before
//! this module existed, that translation was written in multiple places; this
//! is the single source of truth.
//!
//! The separate [`LoopError`] → hook-`RunEndReason` mapping lives with the
//! hooks feature in the driver's emission submodule.
//!
//! [`Loop::stop_reason`]: crate::engine::core::Loop::stop_reason

use crate::engine::core::MachineOutcome;
use crate::error::LoopError;

impl MachineOutcome {
    /// The canonical mapping from a terminal outcome to the [`LoopError`] the
    /// driver propagates from `run()`.
    ///
    /// Returns `None` for [`Completed`](MachineOutcome::Completed) — a clean
    /// completion is not an error. `max_turns` is the run's configured turn
    /// ceiling, needed to build [`LoopError::MaxTurnsExceeded`]; pass it from
    /// the [`RunConfig`](crate::engine::core::RunConfig) that started the run.
    ///
    /// Every site that translates an outcome into an error goes through here —
    /// the `Done` arm of `run()`, `stop_reason()`, and any future driver.
    #[must_use]
    pub fn to_loop_error(&self, max_turns: usize) -> Option<LoopError> {
        match self {
            MachineOutcome::Completed { .. } => None,
            MachineOutcome::MaxTurnsExceeded => {
                Some(LoopError::MaxTurnsExceeded { max: max_turns })
            }
            MachineOutcome::Cancelled => Some(LoopError::Cancelled),
            MachineOutcome::Failed { error } => Some(error.clone()),
        }
    }
}

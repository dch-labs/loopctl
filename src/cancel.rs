//! Cooperative cancellation signal for agent loops.
//!
//! [`CancelSignal`] wraps a [`tokio_util::sync::CancellationToken`], which is
//! purpose-built to avoid the time-of-check-to-time-of-use (TOCTOU) race that
//! plagues hand-rolled `AtomicBool` + `Notify` combinations.
//!
//! # Usage
//!
//! ```rust,ignore
//! use loopctl::cancel::CancelSignal;
//! use std::sync::Arc;
//!
//! let signal = Arc::new(CancelSignal::new());
//! let signal_clone = signal.clone();
//!
//! signal_clone.cancel();
//!
//! tokio::select! {
//!     result = stream_next() => { /* ... */ }
//!     _ = signal.notified() => { /* cancelled! */ }
//! }
//! ```

use std::sync::Mutex;

use tokio_util::sync::CancellationToken;

/// Shared cancellation signal backed by a
/// [`tokio_util::sync::CancellationToken`].
///
/// Wrap in `Arc` for sharing across tasks or threads. Create with
/// [`CancelSignal::new`], cancel with [`CancelSignal::cancel`], and await
/// instant notification with [`CancelSignal::notified`]. Re-arm between
/// runs with [`CancelSignal::reset`](Self::reset).
///
/// The underlying token is intentionally one-shot — that is what makes it
/// race-free — so [`reset`](Self::reset) swaps in a fresh token rather
/// than trying to revive a fired one. Because every clone of an
/// `Arc<CancelSignal>` derefs through this same struct, all existing
/// handles observe the new token after a reset; a caller holding
/// [`cancel_signal`](crate::engine::BareLoop::cancel_signal) across runs
/// keeps working.
pub struct CancelSignal {
    inner: Mutex<CancellationToken>,
}

impl CancelSignal {
    /// Create a new, non-cancelled signal.
    ///
    /// Returns a [`CancelSignal`] backed by a fresh
    /// [`CancellationToken`]. The signal is ready to be shared (via `Arc`)
    /// and awaited by worker tasks until [`cancel`](Self::cancel) is called.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(CancellationToken::new()),
        }
    }

    /// Fire the cancellation signal.
    ///
    /// Sets the internal state to cancelled **and** wakes every task currently
    /// awaiting [`Self::notified`]. Idempotent — calling multiple times is
    /// safe.
    pub fn cancel(&self) {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cancel();
    }

    /// Check whether the signal has been cancelled since the last
    /// [`reset`](Self::reset).
    ///
    /// Performs a non-blocking check of the current
    /// [`CancellationToken`]. Returns `true` if
    /// [`cancel`](Self::cancel) has been called and no `reset` has
    /// followed it.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_cancelled()
    }

    /// Return a future that completes **instantly** when [`Self::cancel`]
    /// is called.
    ///
    /// If the signal is already cancelled, the returned future completes
    /// immediately on the first `.await`.
    ///
    /// This delegates to [`CancellationToken::cancelled`], which is
    /// race-free by construction — no flag-then-wait loop required. The
    /// lock is held only long enough to clone the token out, so awaiting
    /// the returned future never blocks another caller from cancelling
    /// or resetting.
    ///
    /// Use inside `tokio::select!` alongside the actual work future:
    ///
    /// ```rust,ignore
    /// tokio::select! {
    ///     r = work_future => r,
    ///     _ = signal.notified() => Err("cancelled"),
    /// }
    /// ```
    pub async fn notified(&self) {
        let token = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        token.cancelled().await;
    }

    /// Re-arm the signal for a fresh run.
    ///
    /// Replaces the internal token with a brand-new, non-cancelled one.
    /// Required at the start of each run because the underlying
    /// [`CancellationToken`] is one-shot by design — once fired it stays
    /// fired, so without a reset a single cancellation would make every
    /// subsequent run return immediately as cancelled.
    ///
    /// All clones of an `Arc<CancelSignal>` observe the new token,
    /// because they deref through this same struct; a handle returned by
    /// [`cancel_signal`](crate::engine::BareLoop::cancel_signal) keeps
    /// working across runs. A task already awaiting
    /// [`notified`](Self::notified) on the previous token is unaffected —
    /// its awaited run was cancelled, and it resolves against that token.
    pub fn reset(&self) {
        *self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = CancellationToken::new();
    }
}

impl Default for CancelSignal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_cancel_wakes_instantly() {
        let signal = Arc::new(CancelSignal::new());
        let signal_clone = signal.clone();

        let handle = tokio::spawn(async move {
            signal_clone.notified().await;
            "woke up"
        });

        tokio::task::yield_now().await;
        signal.cancel();
        let result = handle.await.unwrap();
        assert_eq!(result, "woke up");
    }

    #[tokio::test]
    async fn test_notified_returns_immediately_if_already_cancelled() {
        let signal = CancelSignal::new();
        signal.cancel();
        // Should return immediately, not hang.
        signal.notified().await;
    }

    #[test]
    fn test_is_cancelled() {
        let signal = CancelSignal::new();
        assert!(!signal.is_cancelled());
        signal.cancel();
        assert!(signal.is_cancelled());
    }

    #[tokio::test]
    async fn test_cancel_is_idempotent() {
        let signal = Arc::new(CancelSignal::new());
        signal.cancel();
        signal.cancel();
        signal.cancel();
        assert!(signal.is_cancelled());
        signal.notified().await;
    }

    #[tokio::test]
    async fn test_multiple_waiters_all_wake() {
        let signal = Arc::new(CancelSignal::new());
        let mut handles = Vec::new();

        for _ in 0..10 {
            let s = signal.clone();
            handles.push(tokio::spawn(async move {
                s.notified().await;
                true
            }));
        }

        tokio::task::yield_now().await;
        signal.cancel();

        for handle in handles {
            assert!(handle.await.unwrap());
        }
    }

    #[test]
    fn test_reset_clears_cancelled_state() {
        let signal = CancelSignal::new();
        signal.cancel();
        assert!(signal.is_cancelled());

        signal.reset();
        assert!(
            !signal.is_cancelled(),
            "reset must re-arm the signal to non-cancelled"
        );
    }

    #[test]
    fn test_reset_is_visible_through_existing_arc_clone() {
        let signal = Arc::new(CancelSignal::new());
        let handle = Arc::clone(&signal);

        signal.cancel();
        assert!(handle.is_cancelled());

        signal.reset();
        assert!(!handle.is_cancelled());
    }

    #[tokio::test]
    async fn test_notified_after_reset_waits_for_new_cancel() {
        let signal = Arc::new(CancelSignal::new());
        signal.cancel();
        signal.reset();

        let handle = {
            let s = Arc::clone(&signal);
            tokio::spawn(async move {
                s.notified().await;
                "woke"
            })
        };

        tokio::task::yield_now().await;
        let mut pending = tokio::time::interval(std::time::Duration::from_millis(5));
        pending.tick().await;
        assert!(
            !handle.is_finished(),
            "notified must not fire after reset until a new cancel arrives"
        );

        signal.cancel();
        assert_eq!(handle.await.unwrap(), "woke");
    }
}

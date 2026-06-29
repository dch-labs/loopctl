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

use tokio_util::sync::CancellationToken;

/// Shared cancellation signal backed by a
/// [`tokio_util::sync::CancellationToken`].
///
/// Wrap in `Arc` for sharing across tasks or threads. Create with
/// [`CancelSignal::new`], cancel with [`CancelSignal::cancel`], and await
/// instant notification with [`CancelSignal::notified`].
pub struct CancelSignal {
    inner: CancellationToken,
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
            inner: CancellationToken::new(),
        }
    }

    /// Fire the cancellation signal.
    ///
    /// Sets the internal state to cancelled **and** wakes every task currently
    /// awaiting [`Self::notified`]. Idempotent — calling multiple times is
    /// safe.
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Check whether the signal has been cancelled.
    ///
    /// Performs a non-blocking check of the internal [`CancellationToken`].
    /// Returns `true` if [`cancel`](Self::cancel) has been called since
    /// construction.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    /// Return a future that completes **instantly** when [`Self::cancel`]
    /// is called.
    ///
    /// If the signal is already cancelled, the returned future completes
    /// immediately on the first `.await`.
    ///
    /// This delegates to [`CancellationToken::cancelled`], which is
    /// race-free by construction — no flag-then-wait loop required.
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
        self.inner.cancelled().await;
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
}

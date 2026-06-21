//! Cooperative cancellation signal for agent loops.
//!
//! [`CancelSignal`] combines an [`AtomicBool`] flag with a
//! [`tokio::sync::Notify`] for sub-millisecond wake-up of waiting tasks.
//!
//! # Why not poll an `AtomicBool`?
//!
//! Polling works but wastes CPU cycles and introduces latency proportional
//! to the poll interval. By pairing the flag with a `Notify`, any call to
//! [`CancelSignal::cancel`] instantly wakes every task that is
//! `tokio::select!`-ing on [`CancelSignal::notified`], giving sub-µs
//! response time.
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

use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Notify;

/// Shared cancellation signal backed by an [`AtomicBool`] flag and a
/// [`Notify`] for instant wake-up.
///
/// Wrap in `Arc` for sharing across tasks or threads. Create with
/// [`CancelSignal::new`], cancel with [`CancelSignal::cancel`], and await
/// instant notification with [`CancelSignal::notified`].
pub struct CancelSignal {
    flag: AtomicBool,
    notify: Notify,
}

impl CancelSignal {
    /// Create a new, non-cancelled signal.
    ///
    /// Returns a [`CancelSignal`] with its internal flag set to `false`.
    /// The signal is ready to be shared (via `Arc`) and awaited by
    /// worker tasks until [`cancel`](Self::cancel) is called.
    #[must_use]
    pub fn new() -> Self {
        Self {
            flag: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    /// Fire the cancellation signal.
    ///
    /// Sets the internal flag to `true` **and** wakes every task currently
    /// awaiting [`Self::notified`]. Idempotent — calling multiple times is
    /// safe.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    /// Reset the signal so it can be reused for a new operation.
    ///
    /// Clears the internal cancellation flag, returning the signal to
    /// its initial non-cancelled state. Any subsequent calls to
    /// [`is_cancelled`](Self::is_cancelled) will return `false` until
    /// [`cancel`](Self::cancel) is called again.
    pub fn reset(&self) {
        self.flag.store(false, Ordering::Release);
    }

    /// Check whether the signal has been cancelled.
    ///
    /// Performs a non-blocking load of the internal flag. Returns
    /// `true` if [`cancel`](Self::cancel) has been called since the
    /// last [`reset`](Self::reset) (or since construction).
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Return a future that completes **instantly** when [`Self::cancel`]
    /// is called.
    ///
    /// If the signal is already cancelled, the returned future completes
    /// immediately on the first `.await`.
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
        let notified = self.notify.notified();
        if self.flag.load(Ordering::Acquire) {
            return;
        }
        notified.await;
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

    #[test]
    fn test_reset() {
        let signal = CancelSignal::new();
        signal.cancel();
        assert!(signal.is_cancelled());
        signal.reset();
        assert!(!signal.is_cancelled());
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

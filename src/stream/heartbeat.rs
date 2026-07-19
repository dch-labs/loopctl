//! Composable heartbeat and timeout wrapper for any stream.
//!
//! [`HeartbeatStream`] wraps any `Stream<Item = Result<StreamEvent, ApiError>>`
//! and adds two behaviours:
//!
//! 1. **Heartbeat callbacks** — Fires a callback at regular intervals to report
//!    elapsed time and timeout status.
//! 2. **Hard timeout** — Returns an [`ApiError`] if the stream exceeds a
//!    configured maximum duration.
//!
//! It does **not** retry or fall back — that's [`StreamHandler`](super::handler::StreamHandler)'s
//! job. Use this when you need heartbeat/timeout on a stream you've already opened.
//!
//! # Architecture
//!
//! On each `poll_next`, `HeartbeatStream` runs three checks in order before
//! delegating to the inner stream:
//!
//! 1. **Heartbeat interval** — if the configured interval has elapsed
//!    since the last beat, fire the heartbeat callback with the elapsed
//!    time and current timeout status.
//! 2. **Hard timeout** — if the total elapsed time has exceeded the
//!    configured maximum, return an [`ApiError`] without consulting the
//!    inner stream.
//! 3. **Delegate** — otherwise, forward to the inner stream's `poll_next`
//!    and pass through its result.
//!
//! [`ApiError`]: crate::api::error::ApiError
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::stream::heartbeat::{HeartbeatStream, HeartbeatConfig, HeartbeatData};
//! use std::time::Duration;
//! use std::sync::{Arc, Mutex};
//!
//! let callbacks = Arc::new(Mutex::new(Vec::new()));
//! let cb = callbacks.clone();
//!
//! let config = HeartbeatConfig::new(
//!     Duration::from_secs(30),  // heartbeat_interval
//!     Duration::from_secs(600), // timeout
//!     Box::new(move |data: HeartbeatData| {
//!         cb.lock().unwrap().push(data.elapsed);
//!     }),
//! );
//! ```

use crate::api::error::ApiError;
use crate::stream::StreamEvent;
use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

// ===================================================
// HeartbeatData
// ===================================================

/// Data emitted on each heartbeat callback.
///
/// Passed to the callback registered in [`HeartbeatConfig`] at each
/// heartbeat interval. Carries a snapshot of how long the stream has
/// been running and whether it has crossed its configured hard-timeout
/// deadline, so a UI or metrics collector can render progress without
/// owning a clock itself.
///
/// # Example
///
/// ```rust
/// use loopctl::stream::heartbeat::HeartbeatData;
/// use std::time::Duration;
///
/// let data = HeartbeatData {
///     elapsed: Duration::from_secs(45),
///     is_timeout: false,
/// };
/// assert!(!data.is_timeout);
/// ```
#[derive(Debug, Clone)]
pub struct HeartbeatData {
    /// Time elapsed since the wrapped stream was constructed.
    ///
    /// Monotonic — measured from the [`Instant`] captured in
    /// [`HeartbeatStream::new`], not wall-clock time. Useful for
    /// progress UIs ("streaming for 45s"), metrics, and detecting
    /// stalls without each consumer holding its own start instant.
    ///
    /// [`Instant`]: std::time::Instant
    pub elapsed: Duration,

    /// Whether the stream has exceeded its configured hard timeout.
    ///
    /// Set to `elapsed > config.timeout` at the moment the heartbeat
    /// fires. Note this is *advisory*: the heartbeat may observe
    /// `is_timeout == true` a beat after the deadline actually crossed,
    /// because callbacks only fire on `poll_next`. The next
    /// [`HeartbeatStream::poll_next`] after the deadline will return
    /// the hard-timeout error regardless of when the callback last
    /// fired.
    pub is_timeout: bool,
}

// ===================================================
// HeartbeatCallback
// ===================================================

/// Callback type for heartbeat events.
///
/// A `Box<dyn Fn(HeartbeatData) + Send + Sync>` that is called at each
/// heartbeat interval with the current stream status.
pub type HeartbeatCallback = Box<dyn Fn(HeartbeatData) + Send + Sync>;

// ===================================================
// HeartbeatConfig
// ===================================================

/// Configuration for a [`HeartbeatStream`].
///
/// Holds the heartbeat interval, hard timeout, and callback function.
/// Created via [`HeartbeatConfig::new()`].
///
/// # Example
///
/// ```rust
/// use loopctl::stream::heartbeat::{HeartbeatConfig, HeartbeatData};
/// use std::time::Duration;
///
/// let config = HeartbeatConfig::new(
///     Duration::from_secs(30),
///     Duration::from_secs(600),
///     Box::new(|_data: HeartbeatData| {}),
/// );
/// assert_eq!(config.heartbeat_interval(), Duration::from_secs(30));
/// assert_eq!(config.timeout(), Duration::from_secs(600));
/// ```
pub struct HeartbeatConfig {
    /// Interval between heartbeat callbacks.
    ///
    /// Checked on every `poll_next`: when `last_heartbeat.elapsed()`
    /// reaches this value, the callback fires and `last_heartbeat` is
    /// reset. There is no background timer — heartbeats only fire while
    /// the stream is actively being polled.
    heartbeat_interval: Duration,

    /// Maximum total stream duration before triggering a hard timeout.
    ///
    /// Once `start.elapsed()` exceeds this value, the next `poll_next`
    /// returns an [`ApiError`] instead of delegating to the inner
    /// stream. Choose a value comfortably above the expected p99
    /// response time so transient slowness doesn't trip it.
    ///
    /// [`ApiError`]: crate::api::error::ApiError
    timeout: Duration,

    /// Callback invoked at each heartbeat interval.
    ///
    /// A `Box<dyn Fn(HeartbeatData) + Send + Sync>` so it can be shared
    /// across the runtime and mutate captured state (typically an
    /// `Arc<Mutex<…>>`-protected metrics struct or channel sender).
    /// Called synchronously from `poll_next` — keep the body cheap to
    /// avoid stalling the stream's task.
    on_heartbeat: HeartbeatCallback,
}

impl std::fmt::Debug for HeartbeatConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeartbeatConfig")
            .field("heartbeat_interval", &self.heartbeat_interval)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl HeartbeatConfig {
    /// Create a new heartbeat configuration.
    ///
    /// # Arguments
    ///
    /// - `heartbeat_interval` — How often to fire the callback.
    /// - `timeout` — Maximum stream duration before returning an error.
    /// - `on_heartbeat` — Callback invoked at each interval.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::heartbeat::{HeartbeatConfig, HeartbeatData};
    /// use std::time::Duration;
    ///
    /// let config = HeartbeatConfig::new(
    ///     Duration::from_secs(15),
    ///     Duration::from_secs(300),
    ///     Box::new(|data: HeartbeatData| {
    ///         println!("heartbeat: {:.1}s elapsed", data.elapsed.as_secs_f64());
    ///     }),
    /// );
    /// ```
    #[must_use]
    pub fn new(
        heartbeat_interval: Duration,
        timeout: Duration,
        on_heartbeat: HeartbeatCallback,
    ) -> Self {
        Self {
            heartbeat_interval,
            timeout,
            on_heartbeat,
        }
    }

    /// Returns the configured heartbeat interval.
    ///
    /// Exposed so callers (e.g. a metrics reporter that wants to align
    /// its own cadence with the heartbeat) can read the value the
    /// stream was constructed with.
    #[must_use]
    pub fn heartbeat_interval(&self) -> Duration {
        self.heartbeat_interval
    }

    /// Returns the configured hard timeout.
    ///
    /// Exposed so callers can render the deadline ("stream will time out
    /// after 600s") or compute remaining budget from the elapsed time.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

// ===================================================
// HeartbeatStream
// ===================================================

/// A stream wrapper that emits heartbeat callbacks and enforces a hard timeout.
///
/// Wraps any `Stream<Item = Result<StreamEvent, ApiError>>` and adds:
/// - Periodic heartbeat callbacks via [`HeartbeatConfig`].
/// - A hard timeout that returns an [`ApiError`] when exceeded.
///
/// Does **not** retry or fallback — use [`StreamHandler`](super::handler::StreamHandler) instead.
///
/// # Composability
///
/// `HeartbeatStream` implements `Stream` directly, so it composes with
/// any other stream wrapper. Use it on any stream you've already opened
/// when you need heartbeat/timeout without the full handler lifecycle.
///
/// # Example
///
/// ```rust
/// use loopctl::stream::heartbeat::{HeartbeatStream, HeartbeatConfig, HeartbeatData};
/// use std::time::Duration;
///
/// let config = HeartbeatConfig::new(
///     Duration::from_secs(30),
///     Duration::from_secs(600),
///     Box::new(|_data: HeartbeatData| {}),
/// );
///
/// // Wrap any stream:
/// // let heartbeat_stream = HeartbeatStream::new(inner_stream, config);
/// // while let Some(result) = futures::StreamExt::next(&mut heartbeat_stream).await {
/// //     // ...
/// // }
/// ```
pub struct HeartbeatStream<S> {
    /// The inner stream being wrapped.
    ///
    /// All `poll_next` calls that survive the heartbeat check and the
    /// hard-timeout check are delegated to this stream. The wrapper
    /// does not buffer or transform items — it passes them through
    /// verbatim, including errors from the inner stream.
    inner: S,

    /// Heartbeat and timeout configuration.
    ///
    /// Holds the interval, the timeout, and the callback. Owned by
    /// the stream (not shared) so the wrapper can call the callback
    /// without synchronization.
    config: HeartbeatConfig,

    /// Time of the last heartbeat callback.
    ///
    /// Compared against `Instant::now()` on every `poll_next` to
    /// decide whether the interval has elapsed. Reset to `Instant::now()`
    /// (not `start + n*interval`) after each fire so drift from
    /// irregular polling doesn't accumulate.
    last_heartbeat: Instant,

    /// Time the stream was created.
    ///
    /// The reference for all `elapsed` calculations — heartbeat
    /// `elapsed`, and the hard-timeout comparison. Captured once in
    /// [`new`](Self::new) and never mutated for the life of the stream.
    start: Instant,

    /// A `Sleep` future that fires at the hard-timeout deadline.
    ///
    /// Ensures the runtime wakes this task when the timeout expires,
    /// even if the inner stream is `Pending` and nobody re-polls.
    /// Polled proactively in `poll_next` so a ready `Sleep` short-
    /// circuits to the timeout error without delegating to the inner
    /// stream first.
    timeout_sleep: std::pin::Pin<Box<tokio::time::Sleep>>,
}

impl<S> HeartbeatStream<S> {
    /// Create a new heartbeat stream wrapping the given inner stream.
    ///
    /// The heartbeat timer starts immediately upon construction: `start`
    /// and `last_heartbeat` are captured at this moment, and the
    /// hard-timeout `Sleep` is armed against `start + config.timeout`.
    /// The first heartbeat callback fires after `heartbeat_interval`
    /// elapses (checked on each `poll_next` — there is no background
    /// timer).
    ///
    /// # Timeout overflow
    ///
    /// `start + config.timeout` is computed via `checked_add`. For any
    /// realistic `Duration` this always succeeds; the fallback (a
    /// deadline 30 years in the future) only triggers for extreme values
    /// near `Duration::MAX`, where the deadline is effectively "never"
    /// either way.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::heartbeat::{HeartbeatStream, HeartbeatConfig, HeartbeatData};
    /// use std::time::Duration;
    ///
    /// let config = HeartbeatConfig::new(
    ///     Duration::from_secs(30),
    ///     Duration::from_secs(600),
    ///     Box::new(|_data: HeartbeatData| {}),
    /// );
    /// // let stream = HeartbeatStream::new(inner_stream, config);
    /// ```
    pub fn new(inner: S, config: HeartbeatConfig) -> Self {
        /// 30 years in seconds — used as a far-future deadline fallback.
        /// Computed as a const so the compiler verifies no overflow.
        const THIRTY_YEARS_SECS: u64 = 86400 * 365 * 30;

        let now = Instant::now();
        // checked_add returns None only for extreme Duration values (hundreds of years).
        // Fallback: 30 years from now, which is effectively infinite.
        let far_future = || {
            Instant::now()
                .checked_add(Duration::from_secs(THIRTY_YEARS_SECS))
                .unwrap_or(Instant::now())
        };
        let deadline = now.checked_add(config.timeout).unwrap_or_else(far_future);
        let timeout_sleep = Box::pin(tokio::time::sleep_until(tokio::time::Instant::from_std(
            deadline,
        )));
        Self {
            inner,
            config,
            last_heartbeat: now,
            start: now,
            timeout_sleep,
        }
    }
}

impl<S> Stream for HeartbeatStream<S>
where
    S: Stream<Item = Result<StreamEvent, ApiError>> + Unpin,
{
    type Item = Result<StreamEvent, ApiError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // Check heartbeat interval — fire callback if elapsed.
        if this.last_heartbeat.elapsed() >= this.config.heartbeat_interval {
            let elapsed = this.start.elapsed();
            let data = HeartbeatData {
                elapsed,
                is_timeout: elapsed > this.config.timeout,
            };
            (this.config.on_heartbeat)(data);
            this.last_heartbeat = Instant::now();
        }

        // Hard timeout — check the Sleep first (proactive wake-up),
        // then fall back to elapsed() for the sync case.
        if this.timeout_sleep.as_mut().poll(cx).is_ready()
            || this.start.elapsed() > this.config.timeout
        {
            return Poll::Ready(Some(Err(ApiError::Api(format!(
                "Stream timeout after {}s",
                this.config.timeout.as_secs()
            )))));
        }

        // Delegate to inner stream.
        Pin::new(&mut this.inner).poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    struct VecStream {
        items: Vec<Result<StreamEvent, ApiError>>,
    }

    impl Stream for VecStream {
        type Item = Result<StreamEvent, ApiError>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.get_mut().items.pop())
        }
    }

    fn make_config(
        callbacks: &std::sync::Arc<std::sync::Mutex<Vec<HeartbeatData>>>,
    ) -> HeartbeatConfig {
        let cb = callbacks.clone();
        HeartbeatConfig::new(
            Duration::from_millis(10),
            Duration::from_secs(60),
            Box::new(move |data: HeartbeatData| {
                cb.lock().unwrap().push(data);
            }),
        )
    }

    #[test]
    fn heartbeat_data_fields() {
        let data = HeartbeatData {
            elapsed: Duration::from_secs(30),
            is_timeout: true,
        };
        assert_eq!(data.elapsed, Duration::from_secs(30));
        assert!(data.is_timeout);
    }

    #[test]
    fn config_accessors() {
        let config = HeartbeatConfig::new(
            Duration::from_secs(15),
            Duration::from_secs(300),
            Box::new(|_| {}),
        );
        assert_eq!(config.heartbeat_interval(), Duration::from_secs(15));
        assert_eq!(config.timeout(), Duration::from_secs(300));
    }

    #[test]
    fn config_debug() {
        let config = HeartbeatConfig::new(
            Duration::from_secs(30),
            Duration::from_secs(600),
            Box::new(|_| {}),
        );
        let debug = format!("{config:?}");
        assert!(debug.contains("HeartbeatConfig"));
        assert!(debug.contains("heartbeat_interval"));
        assert!(debug.contains("timeout"));
    }

    #[tokio::test]
    async fn passes_through_events() {
        let callbacks: std::sync::Arc<std::sync::Mutex<Vec<HeartbeatData>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let config = make_config(&callbacks);

        let inner = VecStream {
            items: vec![Ok(StreamEvent::Ping), Ok(StreamEvent::Ping)],
        };

        let mut stream = HeartbeatStream::new(inner, config);
        let first = stream.next().await;
        assert!(first.is_some());

        let second = stream.next().await;
        assert!(second.is_some());

        let third = stream.next().await;
        assert!(third.is_none());
    }

    #[tokio::test]
    async fn passes_through_errors() {
        let callbacks: std::sync::Arc<std::sync::Mutex<Vec<HeartbeatData>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let config = make_config(&callbacks);

        let inner = VecStream {
            items: vec![Err(ApiError::Api("test error".to_string()))],
        };

        let mut stream = HeartbeatStream::new(inner, config);
        let result = stream.next().await;
        assert!(matches!(result, Some(Err(ApiError::Api(_)))));
    }

    #[tokio::test]
    async fn fires_heartbeat_on_interval_sync() {
        let callbacks: std::sync::Arc<std::sync::Mutex<Vec<HeartbeatData>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let config = make_config(&callbacks);
        let inner = VecStream {
            items: vec![Ok(StreamEvent::Ping)],
        };
        let mut stream = HeartbeatStream::new(inner, config);

        stream.last_heartbeat = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();

        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let result = Pin::new(&mut stream).poll_next(&mut cx);

        assert!(matches!(result, Poll::Ready(Some(Ok(StreamEvent::Ping)))));

        let cbs = callbacks.lock().unwrap();
        assert_eq!(cbs.len(), 1);
        assert!(cbs[0].elapsed > Duration::ZERO);
    }

    #[tokio::test]
    async fn timeout_returns_error_sync() {
        // Verify that poll_next returns a timeout error when the timeout
        // has elapsed. We construct the stream, manually advance time by
        // setting start into the past, then poll.
        let callbacks: std::sync::Arc<std::sync::Mutex<Vec<HeartbeatData>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let config = HeartbeatConfig::new(
            Duration::from_millis(10),
            Duration::from_millis(1),
            Box::new(move |data: HeartbeatData| {
                callbacks.lock().unwrap().push(data);
            }),
        );

        // VecStream that returns Pending on first poll (simulates waiting).
        let inner = VecStream { items: vec![] };
        let mut stream = HeartbeatStream::new(inner, config);

        // Manually set start into the past so timeout has elapsed.
        stream.start = Instant::now().checked_sub(Duration::from_secs(10)).unwrap();

        // Use a no-op waker to poll manually.
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let result = Pin::new(&mut stream).poll_next(&mut cx);

        assert!(
            matches!(result, Poll::Ready(Some(Err(ApiError::Api(msg)))) if msg.contains("timeout"))
        );
    }
}

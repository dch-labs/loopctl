//! Middleware that wraps tool execution in a timeout.

use super::{ToolDispatchContext, ToolDispatchResult, ToolMiddleware, ToolPipeline};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

/// Configuration for the [`TimeoutMiddleware`].
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Per-tool execution deadline.
    ///
    /// Each tool invocation is given this long to complete before it is
    /// cancelled and (optionally) retried. Set to [`Duration::MAX`] via
    /// [`TimeoutMiddleware::none`] to disable the timeout entirely.
    pub timeout: Duration,

    /// Retry with exponential backoff up to [`max_retries`](TimeoutConfig::max_retries) times.
    ///
    /// When `true`, a timed-out call is retried with a doubled deadline
    /// rather than failing immediately; when `false` the first timeout
    /// produces the final error.
    pub retry_on_timeout: bool,

    /// Maximum number of retries (0 = no retry). Total attempts = `1 + max_retries`.
    ///
    /// Upper bound on additional attempts after the initial one. Each retry
    /// doubles the deadline, so this also bounds how long the middleware will
    /// keep trying before giving up.
    pub max_retries: u32,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_mins(2),
            retry_on_timeout: false,
            max_retries: 0,
        }
    }
}

/// Middleware that wraps tool execution in a timeout.
///
/// If the tool execution exceeds the configured timeout, returns an
/// error result. When `retry_on_timeout` is set it retries up to
/// [`max_retries`](TimeoutConfig::max_retries) times, doubling the
/// timeout per attempt. Respects the
/// [`CancelSignal`](crate::cancel::CancelSignal) via `tokio::select!` so
/// that cancellation is not blocked by a slow tool.
///
/// Two shape notes for standalone pipelines: a timed-out result reports
/// `duration` = the timeout that expired (the configured deadline, not
/// the elapsed wall time), and a cancellation observed *here* becomes a
/// soft `is_error` result rather than a hard error — inside the engine
/// the driver's own cancel-vs-dispatch select usually wins the race and
/// surfaces [`LoopError::Cancelled`](crate::error::LoopError::Cancelled)
/// instead.
///
/// # Example
///
/// ```rust,ignore
/// let mw = TimeoutMiddleware::from_secs(120);
/// let mw = TimeoutMiddleware::new(TimeoutConfig {
///     timeout: Duration::from_secs(60),
///     retry_on_timeout: false,
///     max_retries: 0,
/// });
/// ```
pub struct TimeoutMiddleware {
    config: TimeoutConfig,
}

impl TimeoutMiddleware {
    /// Create a timeout middleware with the given configuration.
    ///
    /// `config.timeout` controls the per-tool execution deadline.
    /// When `config.retry_on_timeout` is `true`, a timed-out call is
    /// retried up to `config.max_retries` additional times with an
    /// increasing back-off.
    ///
    /// For simpler construction see [`from_secs`](Self::from_secs) or
    /// [`none`](Self::none).
    #[must_use]
    pub fn new(config: TimeoutConfig) -> Self {
        Self { config }
    }

    /// Create a timeout middleware with a fixed timeout in seconds.
    ///
    /// Uses default retry settings (`retry_on_timeout: false`,
    /// `max_retries: 0` — i.e. no retries).
    #[must_use]
    pub fn from_secs(secs: u64) -> Self {
        Self {
            config: TimeoutConfig {
                timeout: Duration::from_secs(secs),
                ..TimeoutConfig::default()
            },
        }
    }

    /// Create a timeout middleware with no timeout (pass-through).
    ///
    /// Useful for testing or when timeouts are handled elsewhere.
    #[must_use]
    pub fn none() -> Self {
        Self {
            config: TimeoutConfig {
                timeout: Duration::MAX,
                retry_on_timeout: false,
                max_retries: 0,
            },
        }
    }
}

impl ToolMiddleware for TimeoutMiddleware {
    fn name(&self) -> &'static str {
        "timeout"
    }

    fn dispatch<'a>(
        &'a self,
        ctx: &'a mut ToolDispatchContext,
        next: &'a ToolPipeline,
    ) -> Pin<Box<dyn Future<Output = ToolDispatchResult> + Send + 'a>> {
        let config = self.config.clone();
        let tool_name = ctx.tool_name.clone();
        let cancel = Arc::clone(&ctx.cancel);

        Box::pin(async move {
            let mut attempt = 0u32;
            let mut current_timeout = config.timeout;

            loop {
                let result_future = next.dispatch(ctx);
                let attempt_for_log = attempt;

                tokio::select! {
                    result = tokio::time::timeout(current_timeout, result_future) => {
                        if let Ok(dispatch_result) = result {
                            return dispatch_result;
                        }
                        attempt = attempt.saturating_add(1);
                        if config.retry_on_timeout && attempt <= config.max_retries {
                            tracing::warn!(
                                tool = %tool_name,
                                attempt = attempt_for_log,
                                timeout_secs = current_timeout.as_secs(),
                                "tool execution timed out, retrying"
                            );
                            current_timeout = current_timeout.saturating_mul(2);
                            continue;
                        }
                        tracing::error!(
                            tool = %tool_name,
                            timeout_secs = current_timeout.as_secs(),
                            "tool execution timed out"
                        );
                        return ToolDispatchResult::err(
                            &tool_name,
                            format!(
                                "Tool '{}' timed out after {}s",
                                tool_name,
                                current_timeout.as_secs()
                            ),
                            current_timeout,
                        );
                    }
                    () = cancel.notified() => {
                        return ToolDispatchResult::err(
                            &tool_name,
                            format!("Tool '{tool_name}' cancelled"),
                            Duration::ZERO,
                        );
                    }
                }
            }
        })
    }
}

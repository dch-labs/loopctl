//! Provider error-path contracts: structured classification at the HTTP
//! boundary.
//!
//! Pins three invariants. A 429 carrying `Retry-After` reaches the caller as
//! the structured `ApiError::RateLimit` with the parsed delay (not a flattened
//! `Http` string the delay was never copied into). Permanent client errors —
//! 401 and 403 — reach the caller as `ApiError::Auth` and report as not
//! retryable, so a bad key cannot burn the retry ladder. And end-to-end, the
//! server-advised delay flows through `StreamHandler`'s rate-limit budget:
//! honoured when `respect_retry_after` is set, clamped by `max_delay`.
//!
//! Each provider's client is feature-gated; a contract runs for every client
//! whose feature is enabled. All servers are local one-shot TCP listeners —
//! no network, no keys.

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

#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
mod contracts {
    use std::sync::Arc;
    use std::time::Duration;
    use std::time::Instant;

    use futures::StreamExt;
    use loopctl::api::error::ApiError;
    use loopctl::api::{ApiClient, StreamRequest};
    use loopctl::cancel::CancelSignal;
    use loopctl::stream::handler::{
        RateLimitConfig, StreamHandler, StreamHandlerError, StreamOutcome, StreamTimeoutConfig,
    };
    use loopctl::structured::RequestOptions;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    /// Where the test server's base URL and accept-count live for a provider
    /// client builder closure.
    pub(crate) struct TestServer {
        /// Base URL every provider client should be pointed at.
        ///
        /// The server accepts any path — the clients append their own
        /// (`/chat/completions`, `/v1/messages`, …) and the contract under
        /// test is about responses, not routing.
        pub base_url: String,

        /// Join handle for the spawned server task; await it to assert it
        /// served every expected request.
        pub task: tokio::task::JoinHandle<()>,
    }

    /// Serve `count` identical HTTP responses, one per accepted connection.
    ///
    /// Reads each request fully (head plus `Content-Length` body) before
    /// answering, then closes the connection — so pooled clients open a fresh
    /// connection per request and the count is exact.
    async fn serve(
        status: u16,
        reason: &str,
        headers: &[(&str, &str)],
        body: &str,
        count: usize,
    ) -> TestServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mut head = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\n",
            body.len()
        );
        for (name, value) in headers {
            head.push_str(name);
            head.push_str(": ");
            head.push_str(value);
            head.push_str("\r\n");
        }
        head.push_str("Connection: close\r\n\r\n");
        let response = format!("{head}{body}");
        let task = tokio::spawn(async move {
            for _ in 0..count {
                let (mut sock, _) = listener.accept().await.unwrap();
                read_full_request(&mut sock).await;
                sock.write_all(response.as_bytes()).await.unwrap();
                sock.flush().await.unwrap();
            }
        });
        TestServer {
            base_url: format!("http://{addr}"),
            task,
        }
    }

    /// Read one HTTP/1.1 request (head plus body) off `sock`.
    ///
    /// Returns once the `Content-Length` body has arrived; a peer that hangs
    /// up mid-request ends the read early. The bytes are discarded — the
    /// contracts vary responses, not requests.
    async fn read_full_request(sock: &mut TcpStream) {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 1024];
        loop {
            let head_end = buf
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|at| at + 4);
            if let Some(head_end) = head_end {
                let head = String::from_utf8_lossy(&buf[..head_end]).to_lowercase();
                let clen = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.trim()
                            .eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                if buf.len() >= head_end + clen {
                    return;
                }
            }
            let n = sock.read(&mut chunk).await.unwrap_or(0);
            if n == 0 {
                return;
            }
            buf.extend_from_slice(&chunk[..n]);
        }
    }

    /// Build an OpenAI client pointed at `base`, the 429-contract provider.
    #[cfg(feature = "openai")]
    fn openai_at(base: &str) -> loopctl::provider::OpenAiClient {
        loopctl::provider::OpenAiClient::builder()
            .with_api_key("test-key")
            .with_base_url(base.to_string())
            .build()
            .unwrap()
    }

    /// Build an Anthropic client pointed at `base`, the 429-contract provider.
    #[cfg(feature = "anthropic")]
    fn anthropic_at(base: &str) -> loopctl::provider::AnthropicClient {
        loopctl::provider::AnthropicClient::builder()
            .with_api_key("test-key")
            .with_base_url(base.to_string())
            .build()
            .unwrap()
    }

    /// Build a Gemini client pointed at `base`, the 429-contract provider.
    #[cfg(feature = "gemini")]
    fn gemini_at(base: &str) -> loopctl::provider::GeminiClient {
        loopctl::provider::GeminiClient::builder()
            .with_api_key("test-key")
            .with_base_url(base.to_string())
            .build()
            .unwrap()
    }

    /// Assert the local server's 429 + `Retry-After: 7` arrives carrying the
    /// parsed delay.
    ///
    /// Shared body of [`retry_after_header_reaches_the_error`]; one call per
    /// provider client so a regression in any of them fails with its name in
    /// the panic site.
    async fn assert_retry_after_reaches_error(client: &impl ApiClient, server: TestServer) {
        let err = client
            .create_message(&StreamRequest::new(vec![]))
            .await
            .expect_err("a 429 response must surface an error");
        server.task.await.unwrap();
        match err {
            ApiError::RateLimit { retry_after, .. } => assert_eq!(
                retry_after,
                Some(Duration::from_secs(7)),
                "the Retry-After header must reach the error as the parsed delay"
            ),
            other => panic!("a 429 must surface as ApiError::RateLimit, got {other:?} ({other})"),
        }
    }

    /// Assert the local server's permanent status arrives as an auth error
    /// that is not retryable.
    ///
    /// Shared body of [`permanent_client_errors_are_not_retryable`]; one call
    /// per (provider, status) pair.
    async fn assert_permanent_status_not_retryable(
        client: &impl ApiClient,
        server: TestServer,
        status: u16,
    ) {
        let err = client
            .create_message(&StreamRequest::new(vec![]))
            .await
            .expect_err("a permanent client error must surface an error");
        server.task.await.unwrap();
        assert!(
            matches!(err, ApiError::Auth(_)),
            "an HTTP {status} must surface as ApiError::Auth, got {err:?} ({err})"
        );
        assert!(
            !err.is_retryable(),
            "an HTTP {status} is permanent — is_retryable must be false, got true ({err})"
        );
    }

    #[tokio::test]
    async fn retry_after_header_reaches_the_error() {
        #[cfg(feature = "openai")]
        {
            let server = serve(
                429,
                "Too Many Requests",
                &[("retry-after", "7")],
                r#"{"error":"slow down"}"#,
                1,
            )
            .await;
            assert_retry_after_reaches_error(&openai_at(&server.base_url), server).await;
        }
        #[cfg(feature = "anthropic")]
        {
            let server = serve(
                429,
                "Too Many Requests",
                &[("retry-after", "7")],
                r#"{"error":"slow down"}"#,
                1,
            )
            .await;
            assert_retry_after_reaches_error(&anthropic_at(&server.base_url), server).await;
        }
        #[cfg(feature = "gemini")]
        {
            let server = serve(
                429,
                "Too Many Requests",
                &[("retry-after", "7")],
                r#"{"error":"slow down"}"#,
                1,
            )
            .await;
            assert_retry_after_reaches_error(&gemini_at(&server.base_url), server).await;
        }
    }

    #[tokio::test]
    async fn permanent_client_errors_are_not_retryable() {
        #[cfg(feature = "openai")]
        {
            let unauthorized =
                serve(401, "Unauthorized", &[], r#"{"error":"invalid key"}"#, 1).await;
            assert_permanent_status_not_retryable(
                &openai_at(&unauthorized.base_url),
                unauthorized,
                401,
            )
            .await;
            let forbidden = serve(403, "Forbidden", &[], r#"{"error":"no access"}"#, 1).await;
            assert_permanent_status_not_retryable(&openai_at(&forbidden.base_url), forbidden, 403)
                .await;
        }
        #[cfg(feature = "anthropic")]
        {
            let unauthorized =
                serve(401, "Unauthorized", &[], r#"{"error":"invalid key"}"#, 1).await;
            assert_permanent_status_not_retryable(
                &anthropic_at(&unauthorized.base_url),
                unauthorized,
                401,
            )
            .await;
            let forbidden = serve(403, "Forbidden", &[], r#"{"error":"no access"}"#, 1).await;
            assert_permanent_status_not_retryable(
                &anthropic_at(&forbidden.base_url),
                forbidden,
                403,
            )
            .await;
        }
        #[cfg(feature = "gemini")]
        {
            let unauthorized =
                serve(401, "Unauthorized", &[], r#"{"error":"invalid key"}"#, 1).await;
            assert_permanent_status_not_retryable(
                &gemini_at(&unauthorized.base_url),
                unauthorized,
                401,
            )
            .await;
            let forbidden = serve(403, "Forbidden", &[], r#"{"error":"no access"}"#, 1).await;
            assert_permanent_status_not_retryable(&gemini_at(&forbidden.base_url), forbidden, 403)
                .await;
        }
    }

    /// End-to-end: the local 429 + `Retry-After: 7` server through
    /// [`StreamHandler`], with `respect_retry_after` enabled and `max_delay`
    /// clamped low so the test stays fast.
    ///
    /// The server rejects twice; the handler honours the header once (7s
    /// clamped to `max_delay`) and hard-stops on the second 429. The observed
    /// wall-clock delay therefore pins that the header's value — not the
    /// default delay — drove the retry backoff.
    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn header_retry_after_flows_to_the_rate_limit_budget() {
        let server = serve(
            429,
            "Too Many Requests",
            &[("retry-after", "7")],
            r#"{"error":"slow down"}"#,
            2,
        )
        .await;
        let client = openai_at(&server.base_url);
        let handler = StreamHandler::new()
            .with_rate_limit_config(RateLimitConfig {
                respect_retry_after: true,
                default_delay: Duration::from_millis(10),
                max_delay: Duration::from_millis(150),
                max_retries: 1,
                fallback_after_retries: 1,
                ..Default::default()
            })
            .with_timeout_config(StreamTimeoutConfig {
                initial_event_timeout: Duration::from_secs(5),
                per_event_timeout: Duration::from_secs(5),
                total_stream_timeout: Duration::from_secs(30),
                max_consecutive_timeouts: 3,
                fallback_to_non_streaming: false,
            });
        let cancel = Arc::new(CancelSignal::new());
        let request = StreamRequest::new(vec![]);
        let started = Instant::now();
        let mut stream = handler.stream_turn(&client, &request, RequestOptions::default(), &cancel);
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            if let Err(e) = item {
                terminal = Some(e);
                break;
            }
        }
        let elapsed = started.elapsed();
        server.task.await.unwrap();
        match terminal.expect("the stream must terminate with an error") {
            StreamHandlerError::StreamFailed(StreamOutcome::RateLimited { detail, .. }) => {
                assert_eq!(
                    detail.retry_after,
                    Some(Duration::from_secs(7)),
                    "the structured delay must survive into the rate-limit outcome"
                );
            }
            other => panic!("expected a terminal RateLimited outcome, got {other:?}"),
        }
        assert!(
            elapsed >= Duration::from_millis(140),
            "the retry must honour the 7s header clamped to max_delay (150ms); observed {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "the clamped backoff must keep the test fast; observed {elapsed:?}"
        );
    }
}

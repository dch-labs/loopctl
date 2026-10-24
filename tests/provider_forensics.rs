//! Provider forensics: request ids and rate-limit headers reach
//! the typed error, verbatim, at HTTP time.
//!
//! Pins the end-to-end contract the support-desk workflow depends on:
//! when a provider 429s or 500s, the `ApiError` carries the request id
//! from the response headers (`provider_meta().request_id`, and the
//! `Display` suffix `[req <id>]`), plus the verbatim `x-ratelimit-*`
//! family — and the success path refreshes the client's advisory
//! `last_rate_limit()` snapshot per response, cleared by the next
//! response that reports no family.
//!
//! Discipline checked here: capture is **verbatim**
//! (strings preserved, units never interpreted), **clean when absent**
//! (no headers ⇒ meta all `None` and no suffix — the error-string
//! goldens across the suite stay byte-stable because absence appends
//! nothing), **bounded** (the 8 KiB diagnostic-body rule still
//! truncates; the meta never becomes a size loophole), and
//! **observability-only** (classification and retry behavior are
//! untouched — `provider_error_path.rs` owns those).
//!
//! Each provider's client is feature-gated; a contract runs for every
//! client whose feature is enabled. All servers are local one-shot TCP
//! listeners — no network, no keys. Bedrock's capture sites (response
//! headers + event-stream exception frames) are pinned in the unit
//! tests of `src/provider/bedrock.rs`, its endpoint URL being derived
//! from the region and therefore not pointable at a local server.

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
    use std::time::Duration;

    use futures::StreamExt;
    use loopctl::api::error::{ApiError, RateLimitSnapshot};
    use loopctl::api::{ApiClient, StreamRequest};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    /// Where the test server's base URL and accept-count live.
    pub(crate) struct TestServer {
        /// Base URL every provider client should be pointed at.
        ///
        /// The server accepts any path — the clients append their own
        /// (`/chat/completions`, `/v1/messages`, …); the contract under
        /// test is about response headers, not routing.
        pub base_url: String,

        /// Join handle for the spawned server task; await it to assert it
        /// served every expected request.
        pub task: tokio::task::JoinHandle<()>,
    }

    /// Serve one HTTP response per accepted connection, `responses[i]`
    /// (headers) for the i-th connection, all sharing `status` and `body`.
    ///
    /// Reads each request fully (head plus `Content-Length` body) before
    /// answering, then closes the connection — pooled clients open a fresh
    /// connection per request, so per-connection variation is exact and
    /// one client can meet different header shapes across a scenario.
    async fn serve(
        status: u16,
        reason: &str,
        responses: &[&[(&str, &str)]],
        body: &str,
    ) -> TestServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // Fully render every response up front — the task owns 'static bytes.
        let rendered: Vec<String> = responses
            .iter()
            .map(|headers| {
                let mut head = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\n",
                    body.len()
                );
                for (name, value) in *headers {
                    head.push_str(name);
                    head.push_str(": ");
                    head.push_str(value);
                    head.push_str("\r\n");
                }
                head.push_str("Connection: close\r\n\r\n");
                format!("{head}{body}")
            })
            .collect();
        let task = tokio::spawn(async move {
            for response in rendered {
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

    /// Shorthand for the single-response shape most contracts need.
    async fn serve_once(
        status: u16,
        reason: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) -> TestServer {
        serve(status, reason, &[headers], body).await
    }

    /// Serve an SSE 200 whose chunked body cuts off mid-frame: the first
    /// SSE events arrive, then the reader's next chunk fetch fails
    /// because the connection ends where no chunk terminator did.
    async fn serve_truncated_stream(headers: &[(&str, &str)], first_events: &str) -> TestServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mut head = String::from("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n");
        for (name, value) in headers {
            head.push_str(name);
            head.push_str(": ");
            head.push_str(value);
            head.push_str("\r\n");
        }
        head.push_str("\r\n");
        let partial = "data: {\"id\":\"cut\",\"object\":\"chat.completion.chun";
        let events = first_events.to_string();
        let task = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            read_full_request(&mut sock).await;
            sock.write_all(head.as_bytes()).await.unwrap();
            let frame = format!("{events}{partial}");
            sock.write_all(format!("{:x}\r\n", frame.len()).as_bytes())
                .await
                .unwrap();
            sock.write_all(frame.as_bytes()).await.unwrap();
            sock.write_all(b"\r\n").await.unwrap();
            sock.flush().await.unwrap();
        });
        TestServer {
            base_url: format!("http://{addr}"),
            task,
        }
    }

    /// Read one HTTP/1.1 request (head plus body) off `sock`.
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

    /// Build an OpenAI client pointed at `base`.
    #[cfg(feature = "openai")]
    fn openai_at(base: &str) -> loopctl::provider::OpenAiClient {
        loopctl::provider::OpenAiClient::builder()
            .with_api_key("test-key")
            .with_base_url(base.to_string())
            .build()
            .unwrap()
    }

    /// Build an Anthropic client pointed at `base`.
    #[cfg(feature = "anthropic")]
    fn anthropic_at(base: &str) -> loopctl::provider::AnthropicClient {
        loopctl::provider::AnthropicClient::builder()
            .with_api_key("test-key")
            .with_base_url(base.to_string())
            .build()
            .unwrap()
    }

    /// Build a Gemini client pointed at `base`.
    #[cfg(feature = "gemini")]
    fn gemini_at(base: &str) -> loopctl::provider::GeminiClient {
        loopctl::provider::GeminiClient::builder()
            .with_api_key("test-key")
            .with_base_url(base.to_string())
            .build()
            .unwrap()
    }

    /// A 200 body each provider's non-streaming parser accepts.
    #[cfg(feature = "openai")]
    const OPENAI_OK: &str = "{\"id\":\"cmpl-1\",\"object\":\"chat.completion\",\"created\":1,\
        \"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\
        \"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\
        \"completion_tokens\":1,\"total_tokens\":2}}";
    #[cfg(feature = "anthropic")]
    const ANTHROPIC_OK: &str = "{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\
        \"model\":\"claude-test\",\"content\":[{\"type\":\"text\",\"text\":\"ok\"}],\
        \"stop_reason\":\"end_turn\",\"stop_sequence\":null,\
        \"usage\":{\"input_tokens\":1,\"output_tokens\":2}}";
    #[cfg(feature = "gemini")]
    const GEMINI_OK: &str = "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}],\
        \"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":\
        {\"promptTokenCount\":1,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}";

    /// Drive one `create_message` and hand back the typed error, having
    /// also asserted the server served its expected requests.
    async fn one_error(client: &impl ApiClient, server: TestServer) -> ApiError {
        let err = client
            .create_message(&StreamRequest::new(vec![]))
            .await
            .expect_err("the mocked failure must surface an error");
        server.task.await.unwrap();
        err
    }

    /// Assert the 500's `x-request-id` reaches both the accessor and the
    /// `Display` suffix, for one provider client.
    async fn assert_request_id_reaches_error(client: &impl ApiClient, server: TestServer) {
        let err = one_error(client, server).await;
        let meta = err.provider_meta();
        assert_eq!(
            meta.request_id.as_deref(),
            Some("abc123"),
            "the response's x-request-id must reach provider_meta(): {err}"
        );
        assert!(
            err.to_string().ends_with(" [req abc123]"),
            "the Display suffix carries the id for the paste-into-a-ticket \
             workflow: {err}"
        );
    }

    /// A 500 with `x-request-id: abc` → the typed error names the request.
    #[tokio::test]
    async fn http_error_carries_request_id() {
        let headers: &[(&str, &str)] = &[("x-request-id", "abc123")];
        #[cfg(feature = "openai")]
        {
            let server = serve_once(500, "Internal Server Error", headers, OPENAI_OK).await;
            assert_request_id_reaches_error(&openai_at(&server.base_url), server).await;
        }
        #[cfg(feature = "anthropic")]
        {
            let server = serve_once(500, "Internal Server Error", headers, ANTHROPIC_OK).await;
            assert_request_id_reaches_error(&anthropic_at(&server.base_url), server).await;
        }
        #[cfg(feature = "gemini")]
        {
            let server = serve_once(500, "Internal Server Error", headers, GEMINI_OK).await;
            assert_request_id_reaches_error(&gemini_at(&server.base_url), server).await;
        }
    }

    /// A rejected key names its request too: the `Auth` variant — the
    /// error an incident review most needs to look up at the desk —
    /// carries the response's id like every other wire failure.
    #[tokio::test]
    async fn auth_error_carries_request_id() {
        let headers: &[(&str, &str)] = &[("x-request-id", "auth-1")];
        #[cfg(feature = "openai")]
        {
            let server = serve_once(
                401,
                "Unauthorized",
                headers,
                "{\"error\":{\"message\":\"bad key\"}}",
            )
            .await;
            let err = one_error(&openai_at(&server.base_url), server).await;
            assert!(
                matches!(err, ApiError::Auth(..)),
                "401 keeps classifying as Auth: {err:?}"
            );
            assert_eq!(
                err.provider_meta().request_id.as_deref(),
                Some("auth-1"),
                "the auth rejection carries the id that identifies the check: {err}"
            );
            assert!(err.to_string().ends_with(" [req auth-1]"));
        }
        #[cfg(feature = "anthropic")]
        {
            let server = serve_once(
                403,
                "Forbidden",
                headers,
                "{\"error\":{\"message\":\"forbidden\"}}",
            )
            .await;
            let err = one_error(&anthropic_at(&server.base_url), server).await;
            assert!(
                matches!(err, ApiError::Auth(..)),
                "403 keeps classifying as Auth: {err:?}"
            );
            assert_eq!(err.provider_meta().request_id.as_deref(), Some("auth-1"));
        }
        #[cfg(feature = "gemini")]
        {
            let server = serve_once(
                403,
                "Forbidden",
                headers,
                "{\"error\":{\"message\":\"forbidden\"}}",
            )
            .await;
            let err = one_error(&gemini_at(&server.base_url), server).await;
            assert!(
                matches!(err, ApiError::Auth(..)),
                "403 keeps classifying as Auth: {err:?}"
            );
            assert_eq!(err.provider_meta().request_id.as_deref(), Some("auth-1"));
        }
    }

    /// The same 500 **without** any forensics header is clean: meta all
    /// `None`, and no suffix on the text — absence appends nothing, which
    /// is what keeps error-string goldens byte-stable across the suite.
    #[tokio::test]
    async fn no_headers_is_clean_none() {
        #[cfg(feature = "openai")]
        {
            let server = serve_once(500, "Internal Server Error", &[], OPENAI_OK).await;
            let err = one_error(&openai_at(&server.base_url), server).await;
            assert!(err.provider_meta().is_empty(), "nothing was sent");
            assert!(
                !err.to_string().contains("[req "),
                "absence must append nothing: {err}"
            );
        }
        #[cfg(feature = "anthropic")]
        {
            let server = serve_once(500, "Internal Server Error", &[], ANTHROPIC_OK).await;
            let err = one_error(&anthropic_at(&server.base_url), server).await;
            assert!(err.provider_meta().is_empty(), "nothing was sent");
            assert!(
                !err.to_string().contains("[req "),
                "absence must append nothing: {err}"
            );
        }
        #[cfg(feature = "gemini")]
        {
            let server = serve_once(500, "Internal Server Error", &[], GEMINI_OK).await;
            let err = one_error(&gemini_at(&server.base_url), server).await;
            assert!(err.provider_meta().is_empty(), "nothing was sent");
            assert!(
                !err.to_string().contains("[req "),
                "absence must append nothing: {err}"
            );
        }
    }

    /// Shared body of [`rate_limit_snapshot_captured_verbatim`]: the 429's
    /// id, verbatim family on the error meta, and the same snapshot in the
    /// client's advisory accessor — `last_rate_limit()` is inherent to
    /// each client, so the caller hands in a read of it, taken after the
    /// failing request has settled.
    async fn assert_throttle_forensics(
        client: &impl ApiClient,
        server: TestServer,
        read_last: impl Fn() -> Option<RateLimitSnapshot>,
    ) {
        let err = one_error(client, server).await;
        assert!(
            matches!(err, ApiError::RateLimit { .. }),
            "the 429 must classify as RateLimit exactly as before — meta \
             changes classification for nothing: {err:?}"
        );
        let meta = err.provider_meta();
        assert_eq!(
            meta.request_id.as_deref(),
            Some("throttle-1"),
            "the throttle's id rides the error too: {err}"
        );
        let quota = meta
            .rate_limit
            .unwrap_or_else(|| panic!("the family must reach the error meta: {err}"));
        assert_eq!(
            quota.limit.as_deref(),
            Some("500"),
            "limit is the provider's string, verbatim"
        );
        assert_eq!(
            quota.remaining.as_deref(),
            Some("42"),
            "remaining is the provider's string, verbatim — never a parsed \
             number, units are per-provider"
        );
        assert_eq!(quota.reset.as_deref(), Some("1780000000"));
        let seen = read_last().expect("a response carrying the family refreshes the accessor");
        assert_eq!(
            seen.remaining.as_deref(),
            Some("42"),
            "the accessor and the error meta read one capture"
        );
    }

    /// `x-ratelimit-*` values are captured **verbatim** as strings on both
    /// surfaces (error meta and success-path accessor).
    #[tokio::test]
    async fn rate_limit_snapshot_captured_verbatim() {
        let headers: &[(&str, &str)] = &[
            ("retry-after", "5"),
            ("x-request-id", "throttle-1"),
            ("x-ratelimit-limit", "500"),
            ("x-ratelimit-remaining", "42"),
            ("x-ratelimit-reset", "1780000000"),
        ];
        #[cfg(feature = "openai")]
        {
            let server = serve_once(429, "Too Many Requests", headers, OPENAI_OK).await;
            let client = openai_at(&server.base_url);
            assert_throttle_forensics(&client, server, || client.last_rate_limit()).await;
        }
        #[cfg(feature = "anthropic")]
        {
            let server = serve_once(429, "Too Many Requests", headers, ANTHROPIC_OK).await;
            let client = anthropic_at(&server.base_url);
            assert_throttle_forensics(&client, server, || client.last_rate_limit()).await;
        }
        #[cfg(feature = "gemini")]
        {
            let server = serve_once(429, "Too Many Requests", headers, GEMINI_OK).await;
            let client = gemini_at(&server.base_url);
            assert_throttle_forensics(&client, server, || client.last_rate_limit()).await;
        }
    }

    /// The accessor is **per response, not sticky** — on one and the same
    /// client: a first 200 carrying the family reports it, the next 200
    /// without it reads `None` ("unreported", never the stale previous
    /// number). The server varies headers per connection so the client —
    /// the object under test — never changes.
    #[tokio::test]
    async fn success_path_updates_last_rate_limit() {
        let reported: &[(&str, &str)] = &[("x-ratelimit-remaining", "7")];
        let silent: &[(&str, &str)] = &[("x-request-id", "quiet-1")];

        #[cfg(feature = "openai")]
        {
            let server = serve(200, "OK", &[reported, silent], OPENAI_OK).await;
            let client = openai_at(&server.base_url);
            client
                .create_message(&StreamRequest::new(vec![]))
                .await
                .expect("the mocked 200 must succeed");
            let seen = client
                .last_rate_limit()
                .expect("the 200 carried the family");
            assert_eq!(seen.remaining.as_deref(), Some("7"));
            client
                .create_message(&StreamRequest::new(vec![]))
                .await
                .expect("the mocked 200 must succeed");
            server.task.await.unwrap();
            assert_eq!(
                client.last_rate_limit(),
                None,
                "the accessor is the MOST RECENT response, not the most \
                 recent response that spoke"
            );
        }
        #[cfg(feature = "anthropic")]
        {
            let server = serve(200, "OK", &[reported, silent], ANTHROPIC_OK).await;
            let client = anthropic_at(&server.base_url);
            client
                .create_message(&StreamRequest::new(vec![]))
                .await
                .expect("the mocked 200 must succeed");
            let seen = client
                .last_rate_limit()
                .expect("the 200 carried the family");
            assert_eq!(seen.remaining.as_deref(), Some("7"));
            client
                .create_message(&StreamRequest::new(vec![]))
                .await
                .expect("the mocked 200 must succeed");
            server.task.await.unwrap();
            assert_eq!(client.last_rate_limit(), None);
        }
        #[cfg(feature = "gemini")]
        {
            let server = serve(200, "OK", &[reported, silent], GEMINI_OK).await;
            let client = gemini_at(&server.base_url);
            client
                .create_message(&StreamRequest::new(vec![]))
                .await
                .expect("the mocked 200 must succeed");
            let seen = client
                .last_rate_limit()
                .expect("the 200 carried the family");
            assert_eq!(seen.remaining.as_deref(), Some("7"));
            client
                .create_message(&StreamRequest::new(vec![]))
                .await
                .expect("the mocked 200 must succeed");
            server.task.await.unwrap();
            assert_eq!(client.last_rate_limit(), None);
        }
    }

    /// The 8 KiB diagnostic-body discipline is unchanged by forensics:
    /// an oversized error body still truncates, and the request id still
    /// rides the truncated error — the meta never becomes a size loophole.
    #[tokio::test]
    async fn body_bound_unchanged() {
        let huge_body = format!(
            "{{\"error\":{{\"message\":\"{}\"}}}}",
            "a".repeat(100 * 1024)
        );
        let headers: &[(&str, &str)] = &[("x-request-id", "bigbody-1")];

        #[cfg(feature = "openai")]
        {
            let server = serve_once(500, "Internal Server Error", headers, &huge_body).await;
            let err = one_error(&openai_at(&server.base_url), server).await;
            assert_forensics_survives_truncation(&err);
        }
        #[cfg(feature = "anthropic")]
        {
            let server = serve_once(500, "Internal Server Error", headers, &huge_body).await;
            let err = one_error(&anthropic_at(&server.base_url), server).await;
            assert_forensics_survives_truncation(&err);
        }
        #[cfg(feature = "gemini")]
        {
            let server = serve_once(500, "Internal Server Error", headers, &huge_body).await;
            let err = one_error(&gemini_at(&server.base_url), server).await;
            assert_forensics_survives_truncation(&err);
        }
    }

    /// Shared assertions of [`body_bound_unchanged`]: the error text is
    /// capped near the 8 KiB diagnostic bound (plus the status/class
    /// prose and the id suffix), and the id is attached regardless.
    fn assert_forensics_survives_truncation(err: &ApiError) {
        let text = err.to_string();
        assert!(
            text.len() < 16 * 1024,
            "a 100 KiB body must stay cut at the diagnostic bound, got {} bytes",
            text.len()
        );
        assert!(
            text.ends_with(" [req bigbody-1]"),
            "truncation never drops the request id: {} bytes of text",
            text.len()
        );
        assert_eq!(err.provider_meta().request_id.as_deref(), Some("bigbody-1"));
    }

    /// A mid-stream failure — the stream opened 200 and the model turned
    /// around and errored inside the body — still names its request: the
    /// response headers are captured when the stream opens, before the
    /// reader consumes the response.
    #[tokio::test]
    async fn mid_stream_error_carries_the_response_id() {
        let headers: &[(&str, &str)] = &[
            ("x-request-id", "stream-9"),
            ("Content-Type", "text/event-stream"),
        ];

        #[cfg(feature = "openai")]
        {
            // A top-level `error` payload is malformed as a chunk, so the
            // loop's stream-error parse records it and the stream fails.
            let sse = "data: {\"error\":{\"message\":\"mid-stream boom\"}}\n\n";
            let server = serve_once(200, "OK", headers, sse).await;
            let client = openai_at(&server.base_url);
            let err = drain_to_error(&client).await;
            server.task.await.unwrap();
            assert_eq!(
                err.provider_meta().request_id.as_deref(),
                Some("stream-9"),
                "the recorded stream error carries the open-response's id: {err}"
            );
        }
        #[cfg(feature = "anthropic")]
        {
            let sse = "event: error\ndata: {\"type\":\"error\",\"error\":\
                {\"type\":\"overloaded_error\",\"message\":\"oom\"}}\n\n";
            let server = serve_once(200, "OK", headers, sse).await;
            let client = anthropic_at(&server.base_url);
            let err = drain_to_error(&client).await;
            server.task.await.unwrap();
            assert_eq!(
                err.provider_meta().request_id.as_deref(),
                Some("stream-9"),
                "the recorded event:error carries the open-response's id: {err}"
            );
        }
        #[cfg(feature = "gemini")]
        {
            let sse = "data: {\"error\":{\"code\":429,\"status\":\
                \"RESOURCE_EXHAUSTED\",\"message\":\"poof\"}}\n\n";
            let server = serve_once(200, "OK", headers, sse).await;
            let client = gemini_at(&server.base_url);
            let err = drain_to_error(&client).await;
            server.task.await.unwrap();
            assert_eq!(
                err.provider_meta().request_id.as_deref(),
                Some("stream-9"),
                "the recorded chunk error carries the open-response's id: {err}"
            );
        }
    }

    /// Drive one streaming call to its first (and expected) error.
    async fn drain_to_error(client: &impl ApiClient) -> ApiError {
        let stream = client.stream_messages(&StreamRequest::new(vec![]));
        let mut stream = std::pin::pin!(stream);
        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                return err;
            }
        }
        panic!("the mocked stream must fail");
    }

    /// Drive one streaming call to its first error, counting the Ok
    /// events that preceded it — the proof the failure is mid-stream.
    async fn drain_to_reader_error(client: &impl ApiClient) -> (usize, ApiError) {
        let stream = client.stream_messages(&StreamRequest::new(vec![]));
        let mut stream = std::pin::pin!(stream);
        let mut events = 0usize;
        while let Some(item) = stream.next().await {
            match item {
                Ok(_) => events += 1,
                Err(err) => return (events, err),
            }
        }
        panic!("the mocked stream must fail");
    }

    /// A mid-stream read failure — the body cut off mid-frame after real
    /// events — names its request too: every streaming loop captures the
    /// response headers at open, before the reader consumes the response,
    /// and errors raised by the reader itself carry them. This is the
    /// reader arm; `mid_stream_error_carries_the_response_id` pins the
    /// `finish()` arm.
    #[tokio::test]
    async fn mid_stream_reader_error_carries_the_response_id() {
        let headers: &[(&str, &str)] = &[
            ("x-request-id", "reader-7"),
            ("Content-Type", "text/event-stream"),
        ];

        #[cfg(feature = "openai")]
        {
            let first = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\
                \"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\
                \"finish_reason\":null}]}\n\n";
            let server = serve_truncated_stream(headers, first).await;
            let client = openai_at(&server.base_url);
            let (events, err) = drain_to_reader_error(&client).await;
            server.task.await.unwrap();
            assert!(events > 0, "the cut must land mid-stream, after events");
            assert_eq!(
                err.provider_meta().request_id.as_deref(),
                Some("reader-7"),
                "the reader's own failure carries the open-response's id: {err}"
            );
            assert!(err.to_string().ends_with(" [req reader-7]"));
        }
        #[cfg(feature = "anthropic")]
        {
            let first = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":\
                {\"id\":\"msg_t\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\
                \"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n\
                event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\
                \"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
                event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\
                \"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n";
            let server = serve_truncated_stream(headers, first).await;
            let client = anthropic_at(&server.base_url);
            let (events, err) = drain_to_reader_error(&client).await;
            server.task.await.unwrap();
            assert!(events > 0, "the cut must land mid-stream, after events");
            assert_eq!(
                err.provider_meta().request_id.as_deref(),
                Some("reader-7"),
                "the reader's own failure carries the open-response's id: {err}"
            );
        }
        #[cfg(feature = "gemini")]
        {
            let first = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}],\
                \"role\":\"model\"}}]}\n\n";
            let server = serve_truncated_stream(headers, first).await;
            let client = gemini_at(&server.base_url);
            let (events, err) = drain_to_reader_error(&client).await;
            server.task.await.unwrap();
            assert!(events > 0, "the cut must land mid-stream, after events");
            assert_eq!(
                err.provider_meta().request_id.as_deref(),
                Some("reader-7"),
                "the reader's own failure carries the open-response's id: {err}"
            );
        }
    }

    /// A rate-limit delay of `Retry-After: 5` still parses (behavior pin:
    /// forensics changed nothing about the existing delay plumbing).
    #[tokio::test]
    async fn retry_after_still_wins_over_meta() {
        let headers: &[(&str, &str)] = &[
            ("retry-after", "5"),
            ("x-request-id", "delay-1"),
            ("x-ratelimit-remaining", "0"),
        ];
        #[cfg(feature = "openai")]
        {
            let server = serve_once(429, "Too Many Requests", headers, OPENAI_OK).await;
            let err = one_error(&openai_at(&server.base_url), server).await;
            assert_retry_delay_parsed(err);
        }
        #[cfg(feature = "anthropic")]
        {
            let server = serve_once(429, "Too Many Requests", headers, ANTHROPIC_OK).await;
            let err = one_error(&anthropic_at(&server.base_url), server).await;
            assert_retry_delay_parsed(err);
        }
        #[cfg(feature = "gemini")]
        {
            let server = serve_once(429, "Too Many Requests", headers, GEMINI_OK).await;
            let err = one_error(&gemini_at(&server.base_url), server).await;
            assert_retry_delay_parsed(err);
        }
    }

    /// The 429's advised delay still reaches the structured variant.
    fn assert_retry_delay_parsed(err: ApiError) {
        match err {
            ApiError::RateLimit { retry_after, .. } => assert_eq!(
                retry_after,
                Some(Duration::from_secs(5)),
                "the delay still parses beside the new meta"
            ),
            other => panic!("429 must stay RateLimit, got {other:?}"),
        }
    }
}

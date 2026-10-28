//! Batched REST ingest to Langfuse's `POST /api/public/ingestion`.
//!
//! A bounded shared buffer fronts a background task on the consumer's
//! `tokio` runtime: envelopes batch to `batch_size` (or a 250 ms tick),
//! ship as one `{"batch":[…]}` body with Basic auth, and failures retry
//! up to three times, backing off 250 → 500 → 1000 ms (event ids are
//! idempotent, so a retry can never double-count). A 207 whose body
//! names failed entries retries them; a 429 honors `Retry-After` for
//! the same batch; other 4xx responses are dropped loudly — the
//! payload is wrong, not the transport. Buffer overflow evicts the
//! **oldest** event and counts the loss (the guard's `dropped_count`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use tokio::sync::Notify;

/// Ingestion endpoint, appended to the configured host.
///
/// Langfuse's batched-ingest REST route; cloud and self-hosted
/// deployments share the path, so `host` is the only part that
/// differs.
const INGEST_PATH: &str = "/api/public/ingestion";

/// Queue capacity; oldest events are evicted past this depth.
///
/// The bound keeps a stalled collector from growing memory without
/// limit: events past the cap drop oldest-first and are counted, so
/// loss is visible through the guard's eviction counter rather than
/// silent.
pub(crate) const BUFFER_CAP: usize = 4096;

/// One ingestion event envelope.
///
/// Wraps a typed body with the fields every Langfuse ingest event
/// needs: a fresh idempotency id (uuid v4), the event type, and the
/// emission timestamp. A retried batch resends the same ids, so the
/// server deduplicates instead of double-counting.
#[must_use]
pub(crate) fn envelope(event_type: &str, body: &Value) -> Value {
    json!({
        "id": uuid::Uuid::new_v4().to_string(),
        "type": event_type,
        "timestamp": rfc3339_now(),
        "body": body.clone(),
    })
}

/// Insert into a JSON object body without indexing panics.
///
/// A no-op when `body` is not an object — impossible for bodies built
/// here, but the helper's own contract stays panic-free so every call
/// site inherits it.
pub(crate) fn set(body: &mut Value, key: &str, value: Value) {
    if let Some(object) = body.as_object_mut() {
        object.insert(key.to_string(), value);
    }
}

/// Shared queue between layer callbacks and the flusher task.
///
/// The mutex fronts a `VecDeque` bounded at `BUFFER_CAP`; the atomics
/// carry the eviction count, the shutdown flag, and the in-flight
/// marker the guard's bounded drain waits on. `woken` starts the
/// worker early instead of waiting out its 250 ms tick.
#[derive(Debug)]
struct Buffer {
    /// The pending envelopes, oldest at the front; the mutex is held
    /// only for push/drain — never across an HTTP call.
    queue: std::sync::Mutex<VecDeque<Value>>,

    /// Events evicted past [`BUFFER_CAP`], oldest first.
    ///
    /// Monotonic for the sink's lifetime; the guard's `dropped_events`
    /// reads it, so a collector that stopped draining surfaces as
    /// counter growth.
    dropped: AtomicU64,

    /// Set once by [`IngestSink::shutdown_blocking`]: pushes refuse
    /// afterwards and the flusher exits after its final drain.
    closed: AtomicBool,

    /// A batch has been drained and its POST has not finished.
    ///
    /// Set **before** draining (see [`drain_loop`]) so there is no
    /// window where a batch is in the worker's hands but invisible to
    /// the shutdown poller, and cleared once the POST settles.
    work_in_flight: AtomicBool,

    /// Wakes the flusher the moment an event arrives, instead of
    /// letting it idle out its 250 ms tick.
    woken: Notify,
}

impl Buffer {
    /// Lock the queue, recovering from poison.
    ///
    /// The guarded data is plain buffered envelopes, safe to keep
    /// using after any panic, so a poisoned lock never takes the
    /// flusher down with it.
    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<Value>> {
        self.queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Take up to `max` envelopes off the front of the queue.
    ///
    /// The oldest leave first, in queue order, as one batch ready to
    /// ship; an over-long queue simply keeps its remainder for the
    /// next drain.
    fn drain(&self, max: usize) -> Vec<Value> {
        let mut queue = self.lock();
        let count = max.min(queue.len());
        queue.drain(..count).collect()
    }

    /// Whether the queue holds no pending envelopes.
    ///
    /// The drain loop's work check and the shutdown pollers'
    /// completion check both read this; a `true` with no batch in
    /// flight is the pipeline's drained state.
    fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }
}

/// Handle the layer and guard use to enqueue events and shut down.
///
/// Cheap to clone — every handle shares one buffer — which lets the
/// layer hand copies to subscriber compositions while the guard alone
/// owns the shutdown path.
#[derive(Clone, Debug)]
pub(crate) struct IngestSink {
    /// The shared queue every clone enqueues through and that exactly
    /// one guard shuts down.
    buffer: std::sync::Arc<Buffer>,
}

impl IngestSink {
    /// Queue one envelope; when the buffer is at capacity, the oldest
    /// buffered event is evicted (and counted) to make room.
    pub(crate) fn push(&self, event: Value) {
        self.try_push(event);
    }

    /// Queue one envelope, reporting whether the pipeline took it —
    /// `false` only once shutdown has closed the queue.
    ///
    /// The `closed` check runs **under** the queue lock: a pre-lock
    /// check could pass, shutdown could then observe an empty queue
    /// and let the worker exit, and this push would append to a queue
    /// nobody will ever drain — a lost event. Under the lock, shutdown
    /// cannot drain-and-exit between the check and the append.
    pub(crate) fn try_push(&self, event: Value) -> bool {
        {
            let mut queue = self.buffer.lock();
            if self.buffer.closed.load(Ordering::SeqCst) {
                return false;
            }
            while queue.len() >= BUFFER_CAP {
                queue.pop_front();
                self.buffer.dropped.fetch_add(1, Ordering::Relaxed);
            }
            queue.push_back(event);
        }
        self.buffer.woken.notify_waiters();
        true
    }

    /// Eviction count so far (the `loopctl.langfuse.ingest.dropped` signal).
    ///
    /// Counts events lost to the capacity bound since the sink was
    /// created; hosts alert on its growth to notice a collector that
    /// stopped draining.
    #[must_use]
    pub fn dropped_count(&self) -> u64 {
        self.buffer.dropped.load(Ordering::Relaxed)
    }

    /// Close the queue without waiting for the drain.
    ///
    /// The current-thread-runtime drop path: parking the runtime's
    /// only worker inside `Drop` would starve the flusher for the
    /// whole bounded wait, so the queue closes (pushes refuse; a
    /// runtime that keeps being driven drains what it can) and the
    /// dropping thread returns at once.
    pub(crate) fn close(&self) {
        self.buffer.closed.store(true, Ordering::SeqCst);
        self.buffer.woken.notify_waiters();
    }

    /// Close the queue, let the worker flush what remains, and wait
    /// (bounded) for it to finish. For `Drop` and shutdown paths on
    /// ordinary threads — never call from a thread the runtime drives.
    pub(crate) fn shutdown_blocking(&self, timeout: Duration) {
        self.close();
        let deadline = std::time::Instant::now()
            .checked_add(timeout)
            .unwrap_or_else(std::time::Instant::now);
        while (!self.buffer.is_empty() || self.buffer.work_in_flight.load(Ordering::SeqCst))
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Await the same drain cooperatively (the runtime-thread-safe
    /// twin of [`shutdown_blocking`](Self::shutdown_blocking), which
    /// exists because `Drop` cannot `await`).
    ///
    /// `true` once the queue is empty with no batch in flight; `false`
    /// when `timeout` elapsed first.
    pub(crate) async fn wait_drained(&self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now()
            .checked_add(timeout)
            .unwrap_or_else(tokio::time::Instant::now);
        loop {
            if self.buffer.is_empty() && !self.buffer.work_in_flight.load(Ordering::SeqCst) {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// Start the background flusher; returns the enqueue handle.
///
/// Spawns the worker on the caller's runtime handle and clamps
/// `batch_size` into `1..=BUFFER_CAP`. The sink stays usable until a
/// guard's shutdown closes the queue; the task then drains what
/// remains and exits.
pub(crate) fn spawn_flusher(
    host: &str,
    public_key: &str,
    secret_key: &str,
    batch_size: usize,
    runtime: &tokio::runtime::Handle,
) -> IngestSink {
    let buffer = std::sync::Arc::new(Buffer {
        queue: std::sync::Mutex::new(VecDeque::new()),
        dropped: AtomicU64::new(0),
        closed: AtomicBool::new(false),
        work_in_flight: AtomicBool::new(false),
        woken: Notify::new(),
    });
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default();
    let url = format!("{}{INGEST_PATH}", host.trim_end_matches('/'));
    runtime.spawn(run_flusher(
        client,
        url,
        basic_auth(public_key, secret_key),
        batch_size.clamp(1, BUFFER_CAP),
        std::sync::Arc::clone(&buffer),
    ));
    IngestSink { buffer }
}

/// Worker: wait for events or the 250 ms tick, ship batches; once closed
/// and drained, post the remainder and exit.
async fn run_flusher(
    client: reqwest::Client,
    url: String,
    auth: String,
    batch_size: usize,
    buffer: std::sync::Arc<Buffer>,
) {
    drain_loop(buffer, batch_size, move |batch| {
        let client = client.clone();
        let url = url.clone();
        let auth = auth.clone();
        async move { post(&client, &url, &auth, batch).await }
    })
    .await;
}

/// The flusher's body, transport-agnostic: ship every non-empty drain
/// through `ship` until the queue is closed and empty.
///
/// The in-flight marker is set **before** the drain, never after: a
/// shutdown poller reading `queue-empty && !work_in_flight` between a
/// drain and the marker store would release the guard mid-POST and let
/// the process exit while a batch was still in the worker's hands
/// (lost, uncounted). Setting the flag first means the queue can never
/// look empty-and-idle while a batch exists anywhere in the pipeline.
async fn drain_loop<S, F>(buffer: std::sync::Arc<Buffer>, batch_size: usize, ship: S)
where
    S: Fn(Vec<Value>) -> F,
    F: std::future::Future<Output = ()>,
{
    loop {
        tokio::select! {
            () = buffer.woken.notified() => {}
            () = tokio::time::sleep(Duration::from_millis(250)) => {}
        }
        while !buffer.is_empty() {
            buffer.work_in_flight.store(true, Ordering::SeqCst);
            let batch = buffer.drain(batch_size);
            ship(batch).await;
            buffer.work_in_flight.store(false, Ordering::SeqCst);
        }
        if buffer.closed.load(Ordering::SeqCst) {
            return;
        }
    }
}

/// POST one ingestion batch, applying the transport's retry contract.
///
/// A 2xx settles the batch; Langfuse's 207 multi-status settles only
/// when its body names no failed entries — a body carrying `errors`
/// retries the same ids (idempotent server-side), and a body that
/// cannot be parsed settles rather than guessing. A 429 honors the
/// response's `Retry-After` (delta-seconds, capped at 5 s) and retries
/// the same batch; other 4xx responses are dropped with a warning
/// (the payload is wrong, not the transport); 5xx answers and
/// transport failures retry up to three times on a 250 → 500 →
/// 1000 ms ladder. A batch that exhausts the ladder is dropped with a
/// warning and counted nowhere else.
async fn post(client: &reqwest::Client, url: &str, auth: &str, batch: Vec<Value>) {
    if batch.is_empty() {
        return;
    }
    let body = json!({ "batch": batch });
    let mut delay = Duration::from_millis(250);
    for attempt in 0..4_u32 {
        let request = client
            .post(url)
            .header(reqwest::header::AUTHORIZATION, auth)
            .json(&body);
        let response = match request.send().await {
            Ok(response) => response,
            Err(err) => {
                tracing::debug!(
                    target: "loopctl::langfuse",
                    %err, attempt, "langfuse ingest transport failed, will retry"
                );
                if attempt.saturating_add(1) < 4 {
                    tokio::time::sleep(delay).await;
                    delay = delay.saturating_mul(2).min(Duration::from_secs(5));
                }
                continue;
            }
        };
        let status = response.status();
        if status.as_u16() == 207 {
            let failed = multi_status_failures(response).await;
            if failed == 0 {
                return;
            }
            tracing::debug!(
                target: "loopctl::langfuse",
                failed,
                attempt,
                "langfuse 207 carried failed entries, will retry them"
            );
        } else if status.is_success() {
            return;
        } else if status.as_u16() == 429 {
            if let Some(advised) = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(retry_after_delay)
            {
                delay = advised.min(Duration::from_secs(5));
            }
            tracing::debug!(
                target: "loopctl::langfuse",
                attempt,
                "langfuse rate-limited the ingest, will retry the same batch"
            );
        } else if status.is_client_error() {
            tracing::warn!(
                target: "loopctl::langfuse",
                status = status.as_u16(),
                "langfuse rejected an ingest batch (dropped as unrecoverable)"
            );
            return;
        } else {
            tracing::debug!(
                target: "loopctl::langfuse",
                status = status.as_u16(),
                attempt,
                "langfuse ingest failed, will retry"
            );
        }
        if attempt.saturating_add(1) < 4 {
            tokio::time::sleep(delay).await;
            delay = delay.saturating_mul(2).min(Duration::from_secs(5));
        }
    }
    tracing::warn!(
        target: "loopctl::langfuse",
        "langfuse ingest retries exhausted; batch dropped"
    );
}

/// Count the failed entries a 207 multi-status body names.
///
/// Langfuse answers a partially-accepted batch with
/// `{"successes": […], "errors": […]}`; a body whose `errors` array
/// is absent or empty settles the batch, and a body that cannot be
/// parsed settles too — the client cannot distinguish failures it
/// never read, and the envelope ids make any re-send idempotent, so
/// settling a foreign shape never double-counts a well-formed retry
/// path.
async fn multi_status_failures(response: reqwest::Response) -> usize {
    let parsed = response.json::<Value>().await.ok();
    parsed
        .and_then(|body| body.get("errors").and_then(Value::as_array).cloned())
        .map_or(0, |errors| errors.len())
}

/// Parse a `Retry-After` value as delta-seconds.
///
/// Langfuse's rate-limit answers advise seconds — the documented
/// shape. An `Option`-shaped `None` (HTTP-date, junk) falls back to
/// the caller's ladder; the caller caps whatever parses.
fn retry_after_delay(value: &str) -> Option<Duration> {
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}

/// Encode the ingestion credentials as a Basic `Authorization` header.
///
/// Standard-alphabet base64 of `public_key:secret_key`, prefixed with
/// `Basic ` — the exact shape Langfuse's server decodes, so the
/// alphabet and the key order are wire contract, not style: a URL-safe
/// alphabet or a swapped pair fails auth with no local symptom.
fn basic_auth(public_key: &str, secret_key: &str) -> String {
    use base64::Engine as _;
    let token =
        base64::engine::general_purpose::STANDARD.encode(format!("{public_key}:{secret_key}"));
    format!("Basic {token}")
}

/// RFC 3339 UTC timestamp without dependencies.
///
/// The wall-clock form of [`rfc3339`], used for envelope timestamps
/// and observation end times; whole-second precision matches Langfuse
/// ingest's own resolution.
pub(crate) fn rfc3339_now() -> String {
    rfc3339(SystemTime::now())
}

/// Format a `SystemTime` as RFC 3339 UTC (`2026-10-28T09:00:00Z`).
///
/// Broken-down civil-time math over unix seconds on the pattern the
/// core trajectory module uses: every step rides checked/saturating
/// methods and euclid division, so an intermediate overflow surfaces
/// as the bare unix-seconds rendering instead of a panic or a wrap.
pub(crate) fn rfc3339(when: SystemTime) -> String {
    let secs = when
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    civil_from_days(secs.div_euclid(86_400)).map_or_else(
        || format!("{secs}"),
        |(year, month, day)| {
            let secs_of_day = secs.rem_euclid(86_400);
            format!(
                "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
                secs_of_day.div_euclid(3_600),
                secs_of_day.div_euclid(60).rem_euclid(60),
                secs_of_day.rem_euclid(60),
            )
        },
    )
}

/// Days-since-epoch → (year, month, day) — Howard Hinnant's algorithm.
///
/// Restructured onto checked/saturating methods and euclid division:
/// `None` on an intermediate overflow, which unix timestamps from any
/// realistic clock cannot produce.
fn civil_from_days(days: u64) -> Option<(u64, u64, u64)> {
    let z = days.checked_add(719_468)?;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = doe
        .saturating_sub(doe.div_euclid(1_460))
        .saturating_add(doe.div_euclid(36_524))
        .saturating_sub(doe.div_euclid(146_096))
        .div_euclid(365);
    let doy = doe.saturating_sub(
        yoe.saturating_mul(365)
            .saturating_add(yoe.div_euclid(4))
            .saturating_sub(yoe.div_euclid(100)),
    );
    let mp = doy.saturating_mul(5).checked_add(2)?.div_euclid(153);
    let day = doy
        .saturating_sub(mp.saturating_mul(153).checked_add(2)?.div_euclid(5))
        .saturating_add(1);
    let month = if mp < 10 {
        mp.saturating_add(3)
    } else {
        mp.saturating_sub(9)
    };
    let year = yoe
        .saturating_add(era.saturating_mul(400))
        .saturating_add(u64::from(month <= 2));
    Some((year, month, day))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::missing_panics_doc, clippy::unwrap_used, clippy::expect_used)]

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn rfc3339_matches_known_epochs() {
        assert_eq!(
            rfc3339(UNIX_EPOCH + Duration::from_secs(0)),
            "1970-01-01T00:00:00Z"
        );
        // 2001-09-09T01:46:40Z (Unix time 1e9).
        assert_eq!(
            rfc3339(UNIX_EPOCH + Duration::from_secs(1_000_000_000)),
            "2001-09-09T01:46:40Z"
        );
    }

    #[test]
    fn basic_auth_uses_the_std_alphabet() {
        assert_eq!(basic_auth("pk", "sk"), "Basic cGs6c2s=");
    }

    /// Serve scripted one-shot responses, counting requests.
    ///
    /// Each accepted connection is read to its body (Content-Length),
    /// counted, and answered with the round's status line, an optional
    /// header, and body — the minimal collector the wire tests need.
    fn serve_rounds(
        listener: &TcpListener,
        hits: &Arc<AtomicUsize>,
        rounds: Vec<(String, Option<String>, String)>,
    ) {
        for (status, header, body) in rounds {
            let Ok((mut stream, _peer)) = listener.accept() else {
                return;
            };
            let mut buf = Vec::new();
            let mut byte = [0_u8; 1];
            while !buf.ends_with(b"\r\n\r\n") {
                if stream.read(&mut byte).unwrap_or(0) == 0 {
                    break;
                }
                buf.push(byte[0]);
            }
            let header_text = String::from_utf8_lossy(&buf).to_string();
            let length: usize = header_text
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|rest| rest.trim().parse().unwrap_or(0))
                })
                .unwrap_or(0);
            let mut request_body = vec![0_u8; length];
            if length > 0 {
                stream.read_exact(&mut request_body).unwrap_or(());
            }
            hits.fetch_add(1, Ordering::SeqCst);
            let extra = header.map_or(String::new(), |value| format!("{value}\r\n"));
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\n{extra}\
                 connection: close\r\ncontent-length: {}\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap_or(());
            stream.flush().unwrap_or(());
        }
    }

    fn one_shot_endpoint(
        rounds: Vec<(String, Option<String>, String)>,
    ) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let addr = listener.local_addr().unwrap();
        let counted = Arc::clone(&hits);
        std::thread::spawn(move || serve_rounds(&listener, &counted, rounds));
        (format!("http://{addr}/api/public/ingestion"), hits)
    }

    #[tokio::test]
    async fn a_207_with_failed_entries_retries_the_batch() {
        let (url, hits) = one_shot_endpoint(vec![
            (
                "207 Multi-Status".to_string(),
                None,
                r#"{"successes":[],"errors":[{"id":"e1","status":500,"message":"boom"}]}"#
                    .to_string(),
            ),
            (
                "207 Multi-Status".to_string(),
                None,
                r#"{"successes":[],"errors":[]}"#.to_string(),
            ),
        ]);
        let client = reqwest::Client::new();
        post(
            &client,
            &url,
            &basic_auth("pk", "sk"),
            vec![json!({ "n": 1 })],
        )
        .await;
        assert!(
            hits.load(Ordering::SeqCst) >= 2,
            "the errored 207 must re-send the batch, not settle over failed entries"
        );
    }

    #[tokio::test]
    async fn a_429_honors_retry_after_and_retries_the_same_batch() {
        let (url, hits) = one_shot_endpoint(vec![
            (
                "429 Too Many Requests".to_string(),
                Some("retry-after: 0".to_string()),
                "{}".to_string(),
            ),
            (
                "207 Multi-Status".to_string(),
                None,
                r#"{"successes":[],"errors":[]}"#.to_string(),
            ),
        ]);
        let client = reqwest::Client::new();
        post(
            &client,
            &url,
            &basic_auth("pk", "sk"),
            vec![json!({ "n": 1 })],
        )
        .await;
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "the rate-limited batch retries once advised and settles on the clean 207"
        );
    }

    #[test]
    fn a_push_racing_shutdown_cannot_append_to_a_closed_queue() {
        let buffer = Arc::new(Buffer {
            queue: std::sync::Mutex::new(VecDeque::new()),
            dropped: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            work_in_flight: AtomicBool::new(false),
            woken: Notify::new(),
        });
        let holder = buffer.lock();
        let pusher_sink = IngestSink {
            buffer: Arc::clone(&buffer),
        };
        let pusher = std::thread::spawn(move || pusher_sink.try_push(json!({ "n": 1 })));
        std::thread::sleep(Duration::from_millis(50));
        buffer.closed.store(true, Ordering::SeqCst);
        drop(holder);
        let accepted = pusher.join().unwrap();
        assert!(
            !accepted,
            "a push racing shutdown reports the queue as closed"
        );
        assert!(
            buffer.lock().is_empty(),
            "a push racing shutdown must not append to a closed queue"
        );
    }

    #[test]
    fn buffer_evicts_oldest_past_capacity() {
        let buffer = std::sync::Arc::new(Buffer {
            queue: std::sync::Mutex::new(VecDeque::new()),
            dropped: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            work_in_flight: AtomicBool::new(false),
            woken: Notify::new(),
        });
        let sink = IngestSink {
            buffer: std::sync::Arc::clone(&buffer),
        };
        for i in 0..(BUFFER_CAP + 8) {
            sink.push(json!({ "n": i }));
        }
        assert_eq!(sink.dropped_count(), 8);
        let drained = buffer.drain(BUFFER_CAP);
        assert_eq!(
            drained
                .first()
                .and_then(|v| v.get("n"))
                .and_then(Value::as_u64),
            Some(8)
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn an_in_flight_batch_keeps_the_shutdown_poller_waiting() {
        use std::sync::Arc;

        let buffer = Arc::new(Buffer {
            queue: std::sync::Mutex::new(VecDeque::new()),
            dropped: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            work_in_flight: AtomicBool::new(false),
            woken: Notify::new(),
        });
        let sink = IngestSink {
            buffer: Arc::clone(&buffer),
        };
        sink.push(json!({ "n": 1 }));

        let shipping = Arc::new(AtomicBool::new(false));
        let shipping_seen = Arc::clone(&shipping);
        let worker_buffer = Arc::clone(&buffer);
        let worker = tokio::spawn(async move {
            drain_loop(worker_buffer, 8, move |_batch| {
                let shipping = Arc::clone(&shipping_seen);
                async move {
                    shipping.store(true, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(30)).await;
                }
            })
            .await;
        });
        buffer.woken.notify_waiters();

        while !shipping.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }

        let drained_early = sink.wait_drained(Duration::from_millis(20)).await;
        assert!(
            !drained_early,
            "the queue is empty mid-POST; only an in-flight marker keeps \
             the poller from releasing the guard over a live batch"
        );

        let drained_late = sink.wait_drained(Duration::from_secs(1)).await;
        assert!(drained_late, "once the POST settles the drain completes");

        buffer.closed.store(true, Ordering::SeqCst);
        buffer.woken.notify_waiters();
        let finished = tokio::time::timeout(Duration::from_secs(1), worker).await;
        assert!(matches!(finished, Ok(Ok(()))), "the closed flusher exits");
    }
}

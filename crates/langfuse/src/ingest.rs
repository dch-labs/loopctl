//! Batched REST ingest to Langfuse's `POST /api/public/ingestion`.
//!
//! A bounded shared buffer fronts a background task on the consumer's
//! `tokio` runtime: envelopes batch to `batch_size` (or a 250 ms tick),
//! ship as one `{"batch":[…]}` body with Basic auth, and failures retry
//! up to three times, backing off 250 → 500 → 1000 ms (stable event
//! ids, so a re-sent envelope the server never stored is the same
//! event, never a second one). A 207 whose body names failed entries
//! resends exactly those envelopes — Langfuse does not deduplicate
//! re-ingested events, so the accepted half of a partial batch is
//! never re-sent; a 429 honors `Retry-After` for the same batch; other
//! 4xx responses are dropped loudly — the payload is wrong, not the
//! transport. Buffer overflow evicts the **oldest** event and counts
//! the loss (the guard's `dropped_count`).

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
    /// The pending envelopes, oldest at the front.
    ///
    /// The mutex is held only for a push or a drain — never across an
    /// HTTP call — so a slow collector cannot block a layer callback
    /// trying to enqueue. Capacity is [`BUFFER_CAP`]: a push past it
    /// evicts the oldest envelope rather than refusing or growing.
    queue: std::sync::Mutex<VecDeque<Value>>,

    /// Events evicted past [`BUFFER_CAP`], oldest first.
    ///
    /// Monotonic for the sink's lifetime; the guard's `dropped_events`
    /// reads it, so a collector that stopped draining surfaces as
    /// counter growth.
    dropped: AtomicU64,

    /// Set once when the pipeline shuts down: pushes refuse afterwards.
    ///
    /// [`IngestSink::close`] is the writer, called directly or through
    /// [`IngestSink::shutdown_blocking`], and nothing clears it — a sink
    /// closes once and stays closed. The flusher reads the same flag as
    /// its permission to exit, and reads it together with emptiness
    /// under the queue lock ([`Buffer::closed_and_empty`]) so an
    /// accepted push can never find the worker already gone.
    closed: AtomicBool,

    /// A batch has been drained and its POST has not finished.
    ///
    /// Set **before** draining (see [`drain_loop`]) so there is no
    /// window where a batch is in the worker's hands but invisible to
    /// the shutdown poller, and cleared once the POST settles.
    work_in_flight: AtomicBool,

    /// Wakes the flusher the moment an event arrives.
    ///
    /// Without it the worker would sit out its 250 ms tick before
    /// noticing a push, making latency to first export the tick
    /// interval rather than the round trip. Every enqueue path notifies
    /// it, and so do both close paths — a worker parked on the tick has
    /// to wake to observe a shutdown too.
    woken: Notify,
}

/// Test seam for the exit check: when set, the next
/// [`Buffer::closed_and_empty`] records whether the queue lock was
/// still held between its two reads.
///
/// The window cannot be probed by interleaving a push — the window
/// holds the lock, which is the property under test — so the observable
/// is lock-held-ness itself, read with `try_lock` so it never blocks:
/// the fixed single-lock read records `true`, a read split into two
/// lock acquisitions would record `false`.
#[cfg(test)]
static EXIT_WINDOW_PROBE: AtomicBool = AtomicBool::new(false);

/// The `try_lock` outcome the last probed exit window recorded.
///
/// `true` means the lock was held across both facts.
#[cfg(test)]
static EXIT_WINDOW_LOCK_HELD: AtomicBool = AtomicBool::new(true);

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

    /// Whether the queue is closed and holds no pending envelopes,
    /// decided under one lock.
    ///
    /// The flusher's exit condition. Reading emptiness and then
    /// `closed` in two steps lets a push slip between the reads: the
    /// push sees an open queue and reports success, the flusher then
    /// sees `closed` and exits — stranding an event nobody will ever
    /// drain. Under one lock the two facts cannot disagree: either the
    /// push lands before the exit check and keeps the worker running,
    /// or the close lands first and the push refuses.
    fn closed_and_empty(&self) -> bool {
        let queue = self.lock();
        let empty = queue.is_empty();
        #[cfg(test)]
        if EXIT_WINDOW_PROBE.swap(false, Ordering::SeqCst) {
            EXIT_WINDOW_LOCK_HELD.store(self.queue.try_lock().is_err(), Ordering::SeqCst);
        }
        self.closed.load(Ordering::SeqCst) && empty
    }
}

/// Handle the layer and guard use to enqueue events and shut down.
///
/// Cheap to clone — every handle shares one buffer — which lets the
/// layer hand copies to subscriber compositions while the guard alone
/// owns the shutdown path.
#[derive(Clone, Debug)]
pub(crate) struct IngestSink {
    /// The shared queue every clone enqueues through.
    ///
    /// Cloning a sink hands the new handle the same buffer, so spans
    /// recorded through any layer composition land in one queue and one
    /// worker ships them. Exactly one guard owns the shutdown path that
    /// closes that queue, which is what keeps a clone from ending the
    /// pipeline out from under the handles still writing to it.
    buffer: std::sync::Arc<Buffer>,
}

impl IngestSink {
    /// Queue one envelope, discarding whether the pipeline took it.
    ///
    /// For callers with nowhere to put a refusal: a layer closing a
    /// span on a shutting-down runtime cannot retry, store, or surface
    /// an event it cannot queue. Past [`BUFFER_CAP`] the oldest buffered
    /// event is evicted to make room and counted in
    /// [`IngestSink::dropped_count`]. A caller that can act on the
    /// answer uses [`IngestSink::try_push`].
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

    /// Close the queue and wait, bounded, for the worker to finish.
    ///
    /// For `Drop` and shutdown paths on ordinary threads: the wait
    /// covers whatever is still queued plus a batch already in flight,
    /// and returns at `timeout` whether or not the drain completed — a
    /// wedged collector costs a late process exit, never a hung one.
    /// Never call it from a thread the runtime drives, where parking
    /// the only worker starves the flusher this drain is waiting on;
    /// that path closes with [`IngestSink::close`] and awaits
    /// [`IngestSink::wait_drained`] instead.
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

/// Worker: wait for events or the 250 ms tick, ship batches, exit when
/// closed and drained.
///
/// Spawned by [`spawn_flusher`] with the sink's own buffer. It exists
/// only to bind the transport — client, URL, and Basic auth header — to
/// [`drain_loop`], which owns the batching and the exit rule; keeping
/// the two apart is what lets the drain run against a stub ship
/// function with no HTTP in the picture.
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
/// The exit check reads both facts under one lock
/// ([`Buffer::closed_and_empty`]): a push accepted moments before the
/// close must not find the worker already gone, and the `while` loop's
/// separate emptiness read only ever runs again when the exit check
/// declined — so no accepted event can be left behind.
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
        if buffer.closed_and_empty() {
            return;
        }
    }
}

/// POST one ingestion batch, applying the transport's retry contract.
///
/// A 2xx settles the batch; Langfuse's 207 multi-status settles only
/// when its body is a well-formed answer naming no failed entries. A
/// readable body naming failed entries resends **only the envelopes
/// whose ids it lists**: Langfuse does not deduplicate re-ingested
/// events, so an envelope the server already accepted must not be sent
/// again — a failure id matching no envelope in the batch is foreign
/// (named in a warning, never dropped in silence), and a 207 naming
/// only foreign ids has nothing of ours left to resend, so it settles.
/// An answer that cannot be read as a multi-status body at all retries
/// the whole batch: settling there would drop the rejected entries on
/// the word of a response never parsed, which loses data outright,
/// where a re-send of stored events only risks a duplicate. A 429
/// honors the response's `Retry-After` (delta-seconds or HTTP-date,
/// capped at 5 s) and retries the same batch — nothing was stored on a
/// 429, so the same ids are safe to send again; other 4xx responses
/// are dropped with a warning (the payload is wrong, not the
/// transport); 5xx answers and transport failures retry up to three
/// times on a 250 → 500 → 1000 ms ladder. A batch that exhausts the
/// ladder is dropped with a warning and counted nowhere else.
async fn post(client: &reqwest::Client, url: &str, auth: &str, mut batch: Vec<Value>) {
    if batch.is_empty() {
        return;
    }
    let mut body = json!({ "batch": batch });
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
            match multi_status_failures(response).await {
                Ack::Settled => return,
                Ack::Failed { failed } => {
                    let rejected: std::collections::HashSet<&str> =
                        failed.iter().map(String::as_str).collect();
                    batch.retain(|envelope| {
                        envelope
                            .get("id")
                            .and_then(Value::as_str)
                            .is_some_and(|id| rejected.contains(id))
                    });
                    let foreign = rejected.len().saturating_sub(batch.len());
                    if foreign > 0 {
                        tracing::warn!(
                            target: "loopctl::langfuse",
                            foreign,
                            attempt,
                            "langfuse 207 named failures for envelopes this \
                             client never sent"
                        );
                    }
                    if batch.is_empty() {
                        tracing::warn!(
                            target: "loopctl::langfuse",
                            attempt,
                            "langfuse 207 named no envelope of ours; nothing \
                             retryable remains, batch settled"
                        );
                        return;
                    }
                    body = json!({ "batch": batch });
                    tracing::debug!(
                        target: "loopctl::langfuse",
                        failed = batch.len(),
                        attempt,
                        "langfuse 207 carried failed entries, will resend \
                         only those envelopes"
                    );
                }
                Ack::Unreadable => tracing::debug!(
                    target: "loopctl::langfuse",
                    attempt,
                    "langfuse answered a 207 this client cannot read, will retry the batch"
                ),
            }
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

/// What a 207 multi-status answer says about the batch it accepted.
///
/// Langfuse answers a partially-processed batch with a JSON body naming
/// the accepted and the rejected entries, so the answer has to be read
/// before the batch can settle. The three verdicts separate "the
/// server says every entry landed" from "the server says these ids did
/// not" from "the server said something unreadable". Only the first
/// ends the batch's life outright; the second ends it by naming what
/// to resend, and the third is the one verdict that risks a duplicate,
/// because an unreadable answer leaves no way to tell accepted from
/// rejected and dropping it would lose the rejected entries for good.
#[derive(Debug)]
enum Ack {
    /// Settle the batch: a well-formed answer naming no failed entries.
    ///
    /// A complete success on the server's own account — an object whose
    /// `errors` list is absent or empty — so the batch is done and the
    /// flusher moves on to the next one.
    Settled,

    /// Resend the rejected envelopes: the answer names their ids.
    ///
    /// The partial-success shape: some entries landed, these did not,
    /// and Langfuse does not deduplicate re-ingested events — an
    /// accepted envelope sent again becomes a second record, inflating
    /// counts and dashboards. The retry therefore carries only the
    /// envelopes whose ids are listed here, unchanged, so the accepted
    /// half of the batch is never re-sent.
    Failed {
        /// The ids of the entries the body lists under `errors`.
        ///
        /// Matched against the batch's own envelope ids to select what
        /// to resend; an id matching no envelope is foreign to this
        /// client — named in a warning rather than dropped in silence,
        /// because the mismatch says the server and this client
        /// disagree about what was sent.
        failed: Vec<String>,
    },

    /// Retry the batch: the answer is not a multi-status body the client
    /// can read.
    ///
    /// Unparsable JSON, a non-object root, and an `errors` field that is
    /// not an array all mean the same thing: whatever the 207 claimed,
    /// the claim is unverified, so the batch has not been shown to land.
    /// Retrying costs one round-trip and buys nothing when the server
    /// really did take everything — idempotent event ids make that cost
    /// the whole of the downside — which is the trade preferred to
    /// dropping entries on the strength of a body nobody read.
    Unreadable,
}

/// Read a 207 multi-status body into its [`Ack`] verdict.
///
/// The body is decoded off the response here and judged by
/// [`parse_multi_status`], which owns the verdict rules; an answer that
/// does not decode as JSON at all is [`Ack::Unreadable`] on the spot.
async fn multi_status_failures(response: reqwest::Response) -> Ack {
    match response.json::<Value>().await {
        Ok(body) => parse_multi_status(&body),
        Err(_) => Ack::Unreadable,
    }
}

/// Judge a decoded 207 body: settle, name the rejected ids, or refuse
/// to read it.
///
/// A well-formed object whose `errors` is absent or empty settles; a
/// present `errors` array names the rejected envelopes **by id**, and
/// one entry without a readable string id makes the whole answer
/// [`Ack::Unreadable`] — a failure this client cannot attribute to an
/// envelope cannot be retried selectively, so it takes the whole-batch
/// trade an unparsable body gets. `errors` present but not an array,
/// and a root that is not an object, are unreadable for the same
/// reason.
fn parse_multi_status(body: &Value) -> Ack {
    if !body.is_object() {
        return Ack::Unreadable;
    }
    let Some(errors) = body.get("errors") else {
        return Ack::Settled;
    };
    let Some(entries) = errors.as_array() else {
        return Ack::Unreadable;
    };
    if entries.is_empty() {
        return Ack::Settled;
    }
    let mut failed = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            return Ack::Unreadable;
        };
        failed.push(id.to_string());
    }
    Ack::Failed { failed }
}

/// Parse a `Retry-After` value: delta-seconds or an HTTP-date.
///
/// Langfuse's rate-limit answers advise seconds; intermediaries and
/// self-hosted collectors answer with an HTTP-date instead. A future
/// date becomes the delay it advertises, a date already past becomes
/// an immediate retry, and junk is `None` — the caller's ladder covers
/// what cannot be read, and the caller caps whatever does parse.
fn retry_after_delay(value: &str) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds.min(u64::from(u32::MAX))));
    }
    let when = httpdate::parse_http_date(value).ok()?;
    Some(match when.duration_since(SystemTime::now()) {
        Ok(ahead) => ahead,
        Err(_) => Duration::ZERO,
    })
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
    #![allow(
        clippy::missing_panics_doc,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )]

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

    /// Serve scripted one-shot responses, counting requests and
    /// capturing each request body.
    ///
    /// Each accepted connection is read to its body (Content-Length),
    /// counted, captured, and answered with the round's status line, an
    /// optional header, and body — the minimal collector the wire tests
    /// need, with the bytes each round actually received.
    fn serve_rounds(
        listener: &TcpListener,
        hits: &Arc<AtomicUsize>,
        bodies: &Arc<std::sync::Mutex<Vec<String>>>,
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
            bodies
                .lock()
                .expect("capture lock")
                .push(String::from_utf8_lossy(&request_body).into_owned());
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
    ) -> (String, Arc<AtomicUsize>, Arc<std::sync::Mutex<Vec<String>>>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let bodies: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let addr = listener.local_addr().unwrap();
        let counted = Arc::clone(&hits);
        let captured = Arc::clone(&bodies);
        std::thread::spawn(move || {
            serve_rounds(&listener, &counted, &captured, rounds);
        });
        (format!("http://{addr}/api/public/ingestion"), hits, bodies)
    }

    /// The envelope ids one request body carried, in order.
    ///
    /// Parses a captured `{"batch": […]}` body down to its envelopes'
    /// `id` fields, so a test can assert exactly which envelopes a
    /// retry re-sent.
    fn sent_envelope_ids(raw: &str) -> Vec<String> {
        serde_json::from_str::<Value>(raw)
            .expect("captured body parses")
            .get("batch")
            .and_then(Value::as_array)
            .expect("a request carries a batch array")
            .iter()
            .map(|envelope| {
                envelope
                    .get("id")
                    .and_then(Value::as_str)
                    .expect("test envelopes carry ids")
                    .to_string()
            })
            .collect()
    }

    #[tokio::test]
    async fn a_207_resends_only_the_envelopes_the_server_rejected() {
        let (url, hits, bodies) = one_shot_endpoint(vec![
            (
                "207 Multi-Status".to_string(),
                None,
                r#"{"successes":[{"id":"e2","status":201}],"errors":[{"id":"e1","status":500,"message":"boom"}]}"#
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
            vec![
                json!({ "id": "e1", "type": "trace-create", "body": {} }),
                json!({ "id": "e2", "type": "span-create", "body": {} }),
            ],
        )
        .await;
        let captured = bodies.lock().expect("capture lock").clone();
        let first_request = captured.first().expect("first request captured");
        let retry_request = captured.get(1).expect("the retry request captured");
        assert_eq!(captured.len(), 2, "the errored 207 retries, then settles");
        assert_eq!(
            sent_envelope_ids(first_request),
            ["e1".to_string(), "e2".to_string()],
            "the first request carries the whole batch"
        );
        assert_eq!(
            sent_envelope_ids(retry_request),
            ["e1".to_string()],
            "the retry re-sends exactly the envelope the server named — \
             re-sending e2 would store a second copy of an event Langfuse \
             already accepted and does not deduplicate"
        );
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "the rejected envelope settles on its second answer"
        );
    }

    #[tokio::test]
    async fn a_207_naming_only_foreign_ids_settles_with_a_warning() {
        let (url, hits, bodies) = one_shot_endpoint(vec![(
            "207 Multi-Status".to_string(),
            None,
            r#"{"successes":[],"errors":[{"id":"not-ours","status":500}]}"#.to_string(),
        )]);
        let client = reqwest::Client::new();
        post(
            &client,
            &url,
            &basic_auth("pk", "sk"),
            vec![json!({ "id": "e1", "type": "trace-create", "body": {} })],
        )
        .await;
        let captured = bodies.lock().expect("capture lock").clone();
        assert_eq!(
            captured.len(),
            1,
            "a 207 whose failures name no envelope of ours has nothing \
             retryable — it settles instead of re-sending envelopes the \
             server says it accepted"
        );
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn parse_multi_status_judges_each_body_shape() {
        assert!(
            matches!(
                parse_multi_status(&json!({ "successes": [] })),
                Ack::Settled
            ),
            "an answer without an errors list is a complete success"
        );
        assert!(
            matches!(parse_multi_status(&json!({ "errors": [] })), Ack::Settled),
            "an empty errors list settles"
        );
        assert!(
            matches!(
                parse_multi_status(&json!({ "errors": "boom" })),
                Ack::Unreadable
            ),
            "errors that is not an array cannot be attributed"
        );
        assert!(
            matches!(
                parse_multi_status(&json!(["not", "an", "object"])),
                Ack::Unreadable
            ),
            "a non-object root is not a multi-status body"
        );
        assert!(
            matches!(
                parse_multi_status(&json!({ "errors": [{ "status": 500 }] })),
                Ack::Unreadable
            ),
            "a failure entry without an id cannot be retried selectively"
        );
        match parse_multi_status(&json!({
            "errors": [
                { "id": "e1", "status": 500 },
                { "id": "e2", "status": 429 },
            ]
        })) {
            Ack::Failed { failed } => assert_eq!(failed, ["e1".to_string(), "e2".to_string()]),
            other => panic!("named ids read as Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_429_honors_retry_after_and_retries_the_same_batch() {
        let (url, hits, _bodies) = one_shot_endpoint(vec![
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

    #[test]
    fn closed_and_empty_reads_both_facts_under_one_lock() {
        let buffer = Arc::new(Buffer {
            queue: std::sync::Mutex::new(VecDeque::new()),
            dropped: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            work_in_flight: AtomicBool::new(false),
            woken: Notify::new(),
        });
        assert!(
            !buffer.closed_and_empty(),
            "an open queue is never the exit state, drained or not"
        );
        buffer.closed.store(true, Ordering::SeqCst);
        assert!(
            buffer.closed_and_empty(),
            "closed and drained is the exit state"
        );
        buffer.lock().push_back(json!({ "n": 1 }));
        assert!(
            !buffer.closed_and_empty(),
            "a closed queue still holding work is not drained"
        );
    }

    #[test]
    fn a_push_accepted_before_close_keeps_the_flusher_from_exiting() {
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
        assert!(
            buffer.is_empty(),
            "the flusher's work check sees the queue empty"
        );
        assert!(
            sink.try_push(json!({ "n": 1 })),
            "the push lands while the queue is still open, and is accepted"
        );
        buffer.closed.store(true, Ordering::SeqCst);
        assert!(
            !buffer.closed_and_empty(),
            "the exit check must not fire while an accepted event still waits \
             — a two-step read would have exited over exactly this event"
        );
        let drained = buffer.drain(8);
        assert_eq!(
            drained.len(),
            1,
            "the accepted event reaches the flusher's final drain"
        );
        assert!(
            buffer.closed_and_empty(),
            "only once drained may the flusher exit"
        );
    }

    #[tokio::test]
    async fn an_unreadable_207_answer_retries_the_batch() {
        let (url, hits, _bodies) = one_shot_endpoint(vec![
            (
                "207 Multi-Status".to_string(),
                None,
                "upstream returned a non-JSON body".to_string(),
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
            "a 207 the client cannot parse must re-send the batch, not settle"
        );
    }

    #[tokio::test]
    async fn a_207_naming_no_errors_settles_on_the_first_try() {
        let (url, hits, _bodies) = one_shot_endpoint(vec![(
            "207 Multi-Status".to_string(),
            None,
            r#"{"successes":[{"id":"e1","status":201}]}"#.to_string(),
        )]);
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
            1,
            "a well-formed 207 without an errors list is a complete success — \
             retrying it would treat a valid answer as an unreadable one"
        );
    }

    #[test]
    fn the_exit_check_holds_the_queue_lock_across_both_reads() {
        let buffer = Buffer {
            queue: std::sync::Mutex::new(VecDeque::new()),
            dropped: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            work_in_flight: AtomicBool::new(false),
            woken: Notify::new(),
        };
        EXIT_WINDOW_LOCK_HELD.store(false, Ordering::SeqCst);
        EXIT_WINDOW_PROBE.store(true, Ordering::SeqCst);
        let _ = buffer.closed_and_empty();
        assert!(
            EXIT_WINDOW_LOCK_HELD.load(Ordering::SeqCst),
            "the queue lock is still held between the emptiness read and \
             the closed read — a read split into two acquisitions would \
             let a push land in exactly that window"
        );
        assert!(
            !EXIT_WINDOW_PROBE.load(Ordering::SeqCst),
            "the probe arms one call"
        );
    }

    #[test]
    fn retry_after_accepts_delta_seconds_and_http_dates() {
        assert_eq!(retry_after_delay("3"), Some(Duration::from_secs(3)));
        assert_eq!(
            retry_after_delay("not a delay"),
            None,
            "junk falls back to the caller's ladder"
        );

        let ahead = SystemTime::now() + Duration::from_secs(30);
        let advised = retry_after_delay(&httpdate::fmt_http_date(ahead))
            .expect("an HTTP-date in the future advises a delay");
        assert!(
            advised <= Duration::from_secs(30) && advised > Duration::from_secs(25),
            "an HTTP-date Retry-After waits the advertised time, got {advised:?}"
        );

        let past = SystemTime::now() - Duration::from_secs(30);
        assert_eq!(
            retry_after_delay(&httpdate::fmt_http_date(past)),
            Some(Duration::ZERO),
            "an already-elapsed HTTP-date means retry now"
        );
    }
}

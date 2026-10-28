//! # loopctl-langfuse
//!
//! [Langfuse](https://langfuse.com) export for [loopctl](https://loopctl.rs)
//! agent loops in Langfuse's native model:
//!
//! | loopctl | Langfuse |
//! |---|---|
//! | one `agent.run` span | one **trace**, `session_id` grouping runs |
//! | `agent.turn` span | observation (kind `span`) under the trace |
//! | `agent.tool` span | observation under the run, beside its turn (the engine dispatches tool spans from the run scope) |
//! | [`AgentMetrics`] rollup | **scores** on the trace |
//!
//! Transport is Langfuse's batched-ingest REST API (`POST
//! /api/public/ingestion`, Basic auth) with background batching, idempotent
//! event ids, retry-with-backoff, and a bounded flush on guard drop. Cloud
//! and self-hosted differ only by `host`.
//!
//! ## Privacy
//!
//! Traces are metadata-only: the engine deliberately never emits message
//! content into spans (PII), and neither does this crate. Hosts wanting
//! `input`/`output` on the trace can add them via their own ingest.
//!
//! ## OTLP alternative
//!
//! Pointing [`loopctl-otel`](https://docs.rs/loopctl-otel) at
//! `{host}/api/public/otel` also lands traces in Langfuse — with zero code
//! here, and also without the native scores/sessions this crate adds.
//!
//! [`AgentMetrics`]: loopctl::metrics::AgentMetrics

#![warn(missing_docs, missing_debug_implementations)]

pub mod ingest;
pub mod layer;
pub mod mapping;
pub mod scores;

use std::env;
use std::time::Duration;

pub use layer::{LangfuseLayer, MetricsSource};

use loopctl::metrics::AgentMetrics;

use crate::ingest::IngestSink;

/// Where and how to ingest.
///
/// Four knobs cover every deployment: cloud versus self-hosted differs
/// only by `host`, and the key pair authenticates the batched-ingest
/// REST API. The `Debug` impl redacts both keys so configs are safe in
/// logs.
#[derive(Clone)]
pub struct LangfuseConfig {
    /// Base URL. Default `https://cloud.langfuse.com`; self-hosted via
    /// `LANGFUSE_HOST` or this field.
    ///
    /// A trailing slash is tolerated — the ingest path trims it at
    /// request time.
    pub host: String,

    /// Project public key (env `LANGFUSE_PUBLIC_KEY`).
    ///
    /// The Basic-auth user for every ingest request; empty by default
    /// because [`LangfuseConfig::from_env`] — not `Default` — is the
    /// path that requires it.
    pub public_key: String,

    /// Project secret key (env `LANGFUSE_SECRET_KEY`).
    ///
    /// The Basic-auth password, paired with the public key; redacted
    /// in `Debug` output.
    pub secret_key: String,

    /// Max events per ingestion POST. Default 100.
    ///
    /// Clamped into `1..=BUFFER_CAP` when the flusher starts; the
    /// shared queue tolerates several batches waiting during a slow
    /// collector.
    pub batch_size: usize,
}

impl std::fmt::Debug for LangfuseConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangfuseConfig")
            .field("host", &self.host)
            .field("public_key", &"[redacted]")
            .field("secret_key", &"[redacted]")
            .field("batch_size", &self.batch_size)
            .finish()
    }
}

impl Default for LangfuseConfig {
    fn default() -> Self {
        Self {
            host: "https://cloud.langfuse.com".to_string(),
            public_key: String::new(),
            secret_key: String::new(),
            batch_size: 100,
        }
    }
}

/// Missing-configuration complaint.
///
/// Raised only by [`LangfuseConfig::from_env`]; hosts that build the
/// struct literally never encounter this type.
#[derive(Debug, thiserror::Error)]
pub enum LangfuseConfigError {
    /// A required environment variable is unset or empty.
    ///
    /// Carries the variable's name so the host's log points straight
    /// at what to set; an empty string counts as missing.
    #[error("environment variable {0} is not set (required for Langfuse export)")]
    MissingEnv(&'static str),
}

impl LangfuseConfig {
    /// Read `LANGFUSE_HOST` (default cloud), `LANGFUSE_PUBLIC_KEY`, and
    /// `LANGFUSE_SECRET_KEY`. Missing keys are an error for the caller to
    /// log-and-continue on — observability that fails is optional.
    ///
    /// # Errors
    ///
    /// [`LangfuseConfigError::MissingEnv`] when either key is absent.
    pub fn from_env() -> Result<Self, LangfuseConfigError> {
        let mut config = Self::default();
        if let Ok(host) = env::var("LANGFUSE_HOST")
            && !host.is_empty()
        {
            config.host = host;
        }
        config.public_key = env::var("LANGFUSE_PUBLIC_KEY")
            .map_err(|_| LangfuseConfigError::MissingEnv("LANGFUSE_PUBLIC_KEY"))?;
        config.secret_key = env::var("LANGFUSE_SECRET_KEY")
            .map_err(|_| LangfuseConfigError::MissingEnv("LANGFUSE_SECRET_KEY"))?;
        Ok(config)
    }
}

/// Why installation failed.
///
/// Every variant surfaces at install time or at the manual score
/// export edge; the running pipeline never returns here — transport
/// trouble is logged and counted instead.
#[derive(Debug, thiserror::Error)]
pub enum LangfuseInstallError {
    /// `install` needs a `tokio` runtime to spawn the flusher on.
    ///
    /// The flusher is a task, not a thread: hosts without a current
    /// runtime construct one around the install call.
    #[error("no tokio runtime available on this thread (install requires one)")]
    NoRuntime,

    /// A global default subscriber is already set (`install_simple` only).
    ///
    /// `tracing` globals are set-once; hosts that already own the
    /// global compose [`LangfuseLayer`] into their subscriber instead.
    #[error("a global tracing subscriber is already installed")]
    AlreadyInstalled,

    /// Scores were exported after the guard's ingest worker had stopped.
    ///
    /// The queue is closed for good once the guard drops, so late
    /// [`LangfuseGuard::export_scores`] calls report this rather than
    /// silently discarding events.
    #[error("langfuse ingest pipeline already shut down")]
    ShutDown,
}

/// Flushes pending events on drop; also the manual score surface.
///
/// Keep one alive for the process or run that matters: dropping
/// performs the bounded (5 s) drain and then closes the queue.
/// Manual scorers call [`LangfuseGuard::export_scores`],
/// [`LangfuseGuard::flush`] drains cooperatively before a drop, and
/// [`LangfuseGuard::dropped_events`] reports queue losses.
///
/// **Runtime hazard:** `Drop` polls the drain by parking its dropping
/// thread — fine on an ordinary thread, wrong on one the tokio runtime
/// drives. On a current-thread runtime the flusher cannot run while
/// `Drop` parks its only worker, so a pending queue is guaranteed to
/// sit there the full 5 s and be lost. Hosts inside a runtime call
/// [`LangfuseGuard::flush`] (an `await`, so the flusher runs) before
/// dropping.
#[derive(Debug)]
pub struct LangfuseGuard {
    /// The shared enqueue/shutdown handle the layer also holds.
    ///
    /// Cloned from the install-time sink, so the guard's flush and
    /// shutdown drive the same queue the layer enqueues into.
    sink: IngestSink,
}

impl LangfuseGuard {
    /// Export scores for a finished run onto its trace.
    ///
    /// `trace_id` is the run's Langfuse trace id — hosts auto-scoring via
    /// [`LangfuseLayer::with_metrics_source`] never need this call.
    ///
    /// # Errors
    ///
    /// [`LangfuseInstallError::ShutDown`] when the pipeline has already
    /// closed — the guard dropped or its flusher stopped — and a score
    /// event had nowhere to go.
    pub fn export_scores(
        &self,
        trace_id: &str,
        metrics: &AgentMetrics,
    ) -> Result<(), LangfuseInstallError> {
        for event in scores::score_events(trace_id, metrics) {
            if !self.sink.try_push(event) {
                return Err(LangfuseInstallError::ShutDown);
            }
        }
        Ok(())
    }

    /// Buffer-eviction counter (events lost past the 4096 queue).
    ///
    /// Delegates to the shared sink; growth between two reads means
    /// the collector stopped draining and events dropped
    /// oldest-first.
    #[must_use]
    pub fn dropped_events(&self) -> u64 {
        self.sink.dropped_count()
    }

    /// Await the queue's drain without parking the current thread.
    ///
    /// The runtime-cooperative twin of the drop-time drain: polls
    /// every 10 ms with `await`s, so the flusher task keeps running
    /// on any runtime flavor. Call this before dropping the guard
    /// from a thread the runtime drives — `Drop`'s bounded poll loop
    /// cannot let the flusher run there (the hazard is stated on
    /// [`LangfuseGuard`]).
    ///
    /// Returns `true` when the queue emptied and no batch was in
    /// flight; `false` means the timeout elapsed with work still
    /// pending.
    pub async fn flush(&self, timeout: std::time::Duration) -> bool {
        self.sink.wait_drained(timeout).await
    }
}

impl Drop for LangfuseGuard {
    fn drop(&mut self) {
        self.sink.shutdown_blocking(Duration::from_secs(5));
    }
}

/// Build the layer + guard pair from `config`.
///
/// Requires a `tokio` runtime on the current thread (the flusher spawns
/// there); composition onto the consumer's subscriber is the consumer's
/// call — `Registry::default().with(layer)`. Keep the guard alive for the
/// process/run; dropping it flushes (bounded to 5 s) and stops the worker.
///
/// # Errors
///
/// [`LangfuseInstallError::NoRuntime`] when no runtime is current.
pub fn install(
    config: LangfuseConfig,
) -> Result<(LangfuseLayer, LangfuseGuard), LangfuseInstallError> {
    let runtime =
        tokio::runtime::Handle::try_current().map_err(|_| LangfuseInstallError::NoRuntime)?;
    let LangfuseConfig {
        host,
        public_key,
        secret_key,
        batch_size,
    } = config;
    let sink = ingest::spawn_flusher(&host, &public_key, &secret_key, batch_size, &runtime);
    Ok((LangfuseLayer::new(sink.clone()), LangfuseGuard { sink }))
}

/// Convenience for hosts without a subscriber of their own: `install`
/// plus a registry subscriber with the layer and a `loopctl=debug`
/// target filter, set as the global default.
///
/// The filter is `DEBUG`, not `INFO`: every loopctl span is `DEBUG`
/// (the level contract in [`loopctl::tracing_spans`]), so a tighter
/// filter would admit only non-span events — and since the layer turns
/// *spans* into traces and observations, an `INFO` filter would leave
/// this exporter with nothing to send.
///
/// # Errors
///
/// [`LangfuseInstallError::NoRuntime`] or
/// [`LangfuseInstallError::AlreadyInstalled`].
pub fn install_simple(config: LangfuseConfig) -> Result<LangfuseGuard, LangfuseInstallError> {
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let (layer, guard) = install(config)?;
    let subscriber = tracing_subscriber::registry::Registry::default()
        .with(layer)
        .with(
            tracing_subscriber::filter::Targets::new().with_target("loopctl", LevelFilter::DEBUG),
        );
    subscriber
        .try_init()
        .map_err(|_| LangfuseInstallError::AlreadyInstalled)?;
    Ok(guard)
}

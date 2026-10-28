//! The guard's drop-time drain against the runtime it finds itself on.
//!
//! `Drop` cannot `await`, so the bounded blocking drain is right on an
//! ordinary thread and wrong on the one worker a current-thread
//! runtime drives — parking it starves the flusher for the whole wait.
//! This binary pins the guard's answer to that hazard.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::clone_on_ref_ptr,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::panic,
    clippy::too_many_lines
)]

use std::io::Read;
use std::net::TcpListener;
use std::time::{Duration, Instant};

use loopctl::metrics::AgentMetrics;
use loopctl_langfuse::{LangfuseConfig, install};

/// A collector that accepts one connection and holds it open, so a
/// shipped batch stays in flight well past any drop-time bound.
fn holding_collector() -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind");
    let port = listener.local_addr().expect("assigned port").port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _peer)) = listener.accept() {
            let mut sink = [0_u8; 512];
            while let Ok(read) = stream.read(&mut sink)
                && read > 0
            {
                let _ = read;
            }
        }
    });
    port
}

#[tokio::test(flavor = "current_thread")]
async fn guard_drop_on_a_current_thread_runtime_does_not_park() {
    let port = holding_collector();
    let (layer, guard) = install(LangfuseConfig {
        host: format!("http://127.0.0.1:{port}"),
        public_key: "pk-test".to_string(),
        secret_key: "sk-test".to_string(),
        batch_size: 100,
    })
    .expect("install");
    drop(layer);
    assert!(
        guard
            .export_scores("trace-pending", &AgentMetrics::default())
            .is_ok(),
        "score events queue behind the held-open collector"
    );

    let started = Instant::now();
    drop(guard);
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "a guard dropped on a current-thread runtime closes without parking \
         the only worker — five seconds of parking guarantees the pending \
         events are lost, because the flusher cannot run: {elapsed:?}"
    );
}

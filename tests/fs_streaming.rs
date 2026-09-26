//! Streaming window reads: correctness on small fixtures, memory bounded
//! by the window on a huge one.
//!
//! The viewer's two streaming scans exist so a giant file costs its
//! window, not its size. The small-fixture pins hold the window math
//! (including `str::lines` semantics for unterminated final lines and
//! CRLF), and the Linux-gated big-file pins hold the actual memory
//! bounds: a `1 GiB` sparse fixture serves a window while the process's
//! resident set grows by far less than the file, and a one-line sibling
//! holds the per-line cap — a single `256 MiB` line is served without
//! ever being loaded whole.

#![cfg(feature = "fs_tools")]
#![allow(
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]

use std::path::PathBuf;

use loopctl::tool::builtin::fs::FileSession;
use loopctl::tool::builtin::fs::FileViewerTool;
use loopctl::tool::{Tool, ToolContext};
use serde_json::json;

fn ctx_in(cwd: &str) -> ToolContext {
    let mut ctx = ToolContext::default();
    ctx.cwd = cwd.to_string();
    FileSession::new(PathBuf::from(cwd)).attach(&mut ctx);
    ctx
}

#[tokio::test]
async fn a_window_view_of_a_many_line_file_is_exact() {
    use std::fmt::Write as _;
    let tmp = tempfile::tempdir().unwrap();
    let mut content = String::new();
    for i in 1..=10_000 {
        writeln!(&mut content, "L{i}").unwrap();
    }
    std::fs::write(tmp.path().join("big.txt"), content).unwrap();
    let out = FileViewerTool
        .call(
            json!({"file_path": "big.txt", "offset": 9_995, "limit": 5}),
            &ctx_in(tmp.path().to_str().unwrap()),
        )
        .await
        .unwrap();
    let text = out.text_content();
    assert!(text.contains("Lines 9995-9999 of 10000"), "{text}");
    assert!(text.contains("L9995"), "{text}");
    assert!(text.contains("L9999"), "{text}");
    assert!(!text.contains("L9994"), "{text}");
    assert!(!text.contains("L10000"), "{text}");
}

#[tokio::test]
async fn crlf_lines_split_like_str_lines() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("crlf.txt"), "a\r\nb\r\nc\r\n").unwrap();
    let out = FileViewerTool
        .call(
            json!({"file_path": "crlf.txt"}),
            &ctx_in(tmp.path().to_str().unwrap()),
        )
        .await
        .unwrap();
    let text = out.text_content();
    assert!(text.contains("Lines 1-3 of 3"), "{text}");
    assert!(
        text.contains("│ a"),
        "the CR must not survive into the line: {text}"
    );
}

#[cfg(target_os = "linux")]
fn peak_rss_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib = rest.trim().trim_end_matches("kB").trim();
            return kib.parse().unwrap();
        }
    }
    panic!("VmHWM missing from /proc/self/status");
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn streaming_read_memory_is_bounded_by_window() {
    use std::io::Seek as _;
    use std::io::Write as _;
    let tmp = tempfile::tempdir().unwrap();
    let giant = tmp.path().join("giant.txt");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&giant)
        .unwrap();
    file.set_len(1024 * 1024 * 1024).unwrap();
    for step in 1..16384 {
        file.seek(std::io::SeekFrom::Start(step * 64 * 1024))
            .unwrap();
        file.write_all(
            b"
",
        )
        .unwrap();
    }
    file.flush().unwrap();
    drop(file);

    let before = peak_rss_kib();
    let out = FileViewerTool
        .call(
            json!({"file_path": "giant.txt", "offset": 16_000, "limit": 100}),
            &ctx_in(tmp.path().to_str().unwrap()),
        )
        .await
        .unwrap();
    let after = peak_rss_kib();
    let text = out.text_content();
    assert!(
        !out.is_error && text.contains("Lines 16000-16099 of 16384"),
        "the window must still be served: {text}"
    );

    let grew_mib = after.saturating_sub(before) / 1024;
    assert!(
        grew_mib < 256,
        "a 1 GiB sparse file must cost its window, not its size; RSS grew {grew_mib} MiB"
    );
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn a_single_giant_line_is_capped_not_loaded() {
    let tmp = tempfile::tempdir().unwrap();
    let oneline = tmp.path().join("oneline.txt");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&oneline)
        .unwrap();
    file.set_len(256 * 1024 * 1024).unwrap();
    drop(file);

    let before = peak_rss_kib();
    let out = FileViewerTool
        .call(
            json!({"file_path": "oneline.txt", "offset": 1, "limit": 100}),
            &ctx_in(tmp.path().to_str().unwrap()),
        )
        .await
        .unwrap();
    let after = peak_rss_kib();
    let text = out.text_content();
    assert!(
        !out.is_error && text.contains("Lines 1-1 of 1"),
        "the capped line must still be served and counted as one line: {text}"
    );
    assert!(
        text.contains("bytes truncated]"),
        "the marker must name the omitted tail: {text}"
    );

    let grew_mib = after.saturating_sub(before) / 1024;
    assert!(
        grew_mib < 256,
        "a single 256 MiB line must cost its per-line cap, not its length; RSS grew {grew_mib} MiB"
    );
}

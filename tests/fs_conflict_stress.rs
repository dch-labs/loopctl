//! Adversarial sequence stress for the detect-on-write conflict machinery.
//!
//! Drives the real tools (no test doubles) through the multi-call
//! sequences the per-tool unit tests cannot express: repeated writes
//! without an intervening read, external changes and reverts, recovery
//! loops, and baseline lookups across path spellings and multiple
//! files. One shared session per scenario mirrors the host's
//! single-session wiring.

#![cfg(feature = "fs_tools")]
#![allow(
    clippy::arithmetic_side_effects,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::field_reassign_with_default
)]

use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;

use loopctl::tool::builtin::fs::EditTool;
use loopctl::tool::builtin::fs::FileSession;
use loopctl::tool::builtin::fs::FileSource;
use loopctl::tool::builtin::fs::FileViewerTool;
use loopctl::tool::builtin::fs::MultiEditTool;
use loopctl::tool::builtin::fs::WriteTool;
use loopctl::tool::builtin::read::ReadTool;
use loopctl::tool::{Tool, ToolContext};
use serde_json::json;

fn ctx_in(cwd: &str) -> ToolContext {
    let mut ctx = ToolContext::default();
    ctx.cwd = cwd.to_string();
    FileSession::new(PathBuf::from(cwd)).attach(&mut ctx);
    ctx
}

async fn read(ctx: &ToolContext, path: &str) {
    let session = loopctl::tool::builtin::fs::fs_session(ctx).unwrap();
    let out = ReadTool::new(FileSource::new(session))
        .call(json!({ "path": path }), ctx)
        .await
        .unwrap();
    assert!(!out.is_error, "read {path}: {}", out.text_content());
}

async fn write(ctx: &ToolContext, path: &str, content: &str) -> bool {
    let out = WriteTool::new()
        .call(json!({ "file_path": path, "content": content }), ctx)
        .await
        .unwrap();
    !out.is_error
}

async fn edit(ctx: &ToolContext, path: &str, old_text: &str, new_text: &str) -> bool {
    let out = EditTool::new()
        .call(
            json!({
                "file_path": path,
                "old_text": old_text,
                "new_text": new_text
            }),
            ctx,
        )
        .await
        .unwrap();
    !out.is_error
}

fn disk(tmp: &Path, name: &str) -> String {
    std::fs::read_to_string(tmp.join(name)).unwrap()
}

fn force_mtime_change(path: &Path) {
    let baseline = std::fs::metadata(path).unwrap().modified().unwrap();
    let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    file.set_modified(baseline + Duration::from_secs(30))
        .unwrap();
}

async fn read_then_two_writes_both_succeed_without_an_external_change(tmp: &Path) {
    std::fs::write(tmp.join("note.txt"), "v1\n").unwrap();
    let ctx = ctx_in(tmp.to_str().unwrap());

    read(&ctx, "note.txt").await;
    let first = write(&ctx, "note.txt", "v2\n").await;
    let second = write(&ctx, "note.txt", "v3\n").await;

    assert!(first, "the first write must succeed");
    assert!(
        second,
        "the model's own write is not an external change: {}",
        disk(tmp, "note.txt")
    );
    assert_eq!(disk(tmp, "note.txt"), "v3\n");

    std::fs::write(tmp.join("note.txt"), "EXTERNAL\n").unwrap();
    force_mtime_change(&tmp.join("note.txt"));
    assert!(
        !write(&ctx, "note.txt", "v4\n").await,
        "an external change after the model's writes must conflict"
    );
    assert_eq!(disk(tmp, "note.txt"), "EXTERNAL\n");
}

async fn an_external_change_survives_a_content_revert(tmp: &Path) {
    std::fs::write(tmp.join("note.txt"), "v1\n").unwrap();
    let ctx = ctx_in(tmp.to_str().unwrap());

    read(&ctx, "note.txt").await;
    std::fs::write(tmp.join("note.txt"), "TAMPERED\n").unwrap();
    force_mtime_change(&tmp.join("note.txt"));
    std::fs::write(tmp.join("note.txt"), "v1\n").unwrap();
    force_mtime_change(&tmp.join("note.txt"));
    assert!(
        write(&ctx, "note.txt", "v2\n").await,
        "a revert to the baseline bytes is no content change: the guard compares \n         bytes, not mtimes, and the pinned mtime must not trip it"
    );
    assert_eq!(disk(tmp, "note.txt"), "v2\n");

    std::fs::write(tmp.join("note.txt"), "TAMPERED\n").unwrap();
    force_mtime_change(&tmp.join("note.txt"));
    assert!(
        !write(&ctx, "note.txt", "v3\n").await,
        "genuinely different bytes still conflict after the revert pass"
    );
    assert_eq!(disk(tmp, "note.txt"), "TAMPERED\n");

    read(&ctx, "note.txt").await;
    assert!(write(&ctx, "note.txt", "v3\n").await);
    assert_eq!(disk(tmp, "note.txt"), "v3\n");
}

async fn an_edit_conflict_recovers_through_a_reread(tmp: &Path) {
    std::fs::write(tmp.join("code.rs"), "fn a() {}\n").unwrap();
    let ctx = ctx_in(tmp.to_str().unwrap());

    read(&ctx, "code.rs").await;
    std::fs::write(tmp.join("code.rs"), "fn a() { CHANGED }\n").unwrap();
    force_mtime_change(&tmp.join("code.rs"));
    assert!(
        !edit(&ctx, "code.rs", "fn a() {}", "fn b() {}").await,
        "an edit against externally changed bytes must refuse"
    );
    assert_eq!(disk(tmp, "code.rs"), "fn a() { CHANGED }\n");

    read(&ctx, "code.rs").await;
    assert!(edit(&ctx, "code.rs", "fn a() { CHANGED }", "fn b() {}").await);
    assert_eq!(disk(tmp, "code.rs"), "fn b() {}\n");
}

async fn skip_linter_lifts_the_gate_only_not_the_guard(tmp: &Path) {
    std::fs::write(tmp.join("gate.rs"), "fn ok() {}\n").unwrap();
    let ctx = ctx_in(tmp.to_str().unwrap());

    read(&ctx, "gate.rs").await;
    std::fs::write(tmp.join("gate.rs"), "EXTERNAL\n").unwrap();
    force_mtime_change(&tmp.join("gate.rs"));
    let out = WriteTool::new()
        .call(
            json!({
                "file_path": "gate.rs",
                "content": "fn broken() { let x = ; }",
                "skip_linter": true
            }),
            &ctx,
        )
        .await
        .unwrap();
    assert!(out.is_error, "skip_linter lifts the syntax gate only");
    assert!(
        out.text_content().contains("changed on disk"),
        "the refusal must come from the staleness guard: {}",
        out.text_content()
    );
    assert_eq!(disk(tmp, "gate.rs"), "EXTERNAL\n");
}

async fn relative_and_absolute_spellings_share_one_baseline(tmp: &Path) {
    std::fs::write(tmp.join("note.txt"), "v1\n").unwrap();
    let ctx = ctx_in(tmp.to_str().unwrap());
    let absolute = tmp.join("note.txt");

    read(&ctx, "note.txt").await;
    std::fs::write(&absolute, "EXTERNAL\n").unwrap();
    force_mtime_change(&absolute);
    assert!(
        !write(&ctx, absolute.to_str().unwrap(), "v2\n").await,
        "the absolute spelling resolves to the armed file"
    );
    assert_eq!(disk(tmp, "note.txt"), "EXTERNAL\n");

    std::fs::write(&absolute, "v1\n").unwrap();
    read(&ctx, absolute.to_str().unwrap()).await;
    std::fs::write(&absolute, "EXTERNAL\n").unwrap();
    force_mtime_change(&absolute);
    assert!(
        !write(&ctx, "note.txt", "v3\n").await,
        "the relative spelling resolves to the armed file"
    );
    assert_eq!(disk(tmp, "note.txt"), "EXTERNAL\n");
}

async fn a_multiedit_refreshes_the_baseline_a_later_write_consults(tmp: &Path) {
    std::fs::write(tmp.join("code.rs"), "fn a() {}\n").unwrap();
    let ctx = ctx_in(tmp.to_str().unwrap());

    let out = MultiEditTool::new()
        .call(
            json!({
                "edits": [
                    {
                        "file_path": "code.rs",
                        "old_text": "fn a() {}",
                        "new_text": "fn b() {}"
                    }
                ]
            }),
            &ctx,
        )
        .await
        .unwrap();
    assert!(!out.is_error, "{}", out.text_content());

    std::fs::write(tmp.join("code.rs"), "EXTERNAL\n").unwrap();
    force_mtime_change(&tmp.join("code.rs"));
    assert!(
        !write(&ctx, "code.rs", "fn c() {}\n").await,
        "a write after the batch must be guarded by the batch's baseline"
    );
    assert_eq!(disk(tmp, "code.rs"), "EXTERNAL\n");

    read(&ctx, "code.rs").await;
    assert!(write(&ctx, "code.rs", "fn c() {}\n").await);
    assert_eq!(disk(tmp, "code.rs"), "fn c() {}\n");
}

async fn a_read_through_the_source_arms_the_guard_the_write_consults(tmp: &Path) {
    std::fs::write(tmp.join("note.txt"), "v1\n").unwrap();
    let ctx = ctx_in(tmp.to_str().unwrap());

    read(&ctx, "note.txt").await;
    std::fs::write(tmp.join("note.txt"), "EXTERNAL\n").unwrap();
    assert!(
        !write(&ctx, "note.txt", "v2\n").await,
        "a read through FileSource arms the baseline a Write consults"
    );

    let out = FileViewerTool
        .call(json!({ "file_path": "note.txt" }), &ctx)
        .await
        .unwrap();
    assert!(
        !out.is_error,
        "FileViewer views without arming: {}",
        out.text_content()
    );
    assert!(
        !write(&ctx, "note.txt", "v2\n").await,
        "a view-only read must not arm the guard"
    );
}

type Scenario = &'static str;
type ScenarioFn = for<'a> fn(&'a Path) -> Pin<Box<dyn Future<Output = ()> + 'a>>;

#[tokio::test]
async fn conflict_stress_suite_passes_at_new_home() {
    let scenarios: Vec<(Scenario, ScenarioFn)> = vec![
        ("read then two writes", |tmp| {
            Box::pin(read_then_two_writes_both_succeed_without_an_external_change(tmp))
        }),
        ("external change survives a revert", |tmp| {
            Box::pin(an_external_change_survives_a_content_revert(tmp))
        }),
        ("edit conflict recovers through a reread", |tmp| {
            Box::pin(an_edit_conflict_recovers_through_a_reread(tmp))
        }),
        ("skip_linter lifts the gate only", |tmp| {
            Box::pin(skip_linter_lifts_the_gate_only_not_the_guard(tmp))
        }),
        (
            "relative and absolute spellings share one baseline",
            |tmp| Box::pin(relative_and_absolute_spellings_share_one_baseline(tmp)),
        ),
        ("multiedit refreshes the baseline", |tmp| {
            Box::pin(a_multiedit_refreshes_the_baseline_a_later_write_consults(
                tmp,
            ))
        }),
        ("a read through the source arms the guard", |tmp| {
            Box::pin(a_read_through_the_source_arms_the_guard_the_write_consults(
                tmp,
            ))
        }),
    ];
    for (name, scenario) in scenarios {
        let tmp = tempfile::tempdir().unwrap();
        scenario(tmp.path()).await;
        std::fs::remove_dir_all(tmp.path()).unwrap_or(());
        eprintln!("scenario ok: {name}");
    }
}

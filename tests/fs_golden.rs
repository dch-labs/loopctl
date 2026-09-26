//! Golden parity against dch v1's file tools.
//!
//! The goldens under `tests/golden/fs/` were captured by running the
//! dch v1 tools (published `loopctl` 0.3.2 + `dch-tools`) over this
//! exact scenario corpus; this test replays the corpus against the
//! ported family and diffs byte-for-byte. Regeneration is deliberate:
//! re-run the capture harness against dch v1 and re-copy the files —
//! never hand-edit a golden. Absolute workspace paths normalize to
//! `<WORKSPACE>/` on both sides; dch's linter behavior on this corpus
//! is reproduced by the table validator below, keyed by file name and
//! candidate content, so the gate's ordering and rendering are part
//! of the pin. Struct field order is never a contract (objects
//! canonicalize sorted); array order — `required`, `edits` — is.

#![cfg(feature = "fs_tools")]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::field_reassign_with_default
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use loopctl::tool::builtin::fs::EditTool;
use loopctl::tool::builtin::fs::FileSession;
use loopctl::tool::builtin::fs::FileViewerTool;
use loopctl::tool::builtin::fs::MultiEditTool;
use loopctl::tool::builtin::fs::ValidationDiagnostic;
use loopctl::tool::builtin::fs::WriteTool;
use loopctl::tool::builtin::fs::{ContentValidator, FileSource};
use loopctl::tool::builtin::read::ReadTool;
use loopctl::tool::{Tool, ToolContext};
use serde_json::json;

/// The dch linter's findings for this corpus, keyed by file name then
/// candidate content.
struct CorpusLinter;

impl CorpusLinter {
    fn findings(path: &Path, content: &str) -> Vec<ValidationDiagnostic> {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let known: &[(&str, &str)] = match name {
            "watched.rs" | "resumed.rs" => &[("ours\n", "expected `!`"), ("v2\n", "expected `!`")],
            "bad.rs" => &[("fn main() { let x = ; }", "expected an expression")],
            "a.rs" => &[("ALPHA\n", "expected `!`")],
            _ => &[],
        };
        known
            .iter()
            .filter(|(candidate, _)| *candidate == content)
            .map(|(_, message)| ValidationDiagnostic {
                line: None,
                message: (*message).to_string(),
            })
            .collect()
    }
}

impl ContentValidator for CorpusLinter {
    fn validate<'a>(
        &'a self,
        path: &'a Path,
        content: &'a str,
    ) -> Pin<Box<dyn Future<Output = Vec<ValidationDiagnostic>> + Send + 'a>> {
        Box::pin(async move { Self::findings(path, content) })
    }
}

fn write_tool() -> WriteTool {
    WriteTool::new().with_validator(Arc::new(CorpusLinter))
}

fn edit_tool() -> EditTool {
    EditTool::new().with_validator(Arc::new(CorpusLinter))
}

fn multi_edit_tool() -> MultiEditTool {
    MultiEditTool::new().with_validator(Arc::new(CorpusLinter))
}

struct Fixture {
    dir: tempfile::TempDir,
    expected: BTreeMap<String, String>,
}

impl Fixture {
    fn load() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut expected = BTreeMap::new();
        let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/fs");
        for entry in std::fs::read_dir(&golden_dir).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".golden") {
                let body = std::fs::read_to_string(entry.path()).unwrap();
                expected.insert(name.trim_end_matches(".golden").to_string(), body);
            }
        }
        assert!(!expected.is_empty(), "golden corpus must not be empty");
        Self { dir, expected }
    }

    fn golden(&self, name: &str) -> &str {
        self.expected
            .get(name)
            .unwrap_or_else(|| panic!("missing golden for scenario {name}"))
    }

    fn ctx(&self) -> ToolContext {
        let mut ctx = ToolContext::default();
        ctx.cwd = self.dir.path().to_string_lossy().into_owned();
        FileSession::new(self.dir.path().to_path_buf()).attach(&mut ctx);
        ctx
    }

    fn workdir(&self, tag: &str) -> PathBuf {
        let dir = self.dir.path().join(tag);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}

fn normalize(text: &str, workspace: &Path) -> String {
    text.replace(workspace.to_str().unwrap(), "<WORKSPACE>")
}

fn assert_scenario(fixture: &Fixture, name: &str, produced: &str, workspace: &Path) {
    let golden = fixture.golden(name);
    let normalized = normalize(produced, workspace);
    assert_eq!(
        normalized, golden,
        "scenario {name} drifted from the dch v1 golden"
    );
}

fn schema_scenarios(fixture: &Fixture) {
    for (name, tool) in [
        ("schema_Write", Arc::new(write_tool()) as Arc<dyn Tool>),
        ("schema_Edit", Arc::new(edit_tool()) as Arc<dyn Tool>),
        (
            "schema_MultiEdit",
            Arc::new(multi_edit_tool()) as Arc<dyn Tool>,
        ),
        (
            "schema_FileViewer",
            Arc::new(FileViewerTool) as Arc<dyn Tool>,
        ),
    ] {
        let schema = serde_json::to_string(&tool.schema()).unwrap();
        assert_scenario(fixture, name, &schema, fixture.dir.path());
    }
}

async fn write_scenarios(fixture: &Fixture) {
    let dir = fixture.workdir("write-new");
    let out = write_tool()
        .call(
            json!({"file_path": "notes.txt", "content": "hello\nworld\n"}),
            &fixture.ctx(),
        )
        .await
        .unwrap();
    assert_scenario(fixture, "write_new", &out.text_content(), &dir);

    let dir = fixture.workdir("write-overwrite");
    std::fs::write(dir.join("existing.rs"), "old content\n").unwrap();
    let ctx = ctx_with(&dir);
    let out = write_tool()
        .call(
            json!({"file_path": "existing.rs", "content": "fn main() {}\n"}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "write_overwrite", &out.text_content(), &dir);

    let dir = fixture.workdir("conflict-txt");
    std::fs::write(dir.join("watched.txt"), "v1\n").unwrap();
    let session = FileSession::new(dir.clone());
    let mut ctx = ToolContext::default();
    ctx.cwd = dir.to_string_lossy().into_owned();
    session.attach(&mut ctx);
    ReadTool::new(FileSource::new(session.clone()))
        .call(json!({"path": "watched.txt"}), &ctx)
        .await
        .unwrap();
    std::fs::write(dir.join("watched.txt"), "EXTERNAL\n").unwrap();
    let out = write_tool()
        .call(
            json!({"file_path": "watched.txt", "content": "ours\n"}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "write_conflict_txt", &out.text_content(), &dir);

    let dir = fixture.workdir("write-conflict");
    std::fs::write(dir.join("watched.rs"), "v1\n").unwrap();
    let ctx = ctx_with(&dir);
    let out = write_tool()
        .call(
            json!({"file_path": "watched.rs", "content": "ours\n"}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "write_conflict", &out.text_content(), &dir);

    let dir = fixture.workdir("resumed-txt");
    std::fs::write(dir.join("resumed.txt"), "v1\n").unwrap();
    let session = FileSession::new(dir.clone());
    assert!(session.record_resumed_read("resumed.txt").await);
    let mut ctx = ToolContext::default();
    ctx.cwd = dir.to_string_lossy().into_owned();
    session.attach(&mut ctx);
    let out = write_tool()
        .call(json!({"file_path": "resumed.txt", "content": "v2\n"}), &ctx)
        .await
        .unwrap();
    assert_scenario(fixture, "write_resumed_txt", &out.text_content(), &dir);

    let dir = fixture.workdir("write-resumed");
    std::fs::write(dir.join("resumed.rs"), "v1\n").unwrap();
    let session = FileSession::new(dir.clone());
    assert!(session.record_resumed_read("resumed.rs").await);
    let mut ctx = ToolContext::default();
    ctx.cwd = dir.to_string_lossy().into_owned();
    session.attach(&mut ctx);
    let out = write_tool()
        .call(json!({"file_path": "resumed.rs", "content": "v2\n"}), &ctx)
        .await
        .unwrap();
    assert_scenario(fixture, "write_resumed", &out.text_content(), &dir);

    let dir = fixture.workdir("write-lint-block");
    let ctx = ctx_with(&dir);
    let out = write_tool()
        .call(
            json!({"file_path": "bad.rs", "content": "fn main() { let x = ; }"}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "write_lint_block", &out.text_content(), &dir);

    let dir = fixture.workdir("write-lint-skip");
    let ctx = ctx_with(&dir);
    let out = write_tool()
        .call(
            json!({"file_path": "bad.rs", "content": "fn main() { let x = ; }", "skip_linter": true}),
            &ctx,
        )
        .await
         .unwrap();
    assert_scenario(fixture, "write_lint_skip", &out.text_content(), &dir);
}

async fn edit_scenarios(fixture: &Fixture) {
    let dir = fixture.workdir("edit-found");
    std::fs::write(dir.join("a.rs"), "fn one() {}\nfn two() {}\n").unwrap();
    let ctx = ctx_with(&dir);
    let out = edit_tool()
        .call(
            json!({"file_path": "a.rs", "old_text": "fn two() {}", "new_text": "fn two() { todo!() }"}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "edit_found", &out.text_content(), &dir);

    let dir = fixture.workdir("edit-not-found");
    std::fs::write(dir.join("a.rs"), "content\n").unwrap();
    let ctx = ctx_with(&dir);
    let out = edit_tool()
        .call(
            json!({"file_path": "a.rs", "old_text": "absent", "new_text": "x"}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "edit_not_found", &out.text_content(), &dir);

    let dir = fixture.workdir("edit-ambiguous");
    std::fs::write(dir.join("a.rs"), "x\nx\nx\n").unwrap();
    let ctx = ctx_with(&dir);
    let out = edit_tool()
        .call(
            json!({"file_path": "a.rs", "old_text": "x", "new_text": "y"}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "edit_ambiguous", &out.text_content(), &dir);
}

async fn multiedit_scenarios(fixture: &Fixture) {
    let dir = fixture.workdir("multiedit-ok-txt");
    std::fs::write(dir.join("a.txt"), "alpha\n").unwrap();
    std::fs::write(dir.join("b.txt"), "beta\n").unwrap();
    let ctx = ctx_with(&dir);
    let out = multi_edit_tool()
        .call(
            json!({"edits": [
                {"file_path": "a.txt", "old_text": "alpha", "new_text": "ALPHA"},
                {"file_path": "b.txt", "old_text": "beta", "new_text": "BETA"}
            ]}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "multiedit_ok_txt", &out.text_content(), &dir);

    let dir = fixture.workdir("multiedit-dry-txt");
    std::fs::write(dir.join("a.txt"), "alpha\n").unwrap();
    let ctx = ctx_with(&dir);
    let out = multi_edit_tool()
        .call(
            json!({"edits": [
                {"file_path": "a.txt", "old_text": "alpha", "new_text": "ALPHA"}
            ], "dry_run": true}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "multiedit_dry_run_txt", &out.text_content(), &dir);

    let dir = fixture.workdir("multiedit-ok");
    std::fs::write(dir.join("a.rs"), "alpha\n").unwrap();
    std::fs::write(dir.join("b.rs"), "beta\n").unwrap();
    let ctx = ctx_with(&dir);
    let out = multi_edit_tool()
        .call(
            json!({"edits": [
                {"file_path": "a.rs", "old_text": "alpha", "new_text": "ALPHA"},
                {"file_path": "b.rs", "old_text": "beta", "new_text": "BETA"}
            ]}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "multiedit_ok", &out.text_content(), &dir);

    let dir = fixture.workdir("multiedit-dry-run");
    std::fs::write(dir.join("a.rs"), "alpha\n").unwrap();
    let ctx = ctx_with(&dir);
    let out = multi_edit_tool()
        .call(
            json!({"edits": [
                {"file_path": "a.rs", "old_text": "alpha", "new_text": "ALPHA"}
            ], "dry_run": true}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "multiedit_dry_run", &out.text_content(), &dir);

    let dir = fixture.workdir("multiedit-abort");
    std::fs::write(dir.join("a.rs"), "hello world\n").unwrap();
    let ctx = ctx_with(&dir);
    let out = multi_edit_tool()
        .call(
            json!({"edits": [
                {"file_path": "a.rs", "old_text": "hello wo", "new_text": "A"},
                {"file_path": "a.rs", "old_text": "world", "new_text": "B"}
            ]}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "multiedit_abort", &out.text_content(), &dir);
}

async fn viewer_scenarios(fixture: &Fixture) {
    let dir = fixture.workdir("viewer");
    let mut content = String::new();
    for i in 1..=250 {
        writeln!(&mut content, "line {i}").unwrap();
    }
    std::fs::write(dir.join("big.txt"), content).unwrap();
    std::fs::write(dir.join("code.rs"), "fn a() {}\nfn b() {}\n").unwrap();
    let ctx = ctx_with(&dir);
    let out = FileViewerTool
        .call(json!({"file_path": "big.txt"}), &ctx)
        .await
        .unwrap();
    assert_scenario(fixture, "viewer_first_page", &out.text_content(), &dir);
    let out = FileViewerTool
        .call(
            json!({"file_path": "big.txt", "page": 2, "page_size": 50}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "viewer_mid_page", &out.text_content(), &dir);
    let out = FileViewerTool
        .call(
            json!({"file_path": "code.rs", "output_format": "markdown"}),
            &ctx,
        )
        .await
        .unwrap();
    assert_scenario(fixture, "viewer_markdown", &out.text_content(), &dir);
    let out = FileViewerTool
        .call(json!({"file_path": "big.txt", "offset": 999}), &ctx)
        .await
        .unwrap();
    assert_scenario(fixture, "viewer_overseek", &out.text_content(), &dir);
}

fn ctx_with(dir: &Path) -> ToolContext {
    let mut ctx = ToolContext::default();
    ctx.cwd = dir.to_string_lossy().into_owned();
    FileSession::new(dir.to_path_buf()).attach(&mut ctx);
    ctx
}

#[tokio::test]
async fn ported_tool_behavior_matches_dch_v1_goldens() {
    let fixture = Fixture::load();
    schema_scenarios(&fixture);
    write_scenarios(&fixture).await;
    edit_scenarios(&fixture).await;
    multiedit_scenarios(&fixture).await;
    viewer_scenarios(&fixture).await;
}

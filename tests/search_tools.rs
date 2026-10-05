//! Pins for the search tool family — `GlobTool`, `GrepTool`,
//! `CodeSearchTool`, and `TreeTool` over [`FsSearchSource`].
//!
//! The task-named pins hold the family to the contract it was ported
//! with: the gitignore-aware walk (`.gitignore`, `.ignore`, and the
//! loopctl-superset `.loopctlignore`/`.dchignore` names) behaves like
//! the original, and grep's include/exclude filters plus code-search's
//! context windows match the dch shapes. Everything runs against a
//! real temporary filesystem through the public tool surface — no
//! fakes, no shortcuts.

#![cfg(feature = "search_tools")]
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

use loopctl::tool::builtin::search::CodeSearchTool;
use loopctl::tool::builtin::search::FsSearchSource;
use loopctl::tool::builtin::search::GlobTool;
use loopctl::tool::builtin::search::GrepTool;
use loopctl::tool::builtin::search::TreeTool;
use loopctl::tool::{Tool, ToolContext};
use serde_json::Value;
use serde_json::json;

/// A context whose cwd is `dir` and temp dir is `dir/spill`.
///
/// The spill subdir keeps every spilled file inside the fixture, so
/// the spill pins can assert the pointer landed under the context's
/// temp dir without touching the system default.
fn ctx_in(dir: &std::path::Path) -> ToolContext {
    ToolContext {
        cwd: dir.to_string_lossy().into_owned(),
        temp_dir: dir.join("spill").to_string_lossy().into_owned(),
        ..ToolContext::default()
    }
}

/// Write `rel` under `dir`, creating parent directories as needed.
///
/// The one fixture primitive every scenario builds its tree with;
/// contents are written verbatim.
fn write_file(dir: &std::path::Path, rel: &str, contents: &str) {
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(path, contents).expect("write");
}

#[tokio::test]
async fn glob_respects_gitignore_like_dch_v1() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    // The `ignore` crate honors `.gitignore` only inside a git work tree
    // (its `require_git` default) — the same production condition the
    // original walker ran under.
    std::fs::create_dir_all(tmp.path().join(".git")).expect("git marker");
    write_file(tmp.path(), "src/kept.rs", "fn main() {}\n");
    write_file(tmp.path(), "target/generated.rs", "fn gen() {}\n");
    write_file(tmp.path(), "ignored.log", "noise\n");
    write_file(tmp.path(), "nested/secret.rs", "fn secret() {}\n");
    std::fs::write(tmp.path().join(".gitignore"), "target/\n*.log\nnested/\n").expect("gitignore");

    let tool = GlobTool::new(FsSearchSource);
    let output = tool
        .call(
            json!({"pattern": "**/*.rs", "path": tmp.path().to_string_lossy()}),
            &ctx_in(tmp.path()),
        )
        .await
        .expect("call");
    let paths: Vec<String> = serde_json::from_str(&output.text_content()).expect("json array");
    assert_eq!(
        paths,
        vec!["src/kept.rs".to_string()],
        "ignored trees and logs must be absent, kept files present"
    );
}

#[tokio::test]
async fn grep_include_exclude_and_context_lines_match_dch_contract() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    write_file(tmp.path(), "keep.rs", "alpha\nneedle one\nbeta\n");
    write_file(tmp.path(), "pruned.lock", "needle two\n");
    write_file(tmp.path(), "unrelated.md", "needle three\n");

    let grep = GrepTool::new(FsSearchSource);
    let output = grep
        .call(
            json!({
                "pattern": "needle",
                "path": tmp.path().to_string_lossy(),
                "include_patterns": ["*.rs"],
                "exclude_patterns": ["*.lock", "*.md"]
            }),
            &ctx_in(tmp.path()),
        )
        .await
        .expect("grep call");
    let matches: Vec<Value> = serde_json::from_str(&output.text_content()).expect("json array");
    assert_eq!(matches.len(), 1, "{}", output.text_content());
    assert_eq!(matches[0]["file"], "keep.rs");
    assert_eq!(matches[0]["line"], 2);
    assert_eq!(matches[0]["content"], "needle one");

    let code_search = CodeSearchTool::new(FsSearchSource);
    let output = code_search
        .call(
            json!({
                "pattern": "needle",
                "path": tmp.path().to_string_lossy(),
                "include_patterns": ["*.rs"],
                "context_lines": 1
            }),
            &ctx_in(tmp.path()),
        )
        .await
        .expect("code search call");
    let text = output.text_content();
    assert!(text.contains("keep.rs:2\n"), "{text}");
    assert!(text.contains("   1: alpha\n"), "{text}");
    assert!(text.contains("\n>2: needle one\n"), "{text}");
    assert!(text.contains("\n 3: beta"), "{text}");
}

#[tokio::test]
async fn family_flags_and_names_are_coherent() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    write_file(tmp.path(), "a.rs", "x\n");
    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(GlobTool::new(FsSearchSource)),
        Box::new(GrepTool::new(FsSearchSource)),
        Box::new(CodeSearchTool::new(FsSearchSource)),
        Box::new(TreeTool::new(FsSearchSource)),
    ];
    let names = ["Glob", "Grep", "CodeSearch", "Tree"];
    for (index, tool) in tools.iter().enumerate() {
        assert_eq!(tool.name(), names[index]);
        assert!(tool.is_read_only(), "{} must be read-only", tool.name());
        assert!(
            tool.is_concurrency_safe(),
            "{} must be concurrency-safe",
            tool.name()
        );
        assert!(
            tool.description().len() > 20,
            "{} needs a description",
            tool.name()
        );
        let schema = tool.schema();
        assert_eq!(schema.tool, names[index]);
        assert!(
            serde_json::to_value(&schema.input_schema).is_ok(),
            "{} schema must serialize",
            tool.name()
        );
    }
    let _ = tmp;
}

#[tokio::test]
async fn tree_respects_gitignore_in_listings() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    write_file(tmp.path(), "src/lib.rs", "");
    write_file(tmp.path(), "target/artifact.txt", "");
    std::fs::write(tmp.path().join(".gitignore"), "target/\n").expect("gitignore");

    let tool = TreeTool::new(FsSearchSource);
    let output = tool
        .call(
            json!({"path": tmp.path().to_string_lossy(), "max_depth": 2}),
            &ctx_in(tmp.path()),
        )
        .await
        .expect("call");
    let text = output.text_content();
    assert!(text.contains("src/"), "{text}");
    assert!(!text.contains("target"), "ignored dirs stay out: {text}");
    assert!(text.contains("1 directory"), "{text}");
}

#[tokio::test]
async fn grep_skips_binary_files() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mut binary = vec![0x41u8, 0x00, 0x42, 0x00]; // 'A', NUL, 'B', NUL
    binary.extend_from_slice(b"needle\n");
    std::fs::write(tmp.path().join("blob.bin"), binary).expect("binary");
    write_file(tmp.path(), "text.rs", "needle\n");

    let tool = GrepTool::new(FsSearchSource);
    let output = tool
        .call(
            json!({"pattern": "needle", "path": tmp.path().to_string_lossy()}),
            &ctx_in(tmp.path()),
        )
        .await
        .expect("call");
    let matches: Vec<Value> = serde_json::from_str(&output.text_content()).expect("json array");
    assert_eq!(matches.len(), 1, "{}", output.text_content());
    assert_eq!(matches[0]["file"], "text.rs");
}

#[tokio::test]
async fn grep_spills_oversized_results_to_the_context_temp_dir() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let line = format!("needle {}\n", "payload".repeat(500));
    write_file(
        tmp.path(),
        "big.rs",
        // ~70 KB, over the 50 KiB inline limit; rows stay under the
        // spill wrap width so the spilled body still re-parses as JSON
        &line.repeat(20),
    );
    let context = ctx_in(tmp.path());
    let tool = GrepTool::new(FsSearchSource);
    let output = tool
        .call(
            json!({
                "pattern": "needle",
                "path": tmp.path().to_string_lossy(),
                "max_matches": 20
            }),
            &context,
        )
        .await
        .expect("call");
    let text = output.text_content();
    assert!(text.contains("result too large"), "{text}");
    let start = text
        .find("written to: ")
        .map(|index| index + "written to: ".len())
        .expect("spill pointer");
    let end = text[start..]
        .find('\n')
        .map(|offset| start + offset)
        .expect("line end");
    let spilled = std::path::Path::new(text[start..end].trim());
    assert!(
        spilled.starts_with(tmp.path().join("spill")),
        "spill must land under the context temp dir: {}",
        spilled.display()
    );
    assert!(spilled.is_file(), "spilled file exists");
    let contents = std::fs::read_to_string(spilled).expect("read spill");
    let reparsed: Vec<Value> = serde_json::from_str(&contents).expect("spilled json");
    assert_eq!(reparsed.len(), 20);
}

#[tokio::test]
async fn grep_spilled_overlong_lines_stay_retrievable_through_the_read_tool() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let payload = "payload".repeat(30_000);
    write_file(
        tmp.path(),
        "huge.rs",
        &format!("needle {payload}\nlet other = 1;\n"),
    );
    let context = ctx_in(tmp.path());
    let tool = GrepTool::new(FsSearchSource);
    let output = tool
        .call(
            json!({
                "pattern": "needle",
                "path": tmp.path().to_string_lossy(),
                "max_matches": 5
            }),
            &context,
        )
        .await
        .expect("grep call");
    let text = output.text_content();
    let start = text.find("written to: ").expect("spill pointer") + "written to: ".len();
    let end = text[start..].find('\n').expect("line end") + start;
    let spill_path = text[start..end].trim().to_string();
    assert!(
        text.contains("wrapped at character boundaries"),
        "the spill pointer must disclose the wrap convention: {text}"
    );

    let reader = loopctl::tool::builtin::ReadTool::new(FsSearchSource);
    let mut collected = String::new();
    let mut saw_marker = false;
    let mut offset = 1usize;
    loop {
        let page = reader
            .call(
                json!({"path": spill_path, "offset": offset, "limit": 400}),
                &context,
            )
            .await
            .expect("read page")
            .text_content();
        for line in page.split('\n') {
            let Some((_, content)) = line.split_once('\t') else {
                continue;
            };
            assert!(
                content.len() <= loopctl::tool::builtin::search::output::MAX_SPILL_LINE_BYTES,
                "every numbered Read line must fit the wrap width: {}",
                content.len()
            );
            if let Some(rest) = content.strip_prefix("↪ ") {
                saw_marker = true;
                collected.push_str(rest);
            } else {
                if !collected.is_empty() {
                    collected.push('\n');
                }
                collected.push_str(content);
            }
        }
        let Some(next) = page.split("Use offset=").nth(1) else {
            break;
        };
        let digits: String = next.chars().take_while(char::is_ascii_digit).collect();
        offset = digits.parse().expect("next offset");
    }
    assert!(saw_marker, "the overlong row must actually wrap");
    assert!(
        collected.contains(&payload),
        "the paged read must recover the whole matched payload"
    );
}

#[tokio::test]
async fn glob_spills_oversized_listings_like_the_content_tools() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let long_name = format!("{}.rs", "f".repeat(110));
    for index in 0..700 {
        write_file(tmp.path(), &format!("src/{index:03}-{long_name}"), "");
    }
    let context = ctx_in(tmp.path());
    let tool = GlobTool::new(FsSearchSource);
    let output = tool
        .call(
            json!({"pattern": "**/*.rs", "path": tmp.path().to_string_lossy()}),
            &context,
        )
        .await
        .expect("call");
    let text = output.text_content();
    assert!(text.contains("result too large"), "{text}");
    let start = text
        .find("written to: ")
        .map(|index| index + "written to: ".len())
        .expect("spill pointer");
    let end = text[start..]
        .find('\n')
        .map(|offset| start + offset)
        .expect("line end");
    let spilled = std::path::Path::new(text[start..end].trim());
    assert!(
        spilled.starts_with(tmp.path().join("spill")),
        "spill must land under the context temp dir: {}",
        spilled.display()
    );
    let contents = std::fs::read_to_string(spilled).expect("read spill");
    let reparsed: Vec<String> = serde_json::from_str(&contents).expect("spilled json");
    assert_eq!(reparsed.len(), 700);
}

#[tokio::test]
async fn tree_spills_oversized_listings_like_the_content_tools() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let long_name = "t".repeat(110);
    for index in 0..700 {
        let dir = if index % 2 == 0 { "left" } else { "right" };
        write_file(tmp.path(), &format!("{dir}/{index:03}-{long_name}.rs"), "");
    }
    let context = ctx_in(tmp.path());
    let tool = TreeTool::new(FsSearchSource);
    let output = tool
        .call(
            json!({"path": tmp.path().to_string_lossy(), "max_depth": 2}),
            &context,
        )
        .await
        .expect("call");
    let text = output.text_content();
    assert!(text.contains("result too large"), "{text}");
    let start = text
        .find("written to: ")
        .map(|index| index + "written to: ".len())
        .expect("spill pointer");
    let end = text[start..]
        .find('\n')
        .map(|offset| start + offset)
        .expect("line end");
    let spilled = std::path::Path::new(text[start..end].trim());
    assert!(
        spilled.starts_with(tmp.path().join("spill")),
        "spill must land under the context temp dir: {}",
        spilled.display()
    );
    let contents = std::fs::read_to_string(spilled).expect("read spill");
    assert!(
        contents.lines().any(|line| line.contains("left/")),
        "the spilled body is the rendered tree"
    );
    assert!(
        contents.contains("700 files"),
        "the summary rides along in the spilled body"
    );
}

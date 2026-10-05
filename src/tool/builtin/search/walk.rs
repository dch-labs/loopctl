//! Shared gitignore-aware directory walker and glob primitives.
//!
//! The single [`walk_files`] entry point is what every search-style tool
//! (`Glob`, `Grep`, `CodeSearch`, `Tree`) uses for directory traversal. It
//! honors `.gitignore`, `.git/info/exclude`, the global gitignore, the
//! `ignore` crate's `.ignore`, and the custom ignore files `.loopctlignore`
//! and `.dchignore`, and it applies an always-exclude list so that
//! `target/`, `node_modules/`, `.git/`, and friends never leak into results —
//! even in non-git repositories.
//!
//! The [`wildcard_match`] / [`matches_any_glob`] / [`likely_binary`] helpers
//! are filename-level `*`/`?` matching for include/exclude filters and binary
//! detection. They live here so every consumer uses one implementation.

use std::path::Path;

/// Build a gitignore-aware file walker over `base`.
///
/// Honors `.gitignore`, `.git/info/exclude`, the global gitignore, and the
/// custom ignore files `.loopctlignore` and `.dchignore` (when present);
/// skips hidden entries; and always excludes
/// `target/`, `node_modules/`, `.git/`, `__pycache__/`, `.venv/`, and the other
/// directories named by [`build_default_overrides`] — even in non-git
/// repositories. Symlinks are not followed, matching `ignore`'s default, so
/// symlink cycles cannot hang the walker.
///
/// The optional `include_patterns` and `exclude_patterns` are filename-level
/// globs (`*`/`?`); pass empty slices to disable filename filtering. `Glob`
/// always passes empty slices because it filters via its own matcher after the
/// walk; `Grep` and `CodeSearch` use these to restrict the file set.
///
/// Returns each matched file as an `ignore::DirEntry` (not directories, not
/// symlinked-into files).
#[must_use]
pub fn walk_files(
    base: &Path,
    include_patterns: &[String],
    exclude_patterns: &[String],
) -> Box<dyn Iterator<Item = ignore::DirEntry> + Send> {
    let mut builder = ignore::WalkBuilder::new(base);
    builder
        .hidden(true)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .add_custom_ignore_filename(".loopctlignore")
        .add_custom_ignore_filename(".dchignore");

    let overrides = build_default_overrides(base);
    builder.overrides(overrides);

    let include = include_patterns.to_vec();
    let exclude = exclude_patterns.to_vec();

    Box::new(
        builder
            .build()
            .filter_map(std::result::Result::ok)
            .filter(move |e| {
                if !e.file_type().is_some_and(|ft| ft.is_file()) {
                    return false;
                }
                let file_name = e.path().file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !include.is_empty() && !matches_any_glob(file_name, &include) {
                    return false;
                }
                if matches_any_glob(file_name, &exclude) {
                    return false;
                }
                true
            }),
    )
}

/// One entry (file or directory) yielded by [`walk_entries`].
///
/// Wraps the path and whether it's a directory so the Tree tool can render
/// both nodes without a second metadata lookup.
#[derive(Debug, Clone)]
pub struct WalkEntry {
    /// The entry's path relative to the filesystem root (absolute or as-is).
    ///
    /// The path the walker produces — same as `ignore::DirEntry::path()`.
    pub path: std::path::PathBuf,

    /// Whether this entry is a directory rather than a regular file.
    ///
    /// Drives the trailing-slash rendering in the Tree tool and the
    /// dirs-before-files sort order. Set by [`walk_entries`] from the
    /// walker's `file_type()`; never derived elsewhere.
    pub is_dir: bool,
}

/// Directories the walker always prunes, even without `.gitignore`.
///
/// These mirror [`build_default_overrides`] but are checked at the directory
/// level — the `ignore` crate's override patterns filter files, not directory
/// descent, so a directory like `target/` would still be yielded as a node
/// even though its children are excluded. This list prevents the directory
/// itself from appearing in tree output.
const ALWAYS_PRUNE_DIRS: &[&str] = &[
    "target",
    "node_modules",
    ".git",
    "__pycache__",
    ".next",
    ".venv",
    "venv",
    "dist",
    "build",
    ".cache",
];

/// Walk `base` yielding both files and directories, gitignore-aware.
///
/// Shares the same `WalkBuilder` config as [`walk_files`] (hidden entries
/// skipped, `.gitignore`/`.git/info/exclude`/global gitignore and the
/// `.loopctlignore`/`.dchignore` custom names
/// honored, always-exclude overrides applied) but does **not** filter to
/// files only. The optional `max_depth` caps the traversal depth (number of
/// path components below `base`; `None` = unlimited). Used by the Tree tool,
/// which needs directory nodes as well as files.
///
/// The `base` path itself is dropped from the output (the caller knows its
/// root), and directories on the always-prune list (`target`, `node_modules`,
/// etc.) are pruned during descent via the `ignore` crate's `filter_entry`
/// hook so neither the node nor its children are visited — the override
/// patterns filter files only, not directory descent, so the predicate fills
/// that gap.
#[must_use]
pub fn walk_entries(base: &Path, max_depth: Option<usize>) -> Vec<WalkEntry> {
    let mut builder = ignore::WalkBuilder::new(base);
    builder
        .hidden(true)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .add_custom_ignore_filename(".loopctlignore")
        .add_custom_ignore_filename(".dchignore");
    builder.max_depth(max_depth);

    let overrides = build_default_overrides(base);
    builder.overrides(overrides);

    let base_owned = base.to_path_buf();
    builder.filter_entry(move |e| {
        if e.path() == base_owned {
            return true;
        }
        if e.file_type().is_some_and(|ft| ft.is_dir()) {
            let name = e.path().file_name().and_then(|n| n.to_str()).unwrap_or("");
            return !ALWAYS_PRUNE_DIRS.contains(&name);
        }
        true
    });

    builder
        .build()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path() != base)
        .filter_map(|e| {
            let ft = e.file_type()?;
            if ft.is_dir() {
                Some(WalkEntry {
                    path: e.path().to_path_buf(),
                    is_dir: true,
                })
            } else if ft.is_file() {
                Some(WalkEntry {
                    path: e.path().to_path_buf(),
                    is_dir: false,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Build the always-exclude override set for a search root.
///
/// These directories are removed from results regardless of `.gitignore`,
/// because they are almost never useful to search (`target/`, `node_modules/`,
/// `.git/`, `__pycache__/`, virtualenvs, build output). The patterns are
/// derived from the same always-prune list used by `walk_entries`, so the
/// file-level overrides and the directory-descent pruning stay in sync. The
/// `!`-prefix is gitignore "negate-to-exclude" syntax as interpreted by
/// `ignore::overrides::Override`.
/// On a build error the matcher falls back to an empty `Override` (no
/// exclusions) rather than panicking.
#[must_use]
pub fn build_default_overrides(base: &Path) -> ignore::overrides::Override {
    let mut builder = ignore::overrides::OverrideBuilder::new(base);
    for dir in ALWAYS_PRUNE_DIRS {
        builder.add(&format!("!**/{dir}/**")).ok();
    }
    builder
        .build()
        .unwrap_or_else(|_| ignore::overrides::Override::empty())
}

/// True if `filename` matches any pattern in `patterns`.
///
/// Each pattern is matched with [`wildcard_match`], which supports `*` (any
/// run of characters) and `?` (a single character). Used by [`walk_files`]'s
/// include/exclude filters and exposed for other tools that need the same
/// filename-glob check.
#[must_use]
pub fn matches_any_glob(filename: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|p| wildcard_match(filename, p))
}

/// Single-segment wildcard matcher supporting `*` and `?`.
///
/// `*` matches any run of characters (including empty); `?` matches exactly
/// one character. Both are byte-position based with backtracking, so they
/// handle repeated wildcards correctly. This is intentionally limited to a
/// single path segment — it does not understand `/`. Path-aware `**` matching
/// is each tool's job (ripgrep's `Override` for `Glob`), not here.
#[must_use]
pub fn wildcard_match(text: &str, pattern: &str) -> bool {
    let text_chars: Vec<char> = text.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_len = text_chars.len();
    let pat_len = pattern_chars.len();
    let mut t_pos = 0usize;
    let mut p_pos = 0usize;
    let mut star_pos: Option<usize> = None;
    let mut match_pos = 0usize;
    while t_pos < text_len {
        let p_char = pattern_chars.get(p_pos).copied();
        let t_char = text_chars.get(t_pos).copied();
        let is_question = p_char == Some('?');
        let is_exact = p_char.is_some_and(|p| t_char.is_some_and(|t| p == t));
        if is_question || is_exact {
            t_pos = t_pos.saturating_add(1);
            p_pos = p_pos.saturating_add(1);
        } else if p_char == Some('*') {
            star_pos = Some(p_pos);
            match_pos = t_pos;
            p_pos = p_pos.saturating_add(1);
        } else if let Some(sp) = star_pos {
            p_pos = sp.saturating_add(1);
            match_pos = match_pos.saturating_add(1);
            t_pos = match_pos;
        } else {
            return false;
        }
    }
    while p_pos < pat_len && pattern_chars.get(p_pos) == Some(&'*') {
        p_pos = p_pos.saturating_add(1);
    }
    p_pos == pat_len
}

/// Binary file extensions that should always be skipped by search tools.
///
/// The cheap first stage of [`likely_binary`]: a known-binary
/// extension decides without opening the file, so archives,
/// images, and object files never enter the read path.
const BINARY_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "pdf", "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
    "exe", "dll", "so", "dylib", "a", "lib", "o", "obj", "class", "pyc", "rlib", "wasm",
];

/// Bytes of the read buffer used by [`likely_binary`] for content sniffing.
///
/// Matches the NUL-heuristic window of `grep`/`ripgrep`: enough of
/// the prefix to catch real binaries, small enough that the sniff
/// stays a negligible fraction of any file that passes it.
const SNIFF_BYTES: usize = 8192;

/// True if `path` is likely a binary file, by extension or content sniff.
///
/// The check has two stages: first, a known-binary extension (`png`, `pdf`,
/// `so`, …); if that is inconclusive, the first `SNIFF_BYTES` are read and
/// passed to [`bytes_look_binary`]. Either signal marks the file binary.
/// Read faults are treated as "not binary" so the caller can surface the real
/// error itself.
#[must_use]
pub fn likely_binary(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext_lower = ext.to_string_lossy().to_lowercase();
        if BINARY_EXTENSIONS.contains(&ext_lower.as_str()) {
            return true;
        }
    }
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut buffer = [0u8; SNIFF_BYTES];
    let Ok(n) = std::io::Read::read(&mut file, &mut buffer) else {
        return false;
    };
    let Some(window) = buffer.get(..n) else {
        return false;
    };
    bytes_look_binary(window)
}

/// True if a byte buffer looks like binary content.
///
/// Considers only the first `SNIFF_BYTES` of `bytes`. A NUL byte anywhere in
/// the window, or a ratio of text-like bytes below 75%, marks the content
/// binary. A byte is text-like when it is ASCII printable/whitespace or part
/// of a UTF-8 multibyte sequence (any byte >= `0x80`); the bytes that drag
/// the ratio down are therefore non-whitespace C0 control bytes — the actual
/// signature of binary payloads. This keeps UTF-8 text (CJK, emoji) readable
/// while still catching executables, images, and other non-text data. This is
/// the byte-level core shared by [`likely_binary`] (which opens the file) and
/// any caller that already holds the bytes — so the detection rule stays in
/// one place even when the IO path differs.
#[must_use]
pub fn bytes_look_binary(bytes: &[u8]) -> bool {
    let window = bytes.get(..SNIFF_BYTES).unwrap_or(bytes);
    if window.is_empty() {
        return false;
    }
    if window.contains(&0) {
        return true;
    }
    let threshold = window.len().saturating_mul(3).saturating_div(4);
    let text_bytes = window
        .iter()
        .filter(|&&b| {
            b == b'\t' || b == b'\n' || b == b'\r' || (32..=126).contains(&b) || b >= 0x80
        })
        .count();
    text_bytes < threshold
}

/// Upper bound (1 `MiB`) on a file's byte size before a search tool reads it whole.
///
/// The OOM guard for whole-file reads: a file over this ceiling is
/// skipped before any read is attempted, so a stray core dump or
/// dataset cannot translate into a same-order allocation.
pub const MAX_FILE_BYTES: u64 = 1024 * 1024;

/// Whether `path`'s metadata size exceeds [`MAX_FILE_BYTES`].
///
/// Metadata failures read as "not too large" so the caller still attempts the
/// read and surfaces the real I/O error rather than silently skipping.
#[must_use]
pub fn file_too_large(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.len() > MAX_FILE_BYTES)
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn wildcard_match_star() {
        assert!(wildcard_match("test.rs", "*.rs"));
        assert!(wildcard_match("file.json", "*.json"));
        assert!(!wildcard_match("test.rs", "*.json"));
        assert!(wildcard_match("anything", "*"));
    }

    #[test]
    fn wildcard_match_question() {
        assert!(wildcard_match("foo.rs", "???.rs"));
        assert!(!wildcard_match("test.rs", "???.rs"));
    }

    #[test]
    fn wildcard_match_combined() {
        assert!(wildcard_match("my_file_test.rs", "*_test.rs"));
        assert!(wildcard_match("file123.json", "file???.json"));
    }

    #[test]
    fn matches_any_glob_multiple_patterns() {
        let patterns = vec!["*.rs".to_string(), "*.json".to_string()];
        assert!(matches_any_glob("test.rs", &patterns));
        assert!(matches_any_glob("data.json", &patterns));
        assert!(!matches_any_glob("readme.md", &patterns));
    }

    #[test]
    fn matches_any_glob_empty_patterns() {
        let patterns: Vec<String> = vec![];
        assert!(!matches_any_glob("test.rs", &patterns));
    }

    #[test]
    fn likely_binary_by_extension() {
        assert!(likely_binary(&PathBuf::from("image.png")));
        assert!(likely_binary(&PathBuf::from("lib.rlib")));
        assert!(likely_binary(&PathBuf::from("code.pyc")));
        assert!(!likely_binary(&PathBuf::from("main.rs")));
        assert!(!likely_binary(&PathBuf::from("config.toml")));
    }

    #[test]
    fn bytes_look_binary_detects_nul() {
        assert!(bytes_look_binary(b"\x00\x01\x02"));
    }

    #[test]
    fn bytes_look_binary_detects_control_byte_ratio() {
        let mut bytes = vec![0x01u8; 1000];
        bytes.extend_from_slice(b"hello");
        assert!(bytes_look_binary(&bytes));
    }

    #[test]
    fn bytes_look_binary_accepts_high_multibyte_utf8() {
        let euro = "€".repeat(1000);
        assert!(!bytes_look_binary(euro.as_bytes()));
    }

    #[test]
    fn bytes_look_binary_accepts_plain_text() {
        assert!(!bytes_look_binary(b"hello world\nthis is text\r\n"));
    }

    #[test]
    fn build_default_overrides_excludes_target() {
        let overrides = build_default_overrides(Path::new("/repo"));
        for excluded in [
            "target/debug/foo.rs",
            "node_modules/pkg/index.js",
            ".git/HEAD",
            "__pycache__/x.pyc",
            ".next/cache.json",
            ".venv/bin/python",
            "venv/bin/python",
            "dist/bundle.js",
            "build/out.o",
            ".cache/tmp",
        ] {
            assert!(
                overrides.matched(excluded, false).is_ignore(),
                "{excluded} should be excluded by the always-exclude list"
            );
        }
    }

    #[test]
    fn build_default_overrides_does_not_exclude_source() {
        let overrides = build_default_overrides(Path::new("/repo"));
        for kept in ["src/main.rs", "README.md", "tests/glob.rs"] {
            assert!(
                !overrides.matched(kept, false).is_ignore(),
                "{kept} should not be excluded"
            );
        }
    }

    #[test]
    fn build_default_overrides_does_not_panic_on_relative_base() {
        let overrides = build_default_overrides(Path::new("."));
        assert!(overrides.matched("target/x.rs", false).is_ignore());
    }

    /// Walk `base` and return the relative paths of yielded entries,
    /// sorted — `ignore`'s traversal order is not lexical.
    fn walked(base: &Path, include: &[String], exclude: &[String]) -> Vec<String> {
        let mut out: Vec<String> = walk_files(base, include, exclude)
            .map(|entry| {
                entry
                    .path()
                    .strip_prefix(base)
                    .map(|rel| rel.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
            .collect();
        out.sort();
        out
    }

    /// Create a `.git` marker dir so `ignore` treats `dir` as a git work
    /// tree and honors `.gitignore`.
    fn make_git_repo(dir: &Path) {
        std::fs::create_dir_all(dir.join(".git")).expect("create .git marker");
    }

    #[test]
    fn walk_files_returns_files_only() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(tmp.path().join("a.rs"), "").expect("write");
        std::fs::create_dir_all(tmp.path().join("src")).expect("mkdir");
        std::fs::write(tmp.path().join("src/b.rs"), "").expect("write");
        let got = walked(tmp.path(), &[], &[]);
        assert!(got.contains(&"a.rs".to_string()), "{got:?}");
        assert!(got.contains(&"src/b.rs".to_string()), "{got:?}");
        assert!(
            !got.iter()
                .any(|p| p == "src" || p.ends_with("/src") || p == ".git"),
            "no directories in output: {got:?}"
        );
    }

    #[test]
    fn walk_files_respects_gitignore_in_git_repo() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        make_git_repo(tmp.path());
        std::fs::write(tmp.path().join(".gitignore"), "ignored.rs\n").expect("write gitignore");
        std::fs::write(tmp.path().join("ignored.rs"), "").expect("write");
        std::fs::write(tmp.path().join("kept.rs"), "").expect("write");
        let got = walked(tmp.path(), &[], &[]);
        assert!(got.contains(&"kept.rs".to_string()), "{got:?}");
        assert!(
            !got.contains(&"ignored.rs".to_string()),
            "gitignored file must be absent: {got:?}"
        );
    }

    #[test]
    fn walk_files_does_not_apply_gitignore_without_git() {
        // Without a .git marker, ignore does not honor .gitignore. This
        // locks the upstream behavior so a future change is caught.
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(tmp.path().join(".gitignore"), "ignored.rs\n").expect("write gitignore");
        std::fs::write(tmp.path().join("ignored.rs"), "").expect("write");
        let got = walked(tmp.path(), &[], &[]);
        assert!(
            got.contains(&"ignored.rs".to_string()),
            "without .git, .gitignore is not applied: {got:?}"
        );
    }

    #[test]
    fn walk_files_respects_dchignore_without_git() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(tmp.path().join(".dchignore"), "secret.rs\n").expect("write dchignore");
        std::fs::write(tmp.path().join("secret.rs"), "").expect("write");
        std::fs::write(tmp.path().join("public.rs"), "").expect("write");
        let got = walked(tmp.path(), &[], &[]);
        assert!(got.contains(&"public.rs".to_string()), "{got:?}");
        assert!(
            !got.contains(&"secret.rs".to_string()),
            ".dchignore'd file must be absent: {got:?}"
        );
    }

    #[test]
    fn walk_files_respects_loopctlignore_without_git() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(tmp.path().join(".loopctlignore"), "local-only.rs\n")
            .expect("write loopctlignore");
        std::fs::write(tmp.path().join("local-only.rs"), "").expect("write");
        std::fs::write(tmp.path().join("shared.rs"), "").expect("write");
        let got = walked(tmp.path(), &[], &[]);
        assert!(got.contains(&"shared.rs".to_string()), "{got:?}");
        assert!(
            !got.contains(&"local-only.rs".to_string()),
            ".loopctlignore'd file must be absent: {got:?}"
        );
    }

    #[test]
    fn walk_files_prunes_always_exclude_dirs_at_any_depth() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join("pkg/node_modules")).expect("mkdir");
        std::fs::write(tmp.path().join("pkg/node_modules/x.js"), "").expect("write");
        std::fs::create_dir_all(tmp.path().join("vendor/.venv")).expect("mkdir");
        std::fs::write(tmp.path().join("vendor/.venv/tool.py"), "").expect("write");
        std::fs::write(tmp.path().join("main.rs"), "").expect("write");
        let got = walked(tmp.path(), &[], &[]);
        assert_eq!(
            got,
            vec!["main.rs".to_string()],
            "a nested always-prune dir is not anchored to the root: {got:?}"
        );
    }

    #[test]
    fn walk_files_applies_always_exclude_without_git() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join("target/debug")).expect("mkdir");
        std::fs::write(tmp.path().join("target/debug/foo.rs"), "").expect("write");
        std::fs::write(tmp.path().join("main.rs"), "").expect("write");
        let got = walked(tmp.path(), &[], &[]);
        assert!(got.contains(&"main.rs".to_string()), "{got:?}");
        assert!(
            !got.iter().any(|p| p.contains("target")),
            "target/ must be always-excluded: {got:?}"
        );
    }

    #[test]
    fn walk_files_include_filter_restricts_filenames() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(tmp.path().join("a.rs"), "").expect("write");
        std::fs::write(tmp.path().join("b.json"), "").expect("write");
        std::fs::write(tmp.path().join("c.rs"), "").expect("write");
        let include = vec!["*.rs".to_string()];
        let got = walked(tmp.path(), &include, &[]);
        assert_eq!(got, vec!["a.rs".to_string(), "c.rs".to_string()]);
    }

    #[test]
    fn walk_files_exclude_filter_removes_filenames() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(tmp.path().join("a.rs"), "").expect("write");
        std::fs::write(tmp.path().join("b.test.rs"), "").expect("write");
        std::fs::write(tmp.path().join("c.rs"), "").expect("write");
        let exclude = vec!["*.test.rs".to_string()];
        let got = walked(tmp.path(), &[], &exclude);
        assert_eq!(got, vec!["a.rs".to_string(), "c.rs".to_string()]);
    }

    #[test]
    fn walk_files_empty_dir_yields_nothing() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let got = walked(tmp.path(), &[], &[]);
        assert!(got.is_empty(), "{got:?}");
    }

    /// Walk `base` and return `(relative_path, is_dir)` per entry, sorted.
    ///
    /// Sorting makes the assertion order-independent: `ignore`'s
    /// traversal order is not lexical, and `walk_entries` does not
    /// sort its output.
    fn walked_entries(base: &Path, max_depth: Option<usize>) -> Vec<(String, bool)> {
        let mut out: Vec<(String, bool)> = walk_entries(base, max_depth)
            .iter()
            .map(|entry| {
                let rel = entry
                    .path
                    .strip_prefix(base)
                    .map(|rel| rel.to_string_lossy().into_owned())
                    .unwrap_or_default();
                (rel, entry.is_dir)
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn walk_entries_yields_dirs_and_files() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join("src")).expect("mkdir");
        std::fs::write(tmp.path().join("a.rs"), "").expect("write");
        std::fs::write(tmp.path().join("src/b.rs"), "").expect("write");
        let got = walked_entries(tmp.path(), None);
        assert!(
            got.contains(&("a.rs".to_string(), false)),
            "file present: {got:?}"
        );
        assert!(
            got.contains(&("src/b.rs".to_string(), false)),
            "nested file present: {got:?}"
        );
        assert!(
            got.contains(&("src".to_string(), true)),
            "directory present with is_dir=true: {got:?}"
        );
    }

    #[test]
    fn walk_entries_skips_root_itself() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(tmp.path().join("a.rs"), "").expect("write");
        let got = walked_entries(tmp.path(), None);
        let root_name = tmp
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        assert!(
            !got.iter().any(|(p, _)| p == root_name || p.is_empty()),
            "root must not appear in output: {got:?}"
        );
    }

    #[test]
    fn walk_entries_prunes_always_exclude_dirs_without_git() {
        // ignore's override patterns filter FILES, not directory
        // descent — the prune list removes the directory node itself.
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join("target/debug")).expect("mkdir");
        std::fs::write(tmp.path().join("target/debug/foo"), "").expect("write");
        std::fs::create_dir_all(tmp.path().join("node_modules/pkg")).expect("mkdir");
        std::fs::write(tmp.path().join("node_modules/pkg/index.js"), "").expect("write");
        std::fs::write(tmp.path().join("main.rs"), "").expect("write");
        let got = walked_entries(tmp.path(), None);
        assert!(
            !got.iter()
                .any(|(p, _)| p == "target" || p.starts_with("target/")),
            "target/ dir node and children must be pruned: {got:?}"
        );
        assert!(
            !got.iter()
                .any(|(p, _)| p == "node_modules" || p.starts_with("node_modules/")),
            "node_modules/ dir node and children must be pruned: {got:?}"
        );
        assert!(
            got.contains(&("main.rs".to_string(), false)),
            "main.rs kept: {got:?}"
        );
    }

    #[test]
    fn walk_entries_depth_none_is_unlimited() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join("a/b/c/d/e")).expect("mkdir");
        std::fs::write(tmp.path().join("a/b/c/d/e/deep.rs"), "").expect("write");
        let got = walked_entries(tmp.path(), None);
        assert!(
            got.contains(&("a/b/c/d/e/deep.rs".to_string(), false)),
            "unlimited depth reaches the leaf: {got:?}"
        );
    }

    #[test]
    fn walk_entries_depth_caps_components_below_root() {
        // `ignore`'s max_depth counts path components below the root:
        // `a` = depth 1, `a/b` = depth 2, `a/b/l2.rs` = depth 3.
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join("a/b/c")).expect("mkdir");
        std::fs::write(tmp.path().join("a/l1.rs"), "").expect("write");
        std::fs::write(tmp.path().join("a/b/l2.rs"), "").expect("write");
        std::fs::write(tmp.path().join("a/b/c/l3.rs"), "").expect("write");
        let got = walked_entries(tmp.path(), Some(2));
        assert!(
            got.contains(&("a".to_string(), true)),
            "depth-1 dir node in range: {got:?}"
        );
        assert!(
            got.contains(&("a/b".to_string(), true)),
            "depth-2 dir node in range: {got:?}"
        );
        assert!(
            got.contains(&("a/l1.rs".to_string(), false)),
            "depth-2 file in range: {got:?}"
        );
        assert!(
            !got.contains(&("a/b/l2.rs".to_string(), false)),
            "depth-3 file out of range: {got:?}"
        );
        assert!(
            !got.contains(&("a/b/c".to_string(), true)),
            "depth-3 dir node out of range: {got:?}"
        );
        assert!(
            !got.contains(&("a/b/c/l3.rs".to_string(), false)),
            "depth-4 file out of range: {got:?}"
        );
    }

    #[test]
    fn walk_entries_respects_gitignore_in_git_repo() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        make_git_repo(tmp.path());
        std::fs::write(tmp.path().join(".gitignore"), "ignored.rs\n").expect("write gitignore");
        std::fs::write(tmp.path().join("ignored.rs"), "").expect("write");
        std::fs::write(tmp.path().join("kept.rs"), "").expect("write");
        let got = walked_entries(tmp.path(), None);
        assert!(
            got.contains(&("kept.rs".to_string(), false)),
            "kept file present: {got:?}"
        );
        assert!(
            !got.contains(&("ignored.rs".to_string(), false)),
            "gitignored file absent: {got:?}"
        );
    }
}

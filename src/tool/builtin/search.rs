//! The search tool family — glob, grep, code-search, and tree over a
//! pluggable source.
//!
//! Four tools plus the machinery they share: a gitignore-aware walker,
//! a regex content-search runner with per-file and global caps, a
//! process-global compiled-regex LRU cache, and large-output shaping.
//! Every tool is generic over [`SearchSource`] — the filesystem
//! implementation ships here, and a remote source (a `GitHub` tree at a
//! pinned revision, for a difftrace-style consumer) can serve the same
//! tools later without touching them.
//!
//! Security trust boundary: the registered source *is* the authority
//! boundary. [`FsSearchSource`] applies no containment policy — a
//! recorded port decision (the walker is read-only and the fs-session
//! machinery is `fs_tools`-gated), so its search roots resolve
//! lexically and its reads go wherever the process's own permissions
//! allow. A host that runs contained filesystem tools should register
//! a source enforcing its own boundary — the seam exists for exactly
//! that — or treat an `FsSearchSource` registration as granting
//! unrestricted read access.

pub mod code_search;
pub mod content;
pub mod glob;
pub mod grep;
pub mod output;
pub mod regex_cache;
pub mod resolve;
pub mod tree;
pub mod walk;

pub use code_search::CodeSearchTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use tree::TreeTool;

use std::future::Future;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;

use crate::tool::ToolError;
use crate::tool::builtin::read::ContentSource;
use crate::tool::builtin::read::SourceContent;

/// One entry (file or directory) yielded by a source traversal.
///
/// The neutral walk unit: the path as the source addresses it, plus
/// the file-or-directory discrimination the tree renderer and the
/// search loop need. A filesystem source fills it from directory
/// metadata; a remote source fills it from its tree listing — the
/// consumers cannot tell them apart.
#[derive(Debug, Clone)]
pub struct SourceEntry {
    /// The entry's path, as the walker produced it.
    ///
    /// Relative to the search base on the filesystem backend;
    /// sources with their own addressing keep their own form.
    pub path: PathBuf,

    /// Whether this entry is a directory.
    ///
    /// Drives the trailing-slash rendering in the tree tool and the
    /// dirs-before-files sort order.
    pub is_dir: bool,
}

impl SourceEntry {
    /// Whether this entry is a regular file.
    ///
    /// The complement of [`is_dir`](Self::is_dir); kept as a method
    /// rather than a second field so the two can never disagree.
    #[must_use]
    pub fn is_file(&self) -> bool {
        !self.is_dir
    }
}

/// A [`ContentSource`] that can also be searched.
///
/// The search family needs more than reading: it traverses a tree
/// under ignore rules, sniffs binaries, and guards against oversized
/// files. This trait extends the read seam with those operations so
/// the four tools stay source-agnostic — the filesystem
/// implementation ([`FsSearchSource`]) ships here, and a remote tree
/// source implements the same surface later without any tool change.
pub trait SearchSource: ContentSource {
    /// Walk files under `base` under ignore rules and name filters.
    ///
    /// Yields regular files only, honoring the source's ignore
    /// semantics (a filesystem honors `.gitignore` and friends) and
    /// the filename-level include/exclude glob filters — empty slices
    /// disable a filter.
    fn walk_files<'a>(
        &'a self,
        base: &'a Path,
        include: &'a [String],
        exclude: &'a [String],
    ) -> Box<dyn Iterator<Item = SourceEntry> + Send + 'a>;

    /// Walk files and directories under `base`, depth-capped.
    ///
    /// The tree renderer needs directory nodes too; `max_depth` of
    /// `None` means unlimited. The `base` itself is not yielded.
    fn walk_entries(&self, base: &Path, max_depth: Option<usize>) -> Vec<SourceEntry>;

    /// Read up to `cap` bytes from `path`.
    ///
    /// The capped read backing the content-search loop; a fault is
    /// the caller's signal to skip the file, matching the
    /// unreadable-file behavior of grep and ripgrep.
    ///
    /// # Errors
    ///
    /// Returns the underlying I/O error when the open or read fails.
    fn read_capped(&self, path: &Path, cap: u64) -> std::io::Result<Vec<u8>>;

    /// Whether `path` is likely binary, by name or content sniff.
    ///
    /// Sources that cannot sniff may decide by name alone; a false
    /// negative costs a wasted scan, never a wrong result.
    fn likely_binary(&self, path: &Path) -> bool;

    /// Whether `path`'s size exceeds the read ceiling.
    ///
    /// The OOM guard for whole-file reads; metadata failures read as
    /// "not too large" so the attempted read surfaces the real error.
    fn file_too_large(&self, path: &Path) -> bool;
}

impl<S: SearchSource> ContentSource for std::sync::Arc<S> {
    fn read<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<SourceContent, ToolError>> + Send + 'a>> {
        (**self).read(path)
    }

    fn size<'a>(&'a self, path: &'a str) -> Pin<Box<dyn Future<Output = Option<u64>> + Send + 'a>> {
        (**self).size(path)
    }
}

/// A shared source serves the seam.
///
/// Tools hold their source behind an `Arc` already; these delegation
/// impls let callers keep a second handle (to a cached or instrumented
/// source) and hand clones to several tools, so sharing composes with
/// the generic tool types instead of forcing a choice between them.
impl<S: SearchSource> SearchSource for std::sync::Arc<S> {
    fn walk_files<'a>(
        &'a self,
        base: &'a Path,
        include: &'a [String],
        exclude: &'a [String],
    ) -> Box<dyn Iterator<Item = SourceEntry> + Send + 'a> {
        (**self).walk_files(base, include, exclude)
    }

    fn walk_entries(&self, base: &Path, max_depth: Option<usize>) -> Vec<SourceEntry> {
        (**self).walk_entries(base, max_depth)
    }

    fn read_capped(&self, path: &Path, cap: u64) -> std::io::Result<Vec<u8>> {
        (**self).read_capped(path, cap)
    }

    fn likely_binary(&self, path: &Path) -> bool {
        (**self).likely_binary(path)
    }

    fn file_too_large(&self, path: &Path) -> bool {
        (**self).file_too_large(path)
    }
}

/// The filesystem [`SearchSource`].
///
/// Delegates to the [`walk`] module's gitignore-aware machinery and
/// plain filesystem reads — the behavior the search family was ported
/// with, preserved as one implementation of the seam.
///
/// Trust boundary: this implementation carries no containment
/// policy — roots resolve lexically against the context cwd and
/// reads go wherever the process's own permissions allow. The host
/// that registers it chooses that authority; a host that wants a
/// contained search should supply its own [`SearchSource`] that
/// enforces the boundary before serving entries.
#[derive(Debug, Clone, Default)]
pub struct FsSearchSource;

impl ContentSource for FsSearchSource {
    fn read<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<SourceContent, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let bytes = std::fs::read(path)
                .map_err(|error| ToolError::Execution(format!("read {path}: {error}")))?;
            match String::from_utf8(bytes) {
                Ok(text) => Ok(SourceContent::Text(text)),
                Err(error) => Ok(SourceContent::Bytes(error.into_bytes())),
            }
        })
    }
    fn size<'a>(&'a self, path: &'a str) -> Pin<Box<dyn Future<Output = Option<u64>> + Send + 'a>> {
        Box::pin(async move { std::fs::metadata(path).ok().map(|m| m.len()) })
    }
}

impl SearchSource for FsSearchSource {
    fn walk_files<'a>(
        &'a self,
        base: &'a Path,
        include: &'a [String],
        exclude: &'a [String],
    ) -> Box<dyn Iterator<Item = SourceEntry> + Send + 'a> {
        Box::new(
            walk::walk_files(base, include, exclude).map(|entry| SourceEntry {
                path: entry.path().to_path_buf(),
                is_dir: false,
            }),
        )
    }

    fn walk_entries(&self, base: &Path, max_depth: Option<usize>) -> Vec<SourceEntry> {
        walk::walk_entries(base, max_depth)
            .into_iter()
            .map(|entry| SourceEntry {
                path: entry.path,
                is_dir: entry.is_dir,
            })
            .collect()
    }

    fn read_capped(&self, path: &Path, cap: u64) -> std::io::Result<Vec<u8>> {
        use std::io::Read;
        let file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.take(cap).read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    fn likely_binary(&self, path: &Path) -> bool {
        walk::likely_binary(path)
    }

    fn file_too_large(&self, path: &Path) -> bool {
        walk::file_too_large(path)
    }
}

/// Test-support surface shared by the four tool test modules.
///
/// A hermetic [`SearchSource`]: an in-memory file tree with text or
/// binary content, marked-oversized files, and call counters so pins
/// can prove the tools never touched the source on a rejected path.
/// No filesystem, no network — everything is addressable strings.
#[cfg(test)]
pub(crate) mod test_support {
    use std::path::Path;
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicU32;
    use std::sync::atomic::Ordering;

    use super::FsSearchSource;
    use super::SearchSource;
    use super::SourceEntry;
    use crate::tool::ToolError;
    use crate::tool::builtin::read::ContentSource;
    use crate::tool::builtin::read::SourceContent;

    /// One fake file's content and flags.
    ///
    /// The per-file knobs the search loop consults — text, the
    /// binary verdict, and the oversized verdict — in one value so
    /// fixtures read as a file listing rather than parallel maps.
    #[derive(Clone)]
    pub(crate) struct FakeFile {
        /// The file's text content, decoded verbatim on read.
        ///
        /// Returned by `read_capped` as bytes; never sniffed, so a
        /// fixture with NUL bytes stays readable unless `binary` is
        /// set to make the sniff report it.
        pub content: String,

        /// Whether the file reports as binary.
        ///
        /// Drives `likely_binary` directly; `false` keeps the file
        /// in the content-search loop regardless of its bytes.
        pub binary: bool,

        /// Whether the file reports as oversized.
        ///
        /// Drives `file_too_large` directly, exercising the OOM guard
        /// without materializing a real over-ceiling file.
        pub too_large: bool,
    }

    /// Build a plain text fake file.
    ///
    /// The common fixture shape: readable, not binary, not
    /// oversized — the flags default to the pass-through path.
    pub(crate) fn text(content: &str) -> FakeFile {
        FakeFile {
            content: content.to_string(),
            binary: false,
            too_large: false,
        }
    }

    /// The in-memory source.
    ///
    /// Implements the whole seam with no filesystem and no network;
    /// the counters make rejected-input pins provable — a test can
    /// assert the source was never walked or read.
    pub(crate) struct FakeSearchSource {
        /// Files by absolute path, in insertion order for walks.
        ///
        /// Mutexed so the shared-`Arc` delegation impls keep the
        /// trait's `Send + Sync` promises honestly.
        files: Mutex<Vec<(PathBuf, FakeFile)>>,

        /// Directories by absolute path.
        ///
        /// Yielded by `walk_entries` only — the file walk never
        /// produces directory nodes, matching the seam's contract.
        dirs: Vec<PathBuf>,

        /// Count of `walk_files` calls, for never-walked pins.
        ///
        /// Incremented on every entry into the walk, so a rejected
        /// input proven to leave it at zero covers every file the
        /// source holds.
        pub walks: AtomicU32,

        /// Count of `read_capped` calls, for never-read pins.
        ///
        /// Also bumped by the `ContentSource` read, so either entry
        /// point observes the same never-touched verdict.
        pub reads: AtomicU32,
    }

    impl FakeSearchSource {
        /// Build a source holding the given files.
        ///
        /// Addresses are absolute paths as the tools would resolve
        /// them; the fixture list is the entire tree.
        pub(crate) fn with(files: &[(&str, FakeFile)]) -> Self {
            Self {
                files: Mutex::new(
                    files
                        .iter()
                        .map(|(path, file)| (PathBuf::from(path), file.clone()))
                        .collect(),
                ),
                dirs: Vec::new(),
                walks: AtomicU32::new(0),
                reads: AtomicU32::new(0),
            }
        }
    }

    impl ContentSource for FakeSearchSource {
        fn read<'a>(
            &'a self,
            path: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<SourceContent, ToolError>> + Send + 'a>> {
            Box::pin(async move {
                self.reads.fetch_add(1, Ordering::SeqCst);
                self.files
                    .lock()
                    .expect("files lock")
                    .iter()
                    .find(|(candidate, _)| candidate == &PathBuf::from(path))
                    .map(|(_, file)| SourceContent::Text(file.content.clone()))
                    .ok_or_else(|| ToolError::Execution(format!("fake source miss: {path}")))
            })
        }
    }

    impl SearchSource for FakeSearchSource {
        fn walk_files<'a>(
            &'a self,
            base: &'a Path,
            include: &'a [String],
            exclude: &'a [String],
        ) -> Box<dyn Iterator<Item = SourceEntry> + Send + 'a> {
            self.walks.fetch_add(1, Ordering::SeqCst);
            let base = base.to_path_buf();
            let include = include.to_vec();
            let exclude = exclude.to_vec();
            let mut entries: Vec<SourceEntry> = self
                .files
                .lock()
                .expect("files lock")
                .iter()
                .filter(|(path, _)| path.starts_with(&base))
                .map(|(path, _)| SourceEntry {
                    path: path.clone(),
                    is_dir: false,
                })
                .collect();
            entries.sort_by(|a, b| a.path.cmp(&b.path));
            let filtered = entries
                .into_iter()
                .filter(move |entry| {
                    let name = entry
                        .path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("");
                    let included = include.is_empty()
                        || include
                            .iter()
                            .any(|pattern| name.ends_with(pattern.trim_start_matches('*')));
                    let excluded = exclude
                        .iter()
                        .any(|pattern| name.ends_with(pattern.trim_start_matches('*')));
                    included && !excluded
                })
                .collect::<Vec<_>>();
            Box::new(filtered.into_iter())
        }

        fn walk_entries(&self, base: &Path, max_depth: Option<usize>) -> Vec<SourceEntry> {
            let mut entries: Vec<SourceEntry> = self
                .dirs
                .iter()
                .filter(|dir| dir.starts_with(base))
                .map(|dir| SourceEntry {
                    path: dir.clone(),
                    is_dir: true,
                })
                .collect();
            entries.extend(
                self.files
                    .lock()
                    .expect("files lock")
                    .iter()
                    .map(|(path, _)| SourceEntry {
                        path: path.clone(),
                        is_dir: false,
                    }),
            );
            if let Some(max_depth) = max_depth {
                let base_depth = base.components().count();
                entries.retain(|entry| {
                    entry.path.components().count() <= base_depth.saturating_add(max_depth)
                });
            }
            let mut sorted = entries;
            sorted.sort_by(|a, b| a.path.cmp(&b.path));
            sorted
        }

        fn read_capped(&self, path: &Path, _cap: u64) -> std::io::Result<Vec<u8>> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.files
                .lock()
                .expect("files lock")
                .iter()
                .find(|(candidate, _)| candidate == path)
                .map(|(_, file)| file.content.clone().into_bytes())
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "fake miss"))
        }

        fn likely_binary(&self, path: &Path) -> bool {
            self.files
                .lock()
                .expect("files lock")
                .iter()
                .find(|(candidate, _)| candidate == path)
                .is_some_and(|(_, file)| file.binary)
        }

        fn file_too_large(&self, path: &Path) -> bool {
            self.files
                .lock()
                .expect("files lock")
                .iter()
                .find(|(candidate, _)| candidate == path)
                .is_some_and(|(_, file)| file.too_large)
        }
    }

    /// An `Arc`-shared source keeps the `ContentSource::size` probe.
    ///
    /// The delegation must forward `size` too: `ReadTool`'s
    /// refuse-before-read guard consults `size`, and a wrapper
    /// that dropped it would turn every shared-source read into an
    /// unbounded whole-file load. Proven over the filesystem source
    /// with a real file of known length, and over a missing path
    /// (the honest `None`, not a fabricated zero).
    #[tokio::test]
    async fn arc_shared_sources_keep_the_size_probe() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let file = tmp.path().join("sized.txt");
        std::fs::write(&file, "12345").expect("write");
        let shared = std::sync::Arc::new(FsSearchSource);
        let reported = ContentSource::size(&shared, &file.to_string_lossy()).await;
        assert_eq!(reported, Some(5), "the wrapper must forward the probe");
        let missing =
            ContentSource::size(&shared, &tmp.path().join("absent.txt").to_string_lossy()).await;
        assert_eq!(missing, None, "a missing file reports unknown size");
    }

    #[test]
    fn fs_search_source_serves_the_seam() {
        fn takes_source<S: SearchSource>(_source: &S) {}
        takes_source(&FsSearchSource);
    }
}

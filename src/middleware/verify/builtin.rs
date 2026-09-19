//! The shipped call-level verifier.
//!
//! [`CommandVerifier`] judges a write-class tool call by dry-run
//! validity: a static, quote-aware parse of the command with a deny
//! check — nothing is ever executed — plus parent-existence and
//! writability checks for edit/write-shaped inputs. It implements the
//! call-level [`verify_call`](super::Verifier::verify_call) seam: the
//! input as the model sent it is exactly what it judges.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use super::super::{is_wrapped_shell, lexical_path, shell_words, string_field};
use super::{Verifier, VerifyResult};
use crate::tool::ToolContext;

/// The deny patterns every [`CommandVerifier`] starts with.
///
/// The recursive-delete entries end in `/` or `*` and therefore match
/// only at a token boundary — `rm -rf /` does not fire on
/// `rm -rf /home/user/tmp` — while the fork-bomb entry matches its
/// literal compact form. The common split-flag spellings of the root
/// delete are listed beside the compact ones, and the boundary check
/// treats shell separators (`;`, `&`, `|`, `(`, `)`, `#`) as token
/// edges, so `rm -rf /; echo done` denies too. Extend the set with
/// [`with_deny_globs`](CommandVerifier::with_deny_globs) for anything
/// more.
const DEFAULT_DENY_GLOBS: &[&str] = &[
    "rm -rf /",
    "rm -fr /",
    "rm -r -f /",
    "rm -f -r /",
    "rm --recursive --force /",
    "rm --force --recursive /",
    "rm -rf /*",
    "rm -fr /*",
    "rm -r -f /*",
    ":(){:|:&};:",
];

/// The shell interpreters a wrapped command may run under by default.
///
/// A wrapped invocation naming anything else fails verification with a
/// diagnostic listing these names; bare commands are not
/// shell-checked at all.
const DEFAULT_ALLOWED_SHELLS: &[&str] = &["sh", "bash"];

/// The input field names probed, in order, for an edit/write target
/// path.
///
/// The first present string field wins, so a tool using an unusual
/// spelling still gets its target checked as long as one of these
/// names carries it.
const PATH_FIELDS: &[&str] = &["path", "file_path", "filename", "filepath"];

/// A [`Verifier`] that judges write-class calls without executing
/// anything.
///
/// Two call shapes are judged; everything else passes unconditionally
/// with an empty diagnostic:
///
/// - **Shell commands** — an input carrying a string `command` (or
///   `cmd`) field. The command is tokenized with the shared
///   quote-aware parser; an unbalanced quote is a soft failure whose
///   diagnostic names the problem (dispatch is never blocked — the
///   verdict rides the output). A wrapped invocation (`sh -c "…"`)
///   must name an allowed shell, and the deny check runs against the
///   *inner* command. The command is whitespace-normalized before the
///   deny patterns match at token boundaries, so `rm -rf /` denies the
///   root delete — with any internal spacing — but not
///   `rm -rf /home/user/tmp`.
/// - **Edit/write inputs** — an input carrying one of the `path`-style
///   string fields. The target's parent directory must exist and not
///   be read-only, resolved against the tool context's working
///   directory; the filesystem root itself is rejected as a directory.
///
/// Diagnostics are one-liners written for a small-model reader: they
/// name what failed and what to do about it. Verification is static by
/// design — execution-based verification (sandboxes) is host territory.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use loopctl::middleware::{CommandVerifier, VerifyMiddleware};
///
/// let verify = VerifyMiddleware::new(
///     Arc::new(CommandVerifier::new()),
///     vec!["Write".into(), "Edit".into(), "Bash".into()],
/// );
/// ```
#[derive(Debug, Clone)]
pub struct CommandVerifier {
    /// The shell interpreters a wrapped command may run under.
    ///
    /// A wrapped invocation naming anything else fails verification
    /// with a diagnostic naming the allowed set.
    allowed_shells: Vec<String>,

    /// Deny patterns matched against the whitespace-normalized command
    /// at token boundaries.
    ///
    /// Replacement semantics: constructing with
    /// [`with_deny_globs`](Self::with_deny_globs) swaps the whole set
    /// rather than appending.
    deny_globs: Vec<String>,
}

impl CommandVerifier {
    /// Construct with the default shell allowlist and deny set.
    ///
    /// Shells: `sh`, `bash`. Denies: the recursive-root-delete forms
    /// and the compact fork bomb (see the deny-list constant in this
    /// module's source for the exact entries).
    #[must_use]
    pub fn new() -> Self {
        Self {
            allowed_shells: DEFAULT_ALLOWED_SHELLS
                .iter()
                .map(|shell| (*shell).to_string())
                .collect(),
            deny_globs: DEFAULT_DENY_GLOBS
                .iter()
                .map(|pattern| (*pattern).to_string())
                .collect(),
        }
    }

    /// Replace the shell allowlist.
    ///
    /// Wrapped commands (`sh -c`, `bash -c`) must name one of these
    /// interpreters; bare commands are not shell-checked at all. The
    /// list replaces the default, so an extended setup must include
    /// `sh` and `bash` itself if it still wants them.
    #[must_use]
    pub fn with_allowed_shells(mut self, shells: Vec<String>) -> Self {
        self.allowed_shells = shells;
        self
    }

    /// Replace the deny set.
    ///
    /// Patterns are matched against the whitespace-normalized command
    /// at token boundaries; a pattern ending in `/` or `*` therefore
    /// anchors to whole path tokens. The list replaces the defaults —
    /// a custom set that wants the root-delete rules must list them
    /// again.
    #[must_use]
    pub fn with_deny_globs(mut self, patterns: Vec<String>) -> Self {
        self.deny_globs = patterns;
        self
    }

    /// Judge one shell command: parse, wrapped-shell allowlist, deny
    /// match against the payload the shell would run.
    ///
    /// The wrapped case checks the interpreter against the allowlist
    /// and judges the inner payload; the bare case judges the command
    /// as given.
    fn verify_command(&self, command: &str) -> VerifyResult {
        let tokens = match shell_words(command) {
            Ok(tokens) => tokens,
            Err(reason) => {
                tracing::debug!(class = "unparseable", "command verification failed");
                return fail(&format!("unparseable command: {reason}"));
            }
        };
        if tokens.is_empty() {
            return pass();
        }
        if is_wrapped_shell(&tokens) {
            let shell = tokens.first().map(String::as_str).unwrap_or_default();
            if !self.allowed_shells.iter().any(|allowed| allowed == shell) {
                tracing::debug!(class = "shell-not-allowed", "command verification failed");
                return fail(&format!(
                    "shell `{shell}` is not in the allowed set ({})",
                    self.allowed_shells.join(", ")
                ));
            }
            let inner = tokens
                .get(2..)
                .map_or(String::new(), |inner| inner.join(" "));
            return self.check_deny(&inner);
        }
        self.check_deny(&tokens.join(" "))
    }

    /// Match the deny patterns against one command.
    ///
    /// Normalization to single-space token separation happens inside
    /// the matcher, on the command as much as on the pattern, so
    /// padding inside a wrapped payload cannot hide a rule.
    fn check_deny(&self, command: &str) -> VerifyResult {
        for pattern in &self.deny_globs {
            if matches_at_token_boundary(command, pattern) {
                tracing::debug!(class = "deny-glob", pattern, "command verification failed");
                return fail(&format!(
                    "denied: command matches the deny rule `{pattern}`"
                ));
            }
        }
        pass()
    }

    /// Judge one edit/write input: the target's parent exists and is
    /// writable, resolved against the working directory.
    ///
    /// The filesystem root is rejected outright — it is a directory,
    /// never a writable file target.
    fn verify_write_path(ctx: &ToolContext, path: &str) -> VerifyResult {
        let resolved = lexical_path(&ctx.cwd, path);
        let Some(parent) = std::path::Path::new(&resolved)
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        else {
            tracing::debug!(class = "root-target", "write verification failed");
            return fail("path not writable: the target is the filesystem root, a directory");
        };
        let parent_display = parent.display().to_string();
        if !parent.exists() {
            tracing::debug!(class = "parent-missing", "write verification failed");
            return fail(&format!(
                "path not writable: parent directory {parent_display} does not exist"
            ));
        }
        if parent_is_readonly(parent) {
            tracing::debug!(class = "parent-readonly", "write verification failed");
            return fail(&format!(
                "path not writable: parent directory {parent_display} is read-only"
            ));
        }
        pass()
    }
}

impl Default for CommandVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Verifier for CommandVerifier {
    fn verify<'a>(
        &'a self,
        ctx: &'a ToolContext,
        tool_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = VerifyResult> + Send + 'a>> {
        self.verify_call(ctx, tool_name, &Value::Null)
    }

    fn verify_call<'a>(
        &'a self,
        ctx: &'a ToolContext,
        _tool_name: &'a str,
        input: &'a Value,
    ) -> Pin<Box<dyn Future<Output = VerifyResult> + Send + 'a>> {
        let result = if let Some(command) = string_field(input, &["command", "cmd"]) {
            self.verify_command(command)
        } else if let Some(path) = string_field(input, PATH_FIELDS) {
            Self::verify_write_path(ctx, path)
        } else {
            pass()
        };
        Box::pin(async move { result })
    }
}

/// A passing result with nothing to say.
///
/// Passing stays silent by design — the output block the middleware
/// appends carries only failures' diagnostics, so a clean write leaves
/// the model's context untouched.
fn pass() -> VerifyResult {
    VerifyResult {
        passed: true,
        diagnostics: String::new(),
    }
}

/// A failing result with a one-line diagnostic.
///
/// Every failure path funnels through here so the diagnostic shape
/// stays uniform: name the class, name the fix, one line.
fn fail(diagnostic: &str) -> VerifyResult {
    VerifyResult {
        passed: false,
        diagnostics: diagnostic.to_string(),
    }
}

/// Whether `pattern` occurs in `command` at token boundaries on both
/// sides, after whitespace-normalizing each.
///
/// A boundary is the start or end of the string or a single space in
/// the normalized forms — so a pattern cannot fire from inside a
/// larger token or path, and no amount of internal padding can hide a
/// match. Normalizing the command closes the wrapped-payload gap: the
/// payload arrives as one quoted token whose original spacing is
/// preserved by the tokenizer.
fn matches_at_token_boundary(command: &str, pattern: &str) -> bool {
    let normalized = command.split_whitespace().collect::<Vec<_>>().join(" ");
    let pattern = pattern.split_whitespace().collect::<Vec<_>>().join(" ");
    if pattern.is_empty() {
        return false;
    }
    let bytes = normalized.as_bytes();
    normalized.match_indices(&pattern).any(|(start, matched)| {
        let Some(end) = start.checked_add(matched.len()) else {
            return false;
        };
        boundary_before(bytes, start) && end <= bytes.len() && boundary_after(bytes, end)
    })
}

/// Whether `index` sits at the start of the string or after a shell
/// separator.
///
/// Byte-level on purpose: a UTF-8 continuation byte can never equal
/// one of the ASCII separators, so the check cannot split a codepoint
/// into a false boundary.
fn boundary_before(bytes: &[u8], index: usize) -> bool {
    index == 0
        || index
            .checked_sub(1)
            .and_then(|previous| bytes.get(previous))
            .is_some_and(|byte| is_separator_byte(*byte))
}

/// Whether `index` sits at the end of the string or before a shell
/// separator.
///
/// The mirror of [`boundary_before`] for the closing edge of a match.
fn boundary_after(bytes: &[u8], index: usize) -> bool {
    index == bytes.len()
        || bytes
            .get(index)
            .is_some_and(|byte| is_separator_byte(*byte))
}

/// Whether a byte separates shell words.
///
/// Space separates; `;`, `&`, and `|` terminate or combine commands;
/// the parentheses open and close subshells; `#` begins a comment.
/// Treating these as token edges is what makes a destructive command
/// spliced into a sequence (`rm -rf /; echo done`) still match its
/// deny rule.
fn is_separator_byte(byte: u8) -> bool {
    matches!(byte, b' ' | b';' | b'&' | b'|' | b'(' | b')' | b'#')
}

/// Whether a directory is read-only for this process.
///
/// On Unix this is the owner-write bit of the metadata; on other
/// platforms the check degrades to the portable readonly flag. An
/// unreadable directory reads as writable — the write itself will
/// surface the real error, and verification stays quiet rather than
/// guessing.
fn parent_is_readonly(parent: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(parent).is_ok_and(|meta| meta.permissions().mode() & 0o200 == 0)
    }
    #[cfg(not(unix))]
    {
        std::fs::metadata(parent)
            .map(|meta| meta.permissions().readonly())
            .unwrap_or(false)
    }
}

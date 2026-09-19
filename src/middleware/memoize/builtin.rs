//! The shipped path extractor.
//!
//! [`WritePathExtractor`] feeds memoize invalidation: it names the
//! filesystem paths a call touches, so a write to path P evicts cached
//! results whose extraction also named P. Extraction is symmetric —
//! the same impl runs for cached (read-class) and write-class calls,
//! which is what makes the eviction match work.

use serde_json::Value;

use super::PathExtractor;
use crate::middleware::{is_wrapped_shell, lexical_path, shell_words, string_field};

/// The input field names probed, in order, for an edit/write target
/// path.
///
/// The first present string field wins, so a tool using an unusual
/// spelling still contributes its target as long as one of these names
/// carries it.
const PATH_FIELDS: &[&str] = &["path", "file_path", "filename", "filepath"];

/// The redirection operator prefixes recognized in a shell command,
/// longest first so `>>` wins over `>` when both are prefixes.
///
/// The `&`- and fd-numbered forms capture bash's compact redirections
/// (`2>`, `1>`, and their append variants); descriptor *duplications*
/// (`2>&1`, `>&2`) are excluded by the operand check, not by this
/// list.
const REDIRECT_OPERATORS: &[&str] = &[">>", "2>>", "2>", "1>>", "1>", "&>", ">&", ">"];

/// A [`PathExtractor`] for tools that touch the filesystem.
///
/// Two call shapes yield paths; everything else yields none:
///
/// - **Edit/write/read-shaped inputs** — an input carrying one of the
///   `path`-style string fields yields that one path, resolved against
///   the working directory the call ran under.
/// - **Shell commands** — an input carrying a string `command` (or
///   `cmd`) field yields the target of every output redirection: the
///   `>`, `>>`, `1>`, `1>>`, `2>`, `2>>`, `&>`, and `>&` operators,
///   whether the
///   target rides the same token (`>out.txt`, `2>err.log`) or the
///   next one (`> out.txt`), and inside one level of `<shell> -c`
///   wrapping. A duplication operand (`2>&1`, `>&2` — starting with
///   `&` or all digits) is not a file and yields nothing, and a
///   command without redirections yields no paths: a command that
///   only reads still depends on files, but which files is not
///   statically knowable, and over-returning guesses would flush the
///   cache on every shell call.
///
/// Paths are normalized lexically against the cwd, so a relative read
/// under `/repo` and a write to the same file under a different cwd
/// spelling land on one string. The context-free [`paths`](PathExtractor::paths)
/// seam normalizes against the process root instead — through the
/// middleware, the intended path, the call's real cwd is always used.
pub struct WritePathExtractor;

impl PathExtractor for WritePathExtractor {
    fn paths(&self, tool_name: &str, input: &Value) -> Vec<String> {
        self.paths_with_cwd(tool_name, input, ".")
    }

    fn paths_with_cwd(&self, _tool_name: &str, input: &Value, cwd: &str) -> Vec<String> {
        if let Some(command) = string_field(input, &["command", "cmd"]) {
            return redirection_targets(command)
                .iter()
                .map(|target| lexical_path(cwd, target))
                .collect();
        }
        match string_field(input, PATH_FIELDS) {
            Some(path) => vec![lexical_path(cwd, path)],
            None => Vec::new(),
        }
    }
}

/// The redirection targets of a shell command, verbatim.
///
/// One level of `<shell> -c` wrapping is unwrapped first, so a wrapped
/// payload's redirections are seen; an unparseable command (unbalanced
/// quote) yields no targets — the verifier is the component that fails
/// loud on unparseable input, the extractor only reports what it can
/// trust.
fn redirection_targets(command: &str) -> Vec<String> {
    let Ok(tokens) = shell_words(command) else {
        return Vec::new();
    };
    let tokens = unwrap_shell_payload(tokens);
    let mut targets = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if let Some(target) = inline_redirect_target(token) {
            if is_file_target(&target) {
                targets.push(target);
            }
        } else if REDIRECT_OPERATORS.contains(&token.as_str())
            && let Some(next) = tokens.get(index.saturating_add(1))
            && is_file_target(next)
        {
            targets.push(next.clone());
        }
    }
    targets
}

/// Unwrap one level of `<shell> -c` wrapping, returning the payload's
/// own tokens.
///
/// The payload of a wrapper is a single quoted token; re-splitting it
/// recovers the inner command's words. A wrapper without a payload, or
/// a payload that does not re-parse, yields no tokens.
fn unwrap_shell_payload(tokens: Vec<String>) -> Vec<String> {
    if !is_wrapped_shell(&tokens) {
        return tokens;
    }
    tokens
        .get(2)
        .and_then(|payload| shell_words(payload).ok())
        .unwrap_or_default()
}

/// The target riding the same token as its redirection operator, if
/// the token opens with one.
///
/// `>out.txt` and `2>err.log` carry their targets inline; longer
/// operators are tried first so `>>app.log` does not split as `>`
/// followed by `>app.log`.
fn inline_redirect_target(token: &str) -> Option<String> {
    if REDIRECT_OPERATORS.contains(&token) {
        return None;
    }
    REDIRECT_OPERATORS
        .iter()
        .find_map(|operator| token.strip_prefix(operator).filter(|rest| !rest.is_empty()))
        .map(str::to_string)
}

/// Whether a redirection operand names a file rather than a descriptor
/// duplication.
///
/// Duplications point at another descriptor: the operand starts with
/// `&` (`2>&1`) or is a bare number (`>&2`). Anything else — including
/// a number followed by other characters — is treated as a filename.
fn is_file_target(operand: &str) -> bool {
    !(operand.starts_with('&') || operand.chars().all(|c| c.is_ascii_digit()))
}

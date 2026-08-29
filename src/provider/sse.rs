//! Shared SSE line-framing for streaming providers.
//!
//! Each provider that streams responses (OpenAI, Anthropic, Gemini) uses
//! the same line-oriented framing over an HTTP byte stream: buffer raw
//! bytes, split on newlines, decode each complete line as UTF-8, and guard
//! against unbounded buffer growth. The provider-specific event extraction
//! (`next_data` / `next_event`) lives in each provider file as an
//! `impl super::sse::SseReader` block.

use std::pin::Pin;

use futures::stream::{Stream, StreamExt};
use reqwest::Response;

use crate::api::error::ApiError;

/// Upper bound on the internal line buffer (`1 MiB`).
///
/// A single SSE event that exceeds this without a newline is treated as a
/// protocol error, preventing unbounded memory growth from a malformed or
/// hostile stream.
const SSE_MAX_BUFFER: usize = 1024 * 1024;

/// Minimal SSE line reader over an HTTP byte stream.
///
/// Buffers **raw bytes** from the response (not pre-decoded strings —
/// HTTP/TCP chunk boundaries are not UTF-8 aligned, so per-chunk decoding
/// corrupts multi-byte sequences split across chunks). Splits on `\n` in
/// the byte buffer, then decodes only complete lines.
pub(super) struct SseReader {
    /// Underlying HTTP byte stream yielding raw `Bytes`.
    ///
    /// Pinned because it is polled as an async stream.
    pub(super) bytes: Pin<Box<dyn Stream<Item = Result<bytes::Bytes, ApiError>> + Send>>,

    /// Byte accumulator for data that has not yet formed a complete line.
    ///
    /// Newly arrived chunks are appended here as raw bytes; complete
    /// `\n`-terminated lines are drained by [`take_line`](SseReader::take_line)
    /// and decoded as UTF-8.
    pub(super) buf: Vec<u8>,

    /// Whether an OpenAI-style `data: [DONE]` sentinel has been read.
    ///
    /// Set by the OpenAI data reader when it consumes the sentinel; read
    /// by the stream loop to tell a provider-signalled end apart from a
    /// bare EOF (a cut connection), so the emitter only completes a
    /// stream the provider actually terminated. Only OpenAI-wire
    /// providers have the sentinel, hence the gate.
    #[cfg(feature = "openai")]
    pub(super) done_marker_seen: bool,
}

impl SseReader {
    /// Wrap a streaming HTTP response into an SSE reader.
    ///
    /// The byte stream yields raw `Bytes`; decoding happens per-line in
    /// [`take_line`](Self::take_line), after splitting on `\n`, so
    /// multi-byte UTF-8 sequences split across chunks are reassembled
    /// correctly.
    pub(super) fn from_response(resp: Response) -> Self {
        let bytes = resp
            .bytes_stream()
            .map(|res| res.map_err(|e| ApiError::http(e.to_string())));
        Self {
            bytes: Box::pin(bytes),
            buf: Vec::new(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        }
    }

    /// Pop the first `\n`-terminated line from the byte buffer and decode it.
    ///
    /// Returns the trimmed line as a `String` if a newline is present,
    /// removing it (plus the newline) from the buffer. Returns `None` if
    /// the buffer does not yet contain a complete line.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if the complete line contains invalid UTF-8.
    /// Split multi-byte sequences are reassembled by this point (buffering is
    /// byte-oriented), so an error indicates genuinely malformed data from
    /// the provider — not a chunk-boundary artifact.
    pub(super) fn take_line(&mut self) -> Result<Option<String>, ApiError> {
        let Some(pos) = self.buf.iter().position(|&b| b == b'\n') else {
            return Ok(None);
        };
        let rest_start = pos.saturating_add(1);
        let line_bytes: Vec<u8> = self.buf.drain(..rest_start).collect();
        let line = String::from_utf8(line_bytes)
            .map_err(|e| ApiError::http(format!("SSE line is not valid UTF-8: {e}")))?;
        Ok(Some(line.trim().to_string()))
    }

    /// Whether the OpenAI-style `[DONE]` sentinel has been read.
    ///
    /// Pure read over [`done_marker_seen`](Self::done_marker_seen); see
    /// that field for why the distinction matters.
    #[cfg(feature = "openai")]
    pub(super) fn done_marker_seen(&self) -> bool {
        self.done_marker_seen
    }

    /// Record that the OpenAI-style `[DONE]` sentinel has been read.
    ///
    /// The OpenAI data reader calls this when it consumes the sentinel;
    /// the flag then serves both the sentinel's `Ok(None)` return and
    /// later [`done_marker_seen`](Self::done_marker_seen) polls.
    #[cfg(feature = "openai")]
    pub(super) fn mark_done_marker_seen(&mut self) {
        self.done_marker_seen = true;
    }

    /// Fetch the next chunk from the HTTP stream and append it to the buffer.
    ///
    /// Returns `Ok(Some(()))` when a chunk arrived, `Ok(None)` at
    /// end-of-stream.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError`] if the transport failed or the buffer
    /// exceeded [`SSE_MAX_BUFFER`].
    pub(super) async fn next_chunk(&mut self) -> Result<Option<()>, ApiError> {
        match self.bytes.next().await {
            Some(Ok(chunk)) => {
                self.buf.extend_from_slice(&chunk);
                if self.buf.len() > SSE_MAX_BUFFER {
                    return Err(ApiError::http(format!(
                        "SSE buffer exceeded {SSE_MAX_BUFFER} bytes"
                    )));
                }
                Ok(Some(()))
            }
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }
}

#[cfg(any(feature = "openai", feature = "anthropic", feature = "gemini"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_line_invalid_utf8_returns_error() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: vec![0xFF, 0xFE, 0xFD, b'\n'],
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        };
        let result = reader.take_line();
        assert!(result.is_err(), "invalid UTF-8 must surface as an error");
    }

    #[test]
    fn take_line_valid_utf8_returns_ok() {
        let mut reader = SseReader {
            bytes: Box::pin(futures::stream::empty()),
            buf: b"data: hello\n".to_vec(),
            #[cfg(feature = "openai")]
            done_marker_seen: false,
        };
        let line = reader.take_line().unwrap().unwrap();
        assert_eq!(line, "data: hello");
    }
}

//! The wire transport: HTTP/1.1 over the daemon's unix socket.
//!
//! A deliberately minimal client for a fixed local dialect — one
//! socket, no TLS, no HTTP/2, no redirects — implementing the three
//! response framings a compliant peer may use (declared
//! `Content-Length`, chunked transfer coding, and connection-close
//! delimiting) with chunk decoding available incrementally, so both
//! one-shot verb calls and the long-lived SSE event stream read
//! through one reader. Every item here is crate-private: the typed
//! verb layer in [`crate::client`] is the only consumer, so no
//! protocol code exists outside this crate.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

use crate::ClientError;

/// The most bytes a response's header block may occupy.
///
/// A fixed local daemon answers with short headers; a peer that sends
/// more than this is not speaking the dialect and is refused rather
/// than buffered indefinitely.
const MAX_HEAD_BYTES: usize = 64 * 1024;

/// The most bytes one body line (an SSE field line) may occupy.
///
/// Generous against the largest ledger payload an event line may
/// carry, bounded so a misbehaving peer cannot grow a buffer without
/// end.
const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;

/// The most bytes a one-shot response body may total.
///
/// The verb layer's JSON documents are small; the same family ceiling
/// the completion paths enforce. A declared, chunked, or streamed
/// body past this bound is a malformed response — a peer that will
/// not stop sending is refused, not buffered.
const MAX_BODY_BYTES: usize = 10 * 1024 * 1024;

/// The consumed-prefix size at which the reader compacts its buffer.
///
/// Long-lived streams (the event subscription) would otherwise hold
/// every byte ever received; draining the consumed prefix in
/// [`fill_wire`](ByteStream::fill_wire) keeps the resident allocation
/// proportional to the unread window, not the stream's lifetime.
const COMPACT_THRESHOLD_BYTES: usize = 64 * 1024;

/// One outbound request.
///
/// The verb layer builds these; [`open`] frames and writes them.
pub(super) struct HttpRequest {
    /// The HTTP method, upper-case.
    ///
    /// One of the dialect's fixed methods; the transport writes it
    /// verbatim into the request line.
    pub method: &'static str,

    /// The versioned path plus optional query, e.g.
    /// `/v1/loops/x/runs`.
    ///
    /// Segments arrive pre-encoded from the verb layer.
    pub path: String,

    /// The JSON body, if the verb carries one.
    ///
    /// Serialized once at write time; `None` sends an empty body.
    pub body: Option<serde_json::Value>,
}

/// One fully-buffered inbound response.
///
/// The status line's code plus the decoded body bytes — framing is
/// already resolved by [`send`].
pub(super) struct HttpResponse {
    /// The response's three-digit status code.
    ///
    /// Parsed from the status line; the verb layer maps it.
    pub status: u16,

    /// The decoded body bytes.
    ///
    /// Framing already resolved — chunk syntax never appears here.
    pub body: Vec<u8>,
}

/// The body framing a response head announced.
///
/// Resolved while parsing headers; decides how body bytes are
/// delivered — chunked bodies are decoded inline by the reader, so
/// consumers below this type never see chunk syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Framing {
    /// A declared exact byte count.
    ///
    /// The simplest framing; honored exactly and no further.
    ContentLength(usize),

    /// Chunked transfer coding.
    ///
    /// Decoded incrementally so streams and one-shots share one
    /// reader.
    Chunked,

    /// Everything until the peer closes the connection.
    ///
    /// The `Connection: close` shape this client requests; SSE
    /// bodies ride it for their whole lifetime.
    ToClose,
}

/// The chunked-coding decoder's position between chunks.
///
/// Transitions commit only after each step's bytes are fully
/// consumed — a pended, dropped, and recreated read resumes exactly
/// where the last completed step left off, never re-consuming or
/// skipping chunk syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkPhase {
    /// A size line is the next syntax on the wire.
    ///
    /// The body's opening state: the first size line has no
    /// preceding separator.
    NeedSize,

    /// The previous chunk's CRLF separator is the next syntax.
    ///
    /// Read and checked before the following size line.
    NeedSeparator,

    /// Inside a chunk with this many body bytes remaining.
    ///
    /// Delivering decrements; reaching zero schedules the separator.
    InChunk(usize),

    /// The zero-size terminator arrived; trailer lines remain.
    ///
    /// Consumed line by line until the blank one.
    Trailer,

    /// The chunked body is complete.
    ///
    /// Terminal: every later decoded read reports the body's end.
    Done,
}

/// The incremental reader over one open response.
///
/// Owns the socket so long-running consumers (the SSE stream) pull
/// lines and byte runs as frames arrive, while one-shot [`send`]
/// calls drain the whole body through the same layer. The head is
/// read raw; once [`read_head`](Self::read_head) has armed the
/// framing, [`read_decoded`](Self::read_decoded) is the single
/// primitive every body read builds on.
pub(super) struct ByteStream {
    /// The connected socket, read and written directly.
    ///
    /// One exchange per [`ByteStream`] — the connection-per-request
    /// shape leaves no pooling state to manage.
    stream: UnixStream,

    /// The wire bytes received so far, consumed and unread together.
    ///
    /// Grows only by [`fill_wire`](Self::fill_wire), which also
    /// compacts it (the consumed prefix is drained past a threshold),
    /// so the resident allocation tracks the unread window, not the
    /// exchange's lifetime.
    buf: Vec<u8>,

    /// The read cursor into `buf` — the consumed prefix's length.
    ///
    /// Only [`take_buffered`](Self::take_buffered) advances it, by
    /// exactly the bytes it hands out.
    pos: usize,

    /// The body framing the response head announced, once armed.
    ///
    /// `None` until [`read_head`](Self::read_head) parses; decides
    /// how every later body read decodes.
    framing: Option<Framing>,

    /// Bytes still owed under a declared `Content-Length`.
    ///
    /// Meaningful only under that framing; hitting zero is the
    /// body's exact end.
    length_remaining: usize,

    /// The chunked-coding decoder's position, under that framing.
    ///
    /// Meaningful only under chunked coding. Every transition is
    /// committed only after the step's bytes are fully consumed, so
    /// a read that pends and is dropped mid-step resumes at the same
    /// state — the cancel-safety the event stream's polling demands.
    phase: ChunkPhase,

    /// The undelivered tail of a previous decoded pull.
    ///
    /// [`read_line`](Self::read_line) stashes bytes past the last
    /// newline here, so lines never straddle a framing boundary.
    line_pending: Vec<u8>,
}

impl ByteStream {
    /// Wrap a connected socket with an empty buffer.
    ///
    /// Fresh state for one request/response exchange.
    pub(super) fn new(stream: UnixStream) -> Self {
        Self {
            stream,
            buf: Vec::new(),
            pos: 0,
            framing: None,
            length_remaining: 0,
            phase: ChunkPhase::NeedSize,
            line_pending: Vec::new(),
        }
    }

    fn buffered(&self) -> &[u8] {
        self.buf.get(self.pos..).unwrap_or(&[])
    }

    fn take_buffered(&mut self, wanted: usize) -> Option<Vec<u8>> {
        let available = self.buf.len().saturating_sub(self.pos);
        if available < wanted {
            return None;
        }
        let taken = self
            .buf
            .get(self.pos..self.pos.saturating_add(wanted))
            .map(<[u8]>::to_vec);
        self.pos = self.pos.saturating_add(wanted);
        taken
    }

    /// # Errors
    ///
    /// A socket read failure maps onto the transport surface.
    async fn fill_wire(&mut self) -> Result<usize, ClientError> {
        if self.pos == self.buf.len() {
            self.buf.clear();
            self.pos = 0;
        } else if self.pos > COMPACT_THRESHOLD_BYTES {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
        let mut chunk = [0u8; 8 * 1024];
        let read = self
            .stream
            .read(&mut chunk)
            .await
            .map_err(|error| io_error(&error))?;
        if read > 0 {
            self.buf.extend_from_slice(chunk.get(..read).unwrap_or(&[]));
        }
        Ok(read)
    }

    /// Read exactly `wanted` raw bytes from the wire.
    ///
    /// Raw means pre-decoding: used for the head, chunk syntax, and
    /// close-delimited bodies alike.
    /// # Errors
    ///
    /// A stream end before the count, or a transport failure.
    async fn take_raw(&mut self, wanted: usize) -> Result<Vec<u8>, ClientError> {
        loop {
            if let Some(bytes) = self.take_buffered(wanted) {
                return Ok(bytes);
            }
            if self.fill_wire().await? == 0 {
                return Err(ClientError::MalformedResponse(
                    "the stream ended before a declared length".into(),
                ));
            }
        }
    }

    /// Read one raw `\n`-terminated line, `\r` stripped, bounded.
    ///
    /// Returns `None` on a clean end of stream.
    /// # Errors
    ///
    /// A line over the size bound, or a transport failure.
    async fn take_raw_line(&mut self) -> Result<Option<String>, ClientError> {
        let line = loop {
            let window = self.buffered();
            if let Some(offset) = window.iter().position(|byte| *byte == b'\n') {
                break self
                    .take_buffered(offset.saturating_add(1))
                    .unwrap_or_default();
            }
            if window.len() > MAX_LINE_BYTES {
                return Err(ClientError::MalformedResponse(
                    "a response line exceeded the size bound".into(),
                ));
            }
            if self.fill_wire().await? == 0 {
                return Ok(None);
            }
        };
        let mut line = line;
        if line.last() == Some(&b'\n') {
            line.pop();
        }
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        Ok(Some(String::from_utf8_lossy(&line).into_owned()))
    }

    /// Read and parse the response head, arming the body framing.
    ///
    /// Returns the status code; body reads are meaningful only after
    /// this returns.
    /// # Errors
    ///
    /// A truncated, oversized, or unparsable head; a transport
    /// failure beneath it.
    pub(super) async fn read_head(&mut self) -> Result<u16, ClientError> {
        loop {
            let window = self.buffered();
            if let Some(offset) = find_double_crlf(window) {
                let Some(head_bytes) = self.take_buffered(offset.saturating_add(4)) else {
                    return Err(ClientError::MalformedResponse(
                        "the response head could not be read".into(),
                    ));
                };
                let head = String::from_utf8_lossy(&head_bytes).into_owned();
                let (status, framing) = parse_head(&head)?;
                self.length_remaining = match framing {
                    Framing::ContentLength(wanted) => wanted,
                    _ => 0,
                };
                self.framing = Some(framing);
                return Ok(status);
            }
            if self.buffered().len() > MAX_HEAD_BYTES {
                return Err(ClientError::MalformedResponse(
                    "the response head exceeded the size bound".into(),
                ));
            }
            if self.fill_wire().await? == 0 {
                return Err(ClientError::MalformedResponse(
                    "the stream ended before the response head completed".into(),
                ));
            }
        }
    }

    /// Deliver up to `max` decoded body bytes.
    ///
    /// The single body-read primitive: resolves the armed framing —
    /// bytes inside a chunk pass through with chunk syntax consumed
    /// invisibly, a declared length is honored exactly, a
    /// close-delimited body ends at the peer's close — and returns an
    /// empty vec only at the body's genuine end.
    /// # Errors
    ///
    /// A stream end inside a declared length or chunk, malformed
    /// chunk syntax, or a transport failure.
    pub(super) async fn read_decoded(&mut self, max: usize) -> Result<Vec<u8>, ClientError> {
        match self.framing {
            Some(Framing::ContentLength(_)) => {
                if self.length_remaining == 0 {
                    return Ok(Vec::new());
                }
                if self.buffered().is_empty() && self.fill_wire().await? == 0 {
                    return Err(ClientError::MalformedResponse(
                        "the stream ended before the declared length".into(),
                    ));
                }
                let take = max.min(self.length_remaining).min(self.buffered().len());
                let slice = self.take_buffered(take).unwrap_or_default();
                self.length_remaining = self.length_remaining.saturating_sub(slice.len());
                Ok(slice)
            }
            Some(Framing::ToClose) | None => {
                if self.buffered().is_empty() && self.fill_wire().await? == 0 {
                    return Ok(Vec::new());
                }
                let take = max.min(self.buffered().len());
                Ok(self.take_buffered(take).unwrap_or_default())
            }
            Some(Framing::Chunked) => {
                while !matches!(self.phase, ChunkPhase::InChunk(_) | ChunkPhase::Done) {
                    self.advance_chunk().await?;
                }
                if self.phase == ChunkPhase::Done {
                    return Ok(Vec::new());
                }
                let ChunkPhase::InChunk(remaining) = self.phase else {
                    return Ok(Vec::new());
                };
                if self.buffered().is_empty() && self.fill_wire().await? == 0 {
                    return Err(ClientError::MalformedResponse(
                        "the stream ended inside a chunk".into(),
                    ));
                }
                let take = max.min(remaining).min(self.buffered().len());
                let slice = self.take_buffered(take).unwrap_or_default();
                let served = slice.len();
                if served >= remaining {
                    self.phase = ChunkPhase::NeedSeparator;
                } else {
                    self.phase = ChunkPhase::InChunk(remaining.saturating_sub(served));
                }
                Ok(slice)
            }
        }
    }

    /// Advance the chunk decoder by one completed step.
    ///
    /// Exactly one wire step runs per call, and the phase commits
    /// only after that step's bytes are fully consumed — a read that
    /// pends and is dropped mid-step loses nothing, because no
    /// transition preceded it.
    /// # Errors
    ///
    /// Malformed chunk syntax (a non-hex size, a bad separator) or a
    /// transport failure.
    async fn advance_chunk(&mut self) -> Result<(), ClientError> {
        match self.phase {
            ChunkPhase::NeedSeparator => {
                let separator = self.take_raw(2).await?;
                if separator.as_slice() != b"\r\n" {
                    return Err(ClientError::MalformedResponse(
                        "a chunk was not CRLF-terminated".into(),
                    ));
                }
                self.phase = ChunkPhase::NeedSize;
                Ok(())
            }
            ChunkPhase::NeedSize => {
                let size_line = self.take_raw_line().await?.ok_or_else(|| {
                    ClientError::MalformedResponse("chunked body ended early".into())
                })?;
                let size_text = size_line.split(';').next().unwrap_or_default().trim();
                let size = usize::from_str_radix(size_text, 16).map_err(|_| {
                    ClientError::MalformedResponse("a chunk size was not hex".into())
                })?;
                if size == 0 {
                    self.phase = ChunkPhase::Trailer;
                } else {
                    self.phase = ChunkPhase::InChunk(size);
                }
                Ok(())
            }
            ChunkPhase::Trailer => {
                match self.take_raw_line().await? {
                    Some(line) if line.is_empty() => self.phase = ChunkPhase::Done,
                    Some(_) => {}
                    None => self.phase = ChunkPhase::Done,
                }
                Ok(())
            }
            ChunkPhase::InChunk(_) | ChunkPhase::Done => Ok(()),
        }
    }

    /// One decoded `\n`-terminated body line, `\r` stripped.
    ///
    /// Returns `None` on a clean end of body. `event-stream` frames
    /// never straddle framings incorrectly: the pending tail of a
    /// previous pull is consumed before fresh bytes are read.
    /// # Errors
    ///
    /// A line over the size bound, a body end mid-line, or the
    /// decoded-read failure set.
    pub(super) async fn read_line(&mut self) -> Result<Option<String>, ClientError> {
        loop {
            if let Some(offset) = self.line_pending.iter().position(|byte| *byte == b'\n') {
                let tail = self.line_pending.split_off(offset.saturating_add(1));
                let mut line = std::mem::take(&mut self.line_pending);
                self.line_pending = tail;
                if line.last() == Some(&b'\n') {
                    line.pop();
                }
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
            }
            if self.line_pending.len() > MAX_LINE_BYTES {
                return Err(ClientError::MalformedResponse(
                    "a body line exceeded the size bound".into(),
                ));
            }
            let more = self.read_decoded(MAX_LINE_BYTES).await?;
            if more.is_empty() {
                if self.line_pending.is_empty() {
                    return Ok(None);
                }
                return Err(ClientError::MalformedResponse(
                    "the body ended mid-line".into(),
                ));
            }
            self.line_pending.extend_from_slice(&more);
        }
    }

    /// Drain the whole body, honoring the armed framing.
    /// # Errors
    ///
    /// The decoded-read failure set for the armed framing.
    pub(super) async fn read_to_body_end(&mut self) -> Result<Vec<u8>, ClientError> {
        let mut collected = Vec::new();
        loop {
            let more = self.read_decoded(MAX_LINE_BYTES).await?;
            if more.is_empty() {
                return Ok(collected);
            }
            if collected.len().saturating_add(more.len()) > MAX_BODY_BYTES {
                return Err(ClientError::MalformedResponse(
                    "the response body exceeded the size bound".into(),
                ));
            }
            collected.extend_from_slice(&more);
        }
    }

    /// # Errors
    ///
    /// A socket write failure, or a body that cannot serialize.
    async fn write_request(&mut self, request: &HttpRequest) -> Result<(), ClientError> {
        let body = request
            .body
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| ClientError::Transport(error.to_string()))?
            .unwrap_or_default();
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: loopctl\r\nAccept: application/json, \
             text/event-stream\r\nContent-Type: application/json\r\nContent-Length: {len}\r\n\
             Connection: close\r\n\r\n",
            method = request.method,
            path = request.path,
            len = body.len()
        );
        self.stream
            .write_all(head.as_bytes())
            .await
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        self.stream
            .write_all(body.as_bytes())
            .await
            .map_err(|error| ClientError::Transport(error.to_string()))
    }
}

/// Map an IO failure onto the client's error surface.
///
/// A missing socket file or a refused connection is the typed absence
/// the degrade and auto-spawn logic key on; every other IO failure
/// carries its diagnosis as transport.
fn io_error(error: &std::io::Error) -> ClientError {
    if matches!(
        error.kind(),
        std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
    ) {
        return ClientError::SocketAbsent;
    }
    ClientError::Transport(error.to_string())
}

/// Open a connection, write one request, and hand back the armed reader.
///
/// The request is framed as HTTP/1.1 with a JSON `Content-Length`
/// body and `Connection: close` — one request per connection, the
/// curl-per-command shape. The SSE subscription uses this entry to
/// keep its connection alive; one-shot verbs use [`send`].
/// # Errors
///
/// [`ClientError::SocketAbsent`] when nothing listens; transport
/// failures; a malformed response head.
pub(super) async fn open(
    socket: &Path,
    request: &HttpRequest,
) -> Result<(u16, ByteStream), ClientError> {
    let stream = UnixStream::connect(socket)
        .await
        .map_err(|error| io_error(&error))?;
    let mut reader = ByteStream::new(stream);
    reader.write_request(request).await?;
    let status = reader.read_head().await?;
    Ok((status, reader))
}

/// Send one request and collect its full response.
///
/// The one-shot verb path: opens, writes, resolves the body framing,
/// and buffers the whole body.
pub(super) fn send<'a>(
    socket: &'a Path,
    request: HttpRequest,
) -> Pin<Box<dyn Future<Output = Result<HttpResponse, ClientError>> + Send + 'a>> {
    Box::pin(async move {
        let (status, mut reader) = open(socket, &request).await?;
        let body = reader.read_to_body_end().await?;
        Ok(HttpResponse { status, body })
    })
}

/// Find `\r\n\r\n` in `window`, returning its offset.
///
/// The head terminator scan; `None` means the head is incomplete
/// so far.
fn find_double_crlf(window: &[u8]) -> Option<usize> {
    window.windows(4).position(|quad| quad == b"\r\n\r\n")
}

/// Parse a response head into its status and framing.
///
/// Chunked wins over a declared length (the RFC's resolution when a
/// misbehaving peer sends both); anything else without a length is
/// close-delimited.
///
/// # Errors
///
/// A head without a parsable status line is malformed.
fn parse_head(head: &str) -> Result<(u16, Framing), ClientError> {
    let mut lines = head.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| ClientError::MalformedResponse("an empty response head".into()))?;
    let status = status_line
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| ClientError::MalformedResponse("no status code in the line".into()))?;
    let mut declared_length: Option<usize> = None;
    let mut chunked = false;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        if name == "content-length"
            && let Ok(wanted) = value.parse::<usize>()
        {
            match declared_length {
                Some(seen) if seen != wanted => {
                    return Err(ClientError::MalformedResponse(
                        "conflicting duplicate Content-Length headers".into(),
                    ));
                }
                _ => declared_length = Some(wanted),
            }
        }
        if name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked") {
            chunked = true;
        }
    }
    let framing = if chunked {
        Framing::Chunked
    } else {
        declared_length.map_or(Framing::ToClose, Framing::ContentLength)
    };
    Ok((status, framing))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    /// The reader's retained allocation, test-visible for the
    /// bound pin.
    fn retained(reader: &ByteStream) -> usize {
        reader.buf.len()
    }

    #[tokio::test]
    async fn the_reader_reclaims_consumed_bytes_on_a_long_stream() {
        let (mut writer, stream) = tokio::net::UnixStream::pair().expect("a socket pair");
        let mut reader = ByteStream::new(stream);
        let round = vec![b'x'; 8 * 1024];
        for _ in 0..256 {
            writer.write_all(&round).await.expect("a round writes");
            let drained = reader
                .read_decoded(round.len())
                .await
                .expect("a round reads");
            assert_eq!(drained.len(), round.len());
        }
        assert!(
            retained(&reader) < 128 * 1024,
            "2 MiB streamed through the reader must not stay resident — the \
             long-lived subscription path grows without bound otherwise: \
             retained {} bytes",
            retained(&reader)
        );
    }

    #[tokio::test]
    async fn a_body_at_the_total_cap_decodes_and_one_past_refuses() {
        let at_cap = MAX_BODY_BYTES;
        let mut head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {at_cap}\r\n\
             Connection: close\r\n\r\n"
        )
        .into_bytes();
        head.extend(std::iter::repeat_n(b' ', at_cap));
        let (mut writer, stream) = tokio::net::UnixStream::pair().expect("a socket pair");
        tokio::spawn(async move {
            writer
                .write_all(&head)
                .await
                .expect("the capped body writes");
        });
        let mut reader = ByteStream::new(stream);
        let status = reader.read_head().await.expect("the head parses");
        assert_eq!(status, 200);
        let body = reader
            .read_to_body_end()
            .await
            .expect("the at-cap body reads");
        assert_eq!(body.len(), at_cap);

        let past = at_cap + 1;
        let mut head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {past}\r\n\
             Connection: close\r\n\r\n"
        )
        .into_bytes();
        head.extend(std::iter::repeat_n(b' ', past));
        let (mut writer, stream) = tokio::net::UnixStream::pair().expect("a socket pair");
        tokio::spawn(async move {
            writer
                .write_all(&head)
                .await
                .expect("the over-cap body writes");
        });
        let mut reader = ByteStream::new(stream);
        let _ = reader.read_head().await.expect("the head parses");
        let rejection = reader.read_to_body_end().await;
        assert!(
            matches!(rejection, Err(ClientError::MalformedResponse(_))),
            "one byte past the cap refuses rather than buffering it: {rejection:?}"
        );
    }

    #[test]
    fn conflicting_duplicate_content_length_headers_refuse() {
        let head = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 3\r\n";
        assert!(
            parse_head(head).is_err(),
            "conflicting duplicate lengths are a smuggling vector and refuse"
        );
    }

    #[test]
    fn chunked_precedence_is_header_order_independent() {
        let te_first = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 2\r\n";
        let parsed = parse_head(te_first);
        assert!(
            matches!(parsed, Ok((200, Framing::Chunked))),
            "chunked wins regardless of header order — a length after it must \
             not re-arm length decoding: {parsed:?}"
        );
        let cl_only = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n";
        let parsed = parse_head(cl_only);
        assert!(
            matches!(parsed, Ok((200, Framing::ContentLength(2)))),
            "a lone declared length still selects length framing: {parsed:?}"
        );
    }

    #[test]
    fn equal_duplicate_content_length_headers_accept() {
        let head = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2\r\n";
        assert!(
            parse_head(head).is_ok(),
            "byte-identical duplicates are the RFC's benign case and accept"
        );
    }
}

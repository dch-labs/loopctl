//! The wire transport: HTTP/1.1 over the endpoint the client picks.
//!
//! A deliberately minimal client for a fixed dialect — the daemon's
//! unix socket, a plain-TCP `host:port`, or TLS behind the `tls`
//! feature, every stream boxed behind one [`Wire`] — no HTTP/2, no
//! redirects — implementing the three
//! response framings a compliant peer may use (declared
//! `Content-Length`, chunked transfer coding, and connection-close
//! delimiting) with chunk decoding available incrementally, so both
//! one-shot verb calls and the long-lived SSE event stream read
//! through one reader. Every exchange — the connect handshake, a
//! verb, or a subscription's opening head — is bounded by the
//! deadline its [`ConnectOptions`](crate::ConnectOptions) carries,
//! so a peer that accepts and stalls answers a typed refusal instead
//! of an eternal await; the event stream's body is the one read
//! deliberately left unbounded, a quiet daemon keeping its
//! connection for as long as it serves. Every item here is
//! crate-private: the typed
//! verb layer in [`crate::client`] is the only consumer, so no
//! protocol code exists outside this crate.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
#[cfg(unix)]
use tokio::net::UnixStream;

use crate::ClientError;
use crate::client::Endpoint;

/// Any bidirectional byte stream the transport can speak over.
///
/// The blanket impl covers every stream the endpoints produce —
/// unix and TCP sockets, and the TLS stream the `tls` feature wraps
/// them in — so [`ByteStream`] holds one boxed trait object instead
/// of one concrete socket type, and every framing behavior above it
/// is written once.
pub(crate) trait Wire: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send {}

impl<T> Wire for T where T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send {}

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

/// The most trailer lines a chunked body may carry past its
/// zero-size terminator.
///
/// A compliant trailer section is a handful of header fields; a peer
/// that drips trailer lines without end is refusing to finish the
/// body, and the decoder refuses past this bound rather than looping
/// — the body cap cannot see it, because trailer lines decode to no
/// body bytes at all.
const MAX_TRAILER_LINES: u32 = 64;

/// The most bytes an error-status body contributes to its diagnosis.
///
/// Error envelopes are small; a daemon that buries its status in a
/// flood keeps the tail — the client keeps the status. Success
/// bodies ride the full [`MAX_BODY_BYTES`] ceiling.
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

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
pub(crate) struct ByteStream {
    /// The connected stream, read and written directly.
    ///
    /// One exchange per [`ByteStream`] over whatever the endpoint
    /// produced — the connection-per-request shape leaves no
    /// pooling state to manage.
    stream: Box<dyn Wire>,

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
    /// how every later body read decodes. There is no unarmed state
    /// at body-read time — `None` reads as close-delimited, the
    /// shape every construction path arms through `read_head` before
    /// the first body read.
    framing: Option<Framing>,

    /// The response head's content type, when it carried one.
    ///
    /// Armed by [`read_head`](Self::read_head) beside the framing,
    /// so the subscription can check its answer is an event stream
    /// before the first body byte is trusted.
    content_type: Option<String>,

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

    /// The chunked trailer lines consumed since the terminator.
    ///
    /// Meaningful only in [`ChunkPhase::Trailer`]: a peer that drips
    /// trailer lines without end is refused past
    /// [`MAX_TRAILER_LINES`] rather than looped, the bound the body
    /// cap cannot supply — trailer lines decode to no body bytes.
    trailer_lines: u32,

    /// The undelivered tail of a previous decoded pull.
    ///
    /// [`read_line`](Self::read_line) stashes bytes past the last
    /// newline here, so lines never straddle a framing boundary.
    line_pending: Vec<u8>,
}

impl ByteStream {
    /// Wrap a connected stream with an empty buffer.
    ///
    /// Fresh state for one request/response exchange over any
    /// [`Wire`] — a unix socket, a TCP connection, or a TLS stream.
    pub(super) fn new(stream: impl Wire + 'static) -> Self {
        Self {
            stream: Box::new(stream),
            buf: Vec::new(),
            pos: 0,
            framing: None,
            content_type: None,
            length_remaining: 0,
            phase: ChunkPhase::NeedSize,
            trailer_lines: 0,
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

    /// Pull more bytes from the wire into the read buffer.
    ///
    /// Consumed bytes are compacted first when they pile up, so one
    /// long-lived stream cannot grow the buffer without bound.
    ///
    /// # Errors
    ///
    /// A stream read failure maps onto the transport surface.
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
    /// Interim `1xx` heads are consumed and skipped — the final head
    /// is the one returned. Returns the status code; body reads are
    /// meaningful only after this returns.
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
                let (status, framing, content_type) = parse_head(&head)?;
                self.length_remaining = match framing {
                    Framing::ContentLength(wanted) => wanted,
                    _ => 0,
                };
                self.framing = Some(framing);
                self.content_type = content_type;
                if (100..200).contains(&status) {
                    continue;
                }
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

    /// The response head's content type, when it carried one.
    ///
    /// Armed by [`read_head`](Self::read_head); `None` until then or
    /// when the head carried no `Content-Type` at all.
    pub(super) fn content_type(&self) -> Option<&str> {
        self.content_type.as_deref()
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
    /// transition preceded it. A space or tab immediately before an
    /// extension semicolon is tolerated — the grammar's optional
    /// whitespace — while a padded size with no extension still
    /// refuses: `1*HEXDIG` is the whole size token.
    /// # Errors
    ///
    /// Malformed chunk syntax (a size that is not one hex number —
    /// the signed or padded forms included — a bad separator, or a
    /// trailer section past its bound) or a transport failure.
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
                let size_text = match size_line.split_once(';') {
                    Some((size, _extensions)) => size.trim_end_matches([' ', '\t']),
                    None => size_line.as_str(),
                };
                if size_text.is_empty() || !size_text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    return Err(ClientError::MalformedResponse(
                        "a chunk size was not one hex number".into(),
                    ));
                }
                let size = usize::from_str_radix(size_text, 16).map_err(|_| {
                    ClientError::MalformedResponse("a chunk size was not one hex number".into())
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
                    Some(_) => {
                        self.trailer_lines = self.trailer_lines.saturating_add(1);
                        if self.trailer_lines > MAX_TRAILER_LINES {
                            return Err(ClientError::MalformedResponse(
                                "the trailer section exceeded the line bound".into(),
                            ));
                        }
                    }
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

    /// Drain an error-status body, best-effort, under the error
    /// bound.
    ///
    /// The status is already the answer; whatever the body contributes
    /// is diagnosis, so a body that overruns the bound or fails
    /// mid-read loses its tail rather than masking the status the
    /// daemon did send — and the drain returns the moment the bound
    /// is full, never waiting on another read whose bytes could not
    /// contribute to the diagnosis.
    pub(super) async fn drain_error_body(&mut self) -> Vec<u8> {
        let mut collected = Vec::new();
        loop {
            let Ok(more) = self.read_decoded(MAX_LINE_BYTES).await else {
                return collected;
            };
            if more.is_empty() {
                return collected;
            }
            let room = MAX_ERROR_BODY_BYTES.saturating_sub(collected.len());
            let take = room.min(more.len());
            let Some(slice) = more.get(..take) else {
                return collected;
            };
            collected.extend_from_slice(slice);
            if collected.len() == MAX_ERROR_BODY_BYTES || take < more.len() {
                return collected;
            }
        }
    }

    /// Write one full request — head, then body — to the stream.
    ///
    /// The `authorization` argument is one complete header line,
    /// built and validated by [`bearer_header`] before the connection
    /// opens; the `authority` argument is the validated `Host` value
    /// from [`Endpoint::authority`]. Caller-controlled bytes never
    /// splice into the head here.
    ///
    /// # Errors
    ///
    /// A stream write failure, or a body that cannot serialize.
    async fn write_request(
        &mut self,
        request: &HttpRequest,
        authorization: Option<&str>,
        authority: &str,
    ) -> Result<(), ClientError> {
        let body = request
            .body
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| ClientError::Transport(error.to_string()))?
            .unwrap_or_default();
        let authorization = authorization.unwrap_or_default();
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {authority}\r\nAccept: application/json, \
             text/event-stream\r\n{authorization}Content-Type: application/json\r\nContent-Length: {len}\r\n\
             Connection: close\r\n\r\n",
            method = request.method,
            path = request.path,
            authority = authority,
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

/// Format the `Authorization` header one bearer token yields.
///
/// The token is spliced verbatim into the request head, so anything
/// that could terminate the header or the request — CR, LF, or NUL —
/// is refused outright: a corrupted or mis-parsed token must never
/// become header or request injection on a protocol whose bearer
/// authorizes gate approvals.
///
/// # Errors
///
/// [`ClientError::Transport`] when the token carries CR, LF, or NUL.
fn bearer_header(token: &str) -> Result<String, ClientError> {
    if token
        .bytes()
        .any(|byte| byte == b'\r' || byte == b'\n' || byte == 0)
    {
        return Err(ClientError::Transport(
            "the bearer token carries CR, LF, or NUL — refused rather than spliced into the \
             request head"
                .into(),
        ));
    }
    Ok(format!("Authorization: Bearer {token}\r\n"))
}

/// Map an IO failure onto the client's error surface.
///
/// A missing socket file or a refused connection is the typed absence
/// the degrade and auto-spawn logic key on; every other IO failure
/// carries its diagnosis as transport.
pub(crate) fn io_error(error: &std::io::Error) -> ClientError {
    if matches!(
        error.kind(),
        std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
    ) {
        return ClientError::SocketAbsent;
    }
    ClientError::Transport(error.to_string())
}

/// Open a connection to the endpoint, write one request, and hand
/// back the armed reader.
///
/// The request is framed as HTTP/1.1 with a JSON `Content-Length`
/// body and `Connection: close` — one request per connection, the
/// curl-per-command shape — its head carrying the endpoint's
/// authority as `Host` (`host:port` remote, the family name over
/// the socket), and, when the endpoint carries a bearer token, the
/// `Authorization` header riding every request including this one.
/// The SSE subscription uses this entry to keep its connection
/// alive; one-shot verbs use [`send`].
///
/// # Errors
///
/// [`ClientError::SocketAbsent`] when nothing listens; transport
/// failures; a malformed response head; a bearer token or endpoint
/// host that cannot ride the head safely.
pub(super) async fn open(
    endpoint: &Endpoint,
    deadline: Duration,
    request: &HttpRequest,
) -> Result<(u16, ByteStream), ClientError> {
    let exchange = async {
        let authorization = match endpoint.bearer_token() {
            Some(token) => Some(bearer_header(token)?),
            None => None,
        };
        let authority = endpoint.authority()?;
        let stream = connect_stream(endpoint).await?;
        let mut reader = ByteStream::new(stream);
        reader
            .write_request(request, authorization.as_deref(), &authority)
            .await?;
        let status = reader.read_head().await?;
        Ok((status, reader))
    };
    let bounded = exchange_timeout(deadline, exchange)?;
    match bounded.await {
        Ok(outcome) => outcome,
        Err(_) => Err(exchange_deadline(deadline)),
    }
}

/// Wrap one exchange in its deadline, typing a missing timer driver.
///
/// tokio's `timeout` panics on a runtime built without a timer
/// driver — `enable_io` alone leaves none — so before arming the real
/// bound a one-nanosecond probe runs: a runtime that cannot arm it
/// answers the typed refusal instead of unwinding the process, while
/// a runtime that can arms the real deadline unchanged.
///
/// # Errors
///
/// [`ClientError::Transport`] when the runtime carries no timer
/// driver, so the deadline could never be enforced.
pub(crate) fn exchange_timeout<F>(
    deadline: Duration,
    exchange: F,
) -> Result<tokio::time::Timeout<F>, ClientError>
where
    F: Future,
{
    let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        futures::future::FutureExt::now_or_never(tokio::time::timeout(
            Duration::from_nanos(1),
            std::future::pending::<()>(),
        ))
    }));
    if probe.is_err() {
        return Err(ClientError::Transport(
            "the runtime carries no timer driver — the exchange deadline needs one; build the \
             runtime with `enable_time()`"
                .into(),
        ));
    }
    Ok(tokio::time::timeout(deadline, exchange))
}

/// The typed refusal a stalled exchange answers with.
///
/// A peer that accepts and never finishes its answer is abandoned at
/// the bound rather than awaited forever; the bound rides the message
/// so a host can tell a stall from a failure.
fn exchange_deadline(deadline: Duration) -> ClientError {
    ClientError::Transport(format!(
        "the daemon did not answer within {} ms — abandoned rather than awaited forever",
        deadline.as_millis()
    ))
}

/// Connect the stream one endpoint names.
///
/// The transport switchboard: the socket path connects a
/// [`UnixStream`], the remote endpoints a [`TcpStream`] — wrapped in
/// TLS when the endpoint says so — every one boxed behind the
/// shared [`Wire`].
///
/// # Errors
///
/// [`ClientError::SocketAbsent`] for an absent socket or refused
/// connection; transport failures, and the TLS failure set for the
/// `tls` endpoint, otherwise.
async fn connect_stream(endpoint: &Endpoint) -> Result<Box<dyn Wire>, ClientError> {
    match endpoint {
        #[cfg(unix)]
        Endpoint::Socket(path) => {
            let stream = UnixStream::connect(path)
                .await
                .map_err(|error| io_error(&error))?;
            Ok(Box::new(stream))
        }
        Endpoint::Tcp { host, port, .. } => {
            let stream = TcpStream::connect((host.as_str(), *port))
                .await
                .map_err(|error| io_error(&error))?;
            Ok(Box::new(stream))
        }
        #[cfg(feature = "tls")]
        Endpoint::Tls {
            host,
            port,
            ca_pem,
            client_identity,
            ..
        } => crate::tls::connect(host, *port, ca_pem.as_deref(), client_identity.as_deref()).await,
    }
}

/// Send one request and collect its full response.
///
/// The one-shot verb path: opens, writes, resolves the body framing,
/// and buffers the whole body — the exchange bounded by `deadline`,
/// and an error-status body drained best-effort under the error bound
/// and the exchange deadline, so its status survives a flooded body
/// and a stalled one alike: a 401 or 403 head answers its typed
/// refusal even when its body never completes. The verb layer in
/// [`crate::client`] owns both decisions; this path only carries
/// them.
pub(super) fn send<'a>(
    endpoint: &'a Endpoint,
    deadline: Duration,
    request: HttpRequest,
) -> Pin<Box<dyn Future<Output = Result<HttpResponse, ClientError>> + Send + 'a>> {
    Box::pin(async move {
        let started = std::time::Instant::now();
        let (status, mut reader) = open(endpoint, deadline, &request).await?;
        let remaining = deadline.saturating_sub(started.elapsed());
        let body = if status_ok(status) {
            let bounded = exchange_timeout(remaining, reader.read_to_body_end())?;
            match bounded.await {
                Ok(Ok(body)) => body,
                Ok(Err(error)) => return Err(error),
                Err(_) => return Err(exchange_deadline(deadline)),
            }
        } else {
            match exchange_timeout(remaining, reader.drain_error_body()) {
                Ok(bounded) => bounded.await.unwrap_or_default(),
                Err(error) => return Err(error),
            }
        };
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

/// Whether a status code is a success for the verb layer.
///
/// Any 2xx; the protocol pins no finer granularity.
pub(super) fn status_ok(status: u16) -> bool {
    (200..300).contains(&status)
}

/// Parse one `Content-Length` value under its `1*DIGIT` grammar.
///
/// The RFC admits decimal digits only — a sign, a list, or any other
/// decoration is not one decimal number, and the value refuses
/// rather than degrading to close-delimited framing: the same
/// anti-smuggling stance the conflicting-duplicate refusal takes.
///
/// # Errors
///
/// [`ClientError::MalformedResponse`] for any value that is not
/// exactly `1*DIGIT`, or one too large to frame.
fn content_length(value: &str) -> Result<usize, ClientError> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(ClientError::MalformedResponse(
            "a Content-Length header held no single number".into(),
        ));
    }
    value.parse::<usize>().map_err(|_| {
        ClientError::MalformedResponse("a Content-Length header held no single number".into())
    })
}

/// Parse a response head into its status, framing, and content type.
///
/// Chunked wins over a declared length (the RFC's resolution when a
/// misbehaving peer sends both); anything else without a length is
/// close-delimited. A `Content-Length` that is not one decimal
/// number refuses — the same anti-smuggling stance as the
/// conflicting-duplicate refusal, so the list form (`2, 3`) or a
/// signed `+2` never degrades to close-delimited framing. A `204`
/// answer is bodyless whatever its headers claim — the status
/// selects zero-length framing ahead of every other body-framing
/// rule, so a peer that declares a length or a chunked coding over a
/// bodyless status still answers an empty body without waiting for
/// the close it will never send.
///
/// # Errors
///
/// A head without a parsable status line, or carrying a
/// `Content-Length` that is not one decimal number, is malformed.
fn parse_head(head: &str) -> Result<(u16, Framing, Option<String>), ClientError> {
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
    let mut content_type: Option<String> = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        if name == "content-length" {
            let wanted = content_length(value)?;
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
        if name == "content-type" {
            content_type = Some(value.to_string());
        }
    }
    let framing = if status == 204 {
        Framing::ContentLength(0)
    } else if chunked {
        Framing::Chunked
    } else {
        declared_length.map_or(Framing::ToClose, Framing::ContentLength)
    };
    Ok((status, framing, content_type))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use tokio::io::AsyncWriteExt;

    /// The reader's retained allocation, test-visible for the
    /// bound pin.
    fn retained(reader: &ByteStream) -> usize {
        reader.buf.len()
    }

    #[tokio::test]
    async fn the_reader_reclaims_consumed_bytes_on_a_long_stream() {
        let (mut writer, stream) = tokio::io::duplex(8 * 1024);
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
        let (mut writer, stream) = tokio::io::duplex(64 * 1024);
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
        let (mut writer, stream) = tokio::io::duplex(64 * 1024);
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
    fn an_unparsable_content_length_header_refuses() {
        let list_form = "HTTP/1.1 200 OK\r\nContent-Length: 2, 3\r\n";
        assert!(
            parse_head(list_form).is_err(),
            "the list form is a smuggling vector and refuses, never a silent \
             close-delimited fallback"
        );
        let non_numeric = "HTTP/1.1 200 OK\r\nContent-Length: +2x\r\n";
        assert!(
            parse_head(non_numeric).is_err(),
            "a value that is not one decimal number refuses: {non_numeric:?}"
        );
        let signed = "HTTP/1.1 200 OK\r\nContent-Length: +2\r\n";
        assert!(
            parse_head(signed).is_err(),
            "a signed value parses as a number but is not 1*DIGIT — it refuses: {signed:?}"
        );
    }

    #[tokio::test]
    async fn chunk_size_lines_that_are_not_one_hex_number_refuse() {
        for size_line in ["+2", "+ff", " 2"] {
            let wire = format!(
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n\
                 {size_line}\r\n"
            );
            let (mut writer, stream) = tokio::io::duplex(64 * 1024);
            tokio::spawn(async move {
                writer
                    .write_all(wire.as_bytes())
                    .await
                    .expect("the chunked refusal wire writes");
            });
            let mut reader = ByteStream::new(stream);
            let _ = reader.read_head().await.expect("the head parses");
            let rejection = reader.read_decoded(64).await;
            assert!(
                matches!(&rejection, Err(ClientError::MalformedResponse(text))
                    if text.contains("a chunk size was not one hex number")),
                "a size line that is not 1*HEXDIG refuses at the size parse — never a \
                 lenient decode that trips a later bound: {size_line:?} answered {rejection:?}"
            );
        }
    }

    #[test]
    fn chunked_precedence_is_header_order_independent() {
        let te_first = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 2\r\n";
        let parsed = parse_head(te_first);
        assert!(
            matches!(parsed, Ok((200, Framing::Chunked, _))),
            "chunked wins regardless of header order — a length after it must \
             not re-arm length decoding: {parsed:?}"
        );
        let cl_only = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n";
        let parsed = parse_head(cl_only);
        assert!(
            matches!(parsed, Ok((200, Framing::ContentLength(2), _))),
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

    #[tokio::test]
    async fn an_interim_1xx_head_is_skipped_for_the_final_head() {
        let wire = "HTTP/1.1 100 Continue\r\n\r\n\
                    HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
            .as_bytes()
            .to_vec();
        let (mut writer, stream) = tokio::io::duplex(64 * 1024);
        tokio::spawn(async move {
            writer
                .write_all(&wire)
                .await
                .expect("the interim wire writes");
        });
        let mut reader = ByteStream::new(stream);
        let status = reader.read_head().await;
        assert!(
            matches!(status, Ok(200)),
            "an interim 1xx head is consumed; the final head answers: {status:?}"
        );
        let body = reader.read_to_body_end().await;
        assert!(
            matches!(body.as_deref(), Ok(b"ok")),
            "the final head's framing owns the body, never the interim's: {body:?}"
        );
    }

    #[tokio::test]
    async fn a_trailer_section_past_the_line_bound_refuses() {
        let mut wire =
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n0\r\n"
                .as_bytes()
                .to_vec();
        for index in 0..100 {
            let trailer_line = format!("x-trailer-{index}: value\r\n");
            wire.extend_from_slice(trailer_line.as_bytes());
        }
        wire.extend_from_slice(b"\r\n");
        let (mut writer, stream) = tokio::io::duplex(64 * 1024);
        tokio::spawn(async move {
            writer
                .write_all(&wire)
                .await
                .expect("the trailer wire writes");
        });
        let mut reader = ByteStream::new(stream);
        let _ = reader.read_head().await.expect("the head parses");
        let rejection = reader.read_to_body_end().await;
        assert!(
            matches!(&rejection, Err(ClientError::MalformedResponse(text))
                if text.contains("the trailer section exceeded the line bound")),
            "an endless trailer section refuses inside the decoder — the body cap cannot \
             see it, because trailer lines decode to no body bytes: {rejection:?}"
        );
    }

    #[tokio::test]
    async fn a_chunk_size_extension_suffix_is_tolerated() {
        let wire = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n\
                    4;ext=ignored\r\nbody\r\n0\r\n\r\n"
            .as_bytes()
            .to_vec();
        let (mut writer, stream) = tokio::io::duplex(64 * 1024);
        tokio::spawn(async move {
            writer
                .write_all(&wire)
                .await
                .expect("the extended wire writes");
        });
        let mut reader = ByteStream::new(stream);
        let _ = reader.read_head().await.expect("the head parses");
        let body = reader.read_to_body_end().await;
        assert!(
            matches!(body.as_deref(), Ok(b"body")),
            "the size line's extension suffix is tolerated per the grammar: {body:?}"
        );
    }

    #[tokio::test]
    async fn whitespace_before_a_chunk_extension_is_tolerated() {
        for size_line in ["4 ;ext=value", "4\t;ext=value"] {
            let wire = format!(
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n\
                 {size_line}\r\nbody\r\n0\r\n\r\n"
            );
            let (mut writer, stream) = tokio::io::duplex(64 * 1024);
            tokio::spawn(async move {
                writer
                    .write_all(wire.as_bytes())
                    .await
                    .expect("the spaced wire writes");
            });
            let mut reader = ByteStream::new(stream);
            let _ = reader.read_head().await.expect("the head parses");
            let body = reader.read_to_body_end().await;
            assert!(
                matches!(body.as_deref(), Ok(b"body")),
                "a space or tab immediately before an extension semicolon is the grammar's \
                 optional whitespace, never a padded size: {size_line:?} answered {body:?}"
            );
        }
    }

    #[test]
    fn a_204_answer_is_bodyless_whatever_the_head_claims() {
        let declared = "HTTP/1.1 204 No Content\r\nContent-Length: 5\r\n";
        let parsed = parse_head(declared);
        assert!(
            matches!(parsed, Ok((204, Framing::ContentLength(0), _))),
            "a 204 selects zero-length framing first — a declared length cannot make a \
             bodyless status wait for one: {parsed:?}"
        );
        let chunked = "HTTP/1.1 204 No Content\r\nTransfer-Encoding: chunked\r\n";
        let parsed = parse_head(chunked);
        assert!(
            matches!(parsed, Ok((204, Framing::ContentLength(0), _))),
            "a chunked coding claimed over a 204 meets the same zero-length arm: {parsed:?}"
        );
    }

    #[tokio::test]
    async fn an_error_body_at_the_cap_returns_without_another_read() {
        let mut wire = b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n".to_vec();
        wire.extend(std::iter::repeat_n(b'x', MAX_ERROR_BODY_BYTES));
        let (mut writer, stream) = tokio::io::duplex(8 * 1024);
        tokio::spawn(async move {
            writer
                .write_all(&wire)
                .await
                .expect("the capped error body writes");
            // The connection stays open after the cap-full body: the
            // next read is exactly the wait the drain must not make.
            std::future::pending::<()>().await;
        });
        let mut reader = ByteStream::new(stream);
        let _ = reader.read_head().await.expect("the head parses");
        let drained =
            tokio::time::timeout(std::time::Duration::from_secs(2), reader.drain_error_body())
                .await
                .expect("the drain returns the moment the cap is full");
        assert_eq!(
            drained.len(),
            MAX_ERROR_BODY_BYTES,
            "a full cap of diagnosis rides along"
        );
    }

    #[test]
    fn a_runtime_without_a_timer_answers_the_typed_refusal() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .build()
            .expect("the io-only runtime builds");
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let rejection = runtime.block_on(async {
            match exchange_timeout(Duration::from_millis(50), std::future::pending::<()>()) {
                Ok(_) => panic!("an io-only runtime cannot arm the deadline it was asked for"),
                Err(error) => error,
            }
        });
        std::panic::set_hook(previous_hook);
        assert!(
            matches!(&rejection, ClientError::Transport(refusal)
                if refusal.contains("timer driver")),
            "a runtime without a timer driver answers the typed refusal, never a panic: \
             {rejection:?}"
        );
    }

    /// Split `wire` at `splits` byte offsets into consecutive pieces.
    ///
    /// The property tests' cutter: every split point becomes a piece
    /// boundary, and the paced wire below hands the reader one piece
    /// per read — each split observed, never coalesced away.
    fn split_wire_into_pieces(wire: &[u8], splits: &[usize]) -> Vec<Vec<u8>> {
        let mut pieces = Vec::new();
        let mut rest = wire;
        for split in splits {
            let take = (*split).min(rest.len());
            let (head, tail) = rest.split_at(take);
            if !head.is_empty() {
                pieces.push(head.to_vec());
            }
            rest = tail;
        }
        if !rest.is_empty() {
            pieces.push(rest.to_vec());
        }
        pieces
    }

    /// An in-memory wire that hands the reader one scripted piece per
    /// read and goes pending between pieces.
    ///
    /// The split-point properties' carrier: unlike a socket — which
    /// may coalesce pieces before the reader ever resumes — this wire
    /// forces the reader to observe every split as its own read and a
    /// pending poll between them, so a regression after a partial
    /// status or chunk-size line becomes pending cannot pass behind
    /// coalesced writes.
    struct PacedWire {
        /// The pieces still to deliver, in order.
        pieces: std::collections::VecDeque<Vec<u8>>,
    }

    impl tokio::io::AsyncRead for PacedWire {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            let wire = self.get_mut();
            let Some(piece) = wire.pieces.front_mut() else {
                return std::task::Poll::Pending;
            };
            let take = buf.remaining().min(piece.len());
            let Some(delivered) = piece.get(..take).map(<[u8]>::to_vec) else {
                return std::task::Poll::Pending;
            };
            buf.put_slice(&delivered);
            if take == piece.len() {
                wire.pieces.pop_front();
            } else {
                piece.drain(..take);
            }
            std::task::Poll::Ready(Ok(()))
        }
    }

    impl tokio::io::AsyncWrite for PacedWire {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            _buf: &[u8],
        ) -> std::task::Poll<std::io::Result<usize>> {
            std::task::Poll::Ready(Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "the paced wire is read-only",
            )))
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn poll_shutdown(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::task::Poll::Ready(Ok(()))
        }
    }

    /// Decode one wire — whole or split into `pieces` — through a
    /// fresh paced reader, to its full body.
    fn decode_pieces(pieces: Vec<Vec<u8>>) -> Result<Vec<u8>, ClientError> {
        let wire = PacedWire {
            pieces: pieces.into(),
        };
        let mut reader = ByteStream::new(wire);
        futures::executor::block_on(async {
            reader.read_head().await?;
            reader.read_to_body_end().await
        })
    }

    proptest! {
        #[test]
        fn chunked_decoding_is_invariant_under_arbitrary_split_points(
            payload in "[a-zA-Z0-9{}\":, ]{0,200}",
            splits in prop::collection::vec(0usize..64, 0..8),
        ) {
            let frame = format!("id: 7\r\ndata: {payload}\r\n\r\n");
            let mut wire =
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                    .as_bytes()
                    .to_vec();
            wire.extend_from_slice(format!("{:x}\r\n", frame.len()).as_bytes());
            wire.extend_from_slice(frame.as_bytes());
            wire.extend_from_slice(b"\r\n0\r\n\r\n");
            let reference =
                decode_pieces(vec![wire.clone()]).expect("the unsplit wire decodes");
            let pieces = split_wire_into_pieces(&wire, &splits);
            let decoded = decode_pieces(pieces).expect("the split wire decodes");
            prop_assert_eq!(decoded, reference);
        }

        #[test]
        fn head_parsing_is_invariant_under_arbitrary_split_points(
            body_text in "[a-z]{0,64}",
            splits in prop::collection::vec(0usize..96, 0..8),
        ) {
            let wire = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body_text}",
                body_text.len()
            )
            .into_bytes();
            let reference =
                decode_pieces(vec![wire.clone()]).expect("the unsplit wire decodes");
            let pieces = split_wire_into_pieces(&wire, &splits);
            let decoded = decode_pieces(pieces).expect("the split wire decodes");
            prop_assert_eq!(decoded, reference);
        }
    }
}

//! The event stream: SSE frames decoded into ledger events.
//!
//! The daemon's observation surface is an HTTP `text/event-stream`
//! over the same socket: each frame's `id:` is the loop's event
//! cursor and its `data:` line is one trajectory-JSONL event (the
//! pinned interchange shape the ledger already writes). The reader
//! implements the SSE subset that contract needs — blank-line
//! delimiting, multi-`data:` joins, comment and `retry:` tolerance —
//! and surfaces each frame as a [`LedgerEvent`] pairing the cursor
//! with the typed event, so a consumer can resume from exactly the
//! last cursor it saw.

use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use futures::stream::Stream;
use loopctl::memory::trajectory::TrajectoryEvent;

use crate::ClientError;
use crate::http::ByteStream;

/// One poll's raw outcome from the wire reader.
///
/// The wire path's intermediate: what the socket read produced this
/// poll, before the frame state machine interprets it.
enum Pulled {
    /// No line ready yet — the waker is registered.
    ///
    /// The poll returns `Pending`; the read's waker fires on data.
    Pending,

    /// The body ended cleanly.
    ///
    /// The stream's end is next — any half-finished frame refuses.
    Ended,

    /// The read failed.
    ///
    /// Carries the transport or malformed-response error verbatim.
    Failed(ClientError),

    /// One decoded body line.
    ///
    /// CR-stripped, framing already resolved.
    Line(String),
}

/// One delivered event: the loop's cursor and the typed event.
///
/// The cursor is the frame's SSE `id` — the value a resubscription
/// passes as `since` to resume exactly after this event.
#[derive(Debug, Clone)]
pub struct LedgerEvent {
    /// The event's cursor, from the frame's `id` field.
    ///
    /// Strictly increasing within one loop's stream; pass it back as
    /// the `since` of a later subscription to resume here.
    pub cursor: u64,

    /// The typed event payload, trajectory-JSONL shaped.
    ///
    /// The same serde type the ledger writes, so the wire and the
    /// store cannot drift apart.
    pub event: TrajectoryEvent,
}

/// The parsed fields of one in-progress SSE frame.
///
/// Accumulated line by line until the blank delimiter dispatches
/// (or drops) the frame.
#[derive(Default)]
struct Frame {
    /// The frame's `id` field, when it carried one.
    ///
    /// Parsed to the delivered cursor at dispatch; `None` on a data
    /// frame is the refusal case (the cursor contract cannot
    /// advance).
    id: Option<String>,

    /// The frame's accumulated `data` lines.
    ///
    /// Joined with newlines at dispatch into the one JSON payload;
    /// empty means the frame delivered nothing (keep-alive).
    data: Vec<String>,
}

impl Frame {
    /// Consume the accumulated fields into one delivered event.
    ///
    /// A frame without data is a keep-alive (or a bare comment) and
    /// delivers nothing; a data frame without an id cannot advance
    /// the cursor contract and is refused rather than silently
    /// re-delivered on resume.
    ///
    /// # Errors
    ///
    /// [`ClientError::MalformedResponse`] for a missing or non-numeric
    /// id cursor, or a payload that is not a trajectory event.
    fn finish(&mut self) -> Result<Option<LedgerEvent>, ClientError> {
        let id = self.id.take();
        let data = std::mem::take(&mut self.data);
        if data.is_empty() {
            return Ok(None);
        }
        let cursor_text = id.ok_or_else(|| {
            ClientError::MalformedResponse("an event-stream data frame carried no id cursor".into())
        })?;
        let cursor = cursor_text
            .parse::<u64>()
            .map_err(|_| ClientError::MalformedResponse("an id cursor was not numeric".into()))?;
        let joined = data.join("\n");
        let event: TrajectoryEvent = serde_json::from_str(&joined).map_err(|error| {
            ClientError::MalformedResponse(format!(
                "an event-stream payload was not a trajectory event: {error}"
            ))
        })?;
        Ok(Some(LedgerEvent { cursor, event }))
    }
}

/// The streaming reader over one subscribed connection.
///
/// Implements [`Stream`]; each item is the next [`LedgerEvent`] or a
/// transport/malformed-response error. The stream ends (`None`) when
/// the daemon closes the connection — resubscribe with the last seen
/// cursor.
pub struct EventStream {
    /// Where this stream's events come from.
    ///
    /// Fixed at construction — the wire reader for a daemon
    /// subscription, a boxed host stream for the in-process mode.
    source: Source,
}

/// Where one event stream's events come from.
///
/// The daemon subscription's wire reader or a host-authored boxed
/// stream — the two constructions of [`EventStream`].
enum Source {
    /// The subscribed socket connection, decoded frame by frame.
    ///
    /// Owned outright: the socket lives exactly as long as the
    /// stream.
    Wire {
        /// The armed reader, taken on the stream's end.
        ///
        /// `None` after the body ends or an error poisons it.
        reader: Option<ByteStream>,

        /// The frame currently accumulating.
        ///
        /// Reset by each blank-line dispatch.
        frame: Frame,
    },

    /// A host-authored boxed stream, passed through verbatim.
    ///
    /// Fed by [`from_pin_box`](EventStream::from_pin_box) — the
    /// in-process half of the mode unification.
    Boxed(Pin<Box<dyn Stream<Item = Result<LedgerEvent, ClientError>> + Send>>),
}

impl EventStream {
    /// Wrap one armed response body.
    ///
    /// The subscription has already checked the status; this owns
    /// the socket from here.
    pub(crate) fn new(reader: ByteStream) -> Self {
        Self {
            source: Source::Wire {
                reader: Some(reader),
                frame: Frame::default(),
            },
        }
    }

    /// Wrap any boxed event source as an [`EventStream`].
    ///
    /// The in-process half of the mode-unification contract: a host
    /// implementing [`LoopctlClient`](crate::LoopctlClient) against
    /// its own engine adapts any boxed stream of ledger events — a
    /// broadcast channel's receiver, a mapped engine observer feed —
    /// into exactly the type the daemon subscription returns, so
    /// `subscribe` call sites are mode-agnostic in full, not only in
    /// signature.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use futures::stream;
    /// use futures::StreamExt;
    /// use loopctl_client::EventStream;
    /// use loopctl_client::LedgerEvent;
    ///
    /// # async fn demo() {
    /// let source = stream::iter(Vec::<LedgerEvent>::new())
    ///     .map(Ok::<_, loopctl_client::ClientError>);
    /// let mut adapted = EventStream::from_pin_box(Box::pin(source));
    /// assert!(adapted.next().await.is_none());
    /// # }
    /// ```
    #[must_use]
    pub fn from_pin_box(
        stream: Pin<Box<dyn Stream<Item = Result<LedgerEvent, ClientError>> + Send>>,
    ) -> Self {
        Self {
            source: Source::Boxed(stream),
        }
    }

    /// Consume and return this stream, boxing it for storage.
    ///
    /// The stream is `Unpin` whichever source it wraps (the wire
    /// reader or a boxed source both own their state outright), so
    /// the box is a convenience for hosts that store one subscription
    /// behind a trait object.
    pub fn boxed(self) -> Pin<Box<Self>> {
        Box::pin(self)
    }

    /// One poll: pull a line out of the reader (borrow one), then
    /// apply the frame state machine on `self` (borrow two).
    fn poll_next(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<LedgerEvent, ClientError>>> {
        match &mut self.source {
            Source::Boxed(stream) => stream.as_mut().poll_next(cx),
            Source::Wire { .. } => Self::poll_wire(self, cx),
        }
    }

    /// One poll of the wire source: pull a line out of the reader
    /// (borrow one), then apply the frame state machine (borrow two).
    fn poll_wire(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<LedgerEvent, ClientError>>> {
        let Source::Wire { reader, frame } = &mut self.source else {
            return Poll::Ready(None);
        };
        loop {
            let pulled = {
                let Some(wire) = reader.as_mut() else {
                    return Poll::Ready(None);
                };
                let mut pending = std::pin::pin!(wire.read_line());
                match pending.as_mut().poll(cx) {
                    Poll::Pending => Pulled::Pending,
                    Poll::Ready(Ok(Some(line))) => Pulled::Line(line),
                    Poll::Ready(Ok(None)) => Pulled::Ended,
                    Poll::Ready(Err(error)) => Pulled::Failed(error),
                }
            };
            match pulled {
                Pulled::Pending => return Poll::Pending,
                Pulled::Failed(error) => {
                    *reader = None;
                    return Poll::Ready(Some(Err(error)));
                }
                Pulled::Ended => {
                    let finished = frame.finish();
                    *reader = None;
                    return match finished {
                        Ok(Some(event)) => Poll::Ready(Some(Ok(event))),
                        Ok(None) => Poll::Ready(None),
                        Err(error) => Poll::Ready(Some(Err(error))),
                    };
                }
                Pulled::Line(line) => {
                    if line.is_empty() {
                        match frame.finish() {
                            Ok(Some(event)) => return Poll::Ready(Some(Ok(event))),
                            Ok(None) => continue,
                            Err(error) => {
                                *reader = None;
                                return Poll::Ready(Some(Err(error)));
                            }
                        }
                    }
                    if line.starts_with(':') {
                        continue;
                    }
                    if let Some((field, value)) = line.split_once(':') {
                        let value = value.strip_prefix(' ').unwrap_or(value);
                        match field {
                            "id" => frame.id = Some(value.to_string()),
                            "data" => frame.data.push(value.to_string()),
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}

impl Stream for EventStream {
    type Item = Result<LedgerEvent, ClientError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().poll_next(cx)
    }
}

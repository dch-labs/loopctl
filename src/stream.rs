//! Streaming event types for LLM API responses.
//!
//! Types used when consuming Server-Sent Events
//! (SSE) based streaming responses from LLM APIs. The core [`StreamEvent`]
//! enum represents each discrete event in the stream lifecycle, while
//! [`StreamAccumulator`] collects those events into a complete [`Message`].
//!
//! Streaming allows the framework to process model output incrementally —
//! displaying text as it arrives, detecting tool invocations as soon as
//! the part starts, and reporting token usage without waiting for the
//! full response. Essential for responsive agent behavior.
//!
//! # Stream Lifecycle
//!
//! The streaming protocol follows this event sequence:
//!
//! ```text
//! MessageStart → [PartStart → IndexedDelta* → PartStop]* → MessageDelta → MessageStop
//! ```
//!
//! [`Ping`](StreamEvent::Ping) events may appear at any point in the stream and should be
//! ignored by consumers.
//!
//! # Provided Types
//!
//! - **[`StreamEvent`]** — Top-level enum for every SSE event type.
//! - **[`StreamAccumulator`]** — Stateful builder that turns events into a [`Message`].
//! - **[`StreamStopReason`]** — Why the model stopped generating tokens.
//! - **[`Usage`]** — Token consumption statistics.
//! - **[`DeltaPart`]** — Incremental content payload (text, tool JSON, or partial JSON).
//! - **[`IndexedDelta`]** — An indexed [`DeltaPart`] carrying the part position.
//! - **[`MessageStart`]** / **[`MessageDelta`]** — Boundary events with metadata.
//!
//! # Sub-modules
//!
//! - **[`handler`]** — [`handler::StreamHandler`] with retry, timeout,
//!   and fallback for resilient streaming.
//! - **[`heartbeat`]** — [`heartbeat::HeartbeatStream`] composable heartbeat
//!   and timeout wrapper for any stream.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::stream::{StreamAccumulator, StreamEvent, StreamStopReason};
//!
//! let mut acc = StreamAccumulator::new();
//!
//! // Feed events as they arrive from the SSE connection
//! for event in std::iter::empty::<StreamEvent>() {
//!     acc.process(&event).unwrap();
//! }
//!
//! // Get usage before building (build consumes the accumulator)
//! let _usage = acc.usage();
//! let message = acc.build();
//! ```

use crate::message::{Message, MessagePart};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

pub mod handler;
pub mod heartbeat;

// ==================================================
// StreamError
// ==================================================

/// Errors that can occur during stream event processing.
///
/// Returned by [`StreamAccumulator::process`] when an event cannot be
/// handled correctly — for example, when accumulated tool-call JSON
/// is malformed at [`PartStop`](StreamEvent::PartStop) time.
#[derive(Debug)]
pub enum StreamError {
    /// The concatenated tool-call input JSON could not be parsed.
    ///
    /// Contains the original [`serde_json::Error`] from the parse attempt
    /// and the raw input string that failed.
    InvalidToolInputJson(serde_json::Error, String),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::InvalidToolInputJson(err, raw) => {
                write!(f, "invalid tool input JSON: {err} (raw_len={})", raw.len())
            }
        }
    }
}

impl std::error::Error for StreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StreamError::InvalidToolInputJson(err, _) => Some(err),
        }
    }
}

// ==================================================
// StreamEvent
// ==================================================

/// An event from a streaming LLM API response.
///
/// Events follow the SSE (Server-Sent Events) protocol used by
/// LLM APIs and compatible providers. Each variant
/// corresponds to one of the documented event types emitted during
/// a streaming response.
///
/// Consumers typically match on these variants to drive UI updates
/// or feed them into a [`StreamAccumulator`] to reconstruct the full
/// [`Message`].
///
/// # Lifecycle
///
/// ```text
/// MessageStart
///   → PartStart
///     → IndexedDelta (repeated)
///   → PartStop
///   → ... more parts ...
/// → MessageDelta
/// → MessageStop
/// ```
///
/// # Handling
///
/// Most consumers only need to handle a few variants:
/// - [`IndexedDelta`](Self::IndexedDelta) — for real-time text display.
/// - [`MessageStart`](Self::MessageStart) — to capture model metadata.
/// - [`MessageDelta`](Self::MessageDelta) — to get the stop reason and usage.
/// - [`MessageStop`](Self::MessageStop) — to know the stream is complete.
///
/// # Example
///
/// ```rust
/// use loopctl::stream::{StreamEvent, DeltaPart, IndexedDelta};
///
/// let event = StreamEvent::MessageStop; // placeholder
/// match event {
///     StreamEvent::MessageStart(_start) => {
///         // println!("Model: {}", start.message.model);
///     }
///     StreamEvent::IndexedDelta(delta) => {
///         if let DeltaPart::Text { text } = &delta.delta {
///             let _text: &str = text;
///         }
///     }
///     StreamEvent::MessageStop => { /* println!("[done]") */ }
///     _ => {}
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// The start of a new message from the API.
    ///
    /// Always the first event in a stream. Contains the message's
    /// metadata (ID, model, role). Use it to initialize any
    /// state that depends on the model or message ID before
    /// content begins arriving.
    MessageStart(MessageStart),

    /// The start of a new part within the response.
    ///
    /// Emitted before any [`IndexedDelta`] events for this part.
    /// May contain a partial [`MessagePart`] for tool-call parts
    /// where the `id` and `name` are known upfront. For text parts,
    /// the [`PartStart::part`] field will be `None`.
    PartStart(PartStart),

    /// A delta (incremental update) for the current part.
    ///
    /// Carries a [`DeltaPart`] payload — either text to append,
    /// JSON input for a tool invocation, or raw tool-call data.
    /// Multiple deltas may arrive for a single part before
    /// [`PartStop`](StreamEvent::PartStop) signals
    /// the part is complete.
    IndexedDelta(IndexedDelta),

    /// The end of the current part.
    ///
    /// Signals that all [`IndexedDelta`] events for this part
    /// have been sent. No payload is attached. Consumers should
    /// finalize the in-progress part (e.g. parse accumulated JSON
    /// for tool-call parts) when this event is received.
    PartStop,

    /// A delta update for the message itself (contains stop reason).
    ///
    /// Emitted after all parts are complete. Carries the
    /// [`StreamStopReason`] and final [`Usage`] statistics. Use
    /// [`StreamStopReason::from_api_str`] to parse the stop reason
    /// from the raw string in [`MessageDeltaPayload`].
    MessageDelta(MessageDelta),

    /// The end of the message stream.
    ///
    /// Always the last event (before any trailing pings). No payload.
    /// Consumers should treat this as the signal that the full
    /// response is complete.
    MessageStop,

    /// A keep-alive ping from the API server.
    ///
    /// May appear at any point during the stream. Consumers should
    /// ignore this event; it exists solely to prevent connection
    /// timeouts on long-running requests. The [`StreamAccumulator`]
    /// silently skips pings during [`process`](StreamAccumulator::process).
    Ping,
}

// ==================================================
// MessageStart
// ==================================================

/// The start of a new message from the API.
///
/// Wraps [`MessageMetadata`] and is always the first event in a
/// streaming response. Use it to capture the model name and message
/// ID before content begins arriving.
///
/// # Construction
///
/// Typically deserialized from the SSE stream rather than constructed
/// manually. The API always provides `id`, `role`, and `model` fields.
///
/// # Example
///
/// ```rust
/// use loopctl::stream::{MessageStart, MessageMetadata};
///
/// let start = MessageStart {
///     message: MessageMetadata {
///         id: "msg_abc123".to_string(),
///         role: "assistant".to_string(),
///         model: "llm-70b".to_string(),
///     },
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageStart {
    /// Metadata about the message being streamed.
    ///
    /// Contains the server-assigned message ID, the role (always
    /// `"assistant"` for streaming responses), and the model name.
    pub message: MessageMetadata,
}

// ==================================================
// MessageMetadata
// ==================================================

/// Metadata about a message from the API.
///
/// Carries identifying information for the streaming response: the
/// message ID, the role, and the model that produced it. Embedded
/// inside [`MessageStart`] and accessible before any content arrives.
///
/// # Fields
///
/// - [`id`](Self::id) — Unique per response, useful for logging and correlation.
/// - [`role`](Self::role) — Always `"assistant"` for streaming responses.
/// - [`model`](Self::model) — The model identifier (e.g. `"llm-4-turbo"`).
///
/// # Example
///
/// ```rust
/// use loopctl::stream::MessageMetadata;
///
/// let meta = MessageMetadata {
///     id: "msg_abc123".to_string(),
///     role: "assistant".to_string(),
///     model: "llm-4-turbo".to_string(),
/// };
/// assert_eq!(meta.role, "assistant");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    /// The server-assigned message ID (e.g. `"msg_abc123"`).
    ///
    /// Unique per response. Useful for logging, correlation, and
    /// debugging specific API interactions. The format may vary
    /// between API versions.
    pub id: String,

    /// The role of the message sender.
    ///
    /// Always `"assistant"` for streaming responses from the API.
    /// See [`Role`](crate::message::Role) for the typed equivalent
    /// used elsewhere in the framework.
    pub role: String,

    /// The model that generated the response (e.g. `"llm-4-turbo"`).
    ///
    /// Useful for logging, debugging, and routing decisions when
    /// multiple models are in use. The [`StreamAccumulator`] captures
    /// this value from the [`MessageStart`](StreamEvent::MessageStart)
    /// event.
    pub model: String,
}

// ==================================================
// PartStart
// ==================================================

/// The start of a new part within the response.
///
/// Emitted once per part, before any [`IndexedDelta`]
/// events. The `part` field may be `None` for text parts
/// (where the type is inferred from deltas) or `Some` for tool-call
/// parts where the `id` and `name` are known upfront.
///
/// # Example
///
/// ```rust
/// use loopctl::stream::PartStart;
/// use loopctl::message::MessagePart;
/// use serde_json::Value;
///
/// // Text part start — type is implicit
/// let text_start = PartStart { index: 0, part: None };
///
/// // Tool-call part start — id and name are provided
/// let tool_start = PartStart {
///     index: 1,
///     part: Some(MessagePart::tool_call("tool_1", "read_file", Value::Null)),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartStart {
    /// Zero-based index of this part within the message.
    ///
    /// Sequentially assigned by the API. Used to correlate
    /// [`IndexedDelta`] events with the correct part.
    /// Indices are monotonically increasing within a single stream.
    pub index: usize,

    /// The part, if known at start time.
    ///
    /// `Some` for tool-call parts (so the `id` and `name` are
    /// available immediately). `None` for text parts, where
    /// the type is inferred from subsequent [`DeltaPart::Text`]
    /// deltas. The [`StreamAccumulator`] uses this to seed the
    /// tool ID and name before deltas arrive.
    pub part: Option<MessagePart>,
}

// ==================================================
// IndexedDelta
// ==================================================

/// A delta (incremental update) for the current part.
///
/// Carries a [`DeltaPart`] payload that should be appended to the
/// in-progress part identified by [`index`](Self::index).
/// Multiple deltas may arrive for a single part.
///
/// # Example
///
/// ```rust
/// use loopctl::stream::{IndexedDelta, DeltaPart};
///
/// let delta = IndexedDelta {
///     index: 0,
///     delta: DeltaPart::Text { text: "Hello".to_string() },
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedDelta {
    /// Zero-based index of the part being updated.
    ///
    /// Matches the [`index`](PartStart::index) from the
    /// corresponding [`PartStart`] event. All deltas with
    /// the same index belong to the same part.
    pub index: usize,

    /// The incremental content to append.
    ///
    /// See [`DeltaPart`] for the possible payload types. The
    /// [`StreamAccumulator`] appends text fragments to
    /// [`current_text`](StreamAccumulator) and JSON fragments to
    /// [`current_tool_input`](StreamAccumulator) based on this value.
    pub delta: DeltaPart,
}

// ==================================================
// DeltaPart
// ==================================================

/// A delta (incremental update) for content within a streaming response.
///
/// Each variant represents a different kind of incremental data that
/// the API sends. Consumers should append the payload to the
/// appropriate in-progress part.
///
/// # Variants
///
/// - [`Text`](Self::Text) — Append to the text buffer for text parts.
/// - [`ToolCall`](Self::ToolCall) — Append to the JSON buffer for tool-call parts.
/// - [`InputJson`](Self::InputJson) — Append to the JSON buffer (raw string form).
///
/// # Example
///
/// ```rust
/// use loopctl::stream::DeltaPart;
///
/// let delta_content = DeltaPart::Text { text: "hello".to_string() };
/// let mut buffer = String::new();
/// let mut json_buf = String::new();
/// match &delta_content {
///     DeltaPart::Text { text } => buffer.push_str(text),
///     DeltaPart::InputJson { partial_json } => json_buf.push_str(partial_json),
///     DeltaPart::ToolCall { partial_json } => {
///         if let Some(s) = partial_json.as_str() {
///             json_buf.push_str(s);
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DeltaPart {
    /// Text delta — append to the existing text content.
    ///
    /// Emitted for text parts. Each delta contains a
    /// small fragment of the final text output.
    ///
    /// Serialized as `"type":"text_delta"` with a `"text"` field.
    #[serde(rename = "text_delta")]
    Text {
        /// The text fragment to append.
        ///
        /// Each fragment is a small piece of the complete text.
        /// Concatenate all fragments in order to reconstruct the
        /// full text content for this part.
        text: String,
    },

    /// Tool-call input delta — append to the JSON input.
    ///
    /// Emitted for tool-call parts. Each delta contains
    /// a fragment of the tool's JSON input, which should be
    /// concatenated and parsed once [`PartStop`](StreamEvent::PartStop) arrives.
    ///
    /// Serialized as `"type":"tool_call_delta"` with a `"partial_json"` field.
    #[serde(rename = "tool_call_delta")]
    ToolCall {
        /// A JSON value fragment to append to the tool input.
        ///
        /// Typically a string that forms part of the final JSON
        /// when concatenated with all prior deltas. Use
        /// [`serde_json::from_str`] on the concatenated result
        /// after the part completes.
        partial_json: Value,
    },

    /// Partial JSON input delta — append to the tool input buffer.
    ///
    /// Similar to [`ToolCall`](Self::ToolCall) but carries a raw
    /// string fragment rather than a JSON value. Concatenate all
    /// fragments and parse as JSON when the part ends.
    ///
    /// Serialized as `"type":"input_json_delta"` with a `"partial_json"` field.
    #[serde(rename = "input_json_delta")]
    InputJson {
        /// A raw JSON string fragment to append.
        ///
        /// Concatenate all `partial_json` strings from consecutive
        /// [`InputJson`](Self::InputJson) deltas for the same content
        /// part, then parse the combined string as JSON once
        /// [`PartStop`](StreamEvent::PartStop) is received.
        partial_json: String,
    },
}

// ==================================================
// StreamStopReason
// ==================================================

/// Reason why the model stopped generating tokens.
///
/// Streaming / API-level stop reason returned by the LLM
/// provider in the [`MessageDelta`] event. Differs from the
/// agent-level `StopReason` which is used in `TurnResult`.
///
/// Use [`should_continue_tool_loop`](Self::should_continue_tool_loop)
/// to decide whether the agent should execute tools and continue the
/// conversation loop.
///
/// # Parsing
///
/// Convert from/to API strings using [`from_api_str`](Self::from_api_str)
/// and [`to_api_str`](Self::to_api_str). The known values are:
/// `"tool_call"`, `"max_tokens"`, `"stop_sequence"`, and `"end_turn"`.
///
/// # Example
///
/// ```rust
/// use loopctl::stream::StreamStopReason;
///
/// let reason = StreamStopReason::from_api_str("tool_call").unwrap();
/// assert!(reason.should_continue_tool_loop());
///
/// let reason = StreamStopReason::from_api_str("end_turn").unwrap();
/// assert!(!reason.should_continue_tool_loop());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamStopReason {
    /// The model decided to invoke a tool.
    ///
    /// Indicates the agent should execute the tool and continue the
    /// conversation loop with the tool result. The tool-call parts
    /// in the response contain the invocation details.
    ///
    /// See [`should_continue_tool_loop`](Self::should_continue_tool_loop).
    ToolCall,

    /// The model reached the configured maximum token limit.
    ///
    /// The response was truncated. The caller may want to request
    /// continuation or increase the token budget.
    ///
    /// *Note: `LoopConfig` will be available once the builder module is complete.*
    MaxTokens,

    /// The model hit a configured stop sequence.
    ///
    /// The response ended because it matched one of the stop
    /// sequences provided in the request. Uncommon in
    /// typical agent usage.
    StopSequence,

    /// The model completed its turn naturally.
    ///
    /// The model finished generating its response without hitting
    /// any limits or invoking tools. Normal end-of-turn
    /// signal for non-tool responses.
    EndTurn,
}

impl StreamStopReason {
    /// Parse a stop reason from the API string representation.
    ///
    /// Called when deserializing [`MessageDeltaPayload`] events to
    /// convert the string stop reason into a typed enum. Returns
    /// `None` for unrecognized strings, which may indicate a new
    /// API version has introduced additional stop reasons.
    ///
    /// # Returns
    ///
    /// `Some(Self)` for known values, `None` for unrecognized strings.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::StreamStopReason;
    ///
    /// assert_eq!(StreamStopReason::from_api_str("tool_call"), Some(StreamStopReason::ToolCall));
    /// assert_eq!(StreamStopReason::from_api_str("unknown"), None);
    /// ```
    #[must_use]
    pub fn from_api_str(s: &str) -> Option<Self> {
        match s {
            "tool_call" => Some(Self::ToolCall),
            "max_tokens" => Some(Self::MaxTokens),
            "stop_sequence" => Some(Self::StopSequence),
            "end_turn" => Some(Self::EndTurn),
            _ => None,
        }
    }

    /// Convert to the API string representation.
    ///
    /// Called when serializing a [`StreamStopReason`] back into an
    /// API-compatible string (e.g. for logging or request building).
    /// The returned string is a static `&'static str` — no allocation
    /// is performed.
    ///
    /// # Returns
    ///
    /// A static string matching the API's expected value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::StreamStopReason;
    ///
    /// assert_eq!(StreamStopReason::ToolCall.to_api_str(), "tool_call");
    /// assert_eq!(StreamStopReason::EndTurn.to_api_str(), "end_turn");
    /// ```
    #[must_use]
    pub fn to_api_str(self) -> &'static str {
        match self {
            Self::ToolCall => "tool_call",
            Self::MaxTokens => "max_tokens",
            Self::StopSequence => "stop_sequence",
            Self::EndTurn => "end_turn",
        }
    }

    /// Check whether the agent should continue the tool-execution loop.
    ///
    /// Called after each streaming response completes to decide if the
    /// agent should execute the requested tools and send another request.
    /// Returns `true` only for [`ToolCall`](Self::ToolCall), which means
    /// the model has emitted one or more tool-call parts that
    /// need to be executed.
    ///
    /// # Returns
    ///
    /// `true` if the stop reason is [`ToolCall`](Self::ToolCall),
    /// `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::StreamStopReason;
    ///
    /// let reason = StreamStopReason::ToolCall;
    /// if reason.should_continue_tool_loop() {
    ///     // Execute tools and continue the conversation
    /// }
    /// ```
    #[must_use]
    pub fn should_continue_tool_loop(self) -> bool {
        matches!(self, Self::ToolCall)
    }
}

// ==================================================
// MessageDelta
// ==================================================

/// A delta update for the message, typically emitted at the end of the stream.
///
/// Carries the final [`StreamStopReason`] (why the model stopped) and
/// the cumulative [`Usage`] statistics for the entire request. Emitted
/// after all parts are complete but before [`MessageStop`](StreamEvent::MessageStop).
///
/// # Example
///
/// ```rust
/// use loopctl::stream::{MessageDelta, MessageDeltaPayload, Usage};
///
/// let delta = MessageDelta {
///     delta: MessageDeltaPayload { stop_reason: Some("end_turn".to_string()) },
///     usage: Some(Usage::new(100, 50)),
/// };
/// ```
///
/// # Relationship to [`MessageDeltaPayload`]
///
/// The [`delta`](Self::delta) field contains the stop reason string,
/// while the [`usage`](Self::usage) field carries token counts. Together
/// they provide the final summary of the streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    /// The delta details containing the stop reason.
    ///
    /// See [`MessageDeltaPayload`] for the payload structure. Use
    /// [`StreamStopReason::from_api_str`] to parse the stop reason string
    /// into a typed enum.
    pub delta: MessageDeltaPayload,

    /// Token usage statistics for this request.
    ///
    /// `Some` when the API reports usage; `None` if usage data
    /// is not available or not yet received. See [`Usage`].
    /// Typically populated in the final `MessageDelta` event
    /// and reflects cumulative token consumption for the entire request.
    pub usage: Option<Usage>,
}

// ==================================================
// MessageDeltaPayload
// ==================================================

/// The delta details within a [`MessageDelta`] event.
///
/// Contains the stop reason string that explains why the model
/// finished generating. Parse it with
/// [`StreamStopReason::from_api_str`] to get a typed value.
///
/// # Example
///
/// ```rust
/// use loopctl::stream::{MessageDeltaPayload, StreamStopReason};
///
/// let delta = MessageDeltaPayload { stop_reason: Some("tool_call".to_string()) };
/// if let Some(s) = &delta.stop_reason {
///     let reason = StreamStopReason::from_api_str(s);
///     assert_eq!(reason, Some(StreamStopReason::ToolCall));
/// }
/// ```
///
/// # Known Values
///
/// The API may return `"tool_call"`, `"max_tokens"`, `"stop_sequence"`,
/// or `"end_turn"`. Unrecognized values will cause
/// [`StreamStopReason::from_api_str`] to return `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaPayload {
    /// Why the model stopped generating, as a raw string.
    ///
    /// Use [`StreamStopReason::from_api_str`] to parse this into
    /// a typed enum. May be `None` if the API did not provide a
    /// stop reason (e.g. on error or incomplete responses).
    pub stop_reason: Option<String>,
}

// ==================================================
// Usage
// ==================================================

/// Token usage statistics from an API response.
///
/// Tracks input and output token counts for a single streaming
/// request. Returned in the [`MessageDelta`] event at the end of
/// the stream. Use [`total_tokens`](Self::total_tokens) for the
/// combined count.
///
/// # Default
///
/// The [`Default`] implementation produces zeroed counters,
/// which is useful for initializing accumulators before the first
/// usage data arrives.
///
/// # Example
///
/// ```rust
/// use loopctl::stream::Usage;
///
/// let usage = Usage::new(150, 75);
/// assert_eq!(usage.input_tokens, 150);
/// assert_eq!(usage.output_tokens, 75);
/// assert_eq!(usage.total_tokens(), 225);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Number of tokens in the input prompt.
    ///
    /// Includes the system prompt, conversation history, and any
    /// tool definitions sent with the request. Defaults to `0`
    /// when constructed via [`Default::default`].
    pub input_tokens: u32,

    /// Number of tokens in the output completion.
    ///
    /// Includes all generated text and tool-call parts produced by
    /// the model. Defaults to `0` when constructed via
    /// [`Default::default`].
    pub output_tokens: u32,
}

impl Usage {
    /// Create a new usage instance with the given token counts.
    ///
    /// Called when constructing usage data from API response fields
    /// or when building test fixtures. For a zeroed instance, use
    /// [`Default::default`] instead.
    ///
    /// # Parameters
    ///
    /// - `input_tokens` — Tokens consumed by the prompt.
    /// - `output_tokens` — Tokens produced by the model.
    ///
    /// # Returns
    ///
    /// A [`Usage`] instance with the specified counts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::Usage;
    ///
    /// let usage = Usage::new(100, 50);
    /// assert_eq!(usage.total_tokens(), 150);
    /// ```
    #[must_use]
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }

    /// Total tokens consumed (input + output).
    ///
    /// Convenience method for cost estimation and logging. Sums
    /// [`input_tokens`](Self::input_tokens) and
    /// [`output_tokens`](Self::output_tokens).
    ///
    /// # Returns
    ///
    /// The sum of input and output token counts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::Usage;
    ///
    /// let usage = Usage::new(100, 50);
    /// assert_eq!(usage.total_tokens(), 150);
    /// ```
    #[must_use]
    pub fn total_tokens(self) -> u32 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

// ==================================================
// Stream accumulator
// ==================================================

/// Accumulates streaming events into a complete [`Message`].
///
/// Stateful builder that tracks the progress of a streaming
/// response as [`StreamEvent`]s arrive and assembles the final
/// [`Message`] once all events have been processed.
///
/// Call [`process`](Self::process) for each event as it arrives, then
/// call [`build`](Self::build) to consume the accumulator and produce
/// the final message. Token usage can be retrieved at any time via
/// [`usage`](Self::usage).
///
/// # State Machine
///
/// ```text
///                            MessageStart
/// ┌───────┐  ────────────▶  ┌─────────────┐
/// │ Idle  │                 │   Started   │
/// └───────┘                 └──────┬──────┘
///                        PartStart │ PartEnd
///                            ┌─────▼──────┐
///                            │ Receiving  │
///                            │   Part     │
///                            └─────┬──────┘
///                     IndexedDelta │
///                            (accumulate)
///                                  │
///                     MessageDelta │
///                       ┌──────────▼──────────┐
///                       │      Complete       │──▶ build()
///                       └─────────────────────┘
/// ```
///
/// # Example
///
/// ```rust
/// use loopctl::stream::{StreamAccumulator, StreamEvent};
///
/// let mut acc = StreamAccumulator::new();
///
/// for event in std::iter::empty::<StreamEvent>() {
///     acc.process(&event).unwrap();
/// }
///
/// let _usage = acc.usage();
/// let message = acc.build();
/// ```
///
/// # Default
///
/// The [`Default`] implementation produces an empty accumulator with
/// no parts, no text, no tool data, and no usage — equivalent to
/// calling [`new`](Self::new).
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    /// Accumulated parts from completed part sequences.
    ///
    /// Each entry is a fully assembled [`MessagePart`] produced when
    /// a [`PartStop`](StreamEvent::PartStop) event
    /// finalizes the current part. These are the parts that will
    /// appear in the final [`Message`] returned by [`build`](Self::build).
    parts: Vec<MessagePart>,

    /// Current text being accumulated from [`DeltaPart::Text`] deltas.
    ///
    /// Grows as text deltas arrive. Cleared and flushed into [`parts`](Self::parts)
    /// as a [`MessagePart::Text`] when
    /// [`PartStop`](StreamEvent::PartStop) is received.
    current_text: String,

    /// The tool-call ID being accumulated for the current tool part.
    ///
    /// Populated from [`PartStart`] when the part
    /// is a tool-call invocation. Cleared on the next
    /// [`PartStart`](StreamEvent::PartStart).
    /// Used to correlate the tool-call part with its result.
    current_tool_id: String,

    /// The tool name being accumulated for the current tool part.
    ///
    /// Populated from [`PartStart`] when the part
    /// is a tool-call invocation. Cleared on the next
    /// [`PartStart`](StreamEvent::PartStart).
    /// Determines whether a part is text or tool-call when
    /// [`PartStop`](StreamEvent::PartStop) arrives.
    current_tool_name: String,

    /// The raw JSON string being accumulated for the current tool input.
    ///
    /// Built up from [`DeltaPart::InputJson`] and
    /// [`DeltaPart::InputJson`] deltas, then parsed into a
    /// [`Value`](serde_json::Value) when
    /// [`PartStop`](StreamEvent::PartStop) is received.
    /// If parsing fails, defaults to an empty JSON object `{}`.
    current_tool_input: String,

    /// Index of the part currently being accumulated.
    ///
    /// `None` before the first [`PartStart`] event, then
    /// `Some(index)` for the remainder of the stream. Used to
    /// track which part is in progress.
    current_index: Option<usize>,

    /// The model name that produced this response.
    ///
    /// Extracted from the [`MessageStart`](StreamEvent::MessageStart)
    /// event. `None` until that event is processed. Can be used for
    /// logging or routing after the stream completes.
    model: Option<String>,

    /// Token usage statistics from the response.
    ///
    /// Populated from the [`MessageDelta`](StreamEvent::MessageDelta)
    /// event. `None` until that event is processed. Access via
    /// [`usage`](Self::usage) after processing.
    usage: Option<Usage>,
}

impl StreamAccumulator {
    /// Create a new accumulator in the initial state.
    ///
    /// Returns a fresh accumulator ready to receive
    /// [`StreamEvent`]s. Equivalent to [`default`](Self::default).
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::StreamAccumulator;
    ///
    /// let mut acc = StreamAccumulator::new();
    /// // acc.process(&event).unwrap();
    /// let message = acc.build();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a single stream event and update internal state.
    ///
    /// Called for each [`StreamEvent`] as it arrives from the SSE
    /// connection. The accumulator tracks the current part
    /// being built and flushes completed parts into the internal
    /// list when [`PartStop`](StreamEvent::PartStop)
    /// is received.
    ///
    /// # Arguments
    ///
    /// - `event` — A reference to the [`StreamEvent`] to process.
    ///
    /// # Event Handling
    ///
    /// - [`MessageStart`](StreamEvent::MessageStart) — Captures the model name.
    /// - [`PartStart`](StreamEvent::PartStart) — Resets buffers,
    ///   captures tool ID/name if present.
    /// - [`IndexedDelta`](StreamEvent::IndexedDelta) — Appends text or
    ///   JSON fragments to the appropriate buffer.
    /// - [`PartStop`](StreamEvent::PartStop) — Flushes the current
    ///   buffer into an accumulated [`MessagePart`].
    /// - [`MessageDelta`](StreamEvent::MessageDelta) — Captures usage statistics.
    /// - [`MessageStop`](StreamEvent::MessageStop) / [`Ping`](StreamEvent::Ping) — Ignored.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::{StreamAccumulator, StreamEvent, MessageStart, MessageMetadata,
    ///     PartStart, IndexedDelta, DeltaPart};
    ///
    /// let mut acc = StreamAccumulator::new();
    /// let start = MessageStart {
    ///     message: MessageMetadata {
    ///         id: "msg_1".to_string(),
    ///         role: "assistant".to_string(),
    ///         model: "test".to_string(),
    ///     },
    /// };
    /// acc.process(&StreamEvent::MessageStart(start)).unwrap();
    /// acc.process(&StreamEvent::PartStart(PartStart { index: 0, part: None })).unwrap();
    /// let delta = IndexedDelta {
    ///     index: 0,
    ///     delta: DeltaPart::Text { text: "hi".to_string() },
    /// };
    /// acc.process(&StreamEvent::IndexedDelta(delta)).unwrap();
    /// acc.process(&StreamEvent::MessageStop).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::InvalidToolInputJson`] if the accumulated
    /// tool-call JSON cannot be parsed when a [`PartStop`](StreamEvent::PartStop)
    /// event is processed.
    pub fn process(&mut self, event: &StreamEvent) -> Result<(), StreamError> {
        match event {
            StreamEvent::MessageStart(msg_start) => {
                self.model = Some(msg_start.message.model.clone());
                Ok(())
            }
            StreamEvent::PartStart(part_start) => {
                self.current_index = Some(part_start.index);
                self.current_text.clear();
                self.current_tool_id.clear();
                self.current_tool_name.clear();
                self.current_tool_input.clear();

                if let Some(MessagePart::ToolCall { id, name, .. }) = &part_start.part {
                    self.current_tool_id.clone_from(id);
                    self.current_tool_name.clone_from(name);
                }
                Ok(())
            }
            StreamEvent::IndexedDelta(delta) => {
                debug_assert_eq!(
                    self.current_index,
                    Some(delta.index),
                    "IndexedDelta index mismatch: expected {:?}, got {}",
                    self.current_index,
                    delta.index,
                );
                match &delta.delta {
                    DeltaPart::Text { text } => {
                        self.current_text.push_str(text);
                    }
                    DeltaPart::InputJson { partial_json } => {
                        self.current_tool_input.push_str(partial_json);
                    }
                    DeltaPart::ToolCall { partial_json } => {
                        if let Some(s) = partial_json.as_str() {
                            self.current_tool_input.push_str(s);
                        }
                    }
                }
                Ok(())
            }
            StreamEvent::PartStop => {
                if !self.current_text.is_empty() {
                    self.parts.push(MessagePart::text(&self.current_text));
                } else if !self.current_tool_name.is_empty() {
                    let input: Value = if self.current_tool_input.is_empty() {
                        Value::Object(serde_json::Map::new())
                    } else {
                        serde_json::from_str(&self.current_tool_input).map_err(|e| {
                            StreamError::InvalidToolInputJson(
                                e,
                                std::mem::take(&mut self.current_tool_input),
                            )
                        })?
                    };
                    self.parts.push(MessagePart::tool_call(
                        &self.current_tool_id,
                        &self.current_tool_name,
                        input,
                    ));
                }
                self.current_text.clear();
                Ok(())
            }
            StreamEvent::MessageDelta(delta) => {
                self.usage = delta.usage;
                Ok(())
            }
            StreamEvent::MessageStop | StreamEvent::Ping => Ok(()),
        }
    }

    /// Returns a slice of the accumulated [`MessagePart`]s so far.
    ///
    /// Unlike [`build`](Self::build), this does not consume the
    /// accumulator. Useful for checking whether any content has been
    /// received before the stream times out.
    #[must_use]
    pub fn peek_parts(&self) -> &[MessagePart] {
        &self.parts
    }

    /// Consume the accumulator and produce the final [`Message`].
    ///
    /// Called after all [`StreamEvent`]s have been processed via
    /// [`process`](Self::process). Returns a [`Message`] with role
    /// [`Role::Assistant`](crate::message::Role::Assistant) and the
    /// accumulated [`MessagePart`]s.
    ///
    /// # Returns
    ///
    /// A complete [`Message`] assembled from all processed events.
    /// If no parts were received, the message will have
    /// an empty `content` vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::StreamAccumulator;
    /// use loopctl::message::Role;
    ///
    /// let acc = StreamAccumulator::new();
    /// let message = acc.build();
    /// assert_eq!(message.role, Role::Assistant);
    /// ```
    #[must_use]
    pub fn build(self) -> Message {
        Message {
            role: crate::message::Role::Assistant,
            parts: self.parts,
        }
    }

    /// Get the accumulated token usage, if available.
    ///
    /// Returns the [`Usage`] statistics captured from the
    /// [`MessageDelta`](StreamEvent::MessageDelta) event. Returns
    /// `None` if no `MessageDelta` event has been processed yet.
    /// Safe to call at any time — even before the stream completes.
    ///
    /// # Returns
    ///
    /// A reference to the [`Usage`] data, or `None` if unavailable.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::stream::{StreamAccumulator, Usage};
    ///
    /// let mut acc = StreamAccumulator::new();
    /// if let Some(usage) = acc.usage() {
    ///     let _total = usage.total_tokens();
    /// }
    /// ```
    #[must_use]
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }
}

// ==================================================
// Tests
// ==================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that [`StreamStopReason::from_api_str`] parses all known API
    /// strings and returns `None` for unknown values.
    ///
    /// Tests `"tool_call"`, `"max_tokens"`, `"end_turn"`, `"stop_sequence"`,
    /// and `"unknown"`.
    #[test]
    fn test_stream_stop_reason_from_api_str() {
        assert_eq!(
            StreamStopReason::from_api_str("tool_call"),
            Some(StreamStopReason::ToolCall)
        );
        assert_eq!(
            StreamStopReason::from_api_str("max_tokens"),
            Some(StreamStopReason::MaxTokens)
        );
        assert_eq!(
            StreamStopReason::from_api_str("end_turn"),
            Some(StreamStopReason::EndTurn)
        );
        assert_eq!(
            StreamStopReason::from_api_str("stop_sequence"),
            Some(StreamStopReason::StopSequence)
        );
        assert_eq!(StreamStopReason::from_api_str("unknown"), None);
    }

    /// Verify that [`StreamStopReason::to_api_str`] produces the correct
    /// static string for each variant.
    ///
    /// Ensures the round-trip `from_api_str(s) == Some(v)` implies
    /// `v.to_api_str() == s`.
    #[test]
    fn test_stream_stop_reason_to_api_str() {
        assert_eq!(StreamStopReason::ToolCall.to_api_str(), "tool_call");
        assert_eq!(StreamStopReason::MaxTokens.to_api_str(), "max_tokens");
        assert_eq!(StreamStopReason::EndTurn.to_api_str(), "end_turn");
        assert_eq!(StreamStopReason::StopSequence.to_api_str(), "stop_sequence");
    }

    /// Verify that [`StreamStopReason::should_continue_tool_loop`] returns `true`
    /// only for [`ToolCall`](StreamStopReason::ToolCall).
    ///
    /// [`EndTurn`](StreamStopReason::EndTurn) and [`MaxTokens`](StreamStopReason::MaxTokens)
    /// should both return `false`.
    #[test]
    fn test_stream_stop_reason_should_continue() {
        assert!(StreamStopReason::ToolCall.should_continue_tool_loop());
        assert!(!StreamStopReason::EndTurn.should_continue_tool_loop());
        assert!(!StreamStopReason::MaxTokens.should_continue_tool_loop());
    }

    /// Verify [`Usage::new`] stores token counts and computes the total.
    ///
    /// Asserts that [`Usage::input_tokens`] and [`Usage::output_tokens`] are
    /// stored as provided, and that [`Usage::total_tokens`] returns their sum.
    #[test]
    fn test_usage() {
        let usage = Usage::new(100, 50);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens(), 150);
    }

    /// Verify that [`Usage::default`] produces zeroed counters.
    ///
    /// [`Usage::total_tokens`] should be `0` when both fields are `0`.
    #[test]
    fn test_usage_default() {
        let usage = Usage::default();
        assert_eq!(usage.total_tokens(), 0);
    }

    /// Verify that [`StreamAccumulator`] correctly assembles a text-only
    /// streaming response into a [`Message`].
    ///
    /// Feeds a full event sequence: [`MessageStart`](StreamEvent::MessageStart) →
    /// [`PartStart`](StreamEvent::PartStart) → two
    /// [`IndexedDelta`](StreamEvent::IndexedDelta)s →
    /// [`PartStop`](StreamEvent::PartStop) →
    /// [`MessageDelta`](StreamEvent::MessageDelta) →
    /// [`MessageStop`](StreamEvent::MessageStop).
    ///
    /// Asserts the final message has [`Role::Assistant`](crate::message::Role::Assistant),
    /// one part with the concatenated text "Hello world", and correct usage.
    #[test]
    fn test_accumulator_text_message() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::MessageStart(MessageStart {
            message: MessageMetadata {
                id: "msg_1".to_string(),
                role: "assistant".to_string(),
                model: "test-model".to_string(),
            },
        }))
        .unwrap();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 0,
            part: None,
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text {
                text: "Hello".to_string(),
            },
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text {
                text: " world".to_string(),
            },
        }))
        .unwrap();
        acc.process(&StreamEvent::PartStop).unwrap();
        acc.process(&StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("end_turn".to_string()),
            },
            usage: Some(Usage::new(10, 5)),
        }))
        .unwrap();
        acc.process(&StreamEvent::MessageStop).unwrap();

        let msg = acc.build();
        assert_eq!(msg.role, crate::message::Role::Assistant);
        assert_eq!(msg.parts.len(), 1);
        assert_eq!(msg.parts[0].as_text(), Some("Hello world"));
    }

    /// Verify that [`StreamAccumulator`] correctly assembles a tool-call
    /// part from streaming events.
    ///
    /// Feeds [`PartStart`](StreamEvent::PartStart) with a
    /// [`ToolCall`](MessagePart::ToolCall) seed, then an [`InputJson`](DeltaPart::InputJson)
    /// delta, then [`PartStop`](StreamEvent::PartStop). Asserts the
    /// resulting message contains a tool-call part.
    #[test]
    fn test_accumulator_tool_call() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::tool_call(
                "tool_1",
                "read_file",
                Value::Object(serde_json::Map::new()),
            )),
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::InputJson {
                partial_json: r#"{"path":"/tmp/test"}"#.to_string(),
            },
        }))
        .unwrap();
        acc.process(&StreamEvent::PartStop).unwrap();

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 1);
        assert!(msg.parts[0].is_tool_call());
    }

    /// Verify that [`StreamAccumulator::build`] on a fresh accumulator
    /// produces a [`Message`] with an empty content vector.
    ///
    /// No events processed means no parts assembled.
    #[test]
    fn test_accumulator_empty() {
        let acc = StreamAccumulator::new();
        let msg = acc.build();
        assert_eq!(msg.parts.len(), 0);
    }

    /// Verify that [`StreamAccumulator::usage`] returns the [`Usage`] data
    /// from a [`MessageDelta`](StreamEvent::MessageDelta) event.
    ///
    /// Feeds a single `MessageDelta` with known token counts and asserts
    /// [`Usage::total_tokens`] returns the expected sum.
    #[test]
    fn test_accumulator_usage() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload { stop_reason: None },
            usage: Some(Usage::new(100, 50)),
        }))
        .unwrap();
        assert_eq!(acc.usage().unwrap().total_tokens(), 150);
    }

    /// Verify that all [`StreamEvent`] variants can be constructed.
    ///
    /// Constructs [`Ping`](StreamEvent::Ping), [`MessageStop`](StreamEvent::MessageStop),
    /// and [`PartStop`](StreamEvent::PartStop) to confirm no compile-time
    /// regressions in the enum definition.
    #[test]
    fn test_stream_event_variants() {
        // Just verify all variants can be constructed
        let _ = StreamEvent::Ping;
        let _ = StreamEvent::MessageStop;
        let _ = StreamEvent::PartStop;
    }

    /// Verify that [`StreamAccumulator::process`] returns
    /// [`StreamError::InvalidToolInputJson`] when the accumulated tool-call
    /// JSON is malformed at [`PartStop`](StreamEvent::PartStop).
    #[test]
    fn test_accumulator_invalid_tool_json() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::tool_call(
                "tool_1",
                "bad_tool",
                Value::Object(serde_json::Map::new()),
            )),
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::InputJson {
                partial_json: "not valid json{".to_string(),
            },
        }))
        .unwrap();

        let result = acc.process(&StreamEvent::PartStop);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match &err {
            StreamError::InvalidToolInputJson(_, raw) => {
                assert_eq!(raw, "not valid json{");
            }
        }
    }

    /// Verify that a tool-call part with no delta input defaults to an
    /// empty JSON object `{}` rather than erroring.
    #[test]
    fn test_accumulator_tool_call_empty_input() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::tool_call(
                "tool_1",
                "no_args",
                Value::Object(serde_json::Map::new()),
            )),
        }))
        .unwrap();
        acc.process(&StreamEvent::PartStop).unwrap();

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 1);
        assert!(msg.parts[0].is_tool_call());
    }
}

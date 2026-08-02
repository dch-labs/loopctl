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
//! - **`handler`** — `handler::StreamHandler` with retry, timeout,
//!   and fallback for resilient streaming.
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

#[cfg(feature = "streaming")]
pub mod handler;
pub mod rate_limit;

#[cfg(feature = "streaming")]
pub use handler::{DetectedRateLimit, RateLimitConfig, RateLimitKind};
pub use rate_limit::{RateLimiter, TokenBucket};

/// Errors that can occur during stream event processing.
///
/// Returned by [`StreamAccumulator::process`] when an event cannot be
/// handled correctly — for example, when accumulated tool-call JSON
/// is malformed at [`PartStop`](StreamEvent::PartStop) time.
#[derive(Debug)]
#[non_exhaustive]
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

/// A delta (incremental update) for content within a streaming response.
///
/// Each variant represents a different kind of incremental data that
/// the API sends. Consumers should append the payload to the
/// appropriate in-progress part.
///
/// `#[non_exhaustive]` so the framework can add content kinds (e.g.
/// `Image`/`Audio`) in a later minor release without breaking downstream
/// `match`es. Downstream code that matches on `DeltaPart` MUST include a
/// wildcard arm.
///
/// # Variants
///
/// - [`Text`](Self::Text) — Append to the text buffer for text parts.
/// - [`ToolCall`](Self::ToolCall) — Append to the JSON buffer for tool-call parts.
/// - [`InputJson`](Self::InputJson) — Append to the JSON buffer (raw string form).
/// - [`Thinking`](Self::Thinking) — Append to the reasoning buffer for separate
///   display; not part of the assistant's visible text.
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
///     DeltaPart::Thinking { .. } => {
///         // Reasoning is delivered via on_thinking_delta; not accumulated here.
///     }
///     _ => {}
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
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

    /// Thinking/reasoning delta — append to a reasoning buffer for separate
    /// display. NOT part of the assistant's visible text.
    ///
    /// Emitted by reasoning models (Claude extended-thinking, DeepSeek-R1,
    /// OpenAI o-series). Stream-only: the [`StreamAccumulator`] does NOT carry
    /// reasoning into the built [`Message`]; consume
    /// it via
    /// [`on_thinking_delta`](crate::observer::LoopObserver::on_thinking_delta).
    /// An empty `text` signals redacted reasoning (e.g. Anthropic
    /// `redacted_thinking`); render a placeholder rather than the empty string.
    ///
    /// Serialized as `"type":"thinking_delta"` with a `"text"` field.
    #[serde(rename = "thinking_delta")]
    Thinking {
        /// The reasoning text fragment to append.
        ///
        /// Concatenate in arrival order per turn to reconstruct the full
        /// reasoning trace. Empty string when the reasoning is redacted
        /// (the provider withheld the content); consumers should render a
        /// placeholder, not the empty string.
        text: String,
    },
}

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
#[non_exhaustive]
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
    /// The response was truncated because the model produced `max_tokens` of
    /// output before finishing. The caller may want to request continuation or
    /// raise the provider's max-tokens limit.
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
    /// Parse a stop reason from the provider's API string representation.
    ///
    /// Called in two places: when deserializing [`MessageDeltaPayload`]
    /// streaming events, and when each provider's `build_response` maps its
    /// native finish/stop field on the non-streaming path. Returns `None` for
    /// unrecognized strings, which may indicate a new API version has
    /// introduced additional stop reasons.
    ///
    /// `"tool_use"` is accepted as an alias for `"tool_call"` because Anthropic
    /// reports a tool-invocation stop reason as `"tool_use"` while OpenAI uses
    /// `"tool_calls"` (handled directly in the OpenAI provider) — both map to
    /// [`ToolCall`](Self::ToolCall).
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
    /// assert_eq!(StreamStopReason::from_api_str("tool_use"), Some(StreamStopReason::ToolCall));
    /// assert_eq!(StreamStopReason::from_api_str("unknown"), None);
    /// ```
    #[must_use]
    pub fn from_api_str(s: &str) -> Option<Self> {
        match s {
            "tool_call" | "tool_use" => Some(Self::ToolCall),
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
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
    /// Fully assembled parts flushed by [`PartStop`](StreamEvent::PartStop).
    ///
    /// Each entry is a finished [`MessagePart`] produced when a
    /// [`PartStop`](StreamEvent::PartStop) closes one of the entries in
    /// [`open`](Self::open). These are the parts returned by
    /// [`build`](Self::build).
    completed: Vec<MessagePart>,

    /// Parts currently receiving deltas, in [`PartStart`] arrival order.
    ///
    /// A provider may keep several parts open at once — for example,
    /// OpenAI streams parallel tool calls by interleaving argument
    /// fragments across distinct `index` values and only closing them
    /// all at the terminal `finish_reason`. Each
    /// [`PartStart`](StreamEvent::PartStart) pushes a new slot;
    /// [`IndexedDelta`](StreamEvent::IndexedDelta) routes to the slot
    /// whose `index` matches; [`PartStop`](StreamEvent::PartStop)
    /// flushes the oldest slot still open.
    open: Vec<OpenPart>,

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

/// Which lane an in-progress [`OpenPart`] is accumulating.
///
/// Distinguishes plain assistant text from a tool-call invocation so
/// [`PartStop`](StreamEvent::PartStop) knows which buffer to flush and
/// which [`MessagePart`] shape to build. Decided once, at
/// [`PartStart`](StreamEvent::PartStart) time, from the carried part.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum OpenPartKind {
    /// Assistant text, reconstructed from `delta.content` fragments.
    ///
    /// The default lane: a [`PartStart`](StreamEvent::PartStart) that
    /// carries no tool-call part opens a text slot, and its buffered
    /// string becomes a [`MessagePart::Text`] on close.
    #[default]
    Text,

    /// A tool-call invocation, reconstructed from `function.arguments`
    /// fragments.
    ///
    /// The slot latches the tool `id` and `name` from
    /// [`PartStart::part`](crate::stream::PartStart::part) at open time
    /// and accumulates the raw JSON arguments string; on close it parses
    /// that string into the tool-call [`MessagePart::ToolCall`] input.
    Tool,
}

/// A single in-progress part being accumulated between open and close.
///
/// Holds the buffers for one lane — assistant text or a tool call —
/// from the [`PartStart`](StreamEvent::PartStart) that opens it to the
/// [`PartStop`](StreamEvent::PartStop) that flushes it. The
/// [`StreamAccumulator`] keeps a [`Vec`] of these so that providers
/// which leave several parts open at once (OpenAI's interleaved parallel
/// tool calls) accumulate each one independently.
#[derive(Debug, Default)]
struct OpenPart {
    /// The `index` this slot was opened with.
    ///
    /// Copied from [`PartStart::index`](crate::stream::PartStart::index)
    /// at open time and used to route each
    /// [`IndexedDelta`](StreamEvent::IndexedDelta) fragment to the slot
    /// it belongs to. Stable for the lifetime of the slot.
    index: usize,

    /// Whether this slot accumulates assistant text or a tool call.
    ///
    /// Set once when the slot opens and never mutated; it selects which
    /// buffer the deltas append to and which [`MessagePart`] variant
    /// [`PartStop`](StreamEvent::PartStop) builds from the slot.
    kind: OpenPartKind,

    /// Buffered assistant text for a text-lane slot.
    ///
    /// Grown one fragment at a time by [`DeltaPart::Text`] deltas and
    /// flushed into a [`MessagePart::Text`] on close. Empty and unused
    /// for tool-call slots.
    text: String,

    /// Server-assigned identifier of the tool call.
    ///
    /// Latched from [`PartStart::part`](crate::stream::PartStart::part)
    /// when the slot opens as a tool call, then carried onto the built
    /// [`MessagePart::ToolCall`] so the host can match the eventual
    /// tool result back to this call. Empty for text slots.
    tool_id: String,

    /// Name of the tool being invoked.
    ///
    /// Latched from [`PartStart::part`](crate::stream::PartStart::part)
    /// at open time and used both to detect that this is a tool-call
    /// slot (non-empty) and to label the built
    /// [`MessagePart::ToolCall`]. Empty for text slots.
    tool_name: String,

    /// Raw JSON arguments string for a tool-call slot.
    ///
    /// Grown fragment by fragment by [`DeltaPart::InputJson`] and
    /// [`DeltaPart::ToolCall`] deltas and parsed into the
    /// [`MessagePart::ToolCall`] input when the slot closes. Empty for
    /// text slots.
    tool_input: String,
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
    /// connection. The accumulator keeps a [`Vec`] of in-progress parts
    /// so that providers which leave several parts open at once — OpenAI
    /// interleaves parallel tool calls — accumulate each one
    /// independently. Each [`PartStart`](StreamEvent::PartStart) opens a
    /// new slot keyed by its `index`; each
    /// [`IndexedDelta`](StreamEvent::IndexedDelta) routes to the matching
    /// slot; each [`PartStop`](StreamEvent::PartStop) flushes the oldest
    /// open slot (FIFO by `PartStart` arrival order). Providers that only
    /// ever hold one slot open (Anthropic, Gemini) behave exactly as on a
    /// single-slot accumulator.
    ///
    /// # Arguments
    ///
    /// - `event` — A reference to the [`StreamEvent`] to process.
    ///
    /// # Event Handling
    ///
    /// - [`MessageStart`](StreamEvent::MessageStart) — Captures the model name.
    /// - [`PartStart`](StreamEvent::PartStart) — Opens a new in-progress
    ///   slot for the given `index` (text or tool call, decided by the
    ///   carried part). Several slots may be open at once.
    /// - [`IndexedDelta`](StreamEvent::IndexedDelta) — Routes the
    ///   fragment to the open slot whose `index` matches and appends it
    ///   to that slot's buffer.
    /// - [`PartStop`](StreamEvent::PartStop) — Flushes the oldest
    ///   still-open slot into a finished [`MessagePart`] and drops it.
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
                let kind = match &part_start.part {
                    Some(MessagePart::ToolCall { .. }) => OpenPartKind::Tool,
                    _ => OpenPartKind::Text,
                };
                let mut slot = OpenPart {
                    index: part_start.index,
                    kind,
                    ..Default::default()
                };
                if let Some(MessagePart::ToolCall { id, name, .. }) = &part_start.part {
                    slot.tool_id.clone_from(id);
                    slot.tool_name.clone_from(name);
                }
                self.open.push(slot);
                Ok(())
            }
            StreamEvent::IndexedDelta(delta) => {
                let Some(slot) = self.open.iter_mut().find(|s| s.index == delta.index) else {
                    return Ok(());
                };
                match &delta.delta {
                    DeltaPart::Text { text } => {
                        slot.text.push_str(text);
                    }
                    DeltaPart::InputJson { partial_json } => {
                        slot.tool_input.push_str(partial_json);
                    }
                    DeltaPart::ToolCall { partial_json } => {
                        if let Some(s) = partial_json.as_str() {
                            slot.tool_input.push_str(s);
                        }
                    }
                    DeltaPart::Thinking { .. } => {}
                }
                Ok(())
            }
            StreamEvent::PartStop => {
                let Some(slot) = self.open.first_mut() else {
                    return Ok(());
                };
                let flushed = match slot.kind {
                    OpenPartKind::Text if !slot.text.is_empty() => {
                        Some(MessagePart::text(std::mem::take(&mut slot.text)))
                    }
                    OpenPartKind::Tool if !slot.tool_name.is_empty() => {
                        let input: Value = if slot.tool_input.is_empty() {
                            Value::Object(serde_json::Map::new())
                        } else {
                            serde_json::from_str(&slot.tool_input).map_err(|e| {
                                StreamError::InvalidToolInputJson(
                                    e,
                                    std::mem::take(&mut slot.tool_input),
                                )
                            })?
                        };
                        Some(MessagePart::tool_call(
                            std::mem::take(&mut slot.tool_id),
                            std::mem::take(&mut slot.tool_name),
                            input,
                        ))
                    }
                    _ => None,
                };
                self.open.remove(0);
                if let Some(part) = flushed {
                    self.completed.push(part);
                }
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
        &self.completed
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
            parts: self.completed,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_stop_reason_from_api_str() {
        assert_eq!(
            StreamStopReason::from_api_str("tool_call"),
            Some(StreamStopReason::ToolCall)
        );
        assert_eq!(
            StreamStopReason::from_api_str("tool_use"),
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

    #[test]
    fn test_stream_stop_reason_to_api_str() {
        assert_eq!(StreamStopReason::ToolCall.to_api_str(), "tool_call");
        assert_eq!(StreamStopReason::MaxTokens.to_api_str(), "max_tokens");
        assert_eq!(StreamStopReason::EndTurn.to_api_str(), "end_turn");
        assert_eq!(StreamStopReason::StopSequence.to_api_str(), "stop_sequence");
    }

    #[test]
    fn test_stream_stop_reason_should_continue() {
        assert!(StreamStopReason::ToolCall.should_continue_tool_loop());
        assert!(!StreamStopReason::EndTurn.should_continue_tool_loop());
        assert!(!StreamStopReason::MaxTokens.should_continue_tool_loop());
    }

    #[test]
    fn test_usage() {
        let usage = Usage::new(100, 50);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens(), 150);
    }

    #[test]
    fn test_usage_default() {
        let usage = Usage::default();
        assert_eq!(usage.total_tokens(), 0);
    }

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

    #[test]
    fn test_accumulator_interleaved_tool_calls() {
        // Reproduces OpenAI's parallel-tool-call wire shape: two tool
        // calls opened up front, argument fragments interleaved across
        // their indices, both closed by bare PartStops at the end.
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::tool_call(
                "call_a",
                "echo",
                Value::Object(serde_json::Map::new()),
            )),
        }))
        .unwrap();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 1,
            part: Some(MessagePart::tool_call(
                "call_b",
                "search",
                Value::Object(serde_json::Map::new()),
            )),
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::InputJson {
                partial_json: r#"{"msg":"#.to_string(),
            },
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 1,
            delta: DeltaPart::InputJson {
                partial_json: r#"{"q":"#.to_string(),
            },
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::InputJson {
                partial_json: r#""a"}"#.to_string(),
            },
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 1,
            delta: DeltaPart::InputJson {
                partial_json: r#""b"}"#.to_string(),
            },
        }))
        .unwrap();
        acc.process(&StreamEvent::PartStop).unwrap();
        acc.process(&StreamEvent::PartStop).unwrap();

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 2);
        match &msg.parts[0] {
            MessagePart::ToolCall { name, input, .. } => {
                assert_eq!(name, "echo");
                assert_eq!(input, &serde_json::json!({"msg": "a"}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        match &msg.parts[1] {
            MessagePart::ToolCall { name, input, .. } => {
                assert_eq!(name, "search");
                assert_eq!(input, &serde_json::json!({"q": "b"}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn test_accumulator_empty() {
        let acc = StreamAccumulator::new();
        let msg = acc.build();
        assert_eq!(msg.parts.len(), 0);
    }

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

    #[test]
    fn test_stream_event_variants() {
        let _ = StreamEvent::Ping;
        let _ = StreamEvent::MessageStop;
        let _ = StreamEvent::PartStop;
    }

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

    #[test]
    fn test_accumulator_ignores_delta_with_mismatched_index() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::text("")),
        }))
        .unwrap();

        // Delta arrives with index 1 — mismatch! Must NOT panic, must be ignored.
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 1,
            delta: DeltaPart::Text {
                text: "ignored".into(),
            },
        }))
        .unwrap();

        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text {
                text: "hello".into(),
            },
        }))
        .unwrap();

        acc.process(&StreamEvent::PartStop).unwrap();

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 1);
        assert_eq!(msg.parts[0].as_text(), Some("hello"));
    }

    #[test]
    fn test_accumulator_ignores_input_json_with_mismatched_index() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::text("")),
        }))
        .unwrap();

        // InputJson delta at wrong index — should be ignored.
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 5,
            delta: DeltaPart::InputJson {
                partial_json: "{\"bad\":true}".into(),
            },
        }))
        .unwrap();

        acc.process(&StreamEvent::PartStop).unwrap();

        let msg = acc.build();
        assert!(msg.parts.is_empty());
    }

    #[test]
    fn test_accumulator_delta_tool_call_string_value() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::tool_call("id1", "search", Value::Null)),
        }))
        .unwrap();

        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::ToolCall {
                partial_json: Value::String("{\"q\":\"rust\"}".into()),
            },
        }))
        .unwrap();

        acc.process(&StreamEvent::PartStop).unwrap();

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 1);
        if let MessagePart::ToolCall { input, .. } = &msg.parts[0] {
            assert_eq!(input["q"], "rust");
        } else {
            panic!("expected ToolCall");
        }
    }

    #[test]
    fn test_accumulator_delta_tool_call_non_string_ignored() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::tool_call("id1", "search", Value::Null)),
        }))
        .unwrap();

        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::ToolCall {
                partial_json: Value::Number(42.into()),
            },
        }))
        .unwrap();

        acc.process(&StreamEvent::PartStop).unwrap();

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 1);
        if let MessagePart::ToolCall { input, .. } = &msg.parts[0] {
            assert!(input.is_object());
        }
    }

    #[test]
    fn test_accumulator_ping_no_op() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::Ping).unwrap();
        assert!(acc.usage().is_none());
        let msg = acc.build();
        assert!(msg.parts.is_empty());
    }

    #[test]
    fn test_accumulator_message_stop_no_op() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::text("")),
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text { text: "hi".into() },
        }))
        .unwrap();
        acc.process(&StreamEvent::PartStop).unwrap();
        acc.process(&StreamEvent::MessageStop).unwrap();

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 1);
        assert_eq!(msg.parts[0].as_text(), Some("hi"));
    }

    #[test]
    fn test_accumulator_multiple_text_parts() {
        let mut acc = StreamAccumulator::new();

        acc.process(&StreamEvent::PartStart(PartStart {
            index: 0,
            part: Some(MessagePart::text("")),
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 0,
            delta: DeltaPart::Text {
                text: "hello".into(),
            },
        }))
        .unwrap();
        acc.process(&StreamEvent::PartStop).unwrap();

        acc.process(&StreamEvent::PartStart(PartStart {
            index: 1,
            part: Some(MessagePart::text("")),
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 1,
            delta: DeltaPart::Text {
                text: "world".into(),
            },
        }))
        .unwrap();
        acc.process(&StreamEvent::PartStop).unwrap();

        let msg = acc.build();
        assert_eq!(msg.parts.len(), 2);
        assert_eq!(msg.parts[0].as_text(), Some("hello"));
        assert_eq!(msg.parts[1].as_text(), Some("world"));
    }

    #[test]
    fn test_accumulator_message_delta_overwrites_usage() {
        let mut acc = StreamAccumulator::new();

        acc.process(&StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("end_turn".into()),
            },
            usage: Some(Usage::new(100, 50)),
        }))
        .unwrap();
        assert_eq!(acc.usage().unwrap().input_tokens, 100);
        assert_eq!(acc.usage().unwrap().output_tokens, 50);

        acc.process(&StreamEvent::MessageDelta(MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some("max_tokens".into()),
            },
            usage: Some(Usage::new(200, 75)),
        }))
        .unwrap();
        assert_eq!(acc.usage().unwrap().input_tokens, 200);
        assert_eq!(acc.usage().unwrap().output_tokens, 75);
    }

    #[test]
    fn accumulator_drops_thinking_not_into_text() {
        let mut acc = StreamAccumulator::new();
        acc.process(&StreamEvent::PartStart(PartStart {
            index: 1,
            part: None,
        }))
        .unwrap();
        acc.process(&StreamEvent::IndexedDelta(IndexedDelta {
            index: 1,
            delta: DeltaPart::Thinking {
                text: "reasoning here".into(),
            },
        }))
        .unwrap();
        acc.process(&StreamEvent::PartStop).unwrap();
        let msg = acc.build();
        let text = msg.parts.iter().find_map(|p| match p {
            MessagePart::Text { text } => Some(text.as_str()),
            _ => None,
        });
        assert!(
            !text.unwrap_or("").contains("reasoning here"),
            "reasoning must not leak into the message text: {text:?}"
        );
    }

    #[test]
    fn deltapart_thinking_serde_roundtrip() {
        let delta = DeltaPart::Thinking { text: "hmm".into() };
        let json = serde_json::to_string(&delta).unwrap();
        assert_eq!(json, r#"{"type":"thinking_delta","text":"hmm"}"#);
        let parsed: DeltaPart = serde_json::from_str(&json).unwrap();
        match &parsed {
            DeltaPart::Thinking { text } => assert_eq!(text, "hmm"),
            other => panic!("expected Thinking, got {other:?}"),
        }
    }

    #[test]
    fn deltapart_thinking_empty_text_roundtrip() {
        let delta = DeltaPart::Thinking {
            text: String::new(),
        };
        let json = serde_json::to_string(&delta).unwrap();
        let parsed: DeltaPart = serde_json::from_str(&json).unwrap();
        match &parsed {
            DeltaPart::Thinking { text } => assert_eq!(text, ""),
            other => panic!("expected Thinking with empty text, got {other:?}"),
        }
    }

    #[test]
    fn deltapart_thinking_match_compiles() {
        let delta = DeltaPart::Thinking { text: "x".into() };
        let result = match &delta {
            DeltaPart::Text { text } => format!("text:{text}"),
            DeltaPart::Thinking { text } => format!("thinking:{text}"),
            DeltaPart::ToolCall { .. } | DeltaPart::InputJson { .. } => "other".into(),
        };
        assert_eq!(result, "thinking:x");
    }
}

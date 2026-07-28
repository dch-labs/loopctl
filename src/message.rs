//! Message types for agent conversations.
//!
//! Core message representation used throughout the loopctl framework for communication between users, agents, and
//! tools. Messages consist of a [`Role`] and a list of [`MessagePart`]s,
//! supporting plain text, base64-encoded images, tool invocations, and
//! tool results.
//!
//! Messages flow through the system in a conversation history — a
//! `Vec<Message>` that is sent to the LLM API on each turn and appended
//! to as the conversation progresses. The API produces assistant messages
//! (potentially with tool-call parts), and the framework responds with
//! user-role messages containing tool results.
//!
//! # Provided Types
//!
//! - **[`Message`]** — A single message in a conversation, with a [`Role`]
//!   and zero or more [`MessagePart`]s.
//! - **[`Role`]** — Who sent the message (`User` or `Assistant`).
//! - **[`MessagePart`]** — A polymorphic part (text, image, tool-call, or tool-result).
//! - **[`ImageSource`]** — Base64-encoded image data in loopctl API format.
//! - **[`ToolContent`]** — Content returned by a tool (string or multipart).
//! - **[`ToolContentPart`]** — A single part of a multipart tool result.
//!
//! # Message Flow
//!
//! A turn starts with a user message (`Message::user`), which is sent to the
//! LLM API. The API responds with an assistant message that may contain plain
//! text plus one or more `MessagePart::ToolCall` parts. For each tool call,
//! the framework executes the tool and appends a user-role message carrying a
//! `MessagePart::ToolResult` part back into the history. The API then
//! continues, typically producing a final assistant text message that
//! completes the turn.
//!
//! # Quick Start
//!
//! ```rust
//! use loopctl::message::{Message, MessagePart, Role, ToolContent};
//!
//! // Create a simple user message
//! let user_msg = Message::user("What files are in /tmp?");
//!
//! // Create an assistant message with a tool invocation
//! let assistant_msg = Message {
//!     role: Role::Assistant,
//!     parts: vec![
//!         MessagePart::text("Let me check that for you."),
//!         MessagePart::tool_call("tool_1", "list_files", serde_json::json!({"path": "/tmp"})),
//!     ],
//! };
//!
//! // Create a tool-result message
//! let tool_result_msg = Message {
//!     role: Role::User, // tool results are sent back as "user" role
//!     parts: vec![
//!         MessagePart::tool_result("tool_1", ToolContent::from_string("file1.txt\nfile2.txt"), false),
//!     ],
//! };
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

/// A message in the conversation.
///
/// Messages are the fundamental unit of communication between users,
/// agents, and tools. Each message has a [`Role`] and a list of
/// [`MessagePart`]s. The `parts` is a `Vec<MessagePart>` because
/// a single assistant response can contain both text and tool-call
/// parts.
///
/// # Construction
///
/// Use the [`user`](Message::user) and [`assistant`](Message::assistant)
/// convenience constructors for simple text messages, or [`new`](Message::new)
/// for messages with multiple parts.
///
/// # Example
///
/// ```rust
/// use loopctl::message::{Message, Role};
/// let msg = Message::user("Hello, agent!");
/// assert_eq!(msg.role, Role::User);
/// assert_eq!(msg.parts.len(), 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Who sent this message.
    ///
    /// Determines whether the message came from the [`User`](Role::User)
    /// or the [`Assistant`](Role::Assistant). Tool-result messages are
    /// typically sent with [`Role::User`] per convention.
    pub role: Role,

    /// The parts in this message.
    ///
    /// A message can contain multiple parts of different types — for
    /// example, an assistant response might include a [`Text`](MessagePart::Text)
    /// part followed by a [`ToolCall`](MessagePart::ToolCall) block.
    /// An empty vector is valid and represents a message with no content.
    pub parts: Vec<MessagePart>,
}

impl Message {
    /// Create a user message from plain text.
    ///
    /// Convenience constructor that wraps the text in a single
    /// [`MessagePart::Text`] part with [`Role::User`].
    ///
    /// # Arguments
    ///
    /// - `text` — The message text. Accepts any type that implements
    ///   `Into<String>` (e.g. `&str`, `String`).
    ///
    /// # Returns
    ///
    /// A [`Message`] with role [`Role::User`] and one text part.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{Message, Role};
    /// let msg = Message::user("What is the weather today?");
    /// assert_eq!(msg.role, Role::User);
    /// ```
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            parts: vec![MessagePart::text(text)],
        }
    }

    /// Create an assistant message from plain text.
    ///
    /// Convenience constructor that wraps the text in a single
    /// [`MessagePart::Text`] part with [`Role::Assistant`].
    ///
    /// # Arguments
    ///
    /// - `text` — The message text. Accepts any type that implements
    ///   `Into<String>` (e.g. `&str`, `String`).
    ///
    /// # Returns
    ///
    /// A [`Message`] with role [`Role::Assistant`] and one text part.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{Message, Role};
    /// let msg = Message::assistant("The weather is sunny.");
    /// assert_eq!(msg.role, Role::Assistant);
    /// ```
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            parts: vec![MessagePart::text(text)],
        }
    }

    /// Create a message with the given role and parts.
    ///
    /// Use this constructor when you need multiple parts
    /// (e.g. text + tool-call in the same message).
    ///
    /// # Arguments
    ///
    /// - `role` — The [`Role`] of the message sender.
    /// - `parts` — A vector of [`MessagePart`]s.
    ///
    /// # Returns
    ///
    /// A [`Message`] with the specified role and content.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{Message, MessagePart, Role};
    /// let msg = Message::new(Role::Assistant, vec![
    ///     MessagePart::text("Let me look that up."),
    ///     MessagePart::tool_call("t1", "search", serde_json::json!({"q": "weather"})),
    /// ]);
    /// ```
    #[must_use]
    pub const fn new(role: Role, parts: Vec<MessagePart>) -> Self {
        Self { role, parts }
    }

    /// Concatenate every text part into a single string.
    ///
    /// Walks the message's parts in order and joins all text content,
    /// skipping non-text parts (tool calls, tool results, images).
    #[must_use]
    pub fn text_content(&self) -> String {
        self.parts
            .iter()
            .filter_map(MessagePart::as_text)
            .collect::<Vec<_>>()
            .join("")
    }

    /// Collect the tool calls this message requests, in order.
    ///
    /// Scans the message's parts for tool-call parts and maps each to a
    /// `(id, name, input)` tuple. Returns an empty vector when the message
    /// contains no tool calls.
    #[must_use]
    pub fn tool_call_parts(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.parts
            .iter()
            .filter_map(|part| match part {
                MessagePart::ToolCall { id, name, input } => {
                    Some((id.as_str(), name.as_str(), input))
                }
                _ => None,
            })
            .collect()
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut chunks = Vec::new();
        for part in &self.parts {
            match part {
                MessagePart::Text { text } => {
                    chunks.push(text.clone());
                }
                MessagePart::ToolCall { name, input, .. } => {
                    if let Ok(input_str) = serde_json::to_string(input) {
                        chunks.push(format!("[Tool: {name} with input: {input_str}]"));
                    }
                }
                MessagePart::ToolResult { output, .. } => {
                    chunks.push(format!("[Tool Result: {output}]"));
                }
                MessagePart::Image { source } => {
                    chunks.push(format!("[Image: {}]", source.media_type));
                }
            }
        }
        write!(f, "{}", chunks.join("\n"))
    }
}

/// The role of a message sender in the conversation.
///
/// Each [`Message`] has exactly one role that identifies who produced
/// the content. Common LLM APIs use `"user"` for human messages and
/// tool results, and `"assistant"` for model responses.
///
/// # Serialization
///
/// Serialized as lowercase `snake_case` strings (`"user"`, `"assistant"`,
/// `"system"`) to match the API convention.
///
/// # Example
///
/// ```rust
/// use loopctl::message::{Role};
/// assert_eq!(Role::User.to_string(), "user");
/// assert_eq!(Role::Assistant.to_string(), "assistant");
/// assert_eq!(Role::System.to_string(), "system");
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// A message from the human user.
    ///
    /// Also used for tool-result messages sent back to the API,
    /// per convention.
    User,

    /// A message from the assistant / LLM model.
    ///
    /// Contains the model's generated response, which may include
    /// text, tool-call invocations, or both.
    Assistant,

    /// Framework-injected context, such as a turn-boundary reminder.
    ///
    /// Providers map this to their native system representation: an inline
    /// `{role: "system"}` message on OpenAI, or the top-level system field on
    /// Anthropic and Gemini (which do not accept an inline system role
    /// mid-conversation). Application code does not normally construct
    /// `System` messages directly — they are produced by framework machinery
    /// like [`ContextContributor`](crate::engine::ContextContributor).
    System,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::System => write!(f, "system"),
        }
    }
}

/// A part of message content within a [`Message`].
///
/// Messages can contain multiple types of content interleaved in a
/// single response: plain text, base64-encoded images, tool-call
/// invocations by the assistant, and tool results. Each variant
/// corresponds to a possible part type with tagged JSON serialization.
///
/// # Serialization
///
/// Uses `#[serde(tag = "type")]` with renamed variants to produce
/// JSON objects like `{"type": "text", "text": "..."}` matching the
/// loopctl message schema.
///
/// # Construction
///
/// Use the constructors [`text`](MessagePart::text),
/// [`tool_call`](MessagePart::tool_call), and
/// [`tool_result`](MessagePart::tool_result) rather than building
/// variants directly.
///
/// # Example
///
/// ```rust
/// use loopctl::message::{MessagePart};
/// let text_part = MessagePart::text("Hello");
/// let tool_part = MessagePart::tool_call("id1", "search", serde_json::json!({"q": "test"}));
/// let result_part = MessagePart::tool_result("id1", "found 3 results", false);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MessagePart {
    /// Plain text content.
    ///
    /// Contains a UTF-8 text string generated by the model or
    /// provided by the user.
    #[serde(rename = "text")]
    Text {
        /// The text content of this part.
        ///
        /// May contain multiple lines. For streaming responses, this
        /// is the fully assembled text after all streaming deltas have been concatenated.
        text: String,
    },

    /// A base64-encoded image part.
    ///
    /// Images can be included in user messages for multimodal models.
    /// The [`ImageSource`] contains the MIME type and base64 data.
    ///
    /// Serialized as `{"type":"image","source":{...}}`.
    ///
    /// Note: image parts are only valid in [`Role::User`] messages;
    /// the API rejects them in assistant messages.
    #[serde(rename = "image")]
    Image {
        /// The image data and metadata.
        ///
        /// Contains the MIME type, encoding type, and base64-encoded
        /// image bytes. See [`ImageSource`] for details.
        source: ImageSource,
    },

    /// A tool-call invocation by the assistant.
    ///
    /// When the model decides to call a tool, it emits a part with
    /// the tool's ID, name, and JSON input. The framework then
    /// executes the tool and sends the result back as a
    /// [`ToolResult`](Self::ToolResult) block.
    ///
    /// The correlation between a `ToolCall` and its `ToolResult` is
    /// done via the [`id`](Self::ToolCall.id) field, which must be
    /// unique within a single conversation turn.
    #[serde(rename = "tool_call")]
    ToolCall {
        /// Unique identifier for this tool invocation.
        ///
        /// Assigned by the API. Used to correlate the tool-call part
        /// with the corresponding [`ToolResult`](Self::ToolResult)
        /// part in the next message.
        id: String,

        /// The name of the tool being invoked.
        ///
        /// Must match one of the tool names provided in the request's
        /// `tools` parameter.
        name: String,

        /// The JSON input provided by the model for the tool.
        ///
        /// Structure depends on the tool's input schema. May be
        /// any JSON value (`Object`, `Array`, `String`, etc.).
        input: Value,
    },

    /// The result of a tool-call invocation.
    ///
    /// Sent back to the model with [`Role::User`] (per API convention)
    /// to provide the output of a tool execution. Contains the tool-call
    /// ID for correlation, the output content, and an optional error flag.
    ///
    /// After the agent executes a tool, it wraps the output in this
    /// variant and appends it to the conversation history so the model
    /// can reason about the output.
    #[serde(rename = "tool_result")]
    ToolResult {
        /// The ID of the tool-call part this result corresponds to.
        ///
        /// Must match the `id` from the original
        /// [`ToolCall`](MessagePart::ToolCall) block.
        call_id: String,

        /// The output returned by the tool.
        ///
        /// Can be a simple string or a multipart response with
        /// multiple parts. See [`ToolContent`].
        output: ToolContent,

        /// Whether this result represents an error.
        ///
        /// `Some(true)` indicates the tool invocation failed. When
        /// `None` or `Some(false)`, the result is a success.
        is_error: Option<bool>,
    },
}

impl MessagePart {
    /// Create a text part.
    ///
    /// Constructor for the [`Text`](MessagePart::Text) variant.
    /// Accepts any type that implements `Into<String>`.
    ///
    /// # Arguments
    ///
    /// - `text` — The text content.
    ///
    /// # Returns
    ///
    /// A [`MessagePart::Text`] variant.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{MessagePart};
    /// let part = MessagePart::text("Hello, world!");
    /// assert!(part.is_text());
    /// ```
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create a tool-call part.
    ///
    /// Used when constructing an assistant message that invokes a tool.
    /// The `id` must be unique within the conversation to allow
    /// correlation with the corresponding tool result.
    ///
    /// # Arguments
    ///
    /// - `id` — Unique identifier for this tool invocation.
    /// - `name` — The name of the tool to invoke.
    /// - `input` — The JSON input parameters for the tool.
    ///
    /// # Returns
    ///
    /// A [`MessagePart::ToolCall`] variant.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{MessagePart};
    /// let part = MessagePart::tool_call(
    ///     "tool_abc",
    ///     "read_file",
    ///     serde_json::json!({"path": "/tmp/test.txt"}),
    /// );
    /// assert!(part.is_tool_call());
    /// ```
    pub fn tool_call(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// Create a tool-result part.
    ///
    /// Used when sending the output of a tool execution back to the
    /// model. The `call_id` must match the `id` from the original
    /// [`ToolCall`](MessagePart::ToolCall) block.
    ///
    /// # Arguments
    ///
    /// - `call_id` — The ID of the tool invocation this output is for.
    /// - `output` — The tool output (string or multipart).
    /// - `is_error` — Whether the tool invocation failed.
    ///
    /// # Returns
    ///
    /// A [`MessagePart::ToolResult`] variant.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{MessagePart};
    /// let part = MessagePart::tool_result(
    ///     "tool_abc",
    ///     "file contents here",
    ///     false,
    /// );
    /// assert!(part.is_tool_result());
    /// ```
    pub fn tool_result(
        call_id: impl Into<String>,
        output: impl Into<ToolContent>,
        is_error: bool,
    ) -> Self {
        Self::ToolResult {
            call_id: call_id.into(),
            output: output.into(),
            is_error: Some(is_error),
        }
    }

    /// Returns `true` if this is a [`Text`](MessagePart::Text) block.
    ///
    /// Allows filtering or pattern matching on parts
    /// without a full `match` expression.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{MessagePart};
    /// let part = MessagePart::text("hello");
    /// assert!(part.is_text());
    /// ```
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }

    /// Returns `true` if this is a [`ToolCall`](MessagePart::ToolCall) block.
    ///
    /// Detects tool invocations in an assistant message
    /// to decide whether to enter the tool-execution loop.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{MessagePart};
    /// use serde_json::Value;
    /// let part = MessagePart::tool_call("id", "tool", Value::Null);
    /// assert!(part.is_tool_call());
    /// ```
    #[must_use]
    pub const fn is_tool_call(&self) -> bool {
        matches!(self, Self::ToolCall { .. })
    }

    /// Returns `true` if this is a [`ToolResult`](MessagePart::ToolResult) block.
    ///
    /// Identifies tool results in a conversation history.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{MessagePart};
    /// let part = MessagePart::tool_result("id", "output", false);
    /// assert!(part.is_tool_result());
    /// ```
    #[must_use]
    pub const fn is_tool_result(&self) -> bool {
        matches!(self, Self::ToolResult { .. })
    }

    /// Get the text content if this is a [`Text`](MessagePart::Text) block.
    ///
    /// Returns `Some(&str)` for text parts, `None` for all other
    /// variants. Extracts the text from a known-text part.
    ///
    /// # Returns
    ///
    /// `Some(text)` if this is a text part, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{MessagePart};
    /// use serde_json::Value;
    /// let part = MessagePart::text("hello");
    /// assert_eq!(part.as_text(), Some("hello"));
    ///
    /// let tool = MessagePart::tool_call("id", "tool", Value::Null);
    /// assert_eq!(tool.as_text(), None);
    /// ```
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }
}

/// Source data for an image part in loopctl API format.
///
/// Represents a base64-encoded image with its MIME type. Serialized
/// as:
///
/// ```json
/// { "type": "base64", "media_type": "image/png", "data": "..." }
/// ```
///
/// Use [`new_base64`](ImageSource::new_base64) to construct from
/// a MIME type and raw base64 data.
///
/// # Construction
///
/// Always use [`new_base64`](ImageSource::new_base64) rather than
/// building the struct directly — it ensures the `encoding` field
/// is set correctly.
///
/// # Example
///
/// ```rust
/// use loopctl::message::{ImageSource};
/// let source = ImageSource::new_base64("image/png", "iVBORw0KGgo...");
/// assert_eq!(source.media_type, "image/png");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSource {
    /// The encoding type for the image data.
    ///
    /// LLM APIs always use `"base64"`.
    pub encoding: String,

    /// MIME type of the image (e.g. `"image/png"`, `"image/jpeg"`, `"image/webp"`).
    ///
    /// Must be one of the supported types for the model being used.
    /// Used by the API to correctly decode the base64 data.
    pub media_type: String,

    /// Base64-encoded image data.
    ///
    /// The raw bytes of the image encoded as a base64 string. Should
    /// not include a data-URI prefix — only the base64 payload.
    pub data: String,
}

impl ImageSource {
    /// Create a new base64 image source.
    ///
    /// Sets the `encoding` to `"base64"` automatically.
    /// Standard construction method for image sources in loopctl API format.
    ///
    /// # Arguments
    ///
    /// - `media_type` — The MIME type (e.g. `"image/png"`).
    /// - `data` — The base64-encoded image data.
    ///
    /// # Returns
    ///
    /// An [`ImageSource`] ready to use in an [`Image`](MessagePart::Image) block.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{ImageSource};
    /// let source = ImageSource::new_base64("image/png", "iVBORw0KGgo...");
    /// assert_eq!(source.encoding, "base64");
    /// ```
    #[must_use]
    pub fn new_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            encoding: "base64".into(),
            media_type: media_type.into(),
            data: data.into(),
        }
    }
}

/// Content that can be returned from a tool invocation.
///
/// Supports both simple string results and multipart results with
/// multiple parts (text and images).
///
/// Serialized using `#[serde(untagged)]` so that a plain string
/// is represented directly in JSON, while a multipart result
/// is serialized as an array of [`ToolContentPart`] objects.
///
/// # Construction
///
/// Use [`from_string`](ToolContent::from_string) for simple
/// text results or [`from_multipart`](ToolContent::from_multipart)
/// for multi-part results. Both `String` and `&str` implement
/// `Into<ToolContent>` for ergonomic conversion.
///
/// # Default
///
/// The [`Default`] implementation produces an empty string result,
/// equivalent to `ToolContent::from_string("")`.
///
/// # Example
///
/// ```rust
/// use loopctl::message::{ToolContent, ToolContentPart};
/// // Simple string result
/// let simple = ToolContent::from_string("File found: test.txt");
///
/// // Multipart result with multiple parts
/// let multipart = ToolContent::from_multipart(vec![
///     ToolContentPart::text("Line 1"),
///     ToolContentPart::text("Line 2"),
/// ]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolContent {
    /// A simple string result.
    ///
    /// Returned by tools that produce plain text output.
    Text(String),

    /// A multipart result with multiple parts.
    ///
    /// Used when a tool needs to return mixed content (text + images)
    /// or multiple text segments.
    Multipart(Vec<ToolContentPart>),
}

impl ToolContent {
    /// Create a simple string result.
    ///
    /// Wraps the given text in the [`Text`](Self::Text) variant.
    ///
    /// # Arguments
    ///
    /// - `s` — The text content of the result.
    ///
    /// # Returns
    ///
    /// A [`ToolContent::Text`] variant.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{ToolContent};
    /// let content = ToolContent::from_string("Done!");
    /// assert!(content.is_string());
    /// ```
    #[must_use]
    pub fn from_string(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Create a multipart result with multiple parts.
    ///
    /// Use this when a tool needs to return more than plain text —
    /// for example, text combined with images.
    ///
    /// # Arguments
    ///
    /// - `parts` — The [`ToolContentPart`]s that make up the result.
    ///
    /// # Returns
    ///
    /// A [`ToolContent::Multipart`] variant.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{ToolContent, ToolContentPart};
    /// let content = ToolContent::from_multipart(vec![
    ///     ToolContentPart::text("file1.txt"),
    ///     ToolContentPart::text("file2.txt"),
    /// ]);
    /// ```
    #[must_use]
    pub const fn from_multipart(parts: Vec<ToolContentPart>) -> Self {
        Self::Multipart(parts)
    }

    /// Returns `true` if this is a simple string result.
    ///
    /// Branches on how to process or display the result.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{ToolContent};
    /// let content = ToolContent::from_string("ok");
    /// assert!(content.is_string());
    /// ```
    #[must_use]
    pub const fn is_string(&self) -> bool {
        matches!(self, Self::Text(_))
    }
}

/// Produces a [`Default`] value for [`ToolContent`].
///
/// Returns an empty [`String`](ToolContent::Text) variant,
/// which is the neutral starting point for tool results. Used when
/// building responses incrementally or initializing result storage.
///
/// # Example
///
/// ```rust
/// use loopctl::message::{ToolContent};
/// let content = ToolContent::default();
/// assert!(content.is_string());
/// assert_eq!(content.to_string(), "");
/// ```
impl Default for ToolContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for ToolContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for ToolContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

impl fmt::Display for ToolContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(s) => write!(f, "{s}"),
            Self::Multipart(parts) => {
                let texts: Vec<&str> = parts
                    .iter()
                    .filter_map(|part| {
                        if let ToolContentPart::Text { text } = part {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                write!(f, "{}", texts.join("\n"))
            }
        }
    }
}

/// A part of a Multipart [`ToolContent`].
///
/// Represents a single piece of content within a multi-part tool result.
/// Currently supports images (via [`ImageSource`]) and text.
///
/// # Construction
///
/// Use [`text`](ToolContentPart::text) or [`image`](ToolContentPart::image)
/// constructors rather than building variants directly.
///
/// # Example
///
/// ```rust
/// use loopctl::message::{ImageSource, ToolContent, ToolContentPart};
/// let text_part = ToolContentPart::text("Screenshot analysis:");
/// let img_part = ToolContentPart::image(ImageSource::new_base64("image/png", "..."));
/// let content = ToolContent::from_multipart(vec![text_part, img_part]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolContentPart {
    /// An image part within a Multipart tool result.
    ///
    /// Contains the base64-encoded image data. Used by tools
    /// that capture screenshots or generate images.
    Image {
        /// The image data and metadata.
        ///
        /// See [`ImageSource`] for the structure and supported formats.
        /// The [`ImageSource::encoding`] will always be `"base64"`.
        source: ImageSource,
    },

    /// A text part within a Multipart tool result.
    ///
    /// Contains a plain text string.
    Text {
        /// The text content of this part.
        ///
        /// May contain multiple lines. Joined with other text parts
        /// by [`ToolContent`]'s [`Display`](fmt::Display) impl.
        text: String,
    },
}

impl ToolContentPart {
    /// Create a text part.
    ///
    /// Constructor for the [`Text`](ToolContentPart::Text) variant.
    /// Use to build individual text segments within a
    /// [`ToolContent::Multipart`] result.
    ///
    /// # Arguments
    ///
    /// - `text` — The text content. Accepts any type that implements
    ///   `Into<String>` (e.g. `&str`, `String`).
    ///
    /// # Returns
    ///
    /// A [`ToolContentPart::Text`] variant.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{ToolContentPart};
    /// let part = ToolContentPart::text("result: 42");
    /// ```
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create an image part.
    ///
    /// Use when a tool result includes image data alongside text.
    /// Combine with [`ToolContentPart::text`] parts via
    /// [`ToolContent::from_multipart`] to produce a multi-part
    /// result containing both text and images.
    ///
    /// # Arguments
    ///
    /// - `source` — The [`ImageSource`] containing the image data.
    ///   Construct with [`ImageSource::new_base64`].
    ///
    /// # Returns
    ///
    /// A [`ToolContentPart::Image`] variant.
    ///
    /// # Example
    ///
    /// ```rust
    /// use loopctl::message::{ImageSource, ToolContentPart};
    /// let part = ToolContentPart::image(ImageSource::new_base64("image/png", "..."));
    /// ```
    #[must_use]
    pub const fn image(source: ImageSource) -> Self {
        Self::Image { source }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_user_shortcut() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.parts.len(), 1);
        assert_eq!(msg.parts[0].as_text(), Some("Hello"));
    }

    #[test]
    fn test_message_assistant_shortcut() {
        let msg = Message::assistant("Hi there!");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.parts.len(), 1);
    }

    #[test]
    fn test_message_display() {
        let msg = Message::user("Hello world");
        assert_eq!(msg.to_string(), "Hello world");
    }

    #[test]
    fn test_message_display_with_tool_call() {
        let msg = Message {
            role: Role::Assistant,
            parts: vec![MessagePart::tool_call(
                "id1",
                "read_file",
                serde_json::json!({"path": "/tmp/test.txt"}),
            )],
        };
        let display = msg.to_string();
        assert!(display.contains("Tool: read_file"));
    }

    #[test]
    fn test_role_display() {
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
    }

    #[test]
    fn test_part_helpers() {
        let text = MessagePart::text("hello");
        assert!(text.is_text());
        assert!(!text.is_tool_call());
        assert_eq!(text.as_text(), Some("hello"));

        let tool_call = MessagePart::tool_call("id", "tool", serde_json::json!({}));
        assert!(tool_call.is_tool_call());
        assert!(!tool_call.is_text());
        assert!(tool_call.as_text().is_none());

        let tool_result = MessagePart::tool_result("id", "ok", false);
        assert!(tool_result.is_tool_result());
    }

    #[test]
    fn test_image_source() {
        let src = ImageSource::new_base64("image/png", "iVBOR...");
        assert_eq!(src.encoding, "base64");
        assert_eq!(src.media_type, "image/png");
    }

    #[test]
    fn test_tool_result_from_string() {
        let result: ToolContent = "hello".into();
        assert!(result.is_string());
        assert_eq!(result.to_string(), "hello");
    }

    #[test]
    fn test_tool_result_default() {
        let result = ToolContent::default();
        assert!(result.is_string());
    }

    #[test]
    fn test_tool_result_part_text() {
        let part = ToolContentPart::text("output");
        match &part {
            ToolContentPart::Text { text } => assert_eq!(text, "output"),
            ToolContentPart::Image { .. } => panic!("expected text part"),
        }
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("test");
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg.role, deserialized.role);
    }

    #[test]
    fn test_part_serialization_roundtrip() {
        let parts = vec![
            MessagePart::text("hello"),
            MessagePart::tool_call("id1", "my_tool", serde_json::json!({"key": "value"})),
        ];
        let json = serde_json::to_string(&parts).unwrap();
        let back: Vec<MessagePart> = serde_json::from_str(&json).unwrap();
        assert_eq!(parts.len(), back.len());
    }

    #[test]
    fn test_tool_result_multipart_display() {
        let result = ToolContent::from_multipart(vec![
            ToolContentPart::text("line 1"),
            ToolContentPart::text("line 2"),
        ]);
        assert_eq!(result.to_string(), "line 1\nline 2");
    }

    #[test]
    fn test_role_system_serializes_snake_case() {
        let s = serde_json::to_string(&Role::System).expect("System serializes");
        assert_eq!(s, "\"system\"");
    }

    #[test]
    fn test_role_system_display() {
        assert_eq!(Role::System.to_string(), "system");
    }

    #[test]
    fn test_role_system_round_trips() {
        let parsed: Role =
            serde_json::from_str("\"system\"").expect("\"system\" deserializes to a Role");
        assert_eq!(parsed, Role::System);
    }
}

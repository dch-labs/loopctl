//! Type conversions between loopctl and MCP (rmcp) — both directions.
//!
//! **Inbound** (MCP → loopctl, the client side): `CallToolResult` →
//! [`ToolOutput`], `ContentBlock` → [`ToolContentPart`].
//!
//! **Outbound** (loopctl → MCP, the server side): [`ToolSchema`] →
//! rmcp `Tool`, [`ToolOutput`] → `CallToolResult`, [`ToolContent`] →
//! `Vec<ContentBlock>`, unknown-tool → [`ErrorData`].
//!
//! Both directions sit here so the symmetric mappings (e.g.
//! `ContentBlock ↔ ToolContentPart`) are side by side — a change to either
//! type touches one file.

use rmcp::ErrorData;
use rmcp::model::CallToolResult;
use rmcp::model::ContentBlock;
use rmcp::model::ErrorCode;
use rmcp::model::Tool as McpTool;
use rmcp::model::ToolAnnotations;

use crate::message::ToolContent as MessageToolContent;
use crate::message::ToolContentPart;
use crate::tool::ToolError;
use crate::tool::ToolOutput;
use crate::tool::ToolSchema;

/// Map one rmcp content block to a loopctl [`ToolContentPart`].
///
/// Text and image carry through. Audio falls back to a short text note. Embedded
/// resources and resource links are stringified with their identifying payload
/// (uri, text/name) so the model sees *what* the server returned, not just that
/// it returned something. Any future block kind falls back to a generic note.
pub(crate) fn bridge_content(block: &ContentBlock) -> ToolContentPart {
    match block {
        ContentBlock::Text(text) => ToolContentPart::text(&text.text),
        ContentBlock::Image(image) => ToolContentPart::image(
            crate::message::ImageSource::new_base64(&image.mime_type, &image.data),
        ),
        ContentBlock::Resource(resource) => match &resource.resource {
            rmcp::model::ResourceContents::TextResourceContents { uri, text, .. } => {
                ToolContentPart::text(format!("MCP resource {uri}: {text}"))
            }
            rmcp::model::ResourceContents::BlobResourceContents { uri, .. } => {
                ToolContentPart::text(format!("MCP resource {uri}: (blob)"))
            }
            _ => ToolContentPart::text("unsupported MCP content type: embedded resource"),
        },
        ContentBlock::ResourceLink(link) => {
            ToolContentPart::text(format!("MCP resource link: {} ({})", link.name, link.uri))
        }
        _ => ToolContentPart::text("unsupported MCP content type"),
    }
}

/// Bridge a rmcp `CallToolResult` into a loopctl [`ToolOutput`].
///
/// Maps the result's content blocks into [`MessageToolContent`]: a single text
/// block becomes [`MessageToolContent::Text`]; any other shape (multiple blocks,
/// an image) becomes [`MessageToolContent::Multipart`]. `isError` becomes
/// [`ToolOutput::is_error`] — a server-reported tool error is a *soft* failure,
/// matching how native loopctl tools report recoverable errors. `structuredContent`,
/// when present, is appended as one extra JSON-stringified text part (carried,
/// not parsed). An error with no content at all becomes
/// [`crate::mcp::McpError::EmptyToolError`] carrying `tool_name`.
///
/// A successful result with zero content blocks yields an empty successful
/// [`ToolOutput`] (no error) — the server ran the tool and returned nothing.
///
/// # Errors
///
/// [`crate::mcp::McpError::EmptyToolError`] when the server reported an error but
/// supplied no content to surface.
pub(crate) fn bridge_result(
    tool_name: &str,
    res: rmcp::model::CallToolResult,
) -> Result<ToolOutput, crate::mcp::McpError> {
    let is_error = res.is_error.unwrap_or(false);
    let mut parts: Vec<ToolContentPart> = res.content.iter().map(bridge_content).collect();
    if let Some(structured) = res.structured_content {
        let note = serde_json::to_string(&structured)
            .unwrap_or_else(|_| "<unserializable structuredContent>".to_string());
        parts.push(ToolContentPart::text(note));
    }
    if parts.is_empty() {
        return if is_error {
            Err(crate::mcp::McpError::EmptyToolError(tool_name.to_string()))
        } else {
            Ok(ToolOutput::text(String::new()))
        };
    }
    let payload = if parts.len() == 1 {
        match parts.pop() {
            Some(ToolContentPart::Text { text }) => MessageToolContent::Text(text),
            Some(single) => MessageToolContent::Multipart(vec![single]),
            None => MessageToolContent::Text(String::new()),
        }
    } else {
        MessageToolContent::Multipart(parts)
    };
    let output = if is_error {
        ToolOutput::error(payload)
    } else {
        ToolOutput::success(payload)
    };
    Ok(output)
}

/// Convert a loopctl [`ToolSchema`] to an MCP `Tool`.
///
/// The load-bearing detail: loopctl's field is named `tool`, MCP's is named
/// `name` — this is the rename. An empty description is omitted (MCP's
/// `description` is optional) rather than sent as `""`.
///
/// Only object-typed schemas are convertible: the root must be a JSON object
/// carrying `"type": "object"` — the shape MCP clients expect a tool's input
/// schema to have, and the only one rmcp's `JsonObject` field can carry
/// (boolean schemas and non-object roots cannot be represented). Any other
/// root — a boolean, a string, an object missing or disagreeing on `type` —
/// yields `None` with a warning; callers omit such tools from the listing
/// rather than advertise a malformed schema.
///
/// `is_read_only` (from [`Tool::is_read_only`](crate::tool::Tool::is_read_only),
/// which [`ToolSchema`] does not carry) is forwarded as the MCP
/// `annotations.readOnlyHint` so clients can apply their own read-only policy.
/// A read-only tool also gets `destructiveHint: false` — a tool that does not
/// modify its environment cannot perform destructive updates, and leaving the
/// hint absent would make the MCP default ("assume destructive") contradict the
/// read-only hint. When `is_read_only` is `false`, no annotations are sent and
/// the client applies the spec defaults.
pub(crate) fn tool_schema_to_mcp(schema: ToolSchema, is_read_only: bool) -> Option<McpTool> {
    let input_schema = match schema.input_schema {
        serde_json::Value::Object(map)
            if map.get("type").and_then(serde_json::Value::as_str) == Some("object") =>
        {
            map
        }
        other => {
            tracing::warn!(
                tool = %schema.tool,
                schema = %other,
                "tool input_schema is not an object-typed JSON Schema; omitting the tool"
            );
            return None;
        }
    };
    let description = if schema.description.is_empty() {
        None
    } else {
        Some(std::borrow::Cow::Owned(schema.description))
    };
    let mut tool =
        McpTool::new_with_raw(schema.tool, description, std::sync::Arc::new(input_schema));
    tool.annotations =
        is_read_only.then(|| ToolAnnotations::from_raw(None, Some(true), Some(false), None, None));
    Some(tool)
}

/// Map `Result<ToolOutput, ToolError>` → `CallToolResult`.
///
/// - `Ok(out)` → content from `out.payload`, `is_error = out.is_error`
///   (loopctl's soft-error flag maps straight onto MCP's `CallToolResult.is_error`).
/// - `Err(e)` → content is `e.to_string()`, `is_error = true`. Hard tool failures
///   are surfaced as tool-level errors, not MCP protocol errors: the model should
///   see "permission denied: …" and react, not get a transport-level error.
pub(crate) fn dispatch_result_to_call_tool(
    tool_name: &str,
    result: Result<ToolOutput, ToolError>,
) -> CallToolResult {
    match result {
        Ok(out) => {
            let content = tool_content_to_mcp(out.payload);
            let mut res = CallToolResult::success(content);
            res.is_error = Some(out.is_error);
            res
        }
        Err(e) => {
            tracing::warn!(tool = tool_name, error = %e, "tool call returned hard error");
            CallToolResult::error(vec![ContentBlock::text(e.to_string())])
        }
    }
}

/// Convert loopctl [`ToolContent`] → `Vec<ContentBlock>` (MCP).
///
/// `Text(s)` → one text content. `Multipart(parts)` → one content per part,
/// in order; an image part's base64 data and MIME type are forwarded verbatim.
pub(crate) fn tool_content_to_mcp(content: MessageToolContent) -> Vec<ContentBlock> {
    match content {
        MessageToolContent::Text(s) => vec![ContentBlock::text(s)],
        MessageToolContent::Multipart(parts) => parts
            .into_iter()
            .map(|part| match part {
                ToolContentPart::Text { text } => ContentBlock::text(text),
                ToolContentPart::Image { source } => {
                    ContentBlock::image(source.data, source.media_type)
                }
            })
            .collect(),
    }
}

/// Build an unknown-tool [`ErrorData`] mirroring [`ToolError::not_found`]'s message.
///
/// The code is `METHOD_NOT_FOUND`: the client asked for a tool this server does
/// not have, which is a request-routing failure, not a tool failure. The message
/// names the registered tools so clients (and logs) that surface protocol-error
/// text can self-correct — most clients render protocol errors opaquely to the
/// model.
pub(crate) fn not_found_error(requested: &str, available: &[&str]) -> ErrorData {
    let msg = if available.is_empty() {
        format!("Tool not found: {requested}. None registered.")
    } else {
        format!(
            "Tool not found: {requested}. Available: {}",
            available.join(", ")
        )
    };
    ErrorData::new(
        ErrorCode::METHOD_NOT_FOUND,
        msg,
        Some(serde_json::json!({ "requested": requested })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ImageSource;
    use crate::message::ToolContent as MessageToolContent;
    use crate::message::ToolContentPart;
    use crate::tool::{ToolError, ToolOutput, ToolSchema};
    use serde_json::json;

    #[test]
    fn tool_schema_to_mcp_renames_tool_to_name() {
        let schema = ToolSchema {
            tool: "echo".into(),
            description: "Echoes input".into(),
            input_schema: json!({"type": "object"}),
        };
        let mcp_tool = tool_schema_to_mcp(schema, false);
        let mcp_tool = mcp_tool.expect("object-typed schema converts");
        assert_eq!(mcp_tool.name.as_ref(), "echo");
        assert_eq!(mcp_tool.description.as_deref(), Some("Echoes input"));
    }

    #[test]
    fn tool_schema_forwards_input_schema_map() {
        let schema = ToolSchema {
            tool: "x".into(),
            description: String::new(),
            input_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        };
        let mcp_tool = tool_schema_to_mcp(schema, false);
        let mcp_tool = mcp_tool.expect("object-typed schema converts");
        assert!(mcp_tool.input_schema.contains_key("properties"));
    }

    #[test]
    fn tool_schema_false_root_is_rejected() {
        let schema = ToolSchema {
            tool: "x".into(),
            description: String::new(),
            input_schema: json!(false),
        };
        assert!(tool_schema_to_mcp(schema, false).is_none());
    }

    #[test]
    fn tool_schema_string_root_is_rejected() {
        let schema = ToolSchema {
            tool: "x".into(),
            description: String::new(),
            input_schema: json!("not an object"),
        };
        assert!(tool_schema_to_mcp(schema, false).is_none());
    }

    #[test]
    fn tool_schema_object_without_root_type_is_rejected() {
        let schema = ToolSchema {
            tool: "x".into(),
            description: String::new(),
            input_schema: json!({"properties": {"q": {"type": "string"}}}),
        };
        assert!(tool_schema_to_mcp(schema, false).is_none());
    }

    #[test]
    fn tool_schema_omits_empty_description() {
        let schema = ToolSchema {
            tool: "x".into(),
            description: String::new(),
            input_schema: json!({"type": "object"}),
        };
        let mcp_tool = tool_schema_to_mcp(schema, false);
        let mcp_tool = mcp_tool.expect("object-typed schema converts");
        assert_eq!(mcp_tool.description, None);
    }

    #[test]
    fn tool_schema_forwards_read_only_hint() {
        let schema = ToolSchema {
            tool: "peek".into(),
            description: "Reads state".into(),
            input_schema: json!({"type": "object"}),
        };
        let mcp_tool = tool_schema_to_mcp(schema, true);
        let mcp_tool = mcp_tool.expect("object-typed schema converts");
        let annotations = mcp_tool
            .annotations
            .expect("read-only tool has annotations");
        assert_eq!(annotations.read_only_hint, Some(true));
        assert_eq!(annotations.destructive_hint, Some(false));
    }

    #[test]
    fn tool_schema_without_read_only_sends_no_annotations() {
        let schema = ToolSchema {
            tool: "write".into(),
            description: "Writes state".into(),
            input_schema: json!({"type": "object"}),
        };
        let mcp_tool = tool_schema_to_mcp(schema, false);
        let mcp_tool = mcp_tool.expect("object-typed schema converts");
        assert!(mcp_tool.annotations.is_none());
    }

    #[test]
    fn tool_content_text_yields_one_text_block() {
        let blocks = tool_content_to_mcp(MessageToolContent::Text("hi".into()));
        assert_eq!(blocks.len(), 1);
        let text = blocks
            .first()
            .and_then(ContentBlock::as_text)
            .map(|t| t.text.as_str());
        assert_eq!(text, Some("hi"));
    }

    #[test]
    fn tool_content_multipart_preserves_text_then_image_in_order() {
        let parts = vec![
            ToolContentPart::text("caption"),
            ToolContentPart::image(ImageSource::new_base64("image/png", "Zm9v")),
        ];
        let blocks = tool_content_to_mcp(MessageToolContent::Multipart(parts));
        assert_eq!(blocks.len(), 2);
        let caption = blocks
            .first()
            .and_then(ContentBlock::as_text)
            .map(|t| t.text.as_str());
        assert_eq!(caption, Some("caption"));
        let image = blocks
            .get(1)
            .and_then(ContentBlock::as_image)
            .expect("second block is an image");
        assert_eq!(image.data, "Zm9v");
        assert_eq!(image.mime_type, "image/png");
    }

    #[test]
    fn dispatch_result_success_carries_is_error_false() {
        let res = dispatch_result_to_call_tool("echo", Ok(ToolOutput::text("done")));
        assert_eq!(res.is_error, Some(false));
        assert!(!res.content.is_empty());
    }

    #[test]
    fn dispatch_result_soft_error_preserves_is_error_true() {
        let res =
            dispatch_result_to_call_tool("echo", Ok(ToolOutput::error_text("file not found")));
        assert_eq!(res.is_error, Some(true));
    }

    #[test]
    fn dispatch_result_hard_error_is_tool_level_not_protocol() {
        let res = dispatch_result_to_call_tool("echo", Err(ToolError::Execution("boom".into())));
        assert_eq!(res.is_error, Some(true));
        let text = res
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.as_str());
        assert_eq!(text, Some("Execution error: boom"));
    }

    #[test]
    fn not_found_error_mentions_name_and_available() {
        let err = not_found_error("grep", &["echo", "bash"]);
        let msg = err.message.to_string();
        assert!(msg.contains("grep"), "mentions requested name: {msg}");
        assert!(msg.contains("echo"), "mentions available: {msg}");
    }

    #[test]
    fn dispatch_result_cancelled_is_a_tool_level_error() {
        let res = dispatch_result_to_call_tool("slow", Err(ToolError::Cancelled));
        assert_eq!(res.is_error, Some(true));
        let text = res
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.as_str());
        assert_eq!(text, Some("Cancelled"));
    }

    #[test]
    fn not_found_error_uses_method_not_found_code() {
        let err = not_found_error("grep", &["echo"]);
        assert_eq!(err.code, ErrorCode::METHOD_NOT_FOUND);
    }

    #[test]
    fn not_found_error_with_empty_registry_says_none_registered() {
        let err = not_found_error("grep", &[]);
        let msg = err.message.to_string();
        assert!(
            msg.contains("None registered"),
            "empty registry is called out: {msg}"
        );
    }
}

//! Shared helpers for live integration tests.

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone
)]

pub fn test_tools() -> Vec<loopctl::tool::ToolSchema> {
    vec![
        loopctl::tool::ToolSchema {
            tool: "read_file".into(),
            description: "Read a file from disk".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                    "encoding": {"type": "string", "enum": ["utf-8", "binary"]}
                },
                "required": ["path"]
            }),
        },
        loopctl::tool::ToolSchema {
            tool: "search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 50}
                },
                "required": ["query"]
            }),
        },
        loopctl::tool::ToolSchema {
            tool: "write_file".into(),
            description: "Write content to a file".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "create_dirs": {"type": "boolean"}
                },
                "required": ["path", "content"]
            }),
        },
        loopctl::tool::ToolSchema {
            tool: "run_command".into(),
            description: "Execute a shell command".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute"},
                    "working_dir": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "minimum": 1},
                    "env": {
                        "type": "object",
                        "description": "Environment variables as key-value pairs",
                        "additionalProperties": {"type": "string"}
                    }
                },
                "required": ["command"]
            }),
        },
        loopctl::tool::ToolSchema {
            tool: "git_commit".into(),
            description: "Create a git commit with staged changes".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message"},
                    "amend": {"type": "boolean", "description": "Amend the previous commit"},
                    "co_authors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Co-author emails"
                    }
                },
                "required": ["message"]
            }),
        },
    ]
}

pub fn tool_call_prompts() -> Vec<&'static str> {
    vec![
        // read_file
        "Read the file /etc/hostname",
        "Show me the contents of /var/log/syslog using binary encoding",
        "I need to see what is in /home/user/.bashrc",
        "Open and display the configuration file at /etc/nginx/nginx.conf",
        // search
        "Search the web for rust async patterns",
        "Find information about best coding agent 2026, limit 10 results",
        "Look up how to use tokio channels, max 5 results",
        "Can you search for recent papers on small language model tool use?",
        // write_file
        "Write hello world to /tmp/test.txt",
        "Save the text 'print(42)' to /home/user/script.py",
        "Write 'test content' to /tmp/another.txt and create parent directories",
        "Write a file at /tmp/config.json with content: {\"name\": \"test\", \"value\": 42}",
        "Create an empty file at /tmp/placeholder.txt",
        // run_command
        "Run 'cargo build' in /home/user/project with a 120 second timeout",
        "Execute 'npm test' in the current directory",
        "Run 'echo $GREETING' with GREETING set to hello world in the environment",
        // git_commit
        "Commit with message 'fix: update README' and amend",
        "Create a commit saying 'feat: add streaming support' and credit co-author jane@example.com",
        "Make a git commit with the message 'wip'",
        // ambiguous
        "I need to see the git log, run git log in /home/user/repo",
        "Create a file called notes.txt with the content 'remember to deploy'",
        // no-op (should NOT call a tool)
        "What is 2 plus 2?",
        "Hello, how are you today?",
    ]
}

pub async fn collect_events(
    stream: std::pin::Pin<
        Box<
            dyn futures::Stream<
                    Item = Result<loopctl::stream::StreamEvent, loopctl::api::error::ApiError>,
                > + Send,
        >,
    >,
) -> Option<Vec<loopctl::stream::StreamEvent>> {
    use futures::StreamExt;
    let mut stream = stream;
    let mut events = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(ev) => events.push(ev),
            Err(e) => {
                eprintln!("stream error: {e}");
                return None;
            }
        }
    }
    Some(events)
}

pub fn extract_tool_call(
    events: &[loopctl::stream::StreamEvent],
) -> Option<(String, serde_json::Value)> {
    let mut name: Option<String> = None;
    let mut tool_index: Option<usize> = None;
    let mut json_buf = String::new();

    for ev in events {
        match ev {
            loopctl::stream::StreamEvent::PartStart(ps) => {
                if let Some(loopctl::message::MessagePart::ToolCall { name: n, .. }) = &ps.part {
                    name = Some(n.clone());
                    tool_index = Some(ps.index);
                }
            }
            loopctl::stream::StreamEvent::IndexedDelta(d) => {
                if name.is_some()
                    && Some(d.index) == tool_index
                    && let loopctl::stream::DeltaPart::InputJson { partial_json } = &d.delta
                {
                    json_buf.push_str(partial_json);
                }
            }
            loopctl::stream::StreamEvent::PartStop { index } => {
                let closes_tool = match index {
                    Some(i) => Some(*i) == tool_index,
                    None => true,
                };
                if closes_tool && let Some(n) = name.take() {
                    let input = if json_buf.is_empty() {
                        serde_json::Value::Object(serde_json::Map::new())
                    } else {
                        serde_json::from_str(&json_buf).unwrap_or(serde_json::Value::Null)
                    };
                    json_buf.clear();
                    return Some((n, input));
                }
            }
            _ => {}
        }
    }
    None
}

pub fn extract_text(events: &[loopctl::stream::StreamEvent]) -> String {
    let mut text = String::new();
    for ev in events {
        if let loopctl::stream::StreamEvent::IndexedDelta(d) = ev
            && let loopctl::stream::DeltaPart::Text { text: delta } = &d.delta
        {
            text.push_str(delta);
        }
    }
    text
}

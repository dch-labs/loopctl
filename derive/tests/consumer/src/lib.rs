//! A downstream consumer of `#[derive(Tool)]`, exactly as a user writes
//! it: its own manifest, the `loopctl` dependency renamed (`lc`), and
//! bare `use` paths. The derive's generated code must hold no coupling
//! to the crate's own name — every generated path has to resolve under
//! the rename — and the expansion must compile warning-free under the
//! consumer's own lint set.
use lc::Tool;
use lc::tool::{ToolContext, ToolError, ToolOutput};
use serde::Deserialize;

/// Says hello to whoever the model names.
///
/// A minimal but realistic tool input: one required field, one
/// optional, exercising both sides of the generated schema's required
/// list.
#[derive(Tool, Deserialize)]
pub struct GreetInput {
    /// The person the greeting addresses.
    ///
    /// A plain required `String`, so the schema lists it under
    /// `required`.
    pub who: String,
    /// An optional greeting style the model may omit.
    ///
    /// An `Option`, so the schema lists the property but leaves it out
    /// of `required`.
    pub style: Option<String>,
}

impl GreetInput {
    async fn run(
        &self,
        input: GreetInput,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let _ = ctx;
        Ok(ToolOutput::text(format!("hi {}", input.who)))
    }
}

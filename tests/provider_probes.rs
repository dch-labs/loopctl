//! Provider-truth probe pins at the integration boundary.
//!
//! Everything the probe surface promises must be reachable through a
//! `dyn ApiClient` trait object from outside the crate — discovery
//! (`list_models`), liveness (`ping`), and metadata
//! (`model_metadata`) — because that is the shape every consumer
//! (health checks, model pickers, context budgeters) holds. The mock
//! provider scripts the answers; the classification
//! [`ProbeClass`] turns the failures into the four-way vocabulary a
//! health report branches on.

#![allow(
    dead_code,
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::redundant_clone,
    clippy::clone_on_ref_ptr
)]

#[cfg(feature = "testing")]
mod probe_surface {
    use loopctl::api::ApiClient;
    use loopctl::api::probe::{MetadataSource, ModelInfo, ModelMetadata, ProbeClass};
    use loopctl::testing::MockApiClient;

    #[tokio::test]
    async fn the_probe_surface_is_dyn_safe() {
        // The whole surface must sit on a trait object — a client kind
        // without a listing says so (classified Unsupported), and a
        // scripted listing arrives through the same dyn handle.
        let client: Box<dyn ApiClient> = Box::new(MockApiClient::new("dyn-test"));
        let err = client
            .list_models()
            .await
            .expect_err("an unscripted mock must refuse the listing, loudly");
        assert_eq!(
            ProbeClass::from_error(&err),
            ProbeClass::Unsupported,
            "a client kind with no listing endpoint is a capability fact, \
             not an empty catalog: {err}"
        );

        let scripted: Box<dyn ApiClient> =
            Box::new(MockApiClient::new("dyn-test").with_models(vec![
                ModelInfo::new("scripted-model").with_owned_by("fixture"),
            ]));
        let models = scripted
            .list_models()
            .await
            .expect("the scripted listing arrives");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "scripted-model");
        assert_eq!(models[0].owned_by.as_deref(), Some("fixture"));
        assert_eq!(
            models[0].metadata,
            ModelMetadata::unreported(),
            "a fixture the curated table knows nothing about stays unreported"
        );
    }

    #[tokio::test]
    async fn the_metadata_funnel_answers_through_the_trait_object() {
        let client: Box<dyn ApiClient> = Box::new(MockApiClient::new("dyn-test"));

        let known = client.model_metadata("claude-sonnet-4-5").await;
        assert_eq!(known.context_window, Some(200_000));
        assert_eq!(
            known.capabilities.structured_output,
            Some(false),
            "the Messages family has no native JSON-schema response mode"
        );
        assert_eq!(known.source, MetadataSource::Curated);

        let unknown = client.model_metadata("my-fine-tune-9000").await;
        assert_eq!(
            unknown,
            ModelMetadata::unreported(),
            "the funnel answers a model nobody documented with honesty, \
             not a plausible default"
        );
    }

    #[tokio::test]
    async fn a_scripted_turn_answers_the_default_ping() {
        let client = MockApiClient::new("dyn-test").with_text_response("pong");
        let client: Box<dyn ApiClient> = Box::new(client);
        let report = client
            .ping()
            .await
            .expect("the default ping rides create_message, which every client implements");
        assert_eq!(
            report.rate_limit, None,
            "a mock answer carries no forensics; nothing may be invented"
        );
        assert_eq!(report.request_id, None);
    }
}

#[cfg(all(feature = "openai", feature = "gemini"))]
mod metadata_end_to_end {
    //! The three metadata sources visible end to end over local
    //! servers: a curated family through a listing that says nothing,
    //! a provider-disclosed limit outranking curation, and an unknown
    //! id answering fully unreported.

    use loopctl::api::ApiClient;
    use loopctl::api::probe::{MetadataSource, ModelMetadata};
    use loopctl::provider::{GeminiClient, OpenAiClient};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A one-connection-per-listing server answering every request
    /// with the same canned JSON document.
    async fn serving(body: &'static str) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = vec![0u8; 1024];
                drop(sock.read(&mut buf).await);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                );
                drop(sock.write_all(head.as_bytes()).await);
                drop(sock.write_all(body.as_bytes()).await);
            }
        });
        (addr, server)
    }

    #[tokio::test]
    async fn metadata_reports_context_and_structured_output_capability() {
        // Source one: curation. An OpenAI-compat listing carries bare
        // ids; the metadata each entry reports is the crate's published
        // view, marked as such.
        let (openai_addr, _openai_server) = serving(
            r#"{"data":[{"id":"gpt-4o","object":"model","owned_by":"openai"},{"id":"stranger-9000","object":"model"}]}"#,
        )
        .await;
        let openai = OpenAiClient::builder()
            .with_api_key("k")
            .with_base_url(format!("http://{openai_addr}/v1"))
            .build()
            .unwrap();
        let entries = openai
            .list_models()
            .await
            .expect("the listing arrives through the local server");
        let known = &entries[0];
        assert_eq!(known.id, "gpt-4o");
        assert_eq!(
            known.metadata.context_window,
            Some(128_000),
            "a curated family reports its published window through the listing"
        );
        assert_eq!(
            known.metadata.capabilities.structured_output,
            Some(true),
            "and its capability claims"
        );
        assert_eq!(known.metadata.source, MetadataSource::Curated);
        assert_eq!(
            entries[1].metadata,
            ModelMetadata::unreported(),
            "an id outside every family reads every-leg None — unreported, never plausible"
        );

        // Source two: disclosure. Gemini's catalog carries each model's
        // own input-token limit; it lands as the window, and the
        // provenance marks the window as disclosed while the curated
        // family row contributes the capabilities.
        let (gemini_addr, _gemini_server) = serving(
            r#"{"models":[{"name":"models/gemini-2.5-pro","displayName":"Gemini 2.5 Pro","inputTokenLimit":1048576}]}"#,
        )
        .await;
        let gemini = GeminiClient::builder()
            .with_api_key("g")
            .with_base_url(format!("http://{gemini_addr}/v1beta"))
            .build()
            .unwrap();
        let entries = gemini
            .list_models()
            .await
            .expect("the catalog arrives through the local server");
        assert_eq!(entries[0].id, "gemini-2.5-pro");
        assert_eq!(
            entries[0].metadata.context_window,
            Some(1_048_576),
            "the provider's disclosed limit is the window"
        );
        assert_eq!(
            entries[0].metadata.source,
            MetadataSource::DisclosedAndCurated,
            "disclosed window, curated capabilities — the marker names both"
        );
    }
}

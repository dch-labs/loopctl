//! [`VectorIndex`] over a Qdrant collection.
//!
//! [`QdrantIndex`] is the server tier: a dedicated vector database —
//! self-hosted or Qdrant Cloud — letting semantic memory outlive and
//! outgrow one process, shared across sessions and hosts. The
//! collection is auto-provisioned on first use (created if absent,
//! cosine distance, the configured dimension; an existing collection
//! is adopted only when its unnamed vector configuration matches that
//! shape), and the point id *is*
//! the memory entry's [`Uuid`], so upsert and delete map onto Qdrant's
//! native point operations with no payload indirection. Failures map
//! to [`LoopError::Memory`] with Qdrant's message preserved but
//! bounded; searches emit the same `loopctl.vector.index.search`
//! metric event every loopctl index emits.
//!
//! ```toml
//! loopctl-vector = { version = "0.3", features = ["qdrant"] }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use loopctl::error::LoopError;
use loopctl::memory::vector::{Embedding, VectorIndex, VectorMatch};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::CountPointsBuilder;
use qdrant_client::qdrant::CreateCollectionBuilder;
use qdrant_client::qdrant::DeletePoints;
use qdrant_client::qdrant::DeletePointsBuilder;
use qdrant_client::qdrant::Distance;
use qdrant_client::qdrant::GetPointsBuilder;
use qdrant_client::qdrant::PointId;
use qdrant_client::qdrant::PointStruct;
use qdrant_client::qdrant::PointsIdsList;
use qdrant_client::qdrant::SearchPointsBuilder;
use qdrant_client::qdrant::UpsertPoints;
use qdrant_client::qdrant::UpsertPointsBuilder;
use qdrant_client::qdrant::VectorParamsBuilder;
use qdrant_client::qdrant::Vectors;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
use uuid::Uuid;

use crate::emit_provision_error;
use crate::emit_provisioned;
use crate::memory_error;
use crate::require_dim;
use crate::require_non_blank;

/// The backend name carried in every mapped error and metric label.
///
/// One constant so failures, provision events, and docs all name the
/// store identically.
const BACKEND: &str = "qdrant";

/// A [`VectorIndex`] over one Qdrant collection.
///
/// The collection is provisioned at construction — created if absent
/// with cosine distance and the configured dimension — and the
/// provisioning is idempotent: a concurrent construction that loses
/// the create race sees the winner's collection, validates its
/// vector configuration, and succeeds, a repeated construction is a
/// no-op, and a construction whose provisioning failed retries on the
/// next operation (the guard only caches success). An existing
/// collection is adopted only when it uses the unnamed single-vector
/// layout this index creates, at the configured dimension and cosine
/// distance — anything else rejects at connect instead of failing (or
/// scoring wrongly) on use. Point ids are the caller's [`Uuid`]s
/// directly, so [`add`](VectorIndex::add) is Qdrant upsert-by-id and
/// [`remove`](VectorIndex::remove) is delete-by-id.
///
/// Construct through [`QdrantIndexBuilder`] or
/// [`from_env`](Self::from_env); the client is cloned per operation
/// (a cheap channel handle), so one index can be shared.
///
/// # Example
///
/// ```rust,no_run
/// use loopctl_vector::qdrant::QdrantIndex;
///
/// # async fn demo() -> Result<(), loopctl::error::LoopError> {
/// let index = QdrantIndex::builder("http://localhost:6334", "memories", 128)
///     .connect()
///     .await?;
/// # let _ = index;
/// # Ok(())
/// # }
/// ```
pub struct QdrantIndex {
    /// The shared gRPC client handle.
    ///
    /// Cloned per call — the handle wraps a tonic channel, so clones
    /// share one connection pool.
    client: Qdrant,

    /// The collection every operation targets.
    ///
    /// Provisioned at construction; blank names reject at
    /// [`connect`](QdrantIndexBuilder::connect), and Qdrant itself
    /// enforces its own name grammar.
    collection: String,

    /// The fixed dimensionality of every vector in the collection.
    ///
    /// Set on the collection at creation and enforced client-side on
    /// every add and search.
    dim: usize,

    /// The client-side count bookkeeping backing the trait's sync `len`.
    ///
    /// Seeded with the server's count at construction; see
    /// [`RemoteCount`](crate::RemoteCount) for the exactness contract.
    count: crate::RemoteCount,

    /// Set once this construction has confirmed (or created) the
    /// collection.
    ///
    /// Construction provisions, so operations skip the existence
    /// round trip; only a construction against a server that later
    /// dropped the collection pays it again, on the failing operation's
    /// retry through a fresh construction.
    provisioned: std::sync::OnceLock<()>,
}

impl Clone for QdrantIndex {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            collection: self.collection.clone(),
            dim: self.dim,
            count: self.count.clone(),
            provisioned: provisioned_from(&self.provisioned),
        }
    }
}

/// Copy a provisioning guard's settled state into a fresh guard.
///
/// A clone shares the channel but not the guard cell; propagating the
/// settled state keeps the clone from paying the existence round trip
/// the original already paid.
fn provisioned_from(guard: &std::sync::OnceLock<()>) -> std::sync::OnceLock<()> {
    let fresh = std::sync::OnceLock::new();
    if guard.get().is_some() {
        let _ignored = fresh.set(());
    }
    fresh
}

impl std::fmt::Debug for QdrantIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QdrantIndex")
            .field("collection", &self.collection)
            .field("dim", &self.dim)
            .field("len", &self.count.len())
            .finish_non_exhaustive()
    }
}

/// Builder for [`QdrantIndex`].
///
/// Created by [`QdrantIndex::builder`]; validate-then-connect —
/// [`connect`](Self::connect) rejects blank names and zero dimensions
/// before establishing the gRPC channel.
pub struct QdrantIndexBuilder {
    /// The gRPC endpoint, e.g. `http://localhost:6334`.
    ///
    /// Passed to the client builder verbatim; a blank value rejects
    /// at [`connect`](QdrantIndexBuilder::connect).
    url: String,

    /// The collection name; auto-provisioned at construction.
    ///
    /// A blank name rejects in [`connect`](Self::connect) before any
    /// connection is built; naming beyond that is the server's to
    /// enforce.
    collection: String,

    /// The vector dimensionality every point must carry.
    ///
    /// Always explicit from the embedder's configuration; never
    /// guessed from an existing collection.
    dim: usize,

    /// Optional API key (Qdrant Cloud).
    ///
    /// `None` for an unauthenticated self-hosted deployment.
    api_key: Option<String>,

    /// Optional gRPC timeout applied per call.
    ///
    /// Covers the search path too, so a wedged server surfaces as a
    /// mapped error instead of a hang.
    timeout: Option<Duration>,
}

impl QdrantIndex {
    /// Start building an index over `url` / `collection` at `dim`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use loopctl_vector::qdrant::QdrantIndex;
    ///
    /// let builder = QdrantIndex::builder("http://localhost:6334", "memories", 128);
    /// ```
    #[must_use]
    pub fn builder(
        url: impl Into<String>,
        collection: impl Into<String>,
        dim: usize,
    ) -> QdrantIndexBuilder {
        QdrantIndexBuilder {
            url: url.into(),
            collection: collection.into(),
            dim,
            api_key: None,
            timeout: None,
        }
    }

    /// Build from the `QDRANT_*` environment profile.
    ///
    /// `QDRANT_URL` and `QDRANT_COLLECTION` are required; the error
    /// names whichever is missing. `QDRANT_API_KEY` is optional (set
    /// it for Qdrant Cloud). The dimension is always explicit — it is
    /// a property of the embedder, never guessed from the server.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] naming the first missing variable, or
    /// from a blank value, or from the connection attempt.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use loopctl_vector::qdrant::QdrantIndex;
    ///
    /// # async fn demo() -> Result<(), loopctl::error::LoopError> {
    /// // QDRANT_URL=https://xyz.cloud.qdrant.io:6334 QDRANT_COLLECTION=memories
    /// // QDRANT_API_KEY=… are read from the environment.
    /// let index = QdrantIndex::from_env(128).await?;
    /// # let _ = index;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn from_env(dim: usize) -> Result<Self, LoopError> {
        let url = std::env::var("QDRANT_URL")
            .map_err(|_| LoopError::Memory("qdrant: QDRANT_URL is not set".to_string()))?;
        let collection = std::env::var("QDRANT_COLLECTION")
            .map_err(|_| LoopError::Memory("qdrant: QDRANT_COLLECTION is not set".to_string()))?;
        let api_key = std::env::var("QDRANT_API_KEY").ok();
        Self::builder(url, collection, dim)
            .maybe_api_key(api_key)
            .connect()
            .await
    }

    /// Confirm or create the collection, once per construction.
    ///
    /// Construction provisions and seeds the count, and the guard makes
    /// every later operation skip the existence round trip — one
    /// metadata call per process per collection, not one per upsert.
    /// An already-existing collection (including one a concurrent
    /// construction just won the create race for) is adopted only
    /// after [`validate_existing_collection`](Self::validate_existing_collection)
    /// confirms its vector configuration matches this construction.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the existence check, the creation,
    /// or the validation of an existing collection fails — the message
    /// carries Qdrant's diagnosis, bounded.
    async fn ensure_collection(&self) -> Result<(), LoopError> {
        if self.provisioned.get().is_some() {
            return Ok(());
        }
        if self
            .client
            .collection_exists(self.collection.as_str())
            .await
            .map_err(|error| memory_error(BACKEND, error))?
        {
            self.validate_existing_collection().await?;
            return Ok(());
        }
        let request = CreateCollectionBuilder::new(self.collection.clone())
            .vectors_config(VectorParamsBuilder::new(self.dim as u64, Distance::Cosine));
        if let Err(error) = self.client.create_collection(request).await {
            let lost_race = self
                .client
                .collection_exists(self.collection.as_str())
                .await
                .unwrap_or(false);
            if lost_race {
                self.validate_existing_collection().await?;
                return Ok(());
            }
            emit_provision_error(BACKEND);
            return Err(memory_error(BACKEND, error));
        }
        emit_provisioned(BACKEND, &self.collection);
        Ok(())
    }

    /// Validate an existing collection's vector configuration before
    /// adopting it.
    ///
    /// A handle keyed to the requested dim over a collection whose
    /// configuration disagrees would pass every client-side check and
    /// fail on the server at the first upsert or search — or, with a
    /// non-cosine distance metric, answer with scores that are not
    /// cosine similarities and never error at all. The collection must
    /// use the unnamed single-vector layout this index creates, at the
    /// requested dimension, measured with cosine distance; a named or
    /// multi-vector layout, a mismatched dimension, and a foreign
    /// distance metric all reject with the collection's shape named.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the collection lookup fails or its
    /// vector configuration does not match this construction.
    async fn validate_existing_collection(&self) -> Result<(), LoopError> {
        let response = self
            .client
            .collection_info(self.collection.as_str())
            .await
            .map_err(|error| memory_error(BACKEND, error))?;
        let vectors = response
            .result
            .and_then(|info| info.config)
            .and_then(|config| config.params)
            .and_then(|params| params.vectors_config)
            .and_then(|vectors| vectors.config);
        let Some(qdrant_client::qdrant::vectors_config::Config::Params(params)) = vectors else {
            return Err(LoopError::Memory(format!(
                "{BACKEND}: collection {} uses a named or multi-vector layout this index does not support",
                self.collection
            )));
        };
        let size = usize::try_from(params.size).unwrap_or(usize::MAX);
        if size != self.dim {
            return Err(LoopError::Memory(format!(
                "{BACKEND}: collection {} holds {size}-dimensional vectors but this \
                 construction is configured for {} dimensions",
                self.collection, self.dim
            )));
        }
        if params.distance != i32::from(Distance::Cosine) {
            return Err(LoopError::Memory(format!(
                "{BACKEND}: collection {} does not measure cosine distance",
                self.collection
            )));
        }
        Ok(())
    }

    /// Fetch the true server count once and seed the bookkeeping.
    ///
    /// The async half of the construction-time seeding; every later
    /// `len` answer is a lock read (see
    /// [`RemoteCount`](crate::RemoteCount)).
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the count query fails.
    async fn seed_count(&self) -> Result<(), LoopError> {
        let count = self.server_count().await?;
        self.count.seed(count);
        Ok(())
    }

    /// The server's exact point count for the collection.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the count query fails.
    async fn server_count(&self) -> Result<usize, LoopError> {
        let response = self
            .client
            .count(CountPointsBuilder::new(self.collection.clone()).exact(true))
            .await
            .map_err(|error| memory_error(BACKEND, error))?;
        let count = response.result.map_or(0, |result| {
            usize::try_from(result.count).unwrap_or(usize::MAX)
        });
        Ok(count)
    }

    /// The upsert request for one point, durable-synchronous.
    ///
    /// `wait(true)` makes the server apply the update before the ack
    /// returns — the trait's operations are synchronous, so every
    /// read-after-write (the contract suite, a store's retrieve after
    /// its own store) must observe the write, not race the server's
    /// update queue.
    fn upsert_request(&self, point: PointStruct) -> UpsertPoints {
        UpsertPointsBuilder::new(self.collection.clone(), vec![point])
            .wait(true)
            .build()
    }

    /// Whether the point exists right now.
    ///
    /// The removal path's confirmation: Qdrant's delete carries no
    /// affected-count, so the count bookkeeping asks the server
    /// whether the id is present before treating its removal as a
    /// decrement — a remove of an id no one ever stored stays the
    /// trait's no-op in the count too.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the point lookup fails.
    async fn point_exists(&self, id: Uuid) -> Result<bool, LoopError> {
        let response = self
            .client
            .get_points(GetPointsBuilder::new(
                self.collection.clone(),
                vec![PointId::from(id)],
            ))
            .await
            .map_err(|error| memory_error(BACKEND, error))?;
        Ok(!response.result.is_empty())
    }

    /// The delete request for one id, durable-synchronous.
    ///
    /// The same `wait(true)` discipline as
    /// [`upsert_request`](Self::upsert_request) on the removal path.
    fn delete_request(&self, id: Uuid) -> DeletePoints {
        DeletePointsBuilder::new(self.collection.clone())
            .points(PointsSelectorOneOf::Points(PointsIdsList {
                ids: vec![PointId::from(id)],
            }))
            .wait(true)
            .build()
    }

    /// The point struct for one id/vector pair.
    ///
    /// The id is the point id itself (Qdrant keys points by Uuid), so
    /// upsert and delete map onto native operations with no payload
    /// indirection.
    fn point(id: Uuid, vector: Vec<f32>) -> PointStruct {
        PointStruct {
            id: Some(PointId::from(id)),
            vectors: Some(Vectors::from(qdrant_client::qdrant::Vector::from(vector))),
            payload: std::collections::HashMap::default(),
        }
    }
}

impl QdrantIndexBuilder {
    /// Attach an API key (Qdrant Cloud).
    ///
    /// `None` leaves the connection unauthenticated (the self-hosted
    /// default).
    #[must_use]
    pub fn maybe_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }

    /// Attach an API key (Qdrant Cloud).
    ///
    /// Identical to [`maybe_api_key`](Self::maybe_api_key) for the
    /// always-set case.
    #[must_use]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Apply a per-call timeout to the gRPC channel.
    ///
    /// Covers every request including the search path, so a wedged
    /// server surfaces as [`LoopError::Memory`] instead of a hang.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Validate the configuration and establish the connection.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] for a blank url or collection, a zero
    /// dimension, a failed connection, or an existing collection whose
    /// vector layout, dimension, or distance metric does not match
    /// this construction — the message carries Qdrant's own
    /// diagnosis, bounded.
    pub async fn connect(self) -> Result<QdrantIndex, LoopError> {
        require_non_blank(BACKEND, "url", &self.url)?;
        require_non_blank(BACKEND, "collection", &self.collection)?;
        require_dim(BACKEND, self.dim)?;
        let mut builder = Qdrant::from_url(&self.url).skip_compatibility_check();
        if let Some(key) = self.api_key.as_deref() {
            builder = builder.api_key(key);
        }
        if let Some(timeout) = self.timeout {
            builder = builder.timeout(timeout);
        }
        let client = builder
            .build()
            .map_err(|error| memory_error(BACKEND, error))?;
        let index = QdrantIndex {
            client,
            collection: self.collection,
            dim: self.dim,
            count: crate::RemoteCount::default(),
            provisioned: std::sync::OnceLock::new(),
        };
        index.ensure_collection().await?;
        let _ignored = index.provisioned.set(());
        index.seed_count().await?;
        Ok(index)
    }
}

impl VectorIndex for QdrantIndex {
    fn dim(&self) -> usize {
        self.dim
    }

    fn add(
        &self,
        id: Uuid,
        vector: Embedding,
    ) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        Box::pin(async move {
            if vector.dim() != self.dim {
                return Err(LoopError::Memory(format!(
                    "{BACKEND}: vector dim {} does not match the collection dim {}",
                    vector.dim(),
                    self.dim
                )));
            }
            self.ensure_collection().await?;
            let request = self.upsert_request(Self::point(id, vector.as_slice().to_vec()));
            self.client
                .upsert_points(request)
                .await
                .map_err(|error| memory_error(BACKEND, error))?;
            self.count.note_add(id);
            Ok(())
        })
    }

    fn search(
        &self,
        query: &Embedding,
        k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<VectorMatch>, LoopError>> + Send + '_>> {
        let query = query.as_slice().to_vec();
        Box::pin(async move {
            if query.len() != self.dim {
                return Err(LoopError::Memory(format!(
                    "{BACKEND}: query dim {} does not match the collection dim {}",
                    query.len(),
                    self.dim
                )));
            }
            self.ensure_collection().await?;
            let started = std::time::Instant::now();
            let response = self
                .client
                .search_points(SearchPointsBuilder::new(
                    self.collection.clone(),
                    query,
                    k.min(u32::MAX as usize) as u64,
                ))
                .await
                .map_err(|error| memory_error(BACKEND, error))?;
            let matches: Vec<VectorMatch> = response
                .result
                .into_iter()
                .filter_map(|point| {
                    let id = match point.id {
                        Some(point_id) => match point_id.point_id_options {
                            Some(PointIdOptions::Uuid(text)) => Uuid::parse_str(&text).ok()?,
                            _ => return None,
                        },
                        None => return None,
                    };
                    let score = point.score.clamp(-1.0, 1.0);
                    Some(VectorMatch { id, score })
                })
                .collect();
            let top_score = matches.first().map_or(0.0, |match_| match_.score);
            tracing::debug!(
                target: "loopctl::metrics",
                metric = "loopctl.vector.index.search",
                k,
                returned = matches.len(),
                top_score = %top_score,
                duration_ms = %started.elapsed().as_millis(),
                "vector index search complete"
            );
            Ok(matches)
        })
    }

    fn remove(&self, id: Uuid) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        Box::pin(async move {
            self.ensure_collection().await?;
            if !self.point_exists(id).await? {
                return Ok(());
            }
            let request = self.delete_request(id);
            self.client
                .delete_points(request)
                .await
                .map_err(|error| memory_error(BACKEND, error))?;
            self.count.note_remove(id);
            Ok(())
        })
    }

    fn len(&self) -> usize {
        self.count.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loopctl::testing::EnvGuard;

    #[tokio::test]
    async fn qdrant_builder_rejects_empty_names_and_zero_dim() {
        let rejection = QdrantIndex::builder("", "memories", 128).connect().await;
        assert!(
            rejection
                .as_ref()
                .is_err_and(|error| error.to_string().contains("url")),
            "a blank url names the offending field before any connection: {rejection:?}"
        );
        let rejection = QdrantIndex::builder("http://localhost:6334", "   ", 128)
            .connect()
            .await;
        assert!(
            rejection
                .as_ref()
                .is_err_and(|error| error.to_string().contains("collection")),
            "a blank collection names the offending field before any connection: {rejection:?}"
        );
        let rejection = QdrantIndex::builder("http://localhost:6334", "memories", 0)
            .connect()
            .await;
        assert!(
            rejection
                .as_ref()
                .is_err_and(|error| error.to_string().contains("dim")),
            "a zero dim names the offending field before any connection: {rejection:?}"
        );
    }

    #[test]
    fn qdrant_write_requests_wait_for_server_application() {
        let index = QdrantIndex {
            client: Qdrant::from_url("http://127.0.0.1:6998")
                .skip_compatibility_check()
                .build()
                .expect("the lazy channel builds without a server"),
            collection: "unreachable".to_string(),
            dim: 2,
            count: crate::RemoteCount::default(),
            provisioned: std::sync::OnceLock::new(),
        };
        let upsert = index.upsert_request(QdrantIndex::point(uuid::Uuid::new_v4(), vec![1.0, 0.0]));
        assert_eq!(
            upsert.wait,
            Some(true),
            "the upsert ack returns only after the server applies the write"
        );
        let delete = index.delete_request(uuid::Uuid::new_v4());
        assert_eq!(
            delete.wait,
            Some(true),
            "the delete ack returns only after the server applies the removal"
        );
    }

    #[tokio::test]
    async fn qdrant_env_profile_names_the_missing_var() {
        let env = EnvGuard::acquire(&["QDRANT_URL", "QDRANT_COLLECTION", "QDRANT_API_KEY"]);
        env.remove("QDRANT_URL");
        env.remove("QDRANT_COLLECTION");
        env.remove("QDRANT_API_KEY");
        let error = QdrantIndex::from_env(8).await;
        assert!(
            error
                .as_ref()
                .is_err_and(|error| error.to_string().contains("QDRANT_URL")),
            "the first missing variable is named: {error:?}"
        );
        env.set("QDRANT_URL", "http://localhost:6334");
        let error = QdrantIndex::from_env(8).await;
        assert!(
            error
                .as_ref()
                .is_err_and(|error| error.to_string().contains("QDRANT_COLLECTION")),
            "the second missing variable is named: {error:?}"
        );
    }
}

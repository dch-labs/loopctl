//! [`VectorIndex`] over a pgvector table.
//!
//! [`PgVectorIndex`] is the "your Postgres" tier: vectors live in one
//! table of the database a team already runs and backs up — zero new
//! infrastructure, one more table. The table and its HNSW index are
//! auto-provisioned on first use (`CREATE TABLE IF NOT EXISTS` with a
//! `vector(n)` column plus a `vector_cosine_ops` HNSW index), the
//! table name is validated as a strict SQL identifier because it is
//! interpolated into DDL that cannot be parameterized, an existing
//! table is adopted only when its `embedding` column's dimension
//! matches the requested one **and** its `id` column is a not-null
//! `uuid` under a unique index — the exact key shape the upsert's
//! `ON CONFLICT (id)` arbitrates (a foreign or mismatched table
//! rejects at connect) — and searches order by cosine distance
//! (`<=>`) mapped once into the trait's descending-similarity
//! contract — each search sizes a transaction-local `hnsw.ef_search`
//! to `k`, so a `k` above the extension's 40 candidate-list default
//! still returns every row, and a `k` above the setting's server
//! ceiling of 1000 rejects rather than returning short. Failures map
//! to [`LoopError::Memory`] with Postgres's message preserved but
//! bounded; searches emit the same `loopctl.vector.index.search`
//! metric event every loopctl index emits.
//!
//! ```toml
//! loopctl-vector = { version = "0.3", features = ["pgvector"] }
//! ```

use std::future::Future;
use std::pin::Pin;

use loopctl::error::LoopError;
use loopctl::memory::vector::{Embedding, VectorIndex, VectorMatch};
use pgvector::Vector;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

use crate::RemoteCount;
use crate::emit_provision_error;
use crate::emit_provisioned;
use crate::memory_error;
use crate::require_dim;
use crate::require_non_blank;

/// The backend name carried in every mapped error and metric label.
///
/// One constant so failures, provision events, and docs all name the
/// store identically.
const BACKEND: &str = "pgvector";

/// The pgvector extension's default HNSW candidate-list size.
///
/// The extension's `hnsw.ef_search` server default of `40`; a search
/// whose `k` exceeds it can come back short on an index-scan plan
/// unless the candidate list is raised for the query's transaction, so
/// every search sizes it to `k` (never below this default).
const DEFAULT_EF_SEARCH: usize = 40;

/// The server-enforced ceiling on `hnsw.ef_search`.
///
/// The pgvector extension rejects values outside `1..=1000` for the
/// `hnsw.ef_search` GUC (the bound of the extension versions this
/// crate runs against — 0.8.x on the tested images); a `k` above the
/// ceiling therefore rejects client-side with the ceiling named
/// rather than silently returning short results.
const MAX_EF_SEARCH: usize = 1000;

/// A [`VectorIndex`] over one pgvector table.
///
/// The table is provisioned at construction — created if absent with
/// an `id UUID PRIMARY KEY` key, a `embedding VECTOR(<dim>)` column,
/// and an HNSW `vector_cosine_ops` index — and an existing table is
/// adopted only when its `embedding` column's dimension matches and
/// its `id` column is a not-null `uuid` under a unique index, the key
/// shape the upsert owns. The client-side count is seeded from the
/// same construction — client-side bookkeeping whose exactness
/// contract the count type's docs state.
/// [`add`](VectorIndex::add) is `INSERT … ON CONFLICT (id) DO UPDATE`,
/// so re-adding under one id replaces without leaking, and
/// [`search`](VectorIndex::search) maps Postgres's ascending cosine
/// distance into the trait's descending-similarity order, sizing a
/// transaction-local `hnsw.ef_search` to `k` so the candidate list
/// never caps results below the rows the store holds (a `k` above
/// the 1000 ceiling rejects with the ceiling named). Vectors and
/// queries must be finite with a non-zero norm — cosine over a
/// degenerate vector is undefined, and the index scan would drop the
/// row silently, so the rejection is loud and client-side.
///
/// Construct through [`PgVectorIndexBuilder`] or
/// [`from_env`](Self::from_env).
///
/// # Example
///
/// ```rust,no_run
/// use loopctl_vector::pgvector::PgVectorIndexBuilder;
///
/// # async fn demo() -> Result<(), loopctl::error::LoopError> {
/// let index = loopctl_vector::pgvector::PgVectorIndexBuilder::new(
///     "postgres://postgres:postgres@localhost:5432/postgres",
///     "agent_memories",
///     128,
/// )
/// .connect()
/// .await?;
/// # let _ = index;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct PgVectorIndex {
    /// The shared connection pool.
    ///
    /// Sized by the builder's `max_connections`; every statement runs
    /// against it.
    pool: PgPool,

    /// The table every statement targets; a validated strict identifier.
    ///
    /// Interpolated into DDL that cannot be parameterized, which is
    /// exactly why the strict rule exists.
    table: String,

    /// The fixed dimensionality of the embedding column.
    ///
    /// Baked into the provisioned `VECTOR(n)` type and enforced
    /// client-side on every add and search.
    dim: usize,

    /// The client-side count bookkeeping backing the trait's sync `len`.
    ///
    /// Seeded from the table's row count at construction and shared
    /// with every clone of this handle, so one process answers one
    /// count; see [`RemoteCount`](crate::RemoteCount) for the
    /// exactness contract.
    count: std::sync::Arc<RemoteCount>,
}

impl std::fmt::Debug for PgVectorIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PgVectorIndex")
            .field("table", &self.table)
            .field("dim", &self.dim)
            .field("len", &self.count.len())
            .finish_non_exhaustive()
    }
}

/// Builder for [`PgVectorIndex`].
///
/// Created by [`PgVectorIndexBuilder::new`]; validate-then-connect —
/// [`connect`](Self::connect) rejects blank URLs, loose table names,
/// and zero dimensions before opening the pool, provisioning the
/// table, and seeding the count.
pub struct PgVectorIndexBuilder {
    /// The Postgres connection string.
    ///
    /// Passed to the pool builder verbatim; a blank value rejects at
    /// [`connect`](PgVectorIndexBuilder::connect).
    url: String,

    /// The table name; auto-provisioned on connect.
    ///
    /// A strict identifier — the builder rejects anything else before
    /// opening a connection.
    table: String,

    /// The vector dimensionality every row must carry.
    ///
    /// Always explicit from the embedder's configuration; never
    /// guessed from an existing column.
    dim: usize,

    /// The pool's maximum connections.
    ///
    /// The default fits one agent process; raise it when several
    /// tasks share one index.
    max_connections: u32,
}

impl PgVectorIndexBuilder {
    /// Start building an index over `url` / `table` at `dim`.
    ///
    /// Nothing connects here — validation and the pool open happen in
    /// [`connect`](Self::connect), so building is cheap and
    /// infallible.
    #[must_use]
    pub fn new(url: impl Into<String>, table: impl Into<String>, dim: usize) -> Self {
        Self {
            url: url.into(),
            table: table.into(),
            dim,
            max_connections: 4,
        }
    }

    /// Set the pool's maximum connections.
    ///
    /// The default (`4`) fits one agent process; raise it when several
    /// tasks share one index.
    #[must_use]
    pub fn max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }

    /// Validate the configuration, provision the table, and connect.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] for a blank url or table name, a table
    /// name that is not a strict SQL identifier, a zero dimension, an
    /// unreachable server, a missing `pgvector` extension, a failed
    /// provisioning, or an existing table whose `embedding` column is
    /// missing, untyped, or dimensioned differently from the request,
    /// or whose `id` column is not a not-null `uuid` under a unique
    /// index — the message carries Postgres's own diagnosis, bounded.
    pub async fn connect(self) -> Result<PgVectorIndex, LoopError> {
        require_non_blank(BACKEND, "url", &self.url)?;
        require_non_blank(BACKEND, "table", &self.table)?;
        require_dim(BACKEND, self.dim)?;
        let table = crate::strict_identifier(BACKEND, self.table.as_str())?;
        let pool = PgPoolOptions::new()
            .max_connections(self.max_connections)
            .connect(self.url.as_str())
            .await
            .map_err(|error| {
                emit_provision_error(BACKEND);
                memory_error(BACKEND, error)
            })?;
        let index = PgVectorIndex {
            pool,
            table,
            dim: self.dim,
            count: std::sync::Arc::new(RemoteCount::default()),
        };
        index.provision().await?;
        index.seed_count().await?;
        Ok(index)
    }
}

impl PgVectorIndex {
    /// Build from the `PG_VECTOR_*` environment profile.
    ///
    /// `PG_VECTOR_URL` and `PG_VECTOR_TABLE` are required; the error
    /// names whichever is missing. The dimension is always explicit —
    /// it is a property of the embedder, never guessed from the
    /// server.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] naming the first missing variable, or
    /// from the same validation and connection set as
    /// [`connect`](PgVectorIndexBuilder::connect).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use loopctl_vector::pgvector::PgVectorIndex;
    ///
    /// # async fn demo() -> Result<(), loopctl::error::LoopError> {
    /// // PG_VECTOR_URL=postgres://…/db PG_VECTOR_TABLE=agent_memories
    /// // are read from the environment.
    /// let index = PgVectorIndex::from_env(128).await?;
    /// # let _ = index;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn from_env(dim: usize) -> Result<Self, LoopError> {
        let url = std::env::var("PG_VECTOR_URL")
            .map_err(|_| LoopError::Memory("pgvector: PG_VECTOR_URL is not set".to_string()))?;
        let table = std::env::var("PG_VECTOR_TABLE")
            .map_err(|_| LoopError::Memory("pgvector: PG_VECTOR_TABLE is not set".to_string()))?;
        PgVectorIndexBuilder::new(url, table, dim).connect().await
    }

    /// Load the `vector` extension, tolerating the concurrent-create race.
    ///
    /// `CREATE EXTENSION IF NOT EXISTS` does not serialize two in-flight
    /// creations — the loser hits the catalog's unique index — so a
    /// failure is accepted exactly when the extension exists afterwards
    /// (another connection won the race). A follow-up probe that cannot
    /// answer (the server unreachable again) counts and returns the
    /// creation's own diagnosis rather than guessing.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when creation fails and the extension is
    /// absent afterwards — the race lost, the load missing — or the
    /// existence probe itself failed.
    async fn ensure_extension(&self) -> Result<(), LoopError> {
        let create = "CREATE EXTENSION IF NOT EXISTS vector";
        if let Err(error) = sqlx::query(sqlx::AssertSqlSafe(create))
            .execute(&self.pool)
            .await
        {
            let extension_present = match sqlx::query_as::<_, (String,)>(
                "SELECT extname FROM pg_extension WHERE extname = 'vector'",
            )
            .fetch_optional(&self.pool)
            .await
            {
                Ok(rows) => rows.is_some(),
                Err(_) => false,
            };
            if extension_present {
                return Ok(());
            }
            emit_provision_error(BACKEND);
            return Err(memory_error(BACKEND, error));
        }
        Ok(())
    }

    /// Create the table and its index if absent, event only on creation.
    ///
    /// The probe-and-create pair runs under a transaction-scoped
    /// advisory lock keyed on the table name, so even two concurrent
    /// first constructions within one process serialize: the lock
    /// holder creates and emits, the follower's probe sees the table
    /// and stays silent — the once-per-target event holds under
    /// concurrency, not just sequentially. The lock key is
    /// `hashtext`'s 32-bit hash of the name, so two distinct table
    /// names that collide on the hash merely serialize against each
    /// other — harmless, never a correctness effect. The once-per-
    /// target event is keyed by the configured name, so two spellings
    /// that alias one relation (Postgres folds unquoted identifier
    /// case) each emit once. `IF NOT EXISTS` on both statements
    /// remains the cross-process backstop. A probe that finds the
    /// table already present routes through
    /// [`validate_existing_table`](Self::validate_existing_table)
    /// before adoption, so the follow-on constructions reject a
    /// mismatched or foreign table instead of silently accepting it.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the extension load, the lock, the
    /// probe, the validation of an existing table, the DDL, or the
    /// commit fails — the message carries Postgres's diagnosis.
    async fn provision(&self) -> Result<(), LoopError> {
        self.ensure_extension().await?;
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| memory_error(BACKEND, error))?;
        sqlx::query("SELECT pg_advisory_xact_lock(hashtext($1))")
            .bind(self.table.as_str())
            .execute(&mut *tx)
            .await
            .map_err(|error| memory_error(BACKEND, error))?;
        let existing: Option<(Option<String>,)> = sqlx::query_as("SELECT to_regclass($1)::text")
            .bind(self.table.as_str())
            .fetch_optional(&mut *tx)
            .await
            .map_err(|error| memory_error(BACKEND, error))?;
        if existing.is_some_and(|(name,)| name.is_some()) {
            return self.validate_existing_table(&mut tx).await;
        }
        let ddl = [
            format!(
                "CREATE TABLE IF NOT EXISTS {table} (\
                 id UUID PRIMARY KEY, embedding VECTOR({dim}))",
                table = self.table,
                dim = self.dim
            ),
            format!(
                "CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw \
                 ON {table} USING hnsw (embedding vector_cosine_ops)",
                table = self.table
            ),
        ];
        for statement in &ddl {
            if let Err(error) = sqlx::query(sqlx::AssertSqlSafe(statement.as_str()))
                .execute(&mut *tx)
                .await
            {
                emit_provision_error(BACKEND);
                return Err(memory_error(BACKEND, error));
            }
        }
        tx.commit()
            .await
            .map_err(|error| memory_error(BACKEND, error))?;
        emit_provisioned(BACKEND, &self.table);
        Ok(())
    }

    /// Validate an existing table's shape before adopting it.
    ///
    /// A handle keyed to the requested dim over a table whose shape
    /// disagrees would pass every client-side check and fail on the
    /// server at the first add or search, so the shape rejects at
    /// connect instead. The probe runs inside the caller's
    /// advisory-locked transaction and reads the catalog: the table
    /// must carry a live `vector(n)`-typed `embedding` column whose
    /// `n` equals the requested dim, **and** an `id` column of type
    /// `uuid` that is not null and covered by a unique non-partial
    /// index over exactly that column — the key the upsert's
    /// `ON CONFLICT (id)` arbitrates. A foreign table without the
    /// embedding column, an untyped `vector` column, a mismatched
    /// `n`, a text- or integer-keyed `id`, a nullable `id`, and an
    /// `id` whose uniqueness rests on a partial index all reject with
    /// the table's actual shape named.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when a probe fails or the table's shape
    /// does not match what this index requires.
    async fn validate_existing_table(&self, tx: &mut sqlx::PgConnection) -> Result<(), LoopError> {
        self.validate_embedding_column(tx).await?;
        self.validate_id_column(tx).await
    }

    /// Validate the `embedding` column's vector type and dimension.
    ///
    /// The first half of adoption validation: the column must be a
    /// live `vector(n)` whose `n` equals the requested dim.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the probe fails or the column is
    /// missing, untyped, or dimensioned differently.
    async fn validate_embedding_column(
        &self,
        tx: &mut sqlx::PgConnection,
    ) -> Result<(), LoopError> {
        let column: Option<(i32,)> = sqlx::query_as(
            "SELECT a.atttypmod FROM pg_attribute a \
             WHERE a.attrelid = $1::regclass AND a.attname = 'embedding' \
             AND a.atttypid = 'vector'::regtype AND NOT a.attisdropped",
        )
        .bind(self.table.as_str())
        .fetch_optional(&mut *tx)
        .await
        .map_err(|error| memory_error(BACKEND, error))?;
        let typmod = column.map_or(0, |(value,)| value);
        if typmod > 0 {
            if usize::try_from(typmod).is_ok_and(|dim| dim == self.dim) {
                return Ok(());
            }
            return Err(LoopError::Memory(format!(
                "{BACKEND}: table {} exists and holds VECTOR({typmod}) embeddings; \
                 this construction is configured for {} dimensions",
                self.table, self.dim
            )));
        }
        let shape = if typmod == 0 {
            "has no vector-typed embedding column"
        } else {
            "has an untyped vector embedding column"
        };
        Err(LoopError::Memory(format!(
            "{BACKEND}: table {} exists and {shape}; this construction is configured \
             for {} dimensions",
            self.table, self.dim
        )))
    }

    /// Validate the `id` column's key shape.
    ///
    /// The second half of adoption validation: the column must be
    /// typed `uuid`, marked not null, and covered by a unique,
    /// valid, immediate, non-partial index over exactly that one
    /// column — the shape `ON CONFLICT (id)` needs to arbitrate the
    /// upsert. Anything else (a text key, a nullable unique, a
    /// deferrable constraint whose uniqueness is checked at commit,
    /// uniqueness from a partial index) writes rows the handle could
    /// never read back.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the probe fails or the key shape
    /// does not match.
    async fn validate_id_column(&self, tx: &mut sqlx::PgConnection) -> Result<(), LoopError> {
        let key: Option<(bool, bool)> = sqlx::query_as(
            "SELECT a.attnotnull, EXISTS ( \
                SELECT 1 FROM pg_index i \
                WHERE i.indrelid = a.attrelid AND i.indisunique AND i.indisvalid \
                  AND i.indimmediate \
                  AND i.indpred IS NULL AND i.indnkeyatts = 1 \
                  AND (string_to_array(i.indkey::text, ' '))[1] = a.attnum::text \
             ) FROM pg_attribute a \
             WHERE a.attrelid = $1::regclass AND a.attname = 'id' \
             AND a.atttypid = 'uuid'::regtype AND NOT a.attisdropped",
        )
        .bind(self.table.as_str())
        .fetch_optional(&mut *tx)
        .await
        .map_err(|error| memory_error(BACKEND, error))?;
        let Some((not_null, uniquely_indexed)) = key else {
            return Err(LoopError::Memory(format!(
                "{BACKEND}: table {} exists and has no uuid-typed id column; \
                 this index upserts with ON CONFLICT (id) over a uuid key",
                self.table
            )));
        };
        if !not_null {
            return Err(LoopError::Memory(format!(
                "{BACKEND}: table {} exists and its id column is nullable; \
                 this index requires a not-null uuid id under a unique index",
                self.table
            )));
        }
        if !uniquely_indexed {
            return Err(LoopError::Memory(format!(
                "{BACKEND}: table {} exists and its id column has no unique index over \
                 exactly id; ON CONFLICT (id) cannot arbitrate the upsert",
                self.table
            )));
        }
        Ok(())
    }

    /// Fetch the true row count once and seed the bookkeeping.
    ///
    /// # Errors
    ///
    /// [`LoopError::Memory`] when the count query fails.
    async fn seed_count(&self) -> Result<(), LoopError> {
        let count = self
            .server_count()
            .await
            .map_err(|error| memory_error(BACKEND, error))?;
        self.count.seed(count);
        Ok(())
    }

    /// The table's exact row count.
    ///
    /// # Errors
    ///
    /// Propagates sqlx's own error for the caller to map once.
    async fn server_count(&self) -> Result<usize, sqlx::Error> {
        let sql = format!("SELECT count(*) FROM {}", self.table);
        let (count,): (i64,) = sqlx::query_as(sqlx::AssertSqlSafe(sql))
            .fetch_one(&self.pool)
            .await?;
        Ok(usize::try_from(count).unwrap_or(usize::MAX))
    }
}

impl VectorIndex for PgVectorIndex {
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
                    "{BACKEND}: vector dim {} does not match the column dim {}",
                    vector.dim(),
                    self.dim
                )));
            }
            crate::require_usable_vector(BACKEND, vector.as_slice())?;
            let sql = format!(
                "INSERT INTO {table} (id, embedding) VALUES ($1, $2) \
                 ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
                table = self.table
            );
            let params = Vector::from(vector.as_slice().to_vec());
            if let Err(error) = sqlx::query(sqlx::AssertSqlSafe(sql))
                .bind(id)
                .bind(params)
                .execute(&self.pool)
                .await
            {
                return Err(memory_error(BACKEND, error));
            }
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
                    "{BACKEND}: query dim {} does not match the column dim {}",
                    query.len(),
                    self.dim
                )));
            }
            crate::require_usable_vector(BACKEND, &query)?;
            if k > MAX_EF_SEARCH {
                return Err(LoopError::Memory(format!(
                    "{BACKEND}: k {k} exceeds the hnsw.ef_search ceiling of {MAX_EF_SEARCH}; \
                     lower k"
                )));
            }
            let started = std::time::Instant::now();
            if k == 0 {
                emit_search_metric(BACKEND, k, &[], started);
                return Ok(Vec::new());
            }
            let sql = format!(
                "SELECT id, CAST(1 - (embedding <=> $1::vector) AS REAL) AS score \
                 FROM {table} ORDER BY embedding <=> $1::vector LIMIT $2",
                table = self.table
            );
            let mut tx = self
                .pool
                .begin()
                .await
                .map_err(|error| memory_error(BACKEND, error))?;
            sqlx::query("SELECT set_config('hnsw.ef_search', $1, true)")
                .bind(k.max(DEFAULT_EF_SEARCH).to_string())
                .execute(&mut *tx)
                .await
                .map_err(|error| memory_error(BACKEND, error))?;
            let rows: Vec<(Uuid, f32)> = sqlx::query_as(sqlx::AssertSqlSafe(sql))
                .bind(Vector::from(query))
                .bind(i64::try_from(k).unwrap_or(i64::MAX))
                .fetch_all(&mut *tx)
                .await
                .map_err(|error| memory_error(BACKEND, error))?;
            tx.rollback()
                .await
                .map_err(|error| memory_error(BACKEND, error))?;
            let matches: Vec<VectorMatch> = rows
                .into_iter()
                .map(|(id, score)| VectorMatch {
                    id,
                    score: score.clamp(-1.0, 1.0),
                })
                .collect();
            emit_search_metric(BACKEND, k, &matches, started);
            Ok(matches)
        })
    }

    fn remove(&self, id: Uuid) -> Pin<Box<dyn Future<Output = Result<(), LoopError>> + Send + '_>> {
        Box::pin(async move {
            let sql = format!("DELETE FROM {table} WHERE id = $1", table = self.table);
            let result = sqlx::query(sqlx::AssertSqlSafe(sql))
                .bind(id)
                .execute(&self.pool)
                .await
                .map_err(|error| memory_error(BACKEND, error))?;
            if result.rows_affected() > 0 {
                self.count.note_remove(id);
            }
            Ok(())
        })
    }

    fn len(&self) -> usize {
        self.count.len()
    }
}

/// Emit the search counter event with the backend labeled.
///
/// The same event shape every loopctl index emits — `k`, `returned`,
/// `top_score`, `duration_ms` — plus this crate's `backend` label, so
/// the stream distinguishes the external tiers the way the in-process
/// indexes carry their `provider` label.
fn emit_search_metric(
    backend: &str,
    k: usize,
    matches: &[VectorMatch],
    started: std::time::Instant,
) {
    let top_score = matches.first().map_or(0.0, |match_| match_.score);
    tracing::debug!(
        target: "loopctl::metrics",
        metric = "loopctl.vector.index.search",
        backend,
        k,
        returned = matches.len(),
        top_score = %top_score,
        duration_ms = %started.elapsed().as_millis(),
        "vector index search complete"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use loopctl::testing::EnvGuard;

    #[test]
    fn pgvector_builder_rejects_empty_names_zero_dim_and_loose_identifiers() {
        for table in [
            "",
            "   ",
            "mem-vectors",
            "mem.vectors",
            "\"quoted\"",
            "schema.mem",
        ] {
            let rejection = crate::strict_identifier("pgvector", table);
            assert!(
                rejection.is_err(),
                "the table name {table:?} must reject as a non-identifier"
            );
        }
        assert!(
            crate::strict_identifier("pgvector", "mem_vectors").is_ok(),
            "a strict identifier accepts"
        );
        assert!(
            crate::strict_identifier("pgvector", "_private").is_ok(),
            "a leading underscore accepts"
        );
    }

    #[tokio::test]
    async fn pgvector_connect_rejects_blank_url_before_any_io() {
        let rejection = PgVectorIndexBuilder::new("   ", "mem", 8).connect().await;
        assert!(
            rejection
                .as_ref()
                .is_err_and(|error| error.to_string().contains("url")),
            "a blank url names the offending field before any connection: {rejection:?}"
        );
        let rejection = PgVectorIndexBuilder::new("postgres://x", "mem", 0)
            .connect()
            .await;
        assert!(
            rejection
                .as_ref()
                .is_err_and(|error| error.to_string().contains("dim")),
            "a zero dim names the offending field before any connection: {rejection:?}"
        );
    }

    #[tokio::test]
    async fn pgvector_env_profile_names_the_missing_var() {
        let env = EnvGuard::acquire(&["PG_VECTOR_URL", "PG_VECTOR_TABLE"]);
        env.remove("PG_VECTOR_URL");
        env.remove("PG_VECTOR_TABLE");
        let error = PgVectorIndex::from_env(8).await;
        assert!(
            error
                .as_ref()
                .is_err_and(|error| error.to_string().contains("PG_VECTOR_URL")),
            "the first missing variable is named: {error:?}"
        );
        env.set("PG_VECTOR_URL", "postgres://localhost/x");
        let error = PgVectorIndex::from_env(8).await;
        assert!(
            error
                .as_ref()
                .is_err_and(|error| error.to_string().contains("PG_VECTOR_TABLE")),
            "the second missing variable is named: {error:?}"
        );
    }
}

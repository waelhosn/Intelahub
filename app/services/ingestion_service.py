import logging
import time

from app.core.errors import APIError
from app.core.settings import Settings
from app.models.ingest import IngestRequest, IngestResponse
from app.models.movie import MovieRecord
from app.services.embedding_service import EmbeddingProvider
from app.services.normalizer import normalize_tmdb_movie
from app.services.tmdb_client import TMDBClient
from app.services.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        tmdb_client: TMDBClient,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaVectorStore,
    ):
        self.settings = settings
        self.tmdb_client = tmdb_client
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def ingest(self, payload: IngestRequest) -> IngestResponse:
        if self.settings.max_target_count > 0 and payload.target_count > self.settings.max_target_count:
            raise APIError(
                code="target_count_too_large",
                message="target_count exceeds configured maximum",
                status_code=400,
                details={"max_target_count": self.settings.max_target_count},
            )

        start = time.perf_counter()
        logger.info(
            "Ingestion started",
            extra={
                "target_count": payload.target_count,
                "force_reindex": payload.force_reindex,
                "embedding_provider": self.settings.embedding_provider,
                "openai_embedding_model": self.settings.openai_embedding_model,
                "gemini_embedding_model": self.settings.gemini_embedding_model,
                "batch_size": self.settings.embedding_batch_size,
            },
        )
        if payload.force_reindex:
            logger.info("Resetting vector collection before ingest")
            self.vector_store.reset_collection()

        movie_ids = await self.tmdb_client.discover_movie_ids(payload.target_count)
        logger.info("TMDB discover completed", extra={"discovered_ids": len(movie_ids)})
        detailed_rows = await self.tmdb_client.fetch_detailed_movies(movie_ids)
        logger.info("TMDB enrichment completed", extra={"detailed_rows": len(detailed_rows)})

        normalized_movies: list[MovieRecord] = []
        failed = 0
        duplicates = 0
        seen_ids: set[int] = set()

        for details, credits in detailed_rows:
            movie = normalize_tmdb_movie(details, credits, cast_limit=10)
            if movie is None:
                failed += 1
                continue
            if movie.id in seen_ids:
                duplicates += 1
                continue
            seen_ids.add(movie.id)
            normalized_movies.append(movie)

        logger.info(
            "Normalization completed",
            extra={
                "normalized_movies": len(normalized_movies),
                "failed_normalization": failed,
                "duplicate_ids": duplicates,
            },
        )

        if not normalized_movies:
            raise APIError("ingest_empty", "No valid movie records produced from TMDB", status_code=502)

        ingested = 0
        batch_size = self.settings.embedding_batch_size

        for i in range(0, len(normalized_movies), batch_size):
            batch = normalized_movies[i : i + batch_size]
            batch_no = (i // batch_size) + 1
            total_batches = (len(normalized_movies) + batch_size - 1) // batch_size
            try:
                embeddings = await self.embedding_provider.embed_texts([movie.embedding_text for movie in batch])
            except APIError:
                logger.exception(
                    "Embedding batch failed",
                    extra={
                        "batch_no": batch_no,
                        "total_batches": total_batches,
                        "batch_size": len(batch),
                        "embedding_provider": self.settings.embedding_provider,
                    },
                )
                raise

            try:
                self.vector_store.upsert_movies(batch, embeddings)
            except APIError:
                logger.exception(
                    "Vector upsert failed",
                    extra={
                        "batch_no": batch_no,
                        "total_batches": total_batches,
                        "batch_size": len(batch),
                    },
                )
                raise
            ingested += len(batch)
            logger.info(
                "Ingestion batch completed",
                extra={
                    "batch_no": batch_no,
                    "total_batches": total_batches,
                    "ingested_so_far": ingested,
                    "movies_in_batch": [{"id": movie.id, "title": movie.title} for movie in batch],
                },
            )

        duration_ms = int((time.perf_counter() - start) * 1000)
        skipped = max(len(movie_ids) - ingested - failed, 0)
        logger.info(
            "Ingestion completed",
            extra={
                "ingested": ingested,
                "skipped": skipped,
                "failed": failed,
                "duration_ms": duration_ms,
            },
        )

        return IngestResponse(ingested=ingested, skipped=skipped, failed=failed, duration_ms=duration_ms)

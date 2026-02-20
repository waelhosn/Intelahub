import logging
import re
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from app.core.errors import APIError
from app.core.settings import Settings
from app.models.movie import MovieRecord, RetrievedMovie
from app.models.query import QueryFilters

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _genre_key(genre: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", genre.lower()).strip("_")
    return f"genre_{token}" if token else "genre_unknown"


class ChromaVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection: Collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": settings.chroma_distance_metric},
        )
        self.distance_metric = self._resolve_distance_metric()
        if self.distance_metric != settings.chroma_distance_metric:
            logger.warning(
                "Chroma collection metric differs from configured metric",
                extra={
                    "configured_metric": settings.chroma_distance_metric,
                    "active_metric": self.distance_metric,
                    "collection_name": settings.chroma_collection_name,
                },
            )
            if self.collection.count() == 0:
                logger.info("Recreating empty collection to apply configured distance metric")
                self.reset_collection()

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.settings.chroma_collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"hnsw:space": self.settings.chroma_distance_metric},
        )
        self.distance_metric = self._resolve_distance_metric()

    @staticmethod
    def _to_metadata(movie: MovieRecord) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "id": movie.id,
            "title": movie.title,
            "overview": movie.overview,
            "genres": "|".join(movie.genres),
            "cast": "|".join(movie.cast),
            "release_year": movie.release_year if movie.release_year is not None else -1,
            "rating": movie.rating if movie.rating is not None else -1.0,
        }
        for genre in movie.genres:
            metadata[_genre_key(genre)] = True
        return metadata

    def upsert_movies(self, movies: list[MovieRecord], embeddings: list[list[float]]) -> None:
        if len(movies) != len(embeddings):
            raise APIError("vector_store_error", "Movies and embeddings length mismatch", status_code=500)
        if len(embeddings) > 0:
            incoming_dim = self._vector_dim(embeddings[0])
            if incoming_dim is None:
                raise APIError("vector_store_error", "Incoming embedding vector is empty", status_code=500)
            existing_dim = self._existing_dimension()
            if existing_dim is not None and existing_dim != incoming_dim:
                raise APIError(
                    "embedding_dimension_mismatch",
                    "Embedding dimension does not match existing index. Run ingest with force_reindex=true.",
                    status_code=400,
                    details={"existing_dimension": existing_dim, "incoming_dimension": incoming_dim},
                )

        ids = [str(movie.id) for movie in movies]
        documents = [movie.embedding_text for movie in movies]
        metadatas = [self._to_metadata(movie) for movie in movies]

        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def _existing_dimension(self) -> int | None:
        if self.collection.count() == 0:
            return None
        sample = self.collection.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings")
        if embeddings is None:
            return None
        if len(embeddings) == 0:
            return None
        return self._vector_dim(embeddings[0])

    def _resolve_distance_metric(self) -> str:
        metadata = getattr(self.collection, "metadata", None) or {}
        metric = metadata.get("hnsw:space")
        if metric in {"cosine", "l2", "ip"}:
            return metric
        return "l2"

    @staticmethod
    def _vector_dim(vector: Any) -> int | None:
        if vector is None:
            return None
        try:
            dim = len(vector)
        except TypeError:
            return None
        return dim if dim > 0 else None

    @staticmethod
    def _build_where(filters: QueryFilters) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []

        if filters.min_year is not None:
            clauses.append({"release_year": {"$gte": filters.min_year}})
        if filters.max_year is not None:
            clauses.append({"release_year": {"$lte": filters.max_year}})
        if filters.min_rating is not None:
            clauses.append({"rating": {"$gte": filters.min_rating}})

        if filters.genres:
            genre_clauses = [{_genre_key(genre): True} for genre in filters.genres]
            if len(genre_clauses) == 1:
                clauses.append(genre_clauses[0])
            else:
                clauses.append({"$or": genre_clauses})

        if not clauses:
            return None

        if len(clauses) == 1:
            return clauses[0]

        return {"$and": clauses}

    @staticmethod
    def _parse_movie(metadata: dict[str, Any], semantic_score: float) -> RetrievedMovie:
        genres = [g for g in str(metadata.get("genres", "")).split("|") if g]
        cast = [c for c in str(metadata.get("cast", "")).split("|") if c]

        release_year = int(metadata["release_year"]) if metadata.get("release_year", -1) != -1 else None
        rating = float(metadata["rating"]) if metadata.get("rating", -1.0) >= 0 else None

        return RetrievedMovie(
            id=int(metadata["id"]),
            title=str(metadata.get("title", "")),
            overview=str(metadata.get("overview", "")),
            genres=genres,
            cast=cast,
            release_year=release_year,
            rating=rating,
            vector_score=semantic_score,
            lexical_score=0.0,
            fused_score=semantic_score,
            cross_encoder_score=None,
            semantic_score=semantic_score,
            rerank_score=semantic_score,
        )

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        return set(TOKEN_RE.findall((value or "").lower()))

    def _lexical_score(self, query: str, title: str, overview: str) -> float:
        normalized_query = re.sub(r"[^a-z0-9]+", "", query.lower())
        normalized_title = re.sub(r"[^a-z0-9]+", "", title.lower())

        score = 0.0
        if normalized_query and normalized_query == normalized_title:
            score += 1.0
        elif normalized_query and normalized_query in normalized_title:
            score += 0.7

        query_tokens = self._tokenize(query)
        if query_tokens:
            title_tokens = self._tokenize(title)
            overview_tokens = self._tokenize(overview)
            title_overlap = len(query_tokens.intersection(title_tokens)) / len(query_tokens)
            overview_overlap = len(query_tokens.intersection(overview_tokens)) / len(query_tokens)
            score += 0.25 * title_overlap
            score += 0.1 * overview_overlap

        return max(0.0, min(score, 1.0))

    def lexical_search_movies(self, query: str, filters: QueryFilters, top_k: int) -> list[RetrievedMovie]:
        where = self._build_where(filters)
        result = self.collection.get(where=where, include=["metadatas"])
        metadatas = result.get("metadatas")
        if metadatas is None or len(metadatas) == 0:
            return []

        scored_movies: list[RetrievedMovie] = []
        for metadata in metadatas:
            if not metadata:
                continue
            movie = self._parse_movie(metadata, semantic_score=0.0)
            lexical_score = self._lexical_score(query, movie.title, movie.overview)
            if lexical_score <= 0.0:
                continue
            movie.vector_score = 0.0
            movie.lexical_score = lexical_score
            movie.fused_score = lexical_score
            movie.cross_encoder_score = None
            movie.semantic_score = lexical_score
            movie.rerank_score = lexical_score
            scored_movies.append(movie)

        scored_movies.sort(key=lambda item: item.semantic_score, reverse=True)
        return scored_movies[:top_k]

    def query_movies(self, query_embedding: list[float], filters: QueryFilters, top_k: int) -> list[RetrievedMovie]:
        where = self._build_where(filters)
        query_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )

        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        movies: list[RetrievedMovie] = []
        for metadata, distance in zip(metadatas, distances):
            if not metadata:
                continue
            semantic_score = self._distance_to_similarity(float(distance))
            movies.append(self._parse_movie(metadata, semantic_score))

        return movies

    def count(self) -> int:
        return self.collection.count()

    def _distance_to_similarity(self, distance: float) -> float:
        # chroma returns distance so convert it to similarity in range 0 to 1
        if self.distance_metric == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))
        if self.distance_metric == "l2":
            return 1.0 / (1.0 + max(distance, 0.0))
        # default mapping for ip or unknown metrics
        return 1.0 / (1.0 + max(distance, 0.0))

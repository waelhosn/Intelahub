import pytest

from app.core.settings import Settings
from app.models.movie import RetrievedMovie
from app.models.query import QueryFilters, QueryRequest
from app.services.reranker import ReRanker
from app.services.retrieval_service import RetrievalService


class _EmbeddingProviderStub:
    async def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        del texts, task_type
        return [[0.1, 0.2, 0.3]]


class _VectorStoreStub:
    def __init__(self, semantic_movies: list[RetrievedMovie], lexical_movies: list[RetrievedMovie] | None = None):
        self._semantic_movies = semantic_movies
        self._lexical_movies = lexical_movies if lexical_movies is not None else semantic_movies

    def query_movies(self, query_embedding: list[float], filters: QueryFilters, top_k: int) -> list[RetrievedMovie]:
        del query_embedding, filters, top_k
        return self._semantic_movies

    def lexical_search_movies(self, query: str, filters: QueryFilters, top_k: int) -> list[RetrievedMovie]:
        del query, filters, top_k
        return self._lexical_movies


class _LLMServiceStub:
    def __init__(self, summary: str = "summary"):
        self.summary = summary
        self.called = False

    async def summarize_results(self, query: str, movies: list[RetrievedMovie]) -> tuple[str, str, bool]:
        del query, movies
        self.called = True
        return self.summary, "openai", False

    async def parse_query_intent(self, query: str):
        del query
        from app.services.llm_service import QueryIntent

        return QueryIntent()


def _movie(
    semantic_score: float,
    rating: float = 7.0,
    release_year: int = 2020,
    vector_score: float | None = None,
    lexical_score: float = 0.0,
) -> RetrievedMovie:
    resolved_vector = semantic_score if vector_score is None else vector_score
    return RetrievedMovie(
        id=7,
        title="Threshold Test",
        overview="Overview",
        genres=["drama"],
        cast=["Actor"],
        release_year=release_year,
        rating=rating,
        vector_score=resolved_vector,
        lexical_score=lexical_score,
        fused_score=semantic_score,
        semantic_score=semantic_score,
        rerank_score=semantic_score,
    )


@pytest.mark.asyncio
async def test_no_relevant_results_when_below_threshold() -> None:
    settings = Settings(environment="test", relevance_threshold=0.95)
    llm = _LLMServiceStub()
    service = RetrievalService(
        settings=settings,
        embedding_provider=_EmbeddingProviderStub(),
        vector_store=_VectorStoreStub(semantic_movies=[_movie(semantic_score=0.1, rating=1.0, release_year=1990)]),
        reranker=ReRanker(settings),
        llm_service=llm,
    )

    response = await service.query(QueryRequest(query="some random query", top_k=5))
    assert response.meta.no_relevant_results is True
    assert response.meta.no_results_reason in {"threshold", "low_signal"}
    assert response.results == []
    assert response.meta.llm_provider == "none"
    assert response.meta.latency_breakdown is not None
    assert llm.called is False


@pytest.mark.asyncio
async def test_returns_results_when_above_threshold() -> None:
    settings = Settings(environment="test", relevance_threshold=0.4)
    llm = _LLMServiceStub(summary="ok summary")
    service = RetrievalService(
        settings=settings,
        embedding_provider=_EmbeddingProviderStub(),
        vector_store=_VectorStoreStub(semantic_movies=[_movie(semantic_score=0.9, rating=9.0, release_year=2024)]),
        reranker=ReRanker(settings),
        llm_service=llm,
    )

    response = await service.query(QueryRequest(query="top new movies", top_k=5))
    assert response.meta.no_relevant_results is False
    assert len(response.results) == 1
    assert response.meta.llm_provider == "openai"
    assert response.meta.latency_breakdown is not None
    assert llm.called is True


@pytest.mark.asyncio
async def test_exact_title_query_boosts_score_and_bypasses_threshold() -> None:
    settings = Settings(environment="test", relevance_threshold=0.7, max_top_k=50)
    llm = _LLMServiceStub(summary="exact title summary")
    service = RetrievalService(
        settings=settings,
        embedding_provider=_EmbeddingProviderStub(),
        vector_store=_VectorStoreStub(
            semantic_movies=[
                _movie(semantic_score=0.2, rating=7.0, release_year=2010),
            ],
            lexical_movies=[
                RetrievedMovie(
                    id=10,
                    title="Inception",
                    overview="A dream heist movie",
                    genres=["science fiction"],
                    cast=["Leonardo DiCaprio"],
                    release_year=2010,
                    rating=8.8,
                    semantic_score=0.2,
                    rerank_score=0.2,
                ),
            ],
        ),
        reranker=ReRanker(settings),
        llm_service=llm,
    )

    response = await service.query(QueryRequest(query="Inception", top_k=5))
    assert response.meta.no_relevant_results is False
    assert len(response.results) >= 1
    assert response.results[0].title == "Inception"


@pytest.mark.asyncio
async def test_no_results_safety_blocks_open_query() -> None:
    settings = Settings(
        environment="test",
        reranker_mode="heuristic",
        relevance_threshold=0.1,
        enable_no_results_safety_check=True,
        no_results_min_vector_score=0.35,
        no_results_min_lexical_score=0.03,
    )
    llm = _LLMServiceStub()
    service = RetrievalService(
        settings=settings,
        embedding_provider=_EmbeddingProviderStub(),
        vector_store=_VectorStoreStub(
            semantic_movies=[_movie(semantic_score=0.9, vector_score=0.2, lexical_score=0.0)],
            lexical_movies=[_movie(semantic_score=0.9, vector_score=0.2, lexical_score=0.02)],
        ),
        reranker=ReRanker(settings),
        llm_service=llm,
    )

    response = await service.query(QueryRequest(query="strange abstract space tax opera", top_k=5))
    assert response.meta.no_relevant_results is True
    assert response.meta.no_results_reason == "low_signal"
    assert response.results == []


@pytest.mark.asyncio
async def test_raw_signal_gate_skips_when_explicit_filters_present() -> None:
    settings = Settings(
        environment="test",
        reranker_mode="heuristic",
        relevance_threshold=0.1,
        enable_no_results_safety_check=True,
        no_results_min_vector_score=0.35,
        no_results_min_lexical_score=0.03,
    )
    llm = _LLMServiceStub()
    service = RetrievalService(
        settings=settings,
        embedding_provider=_EmbeddingProviderStub(),
        vector_store=_VectorStoreStub(
            semantic_movies=[_movie(semantic_score=0.9, vector_score=0.2, lexical_score=0.0)],
            lexical_movies=[_movie(semantic_score=0.9, vector_score=0.2, lexical_score=0.02)],
        ),
        reranker=ReRanker(settings),
        llm_service=llm,
    )

    response = await service.query(
        QueryRequest(
            query="strange abstract space tax opera",
            top_k=5,
            filters=QueryFilters(genres=["drama"]),
        )
    )
    assert response.meta.no_relevant_results is False
    assert response.meta.no_results_reason is None
    assert len(response.results) == 1

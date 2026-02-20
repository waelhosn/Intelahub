import pytest

from app.core.settings import Settings
from app.models.movie import RetrievedMovie
from app.models.query import QueryFilters, QueryRequest
from app.services.llm_service import QueryIntent
from app.services.reranker import ReRanker
from app.services.retrieval_service import RetrievalService


def _movie() -> RetrievedMovie:
    return RetrievedMovie(
        id=42,
        title="Interstellar",
        overview="Explorers travel through a wormhole in space.",
        genres=["science fiction", "drama"],
        cast=["Matthew McConaughey"],
        release_year=2014,
        rating=8.4,
        semantic_score=0.9,
        rerank_score=0.9,
    )


class _EmbeddingProviderSpy:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        del task_type
        self.calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class _VectorStoreSpy:
    def __init__(self) -> None:
        self.lexical_queries: list[str] = []

    def query_movies(self, query_embedding: list[float], filters: QueryFilters, top_k: int) -> list[RetrievedMovie]:
        del query_embedding, filters, top_k
        return [_movie()]

    def lexical_search_movies(self, query: str, filters: QueryFilters, top_k: int) -> list[RetrievedMovie]:
        del filters, top_k
        self.lexical_queries.append(query)
        return [_movie()]


class _LLMStub:
    async def summarize_results(self, query: str, movies: list[RetrievedMovie]) -> tuple[str, str, bool]:
        del query, movies
        return "summary", "openai", False

    async def parse_query_intent(self, query: str) -> QueryIntent:
        del query
        return QueryIntent(
            genres=["science fiction"],
            must_terms=["space exploration", "astronaut"],
            expansion_terms=["wormhole mission", "deep space survival"],
            query_type="theme",
            confidence=0.9,
        )


class _LLMLowConfidenceStub(_LLMStub):
    async def parse_query_intent(self, query: str) -> QueryIntent:
        del query
        return QueryIntent(
            genres=["science fiction"],
            must_terms=["space exploration"],
            expansion_terms=["space survival"],
            query_type="theme",
            confidence=0.1,
        )


async def _run_query(llm_stub) -> tuple[_EmbeddingProviderSpy, _VectorStoreSpy]:
    settings = Settings(
        environment="test",
        reranker_mode="heuristic",
        generation_provider="openai",
        relevance_threshold=0.1,
        enable_llm_query_expansion=True,
        llm_query_expansion_min_confidence=0.75,
        llm_query_expansion_trigger_score=1.0,
        llm_query_expansion_max_query_tokens=4,
    )
    embedding = _EmbeddingProviderSpy()
    vector = _VectorStoreSpy()
    service = RetrievalService(
        settings=settings,
        embedding_provider=embedding,
        vector_store=vector,
        reranker=ReRanker(settings),
        llm_service=llm_stub,
    )
    await service.query(QueryRequest(query="space movie", top_k=5))
    return embedding, vector


class _LowVectorStoreSpy(_VectorStoreSpy):
    def query_movies(self, query_embedding: list[float], filters: QueryFilters, top_k: int) -> list[RetrievedMovie]:
        del query_embedding, filters, top_k
        movie = _movie()
        movie.vector_score = 0.2
        movie.semantic_score = 0.2
        movie.rerank_score = 0.2
        movie.fused_score = 0.2
        return [movie]

    def lexical_search_movies(self, query: str, filters: QueryFilters, top_k: int) -> list[RetrievedMovie]:
        del filters, top_k
        self.lexical_queries.append(query)
        movie = _movie()
        movie.lexical_score = 0.9
        movie.semantic_score = 0.9
        movie.rerank_score = 0.9
        movie.fused_score = 0.9
        return [movie]


@pytest.mark.asyncio
async def test_high_confidence_intent_expands_query_variants() -> None:
    embedding, vector = await _run_query(_LLMStub())
    assert len(embedding.calls) == 2
    assert embedding.calls[0][0] == "space movie"
    assert any("wormhole mission" in q for q in embedding.calls[1])
    assert len(vector.lexical_queries) == 2
    assert any("wormhole mission" in q for q in vector.lexical_queries)


@pytest.mark.asyncio
async def test_low_confidence_intent_does_not_expand_query_variants() -> None:
    embedding, vector = await _run_query(_LLMLowConfidenceStub())
    assert len(embedding.calls) == 1
    assert embedding.calls[0] == ["space movie"]
    assert vector.lexical_queries == ["space movie"]


@pytest.mark.asyncio
async def test_explicit_filters_disable_query_expansion() -> None:
    settings = Settings(
        environment="test",
        reranker_mode="heuristic",
        generation_provider="openai",
        relevance_threshold=0.1,
        enable_llm_query_expansion=True,
        llm_query_expansion_min_confidence=0.75,
        llm_query_expansion_trigger_score=1.0,
        llm_query_expansion_max_query_tokens=4,
    )
    embedding = _EmbeddingProviderSpy()
    vector = _VectorStoreSpy()
    service = RetrievalService(
        settings=settings,
        embedding_provider=embedding,
        vector_store=vector,
        reranker=ReRanker(settings),
        llm_service=_LLMStub(),
    )
    await service.query(
        QueryRequest(
            query="space movie",
            top_k=5,
            filters=QueryFilters(genres=["science fiction"]),
        )
    )
    assert embedding.calls[0] == ["space movie"]
    assert vector.lexical_queries == ["space movie"]


@pytest.mark.asyncio
async def test_expansion_gate_uses_vector_confidence_not_rerank_confidence() -> None:
    settings = Settings(
        environment="test",
        reranker_mode="heuristic",
        generation_provider="openai",
        relevance_threshold=0.1,
        enable_llm_query_expansion=True,
        llm_query_expansion_min_confidence=0.75,
        llm_query_expansion_trigger_score=0.55,
        llm_query_expansion_max_query_tokens=8,
    )
    embedding = _EmbeddingProviderSpy()
    vector = _LowVectorStoreSpy()
    service = RetrievalService(
        settings=settings,
        embedding_provider=embedding,
        vector_store=vector,
        reranker=ReRanker(settings),
        llm_service=_LLMStub(),
    )
    await service.query(QueryRequest(query="space adventure", top_k=5))
    assert len(embedding.calls) == 2
    assert len(vector.lexical_queries) == 2

import logging

from app.core.settings import Settings
from app.services.embedding_service import GeminiEmbeddingProvider, OpenAIEmbeddingProvider
from app.services.guardrails import DeterministicGuardrails
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.reranker import ReRanker
from app.services.retrieval_service import RetrievalService
from app.services.tmdb_client import TMDBClient
from app.services.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class AppContainer:
    def __init__(self, settings: Settings):
        self.settings = settings

        self.tmdb_client = TMDBClient(settings)
        if settings.embedding_provider == "gemini":
            self.embedding_provider = GeminiEmbeddingProvider(settings)
        else:
            self.embedding_provider = OpenAIEmbeddingProvider(settings)
        self.vector_store = ChromaVectorStore(settings)
        self.guardrails = DeterministicGuardrails(settings)
        self.llm_service = LLMService(settings)
        self.reranker = ReRanker(settings)
        cross_encoder_warmed_up = False
        if settings.reranker_mode == "cross_encoder" and settings.warmup_cross_encoder_on_startup:
            cross_encoder_warmed_up = self.reranker.warmup()

        self.ingestion_service = IngestionService(
            settings=settings,
            tmdb_client=self.tmdb_client,
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
        )
        self.retrieval_service = RetrievalService(
            settings=settings,
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            reranker=self.reranker,
            llm_service=self.llm_service,
        )

        logger.info(
            "App container initialized",
            extra={
                "chroma_distance_metric": settings.chroma_distance_metric,
                "embedding_provider": settings.embedding_provider,
                "embedding_model_openai": settings.openai_embedding_model,
                "embedding_model_gemini": settings.gemini_embedding_model,
                "generation_provider": settings.generation_provider,
                "generation_model_openai": settings.openai_generation_model,
                "generation_model_gemini": settings.gemini_generation_model,
                "relevance_threshold": settings.relevance_threshold,
                "enable_no_results_safety_check": settings.enable_no_results_safety_check,
                "no_results_min_vector_score": settings.no_results_min_vector_score,
                "no_results_min_lexical_score": settings.no_results_min_lexical_score,
                "enable_llm_query_parser": settings.enable_llm_query_parser,
                "enable_llm_query_expansion": settings.enable_llm_query_expansion,
                "llm_query_expansion_min_confidence": settings.llm_query_expansion_min_confidence,
                "llm_query_expansion_trigger_score": settings.llm_query_expansion_trigger_score,
                "llm_query_expansion_max_query_tokens": settings.llm_query_expansion_max_query_tokens,
                "max_query_expansion_terms": settings.max_query_expansion_terms,
                "reranker_mode": settings.reranker_mode,
                "cross_encoder_model": settings.cross_encoder_model,
                "warmup_cross_encoder_on_startup": settings.warmup_cross_encoder_on_startup,
                "cross_encoder_warmed_up": cross_encoder_warmed_up,
            },
        )

    async def close(self) -> None:
        await self.tmdb_client.close()
        await self.embedding_provider.close()
        await self.llm_service.close()

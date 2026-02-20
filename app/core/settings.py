from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Intelahub Retrieval API"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    tmdb_api_key: str | None = None
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    tmdb_timeout_seconds: float = 20.0
    tmdb_max_retries: int = 3
    tmdb_requests_per_second: float = 3.0

    chroma_persist_path: str = "./data/chroma"
    chroma_collection_name: str = "movies_v1"
    chroma_distance_metric: Literal["cosine", "l2", "ip"] = "cosine"

    embedding_provider: Literal["openai", "gemini"] = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "gemini-embedding-001"
    # keep old env var names working
    embedding_model: str = "gemini-embedding-001"
    embedding_batch_size: int = 32

    gemini_api_key: str | None = None
    generation_provider: Literal["openai", "gemini"] = "openai"
    gemini_generation_model: str = "gemini-2.5-flash"

    openai_api_key: str | None = None
    openai_generation_model: str = "gpt-4.1-mini"

    default_top_k: int = 10
    max_top_k: int = 50
    relevance_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    enable_no_results_safety_check: bool = True
    no_results_min_vector_score: float = Field(default=0.35, ge=0.0, le=1.0)
    no_results_min_lexical_score: float = Field(default=0.03, ge=0.0, le=1.0)
    enable_llm_query_parser: bool = True
    enable_llm_query_expansion: bool = True
    llm_query_expansion_min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    llm_query_expansion_trigger_score: float = Field(default=0.55, ge=0.0, le=1.0)
    llm_query_expansion_max_query_tokens: int = Field(default=8, ge=1, le=20)
    max_query_expansion_terms: int = Field(default=4, ge=0, le=12)

    reranker_mode: Literal["cross_encoder", "heuristic"] = "cross_encoder"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    warmup_cross_encoder_on_startup: bool = True

    rate_limit_per_minute: int = 60
    ingest_rate_limit_per_minute: int = 6

    max_query_chars: int = 300
    # 0 means no app level cap
    max_target_count: int = Field(default=0, ge=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()

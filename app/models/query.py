from pydantic import BaseModel, Field, field_validator


class QueryFilters(BaseModel):
    genres: list[str] | None = Field(default=None)
    min_year: int | None = Field(default=None, ge=1888, le=2100)
    max_year: int | None = Field(default=None, ge=1888, le=2100)
    min_rating: float | None = Field(default=None, ge=0.0, le=10.0)

    @field_validator("genres")
    @classmethod
    def normalize_genres(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [g.strip().lower() for g in value if g.strip()]
        return cleaned or None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    filters: QueryFilters | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    sort: str = Field(default="relevance", pattern="^(relevance|rating|year)$")


class MovieResult(BaseModel):
    id: int
    title: str
    overview: str
    genres: list[str]
    cast: list[str]
    release_year: int | None
    rating: float | None
    vector_score: float
    lexical_score: float
    fused_score: float
    cross_encoder_score: float | None
    semantic_score: float
    rerank_score: float
    why_matched: str


class QueryInterpretation(BaseModel):
    original_query: str
    parsed_filters: QueryFilters
    query_variants: list[str] | None = None


class ConfidenceInfo(BaseModel):
    top_score: float | None
    threshold: float


class LatencyBreakdown(BaseModel):
    intent_parse_ms: int
    embedding_ms: int
    retrieval_ms: int
    summary_ms: int
    total_ms: int


class QueryResponseMeta(BaseModel):
    top_k: int
    sort: str
    llm_provider: str
    fallback_used: bool
    no_relevant_results: bool
    confidence: ConfidenceInfo
    latency_ms: int
    latency_breakdown: LatencyBreakdown | None = None
    expansion_applied: bool | None = None
    expansion_reason: str | None = None
    no_results_reason: str | None = None


class QueryResponse(BaseModel):
    query_interpretation: QueryInterpretation
    applied_filters: QueryFilters
    results: list[MovieResult]
    summary: str
    meta: QueryResponseMeta

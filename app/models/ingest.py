from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    target_count: int = Field(default=250, ge=1, description="number of movies to ingest")
    force_reindex: bool = Field(default=False, description="rebuild index before ingestion")


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    failed: int
    duration_ms: int

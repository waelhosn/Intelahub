from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.core.container import AppContainer
from app.models.ingest import IngestRequest, IngestResponse

router = APIRouter(prefix="/v1", tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_movies(payload: IngestRequest, container: AppContainer = Depends(get_container)) -> IngestResponse:
    return await container.ingestion_service.ingest(payload)

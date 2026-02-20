from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.core.container import AppContainer
from app.models.query import QueryRequest, QueryResponse

router = APIRouter(prefix="/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_movies(payload: QueryRequest, container: AppContainer = Depends(get_container)) -> QueryResponse:
    container.guardrails.validate_request(payload)
    return await container.retrieval_service.query(payload)

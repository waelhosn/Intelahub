from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from app.core.container import AppContainer
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.core.rate_limit import RateLimitMiddleware
from app.core.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.settings = settings
    app.state.container = AppContainer(settings)
    yield
    await app.state.container.close()


app = FastAPI(title="Intelahub Retrieval API", version="0.1.0", lifespan=lifespan)

settings = get_settings()
app.add_middleware(RateLimitMiddleware, settings=settings)
register_error_handlers(app)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)

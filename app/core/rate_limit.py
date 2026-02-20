import asyncio
import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.errors import APIError
from app.core.settings import Settings


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._events: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def allow(self, key: tuple[str, str], limit: int, window_seconds: int = 60) -> bool:
        now = time.monotonic()
        async with self._lock:
            bucket = self._events[key]
            cutoff = now - window_seconds
            while bucket and bucket[0] < cutoff:
                bucket.pop(0)
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.limiter = InMemoryRateLimiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if path.startswith("/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        limit = self.settings.rate_limit_per_minute
        if path.startswith("/v1/ingest"):
            limit = self.settings.ingest_rate_limit_per_minute

        allowed = await self.limiter.allow(key=(path, client_ip), limit=limit)
        if not allowed:
            raise APIError(
                code="rate_limited",
                message="Too many requests",
                status_code=429,
                details={"path": path, "limit_per_minute": limit},
            )

        return await call_next(request)

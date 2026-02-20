import asyncio
import logging
from typing import Any

import httpx

from app.core.errors import APIError
from app.core.settings import Settings

logger = logging.getLogger(__name__)


class TMDBClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(base_url=settings.tmdb_base_url, timeout=settings.tmdb_timeout_seconds)
        self._min_interval_seconds = 1.0 / max(settings.tmdb_requests_per_second, 0.5)
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval_seconds:
                await asyncio.sleep(self._min_interval_seconds - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()

    async def _request(self, method: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.settings.tmdb_api_key:
            raise APIError("config_error", "TMDB_API_KEY is not set", status_code=500)

        merged_params = {"api_key": self.settings.tmdb_api_key}
        if params:
            merged_params.update(params)

        backoff_seconds = 0.6
        for attempt in range(1, self.settings.tmdb_max_retries + 1):
            await self._throttle()
            try:
                response = await self._client.request(method, path, params=merged_params)
            except (httpx.TransportError, httpx.RemoteProtocolError, httpx.ReadTimeout) as exc:
                if attempt < self.settings.tmdb_max_retries:
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds *= 2
                    continue
                raise APIError(
                    "tmdb_upstream_unavailable",
                    "TMDB upstream is temporarily unavailable",
                    status_code=502,
                    details={"path": path, "error_type": exc.__class__.__name__},
                ) from exc

            if response.status_code < 400:
                return response.json()

            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.settings.tmdb_max_retries:
                await asyncio.sleep(backoff_seconds)
                backoff_seconds *= 2
                continue

            if response.status_code == 401:
                raise APIError("tmdb_auth_error", "TMDB API key is invalid", status_code=502)

            raise APIError(
                "tmdb_request_failed",
                "TMDB request failed",
                status_code=502,
                details={"path": path, "status_code": response.status_code, "response": response.text[:200]},
            )

        raise APIError("tmdb_request_failed", "TMDB request retries exhausted", status_code=502)

    async def discover_movie_ids(self, target_count: int) -> list[int]:
        ids: list[int] = []
        page = 1

        while len(ids) < target_count:
            payload = await self._request(
                "GET",
                "/discover/movie",
                params={
                    "language": "en-US",
                    "include_adult": "false",
                    "sort_by": "popularity.desc",
                    "page": page,
                },
            )

            results = payload.get("results", [])
            if not results:
                break

            for item in results:
                movie_id = item.get("id")
                if isinstance(movie_id, int):
                    ids.append(movie_id)
                    if len(ids) >= target_count:
                        break

            total_pages = payload.get("total_pages", page)
            if page >= total_pages:
                break
            page += 1

        unique_ids = list(dict.fromkeys(ids))
        logger.info("collected TMDB movie ids", extra={"count": len(unique_ids)})
        return unique_ids

    async def fetch_movie_details(self, movie_id: int) -> dict[str, Any]:
        return await self._request("GET", f"/movie/{movie_id}", params={"language": "en-US"})

    async def fetch_movie_credits(self, movie_id: int) -> dict[str, Any]:
        return await self._request("GET", f"/movie/{movie_id}/credits", params={"language": "en-US"})

    async def fetch_detailed_movies(self, ids: list[int], concurrency: int = 8) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _fetch(movie_id: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
            async with semaphore:
                try:
                    details, credits = await asyncio.gather(
                        self.fetch_movie_details(movie_id),
                        self.fetch_movie_credits(movie_id),
                    )
                    return details, credits
                except APIError:
                    logger.exception("Failed TMDB fetch for movie", extra={"movie_id": movie_id})
                    return None

        results = await asyncio.gather(*[_fetch(movie_id) for movie_id in ids])
        return [entry for entry in results if entry is not None]

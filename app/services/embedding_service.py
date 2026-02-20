import asyncio
import logging
from typing import Protocol

import httpx

from app.core.errors import APIError
from app.core.settings import Settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        ...

    async def close(self) -> None:
        ...


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=30)
        self._max_retries = 4

    async def close(self) -> None:
        await self._client.aclose()

    async def _embed_batch_request(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.openai_api_key:
            raise APIError("config_error", "OPENAI_API_KEY is not set", status_code=500)

        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload = {"model": self.settings.openai_embedding_model, "input": texts}
        response = await self._client.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=payload,
        )
        if response.status_code in {429, 500, 502, 503, 504}:
            raise APIError(
                "embedding_retryable_error",
                "OpenAI embeddings request returned retryable status",
                status_code=502,
                details={"status_code": response.status_code},
            )
        if response.status_code >= 400:
            raise APIError(
                "embedding_failed",
                "OpenAI embedding request failed",
                status_code=502,
                details={"status_code": response.status_code, "response": response.text[:200]},
            )

        data = response.json()
        items = data.get("data", [])
        if not items:
            raise APIError("embedding_failed", "Embedding response missing data", status_code=502)

        vectors: list[list[float]] = []
        for item in sorted(items, key=lambda x: x.get("index", 0)):
            vector = item.get("embedding", [])
            if not vector:
                raise APIError("embedding_failed", "Embedding vector missing", status_code=502)
            vectors.append([float(v) for v in vector])
        if len(vectors) != len(texts):
            raise APIError(
                "embedding_failed",
                "Embedding response count mismatch",
                status_code=502,
                details={"expected": len(texts), "received": len(vectors)},
            )
        return vectors

    async def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        backoff = 0.8
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._embed_batch_request(texts)
            except APIError as exc:
                last_exc = exc
                if exc.code != "embedding_retryable_error" or attempt == self._max_retries:
                    break
            except (httpx.TransportError, httpx.RemoteProtocolError, httpx.ReadTimeout) as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    break
            await asyncio.sleep(backoff)
            backoff *= 2

        if len(texts) > 1:
            # if a full batch keeps failing split it and retry
            mid = len(texts) // 2
            logger.warning(
                "Embedding batch failed after retries; splitting batch",
                extra={"batch_size": len(texts), "left_size": mid, "right_size": len(texts) - mid},
            )
            left = await self._embed_batch_with_retry(texts[:mid])
            right = await self._embed_batch_with_retry(texts[mid:])
            return left + right

        if isinstance(last_exc, APIError):
            raise last_exc
        raise APIError(
            "embedding_upstream_unavailable",
            "OpenAI embeddings upstream is temporarily unavailable",
            status_code=502,
            details={"error_type": last_exc.__class__.__name__ if last_exc else "unknown"},
        )

    async def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        del task_type  # openai embeddings api does not use task_type
        if not texts:
            return []
        return await self._embed_batch_with_retry(texts)


class GeminiEmbeddingProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=30)
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def close(self) -> None:
        await self._client.aclose()

    async def _embed_one(self, text: str, task_type: str) -> list[float]:
        if not self.settings.gemini_api_key:
            raise APIError("config_error", "GEMINI_API_KEY is not set", status_code=500)

        model = self.settings.gemini_embedding_model or self.settings.embedding_model
        url = f"{self._base_url}/models/{model}:embedContent"
        params = {"key": self.settings.gemini_api_key}
        payload = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }

        try:
            response = await self._client.post(url, params=params, json=payload)
        except (httpx.TransportError, httpx.RemoteProtocolError, httpx.ReadTimeout) as exc:
            raise APIError(
                "embedding_upstream_unavailable",
                "Gemini embeddings upstream is temporarily unavailable",
                status_code=502,
                details={"error_type": exc.__class__.__name__},
            ) from exc
        if response.status_code >= 400:
            raise APIError(
                "embedding_failed",
                "Gemini embedding request failed",
                status_code=502,
                details={"status_code": response.status_code, "response": response.text[:200]},
            )

        data = response.json()
        values = (((data.get("embedding") or {}).get("values")) or [])
        if not values:
            raise APIError("embedding_failed", "Embedding response missing values", status_code=502)
        return [float(v) for v in values]

    async def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        if not texts:
            return []

        semaphore = asyncio.Semaphore(6)

        async def _run(text: str) -> list[float]:
            async with semaphore:
                return await self._embed_one(text, task_type)

        return await asyncio.gather(*[_run(text) for text in texts])

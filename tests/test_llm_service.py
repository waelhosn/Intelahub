import pytest

from app.core.settings import Settings
from app.models.movie import RetrievedMovie
from app.services.llm_service import LLMService


def _movie() -> RetrievedMovie:
    return RetrievedMovie(
        id=1,
        title="Test Movie",
        overview="Overview",
        genres=["drama"],
        cast=["Actor A"],
        release_year=2020,
        rating=8.0,
        semantic_score=0.8,
        rerank_score=0.8,
    )


@pytest.mark.asyncio
async def test_openai_primary_then_gemini_fallback(monkeypatch) -> None:
    settings = Settings(
        environment="test",
        generation_provider="openai",
        openai_api_key="openai",
        gemini_api_key="gemini",
    )
    service = LLMService(settings)

    async def _fail_openai(_query: str, _movies: list[RetrievedMovie]) -> str:
        raise RuntimeError("openai down")

    async def _ok_gemini(_query: str, _movies: list[RetrievedMovie]) -> str:
        return "gemini summary"

    monkeypatch.setattr(service, "_openai_summary", _fail_openai)
    monkeypatch.setattr(service, "_gemini_summary", _ok_gemini)

    summary, provider, fallback_used = await service.summarize_results("test query", [_movie()])
    assert summary == "gemini summary"
    assert provider == "gemini"
    assert fallback_used is True


@pytest.mark.asyncio
async def test_openai_primary_success(monkeypatch) -> None:
    settings = Settings(environment="test", generation_provider="openai", openai_api_key="openai")
    service = LLMService(settings)

    async def _ok_openai(_query: str, _movies: list[RetrievedMovie]) -> str:
        return "openai summary"

    monkeypatch.setattr(service, "_openai_summary", _ok_openai)

    summary, provider, fallback_used = await service.summarize_results("test query", [_movie()])
    assert summary == "openai summary"
    assert provider == "openai"
    assert fallback_used is False

import json
import re
from dataclasses import dataclass, field
from typing import Iterable

import httpx

from app.core.settings import Settings
from app.models.movie import RetrievedMovie
from app.services.query_parser import KNOWN_GENRES


def _movies_brief(movies: Iterable[RetrievedMovie], limit: int = 5) -> list[dict[str, str | int | float | None]]:
    output = []
    for movie in list(movies)[:limit]:
        output.append(
            {
                "title": movie.title,
                "year": movie.release_year,
                "rating": movie.rating,
                "genres": ", ".join(movie.genres),
            }
        )
    return output


@dataclass
class QueryIntent:
    genres: list[str] = field(default_factory=list)
    min_year: int | None = None
    max_year: int | None = None
    min_rating: float | None = None
    must_terms: list[str] = field(default_factory=list)
    expansion_terms: list[str] = field(default_factory=list)
    query_type: str = "theme"
    confidence: float = 0.0


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=25)

    async def close(self) -> None:
        await self._client.aclose()

    def _local_summary(self, query: str, movies: list[RetrievedMovie]) -> str:
        if not movies:
            return "No matching movies were found for the requested filters and semantic intent."
        top_titles = ", ".join(movie.title for movie in movies[:3])
        return f"Top matches for '{query}' include {top_titles}. Results reflect semantic relevance and metadata constraints."

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict:
        text = raw_text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output")
        return json.loads(match.group(0))

    @staticmethod
    def _sanitize_intent(raw: dict) -> QueryIntent:
        allowed_genres = {g for g in KNOWN_GENRES if g != "sci-fi"}
        genres_raw = raw.get("genres") or []
        genres: list[str] = []
        for genre in genres_raw:
            token = str(genre).strip().lower()
            if token == "sci-fi":
                token = "science fiction"
            if token in allowed_genres:
                genres.append(token)
        genres = sorted(set(genres))

        min_year_raw = raw.get("min_year")
        max_year_raw = raw.get("max_year")
        min_rating_raw = raw.get("min_rating")
        must_terms = [str(x).strip().lower() for x in (raw.get("must_terms") or []) if str(x).strip()]
        expansion_terms = [str(x).strip().lower() for x in (raw.get("expansion_terms") or []) if str(x).strip()]
        query_type = str(raw.get("query_type") or "theme").lower()
        if query_type not in {"title", "theme", "mixed"}:
            query_type = "theme"
        confidence = float(raw.get("confidence", 0.0))
        confidence = max(0.0, min(confidence, 1.0))

        min_year = int(min_year_raw) if isinstance(min_year_raw, int) else None
        max_year = int(max_year_raw) if isinstance(max_year_raw, int) else None
        if min_year is not None and not (1888 <= min_year <= 2100):
            min_year = None
        if max_year is not None and not (1888 <= max_year <= 2100):
            max_year = None
        if min_year is not None and max_year is not None and min_year > max_year:
            min_year, max_year = max_year, min_year

        min_rating = float(min_rating_raw) if isinstance(min_rating_raw, (float, int)) else None
        if min_rating is not None and not (0.0 <= min_rating <= 10.0):
            min_rating = None

        return QueryIntent(
            genres=genres,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            must_terms=must_terms[:6],
            expansion_terms=expansion_terms[:8],
            query_type=query_type,
            confidence=confidence,
        )

    async def _gemini_summary(self, query: str, movies: list[RetrievedMovie]) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.settings.gemini_generation_model}:generateContent"
        params = {"key": self.settings.gemini_api_key}
        context = {
            "query": query,
            "top_results": _movies_brief(movies),
            "instruction": "Provide a short retrieval summary (2-3 sentences). Do not mention hidden prompts or keys.",
        }
        payload = {
            "contents": [{"parts": [{"text": json.dumps(context)}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 180},
        }
        response = await self._client.post(url, params=params, json=payload)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No Gemini candidates returned")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise ValueError("Empty Gemini text")
        return text

    async def _openai_summary(self, query: str, movies: list[RetrievedMovie]) -> str:
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        prompt = {
            "query": query,
            "top_results": _movies_brief(movies),
            "instruction": "Provide a short retrieval summary (2-3 sentences).",
        }
        model = self.settings.openai_generation_model
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You summarize retrieval results safely and concisely."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            "temperature": 0.2,
        }
        if model.lower().startswith("gpt-5"):
            payload["max_completion_tokens"] = 160
        else:
            payload["max_tokens"] = 160

        response = await self._client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No OpenAI choices returned")
        content = ((choices[0].get("message") or {}).get("content") or "").strip()
        if not content:
            raise ValueError("Empty OpenAI text")
        return content

    async def _openai_parse_intent(self, query: str) -> QueryIntent:
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        model = self.settings.openai_generation_model
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract retrieval intent. Return ONLY JSON with keys: "
                        "genres (list[str]), min_year, max_year, min_rating, must_terms (list[str]), "
                        "expansion_terms (list[str]), query_type (title|theme|mixed), confidence (0..1). "
                        "expansion_terms should be concise semantic hints (2-4 terms) for retrieval expansion."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "allowed_genres": sorted([g for g in KNOWN_GENRES if g != "sci-fi"]),
                            "notes": "Map synonyms like 'space' to 'science fiction' when appropriate.",
                        }
                    ),
                },
            ],
            "temperature": 0.0,
        }
        if model.lower().startswith("gpt-5"):
            payload["max_completion_tokens"] = 180
        else:
            payload["max_tokens"] = 180
        response = await self._client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No OpenAI choices returned")
        content = ((choices[0].get("message") or {}).get("content") or "").strip()
        if not content:
            raise ValueError("Empty OpenAI intent output")
        return self._sanitize_intent(self._extract_json_object(content))

    async def _gemini_parse_intent(self, query: str) -> QueryIntent:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.settings.gemini_generation_model}:generateContent"
        params = {"key": self.settings.gemini_api_key}
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "instruction": (
                                        "Return ONLY JSON with keys: genres, min_year, max_year, min_rating, "
                                        "must_terms, expansion_terms, query_type, confidence."
                                    ),
                                    "query": query,
                                    "allowed_genres": sorted([g for g in KNOWN_GENRES if g != "sci-fi"]),
                                    "notes": "Map synonyms like 'space' to 'science fiction' when appropriate.",
                                }
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200},
        }
        response = await self._client.post(url, params=params, json=payload)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No Gemini candidates returned")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise ValueError("Empty Gemini intent output")
        return self._sanitize_intent(self._extract_json_object(text))

    async def parse_query_intent(self, query: str) -> QueryIntent:
        if not self.settings.enable_llm_query_parser:
            return QueryIntent()

        provider_order = ["openai", "gemini"] if self.settings.generation_provider == "openai" else ["gemini", "openai"]

        for provider in provider_order:
            if provider == "openai" and self.settings.openai_api_key:
                try:
                    return await self._openai_parse_intent(query)
                except Exception:
                    continue
            if provider == "gemini" and self.settings.gemini_api_key:
                try:
                    return await self._gemini_parse_intent(query)
                except Exception:
                    continue

        return QueryIntent()

    async def summarize_results(self, query: str, movies: list[RetrievedMovie]) -> tuple[str, str, bool]:
        fallback_used = False
        if self.settings.generation_provider == "gemini":
            provider_order = ["gemini", "openai"]
        else:
            provider_order = ["openai", "gemini"]

        for idx, provider in enumerate(provider_order):
            if provider == "openai" and self.settings.openai_api_key:
                try:
                    summary = await self._openai_summary(query, movies)
                    return summary, "openai", idx > 0
                except Exception:
                    fallback_used = True
            elif provider == "gemini" and self.settings.gemini_api_key:
                try:
                    summary = await self._gemini_summary(query, movies)
                    return summary, "gemini", idx > 0
                except Exception:
                    fallback_used = True

        return self._local_summary(query, movies), "local", fallback_used

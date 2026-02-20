import time
import re
from difflib import SequenceMatcher

from app.core.errors import APIError
from app.core.settings import Settings
from app.models.movie import RetrievedMovie
from app.models.query import (
    ConfidenceInfo,
    LatencyBreakdown,
    MovieResult,
    QueryFilters,
    QueryInterpretation,
    QueryRequest,
    QueryResponse,
    QueryResponseMeta,
)
from app.services.embedding_service import EmbeddingProvider
from app.services.llm_service import LLMService, QueryIntent
from app.services.query_parser import merge_filters, parse_filters_from_query
from app.services.reranker import ReRanker
from app.services.vector_store import ChromaVectorStore


class RetrievalService:
    RRF_K = 60
    NOISE_TERMS = {"movie", "movies", "film", "films", "best", "top"}
    EXPANSION_STOPWORDS = {
        "a",
        "an",
        "and",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }

    def __init__(
        self,
        settings: Settings,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        reranker: ReRanker,
        llm_service: LLMService,
    ):
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm_service = llm_service

    @staticmethod
    def _apply_sort(results: list, sort: str) -> list:
        if sort == "rating":
            return sorted(results, key=lambda x: (x.rating or -1), reverse=True)
        if sort == "year":
            return sorted(results, key=lambda x: (x.release_year or -1), reverse=True)
        return sorted(results, key=lambda x: x.rerank_score, reverse=True)

    @staticmethod
    def _normalize_title(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    @classmethod
    def _rrf(cls, rank: int) -> float:
        return 1.0 / (cls.RRF_K + rank)

    @classmethod
    def _is_strong_title_match(cls, query: str, title: str) -> bool:
        normalized_query = cls._normalize_title(query)
        normalized_title = cls._normalize_title(title)
        if not normalized_query or not normalized_title:
            return False
        if normalized_query == normalized_title:
            return True
        if len(normalized_query) >= 4 and (
            normalized_query in normalized_title or normalized_title in normalized_query
        ):
            return True
        return SequenceMatcher(a=normalized_query, b=normalized_title).ratio() >= 0.92

    @classmethod
    def _fuse_candidates(
        cls,
        semantic_candidates: list[RetrievedMovie],
        lexical_candidates: list[RetrievedMovie],
    ) -> list[RetrievedMovie]:
        by_id: dict[int, RetrievedMovie] = {}
        semantic_ranks: dict[int, int] = {}
        lexical_ranks: dict[int, int] = {}

        for rank, movie in enumerate(semantic_candidates, start=1):
            by_id[movie.id] = movie
            semantic_ranks[movie.id] = rank

        for rank, movie in enumerate(lexical_candidates, start=1):
            if movie.id not in by_id:
                by_id[movie.id] = movie
            else:
                # keep the semantic row and copy lexical score
                by_id[movie.id].lexical_score = movie.lexical_score
            lexical_ranks[movie.id] = rank

        raw_scores: dict[int, float] = {}
        for movie_id in by_id:
            score = 0.0
            if movie_id in semantic_ranks:
                score += 0.65 * cls._rrf(semantic_ranks[movie_id])
            if movie_id in lexical_ranks:
                score += 0.35 * cls._rrf(lexical_ranks[movie_id])
            raw_scores[movie_id] = score

        if not raw_scores:
            return []

        max_score = max(raw_scores.values()) or 1.0
        fused: list[RetrievedMovie] = []
        for movie_id, movie in by_id.items():
            movie.fused_score = raw_scores[movie_id] / max_score
            movie.semantic_score = movie.fused_score
            movie.rerank_score = movie.fused_score
            fused.append(movie)
        return fused

    @staticmethod
    def _token_overlap(query: str, movie: RetrievedMovie) -> tuple[float, float]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_tokens:
            return 0.0, 0.0
        title_tokens = set(re.findall(r"[a-z0-9]+", movie.title.lower()))
        overview_tokens = set(re.findall(r"[a-z0-9]+", movie.overview.lower()))
        title_overlap = len(query_tokens.intersection(title_tokens)) / len(query_tokens)
        overview_overlap = len(query_tokens.intersection(overview_tokens)) / len(query_tokens)
        return title_overlap, overview_overlap

    @staticmethod
    def _filter_evidence(movie: RetrievedMovie, filters: QueryFilters) -> list[str]:
        evidence: list[str] = []
        if filters.genres:
            matched_genres = sorted(set(filters.genres).intersection(set(movie.genres)))
            if matched_genres:
                evidence.append(f"genre={','.join(matched_genres)}")
        if filters.min_year is not None and movie.release_year is not None and movie.release_year >= filters.min_year:
            evidence.append(f"year>={filters.min_year}")
        if filters.max_year is not None and movie.release_year is not None and movie.release_year <= filters.max_year:
            evidence.append(f"year<={filters.max_year}")
        if filters.min_rating is not None and movie.rating is not None and movie.rating >= filters.min_rating:
            evidence.append(f"rating>={filters.min_rating}")
        return evidence

    def _why_matched(
        self,
        movie: RetrievedMovie,
        query: str,
        filters: QueryFilters,
        is_title_override: bool,
    ) -> str:
        title_overlap, overview_overlap = self._token_overlap(query, movie)
        evidence = self._filter_evidence(movie, filters)
        reasons: list[str] = []

        if is_title_override:
            reasons.append("strong title match")
        if movie.lexical_score > 0:
            reasons.append(f"lexical={movie.lexical_score:.3f}")
        if movie.vector_score > 0:
            reasons.append(f"vector={movie.vector_score:.3f}")
        reasons.append(f"fused={movie.fused_score:.3f}")
        if movie.cross_encoder_score is not None:
            reasons.append(f"cross_encoder={movie.cross_encoder_score:.3f}")
        reasons.append(f"rerank={movie.rerank_score:.3f}")
        reasons.append(f"title_overlap={title_overlap:.2f}")
        reasons.append(f"overview_overlap={overview_overlap:.2f}")
        if evidence:
            reasons.append("filters=" + ";".join(evidence))

        return " | ".join(reasons)

    @staticmethod
    def _intent_to_filters(intent: QueryIntent) -> QueryFilters:
        min_year = intent.min_year if intent.min_year is not None and 1888 <= intent.min_year <= 2100 else None
        max_year = intent.max_year if intent.max_year is not None and 1888 <= intent.max_year <= 2100 else None
        if min_year is not None and max_year is not None and min_year > max_year:
            min_year, max_year = max_year, min_year
        min_rating = intent.min_rating if intent.min_rating is not None and 0.0 <= intent.min_rating <= 10.0 else None
        return QueryFilters(
            genres=intent.genres or None,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
        )

    @staticmethod
    def _combine_parsed_filters(base: QueryFilters, llm_filters: QueryFilters) -> QueryFilters:
        return QueryFilters(
            genres=base.genres if base.genres is not None else llm_filters.genres,
            min_year=base.min_year if base.min_year is not None else llm_filters.min_year,
            max_year=base.max_year if base.max_year is not None else llm_filters.max_year,
            min_rating=base.min_rating if base.min_rating is not None else llm_filters.min_rating,
        )

    @staticmethod
    def _has_explicit_filters(filters: QueryFilters | None) -> bool:
        if filters is None:
            return False
        return (
            bool(filters.genres)
            or filters.min_year is not None
            or filters.max_year is not None
            or filters.min_rating is not None
        )

    @staticmethod
    def _query_tokens(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", value.lower())

    @classmethod
    def _clean_expansion_term(cls, term: str, query_tokens: set[str]) -> str | None:
        normalized = re.sub(r"\s+", " ", str(term).strip().lower())
        if not normalized:
            return None
        tokens = [
            token
            for token in cls._query_tokens(normalized)
            if token not in cls.NOISE_TERMS and token not in cls.EXPANSION_STOPWORDS
        ]
        if not tokens:
            return None
        if all(token in query_tokens for token in tokens):
            return None
        cleaned = " ".join(tokens[:4]).strip()
        return cleaned or None

    def _build_expansion_queries(self, query: str, intent: QueryIntent) -> list[str]:
        base = query.strip()
        if not base:
            return []

        normalized_query = self._normalize_title(base)
        query_tokens = set(self._query_tokens(base))
        cap = max(0, self.settings.max_query_expansion_terms)
        if cap == 0:
            return []

        # try expansion terms first then must terms then genres
        seed_terms = [*intent.expansion_terms, *intent.must_terms, *intent.genres]
        expansion_terms: list[str] = []
        for term in seed_terms:
            cleaned = self._clean_expansion_term(str(term), query_tokens)
            if not cleaned:
                continue
            if cleaned not in expansion_terms:
                expansion_terms.append(cleaned)
            if len(expansion_terms) >= cap:
                break

        if not expansion_terms:
            return []

        variants: list[str] = []
        variant_with_context = f"{base} {' '.join(expansion_terms)}".strip()
        if self._normalize_title(variant_with_context) != normalized_query:
            variants.append(variant_with_context)

        # add a keyword heavy variant to help lexical matching
        if len(expansion_terms) >= 2:
            keyword_variant = f"{base} about {', '.join(expansion_terms[:3])}".strip()
            if self._normalize_title(keyword_variant) != normalized_query and keyword_variant not in variants:
                variants.append(keyword_variant)

        return variants[:2]

    @staticmethod
    def _query_token_count(query: str) -> int:
        return len(re.findall(r"[a-z0-9]+", query.lower()))

    def _expansion_decision(
        self,
        payload: QueryRequest,
        intent: QueryIntent,
        base_top_vector_score: float | None,
    ) -> tuple[bool, str]:
        if not self.settings.enable_llm_query_expansion:
            return False, "disabled"
        if intent.query_type == "title":
            return False, "title_query"
        if intent.confidence < self.settings.llm_query_expansion_min_confidence:
            return False, "low_intent_confidence"
        if self._has_explicit_filters(payload.filters):
            return False, "explicit_filters"
        if self._query_token_count(payload.query) > self.settings.llm_query_expansion_max_query_tokens:
            return False, "query_too_long"
        if (
            base_top_vector_score is not None
            and base_top_vector_score >= self.settings.llm_query_expansion_trigger_score
        ):
            return False, "base_vector_score_high"
        return True, "eligible"

    def _no_results_safety_decision(
        self,
        payload: QueryRequest,
        has_title_override: bool,
        top_vector_score: float | None,
        top_lexical_score: float | None,
    ) -> tuple[bool, str | None]:
        if not self.settings.enable_no_results_safety_check:
            return False, None
        if has_title_override:
            return False, None
        if self._has_explicit_filters(payload.filters):
            return False, None
        if top_vector_score is None:
            return True, "signal_missing"

        lexical = top_lexical_score or 0.0
        if (
            top_vector_score < self.settings.no_results_min_vector_score
            and lexical < self.settings.no_results_min_lexical_score
        ):
            return True, "low_signal"
        return False, None

    @staticmethod
    def _merge_semantic_candidates(
        candidate_sets: list[list[RetrievedMovie]],
        top_k: int,
    ) -> list[RetrievedMovie]:
        by_id: dict[int, RetrievedMovie] = {}
        for candidates in candidate_sets:
            for movie in candidates:
                existing = by_id.get(movie.id)
                if existing is None:
                    by_id[movie.id] = movie
                    continue
                if movie.vector_score > existing.vector_score:
                    by_id[movie.id] = movie
                else:
                    existing.vector_score = max(existing.vector_score, movie.vector_score)
                    existing.semantic_score = existing.vector_score
                    existing.rerank_score = existing.vector_score
                    existing.fused_score = existing.vector_score
        merged = sorted(by_id.values(), key=lambda item: item.vector_score, reverse=True)
        return merged[:top_k]

    @staticmethod
    def _merge_lexical_candidates(
        candidate_sets: list[list[RetrievedMovie]],
        top_k: int,
    ) -> list[RetrievedMovie]:
        by_id: dict[int, RetrievedMovie] = {}
        for candidates in candidate_sets:
            for movie in candidates:
                existing = by_id.get(movie.id)
                if existing is None:
                    by_id[movie.id] = movie
                    continue
                if movie.lexical_score > existing.lexical_score:
                    by_id[movie.id] = movie
                else:
                    existing.lexical_score = max(existing.lexical_score, movie.lexical_score)
                    existing.semantic_score = existing.lexical_score
                    existing.rerank_score = existing.lexical_score
                    existing.fused_score = existing.lexical_score
        merged = sorted(by_id.values(), key=lambda item: item.lexical_score, reverse=True)
        return merged[:top_k]

    async def query(self, payload: QueryRequest) -> QueryResponse:
        start = time.perf_counter()
        intent_start = time.perf_counter()
        parsed_filters = parse_filters_from_query(payload.query)
        llm_intent = await self.llm_service.parse_query_intent(payload.query)
        llm_filters = self._intent_to_filters(llm_intent)
        parsed_filters = self._combine_parsed_filters(parsed_filters, llm_filters)
        applied_filters: QueryFilters = merge_filters(parsed_filters, payload.filters)
        intent_parse_ms = int((time.perf_counter() - intent_start) * 1000)

        candidate_k = min(max(payload.top_k * 4, payload.top_k), self.settings.max_top_k)
        query_text = payload.query.strip() or payload.query
        query_variants = [query_text]

        embedding_ms = 0
        retrieval_ms = 0
        summary_ms = 0

        base_embed_start = time.perf_counter()
        base_embeddings = await self.embedding_provider.embed_texts([query_text], task_type="RETRIEVAL_QUERY")
        if not base_embeddings:
            raise APIError("embedding_failed", "Query embedding response was empty", status_code=502)
        base_embedding = base_embeddings[0]
        embedding_ms += int((time.perf_counter() - base_embed_start) * 1000)

        base_retrieval_start = time.perf_counter()
        base_semantic_candidates = self.vector_store.query_movies(base_embedding, applied_filters, candidate_k)
        base_lexical_candidates = self.vector_store.lexical_search_movies(query_text, applied_filters, candidate_k)
        semantic_candidates = base_semantic_candidates
        lexical_candidates = base_lexical_candidates
        fused_candidates = self._fuse_candidates(semantic_candidates, lexical_candidates)
        reranked = self.reranker.rerank(query_text, fused_candidates)
        retrieval_ms += int((time.perf_counter() - base_retrieval_start) * 1000)

        expansion_applied = False
        expansion_reason: str | None = None
        base_title_override_ids = {
            movie.id for movie in base_lexical_candidates if self._is_strong_title_match(payload.query, movie.title)
        }
        base_sorted = sorted(reranked, key=lambda x: x.rerank_score, reverse=True)
        base_top_vector_score = max((movie.vector_score for movie in base_semantic_candidates), default=None)
        base_has_title_override = any(movie.id in base_title_override_ids for movie in base_sorted[: payload.top_k])
        should_expand, expansion_reason = self._expansion_decision(payload, llm_intent, base_top_vector_score)

        if should_expand and not base_has_title_override:
            expanded_queries = self._build_expansion_queries(query_text, llm_intent)
            if expanded_queries:
                query_variants = [query_text, *expanded_queries]
                expansion_embed_start = time.perf_counter()
                expanded_embeddings = await self.embedding_provider.embed_texts(expanded_queries, task_type="RETRIEVAL_QUERY")
                embedding_ms += int((time.perf_counter() - expansion_embed_start) * 1000)
                pair_count = min(len(expanded_queries), len(expanded_embeddings))
                if pair_count == 0:
                    expansion_reason = "expansion_embedding_empty"
                    query_variants = [query_text]
                else:
                    expansion_retrieval_start = time.perf_counter()
                    semantic_candidate_sets = [base_semantic_candidates]
                    lexical_candidate_sets = [base_lexical_candidates]
                    for expanded_query, expanded_embedding in zip(expanded_queries[:pair_count], expanded_embeddings[:pair_count]):
                        expanded_semantic_candidates = self.vector_store.query_movies(
                            expanded_embedding,
                            applied_filters,
                            candidate_k,
                        )
                        expanded_lexical_candidates = self.vector_store.lexical_search_movies(
                            expanded_query,
                            applied_filters,
                            candidate_k,
                        )
                        semantic_candidate_sets.append(expanded_semantic_candidates)
                        lexical_candidate_sets.append(expanded_lexical_candidates)
                    semantic_candidates = self._merge_semantic_candidates(
                        semantic_candidate_sets,
                        candidate_k,
                    )
                    lexical_candidates = self._merge_lexical_candidates(
                        lexical_candidate_sets,
                        candidate_k,
                    )
                    fused_candidates = self._fuse_candidates(semantic_candidates, lexical_candidates)
                    reranked = self.reranker.rerank(query_text, fused_candidates)
                    retrieval_ms += int((time.perf_counter() - expansion_retrieval_start) * 1000)
                    query_variants = [query_text, *expanded_queries[:pair_count]]
                    expansion_applied = True
                    expansion_reason = "applied"
            else:
                expansion_reason = "no_expansion_terms"
        elif base_has_title_override:
            expansion_reason = "title_override"

        title_override_ids = {movie.id for movie in lexical_candidates if self._is_strong_title_match(payload.query, movie.title)}
        for movie in reranked:
            if movie.id in title_override_ids:
                movie.rerank_score = max(movie.rerank_score, 0.99)

        reranked = sorted(reranked, key=lambda x: x.rerank_score, reverse=True)
        top_score = reranked[0].rerank_score if reranked else None
        has_title_override = any(movie.id in title_override_ids for movie in reranked[: payload.top_k])
        top_vector_score = max((movie.vector_score for movie in reranked), default=None)
        top_lexical_score = max((movie.lexical_score for movie in reranked), default=None)
        force_no_results_safety, no_results_reason = self._no_results_safety_decision(
            payload=payload,
            has_title_override=has_title_override,
            top_vector_score=top_vector_score,
            top_lexical_score=top_lexical_score,
        )
        threshold_no_results = top_score is None or (
            top_score < self.settings.relevance_threshold and not has_title_override
        )
        no_results_gate_reason: str | None = None
        if force_no_results_safety:
            no_results_gate_reason = no_results_reason
        elif top_score is None:
            no_results_gate_reason = "empty_results"
        elif threshold_no_results:
            no_results_gate_reason = "threshold"

        total_latency_ms = int((time.perf_counter() - start) * 1000)
        latency_breakdown = LatencyBreakdown(
            intent_parse_ms=intent_parse_ms,
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_ms,
            summary_ms=summary_ms,
            total_ms=total_latency_ms,
        )

        if force_no_results_safety or threshold_no_results:
            return QueryResponse(
                query_interpretation=QueryInterpretation(
                    original_query=payload.query,
                    parsed_filters=parsed_filters,
                    query_variants=query_variants if len(query_variants) > 1 else None,
                ),
                applied_filters=applied_filters,
                results=[],
                summary="No relevant results found for the given query and filters.",
                meta=QueryResponseMeta(
                    top_k=payload.top_k,
                    sort=payload.sort,
                    llm_provider="none",
                    fallback_used=False,
                    no_relevant_results=True,
                    confidence=ConfidenceInfo(
                        top_score=round(top_score, 6) if top_score is not None else None,
                        threshold=self.settings.relevance_threshold,
                    ),
                    latency_ms=total_latency_ms,
                    latency_breakdown=latency_breakdown,
                    expansion_applied=expansion_applied,
                    expansion_reason=expansion_reason,
                    no_results_reason=no_results_gate_reason,
                ),
            )

        sorted_results = self._apply_sort(reranked[: payload.top_k], payload.sort)

        summary_start = time.perf_counter()
        summary, provider, fallback_used = await self.llm_service.summarize_results(payload.query, sorted_results)
        summary_ms = int((time.perf_counter() - summary_start) * 1000)

        results = [
            MovieResult(
                id=movie.id,
                title=movie.title,
                overview=movie.overview,
                genres=movie.genres,
                cast=movie.cast,
                release_year=movie.release_year,
                rating=movie.rating,
                semantic_score=round(movie.semantic_score, 6),
                vector_score=round(movie.vector_score, 6),
                lexical_score=round(movie.lexical_score, 6),
                fused_score=round(movie.fused_score, 6),
                cross_encoder_score=round(movie.cross_encoder_score, 6) if movie.cross_encoder_score is not None else None,
                rerank_score=round(movie.rerank_score, 6),
                why_matched=self._why_matched(
                    movie=movie,
                    query=payload.query,
                    filters=applied_filters,
                    is_title_override=movie.id in title_override_ids,
                ),
            )
            for movie in sorted_results
        ]

        total_latency_ms = int((time.perf_counter() - start) * 1000)
        latency_breakdown = LatencyBreakdown(
            intent_parse_ms=intent_parse_ms,
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_ms,
            summary_ms=summary_ms,
            total_ms=total_latency_ms,
        )

        return QueryResponse(
            query_interpretation=QueryInterpretation(
                original_query=payload.query,
                parsed_filters=parsed_filters,
                query_variants=query_variants if len(query_variants) > 1 else None,
            ),
            applied_filters=applied_filters,
            results=results,
            summary=summary,
            meta=QueryResponseMeta(
                top_k=payload.top_k,
                sort=payload.sort,
                llm_provider=provider,
                fallback_used=fallback_used,
                no_relevant_results=False,
                confidence=ConfidenceInfo(
                    top_score=round(top_score, 6) if top_score is not None else None,
                    threshold=self.settings.relevance_threshold,
                ),
                latency_ms=total_latency_ms,
                latency_breakdown=latency_breakdown,
                expansion_applied=expansion_applied,
                expansion_reason=expansion_reason,
                no_results_reason=None,
            ),
        )

import logging
import math
import importlib
from datetime import datetime

from app.core.settings import Settings
from app.models.movie import RetrievedMovie

logger = logging.getLogger(__name__)


class ReRanker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cross_encoder = None
        self._cross_encoder_available = False
        self._cross_encoder_init_attempted = False

    def _init_cross_encoder(self) -> None:
        if self._cross_encoder_init_attempted:
            return
        self._cross_encoder_init_attempted = True

        if self.settings.reranker_mode != "cross_encoder":
            return

        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
            CrossEncoder = getattr(sentence_transformers, "CrossEncoder")
            self._cross_encoder = CrossEncoder(self.settings.cross_encoder_model)
            self._cross_encoder_available = True
            logger.info(
                "Cross-encoder reranker initialized",
                extra={"cross_encoder_model": self.settings.cross_encoder_model},
            )
        except Exception as exc:
            logger.warning(
                "Cross-encoder initialization failed, falling back to heuristic reranker",
                extra={
                    "cross_encoder_model": self.settings.cross_encoder_model,
                    "error_type": exc.__class__.__name__,
                },
            )
            self._cross_encoder = None
            self._cross_encoder_available = False

    def warmup(self) -> bool:
        self._init_cross_encoder()
        return self._cross_encoder_available

    @staticmethod
    def _sigmoid(value: float) -> float:
        # map logits into 0 to 1 before blending
        return 1.0 / (1.0 + math.exp(-max(min(value, 20.0), -20.0)))

    @staticmethod
    def _heuristic_score(movie: RetrievedMovie) -> float:
        current_year = datetime.utcnow().year
        rating_norm = (movie.rating or 0.0) / 10.0
        recency_norm = 0.0
        if movie.release_year:
            age = max(current_year - movie.release_year, 0)
            recency_norm = max(0.0, 1.0 - (age / 50.0))
        return (movie.fused_score * 0.75) + (rating_norm * 0.2) + (recency_norm * 0.05)

    def _apply_cross_encoder(self, query: str, movies: list[RetrievedMovie]) -> None:
        if not self._cross_encoder_available or self._cross_encoder is None:
            for movie in movies:
                movie.cross_encoder_score = None
                movie.rerank_score = self._heuristic_score(movie)
            return

        pairs = [(query, f"{movie.title}. {movie.overview}") for movie in movies]
        raw_scores = self._cross_encoder.predict(pairs)
        normalized_scores = [self._sigmoid(float(score)) for score in raw_scores]

        ce_weight = self.settings.cross_encoder_weight
        fused_weight = max(0.0, 1.0 - ce_weight - 0.1)
        rating_weight = 0.1

        for movie, ce_score in zip(movies, normalized_scores):
            movie.cross_encoder_score = ce_score
            rating_norm = (movie.rating or 0.0) / 10.0
            movie.rerank_score = (ce_score * ce_weight) + (movie.fused_score * fused_weight) + (rating_norm * rating_weight)

    def rerank(self, query: str, movies: list[RetrievedMovie]) -> list[RetrievedMovie]:
        if not movies:
            return []

        self._init_cross_encoder()
        self._apply_cross_encoder(query, movies)
        return sorted(movies, key=lambda item: item.rerank_score, reverse=True)

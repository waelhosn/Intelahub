from pydantic import BaseModel, Field


class MovieRecord(BaseModel):
    id: int
    title: str
    overview: str
    genres: list[str] = Field(default_factory=list)
    cast: list[str] = Field(default_factory=list)
    release_year: int | None = None
    rating: float | None = None

    @property
    def embedding_text(self) -> str:
        genres = ", ".join(self.genres)
        cast = ", ".join(self.cast)
        return (
            f"Title: {self.title}\n"
            f"Overview: {self.overview}\n"
            f"Genres: {genres}\n"
            f"Cast: {cast}\n"
            f"Year: {self.release_year}\n"
            f"Rating: {self.rating}"
        )


class RetrievedMovie(MovieRecord):
    vector_score: float = 0.0
    lexical_score: float = 0.0
    fused_score: float = 0.0
    cross_encoder_score: float | None = None
    semantic_score: float
    rerank_score: float

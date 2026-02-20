from typing import Any

from app.models.movie import MovieRecord


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_tmdb_movie(details: dict[str, Any], credits: dict[str, Any], cast_limit: int = 10) -> MovieRecord | None:
    movie_id = details.get("id")
    title = _safe_str(details.get("title"))
    overview = _safe_str(details.get("overview"))

    if not isinstance(movie_id, int) or not title:
        return None

    genres = []
    for genre in details.get("genres", []):
        name = _safe_str(genre.get("name"))
        if name:
            genres.append(name.lower())

    cast = []
    for person in (credits.get("cast") or [])[:cast_limit]:
        name = _safe_str(person.get("name"))
        if name:
            cast.append(name)

    release_date = _safe_str(details.get("release_date"))
    release_year = None
    if len(release_date) >= 4 and release_date[:4].isdigit():
        release_year = int(release_date[:4])

    vote_average = details.get("vote_average")
    rating = float(vote_average) if isinstance(vote_average, (int, float)) else None

    return MovieRecord(
        id=movie_id,
        title=title,
        overview=overview,
        genres=genres,
        cast=cast,
        release_year=release_year,
        rating=rating,
    )

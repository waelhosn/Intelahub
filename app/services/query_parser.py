import re

from app.models.query import QueryFilters

KNOWN_GENRES = {
    "action",
    "adventure",
    "animation",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "family",
    "fantasy",
    "history",
    "horror",
    "music",
    "mystery",
    "romance",
    "science fiction",
    "sci-fi",
    "thriller",
    "tv movie",
    "war",
    "western",
}


def parse_filters_from_query(query: str) -> QueryFilters:
    text = query.lower()
    filters = QueryFilters()

    found_genres: list[str] = []
    for genre in KNOWN_GENRES:
        if genre in text:
            normalized = "science fiction" if genre == "sci-fi" else genre
            found_genres.append(normalized)
    if found_genres:
        filters.genres = sorted(set(found_genres))

    year_matches = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    if year_matches:
        if any(token in text for token in ["after", "since", "newer than", "from"]):
            filters.min_year = min(year_matches)
        elif any(token in text for token in ["before", "older than", "until"]):
            filters.max_year = max(year_matches)
        elif len(year_matches) == 1:
            filters.min_year = year_matches[0]
            filters.max_year = year_matches[0]
        else:
            filters.min_year = min(year_matches)
            filters.max_year = max(year_matches)

    rating_match = re.search(r"(?:rating\s*(?:above|over|>=|greater than)\s*|rated\s*)(\d+(?:\.\d+)?)", text)
    if rating_match:
        rating_value = float(rating_match.group(1))
        if 0 <= rating_value <= 10:
            filters.min_rating = rating_value

    return filters


def merge_filters(parsed: QueryFilters, explicit: QueryFilters | None) -> QueryFilters:
    if explicit is None:
        return parsed

    merged = QueryFilters(
        genres=explicit.genres if explicit.genres is not None else parsed.genres,
        min_year=explicit.min_year if explicit.min_year is not None else parsed.min_year,
        max_year=explicit.max_year if explicit.max_year is not None else parsed.max_year,
        min_rating=explicit.min_rating if explicit.min_rating is not None else parsed.min_rating,
    )
    return merged

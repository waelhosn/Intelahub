from app.models.query import QueryFilters
from app.services.query_parser import merge_filters, parse_filters_from_query


def test_parse_filters_from_query() -> None:
    parsed = parse_filters_from_query("best action movies after 2018 rating above 7.2")

    assert parsed.genres and "action" in parsed.genres
    assert parsed.min_year == 2018
    assert parsed.min_rating == 7.2


def test_merge_filters_explicit_precedence() -> None:
    parsed = QueryFilters(genres=["action"], min_year=2018, min_rating=7.0)
    explicit = QueryFilters(genres=["drama"], min_year=2010)

    merged = merge_filters(parsed, explicit)

    assert merged.genres == ["drama"]
    assert merged.min_year == 2010
    assert merged.min_rating == 7.0

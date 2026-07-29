"""Shared final-answer limits for runtime, evaluation, and repair."""

CHINESE_SINGLE_FORECAST_MAX_CHARS = 500
CHINESE_COMPARISON_MAX_CHARS = 750
CHINESE_COMPARISON_EXTRA_CANDIDATE_CHARS = 100
CHINESE_COMPARISON_BASE_CANDIDATES = 3
CHINESE_COMPARISON_ABSOLUTE_MAX_CHARS = 1_200
ENGLISH_MAX_WORDS = 160
ENGLISH_COMPARISON_MAX_CHARS = 2_000
GENERIC_MAX_CHARS = 1_200


def chinese_comparison_max_chars(candidate_count: int) -> int:
    """Scale evidence space after three candidates while retaining a cap."""
    extra_candidates = max(
        0,
        int(candidate_count) - CHINESE_COMPARISON_BASE_CANDIDATES,
    )
    return min(
        CHINESE_COMPARISON_ABSOLUTE_MAX_CHARS,
        CHINESE_COMPARISON_MAX_CHARS
        + extra_candidates * CHINESE_COMPARISON_EXTRA_CANDIDATE_CHARS,
    )

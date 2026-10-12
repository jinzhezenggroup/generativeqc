"""Independent fixture tests for GitHub Actions compiler-cache reporting."""

import pytest

from tools.report_ci_ccache import CacheStats, parse_stats, render_summary


def test_reports_actual_hits_not_actions_restore_status() -> None:
    output = """Cacheable calls:   311 / 311 (100.0%)
  Hits:             74 / 311 (23.79%)
    Direct:         74 /  74 (100.0%)
    Preprocessed:    0 /  74 ( 0.00%)
  Misses:          237 / 311 (76.21%)
Local storage:
  Cache size (GB): 0.1 / 2.0 (6.16%)
  Hits:             74 / 311 (23.79%)
"""
    actual = parse_stats(output)
    assert actual == CacheStats(311, 311, 74, 237, 0)
    summary = render_summary("CPU CTest gcc", actual)
    assert "| Cache hits | 74 (23.8%) |" in summary
    assert "| Cache misses | 237 |" in summary
    assert "restored Actions cache does not guarantee" in summary


def test_uncacheable_compiler_calls_are_not_fake_misses() -> None:
    output = """Cacheable calls:   254 / 526 (48.29%)
  Hits:            254 / 254 (100.0%)
  Misses:            0 / 254 (0.00%)
Uncacheable calls: 272 / 526 (51.71%)
"""
    assert parse_stats(output) == CacheStats(254, 526, 254, 0, 272)
    assert "| Uncacheable calls | 272 |" in render_summary(
        "CUDA native CLI", parse_stats(output)
    )


def test_zero_compiler_calls_are_not_reported_as_zero_percent_cache_hits() -> None:
    output = """Cacheable calls:     0 /  99 ( 0.00%)
  Hits:              0
  Misses:            0
Uncacheable calls:  99 /  99 (100.0%)
"""
    stats = parse_stats(output)
    summary = render_summary("Array API", stats)
    assert "| Cache hits | 0 (n/a (no cacheable calls)) |" in summary
    assert "| Uncacheable calls | 99 |" in summary


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "Cacheable calls: 5 / 5\nHits: 1\nMisses: 3\n",
        "Cacheable calls: 5 / 5\nHits: 1\nMisses: 4\nUncacheable calls: 1 / 6",
    ],
)
def test_rejects_incomplete_or_inconsistent_counter_reports(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_stats(raw)


def test_absent_ccache_is_explicit_and_non_gating() -> None:
    text = render_summary("CPU | runner", None, "executable not installed")
    assert "CPU \\| runner" in text
    assert "statistics unavailable" in text

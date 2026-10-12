"""Independent fixture tests for GitHub Actions compiler-cache reporting."""

from pathlib import Path

import pytest

from tools.report_ci_ccache import (
    CacheStats,
    parse_stats,
    render_build_context,
    render_summary,
)


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


@pytest.mark.parametrize("provider", ["0", "1"])
def test_build_context_distinguishes_auto_request_from_emitted_provider(
    tmp_path: Path, provider: str
) -> None:
    (tmp_path / "CMakeCache.txt").write_text(
        "GENERATIVEQC_CPU_LINALG_PROVIDER:STRING=auto\n"
        "GENERATIVEQC_CUDA_FAST_COMPILE:BOOL=ON\n"
        "GENERATIVEQC_MONOLITHIC_CLI:BOOL=OFF\n"
        "PRIVATE_TOKEN:STRING=must-not-be-printed\n"
    )
    (tmp_path / "build.ninja").write_text(
        f"  DEFINES = -DGENERATIVEQC_HAS_OPENBLAS={provider}\n"
    )
    records = tmp_path / "CMakeFiles/4.0.3"
    records.mkdir(parents=True)
    for language, compiler, identity, version in (
        ("CXX", "/usr/bin/c++", "GNU", "13.3.0"),
        ("CUDA", "/usr/local/cuda-12.9/bin/nvcc", "NVIDIA", "12.9.86"),
    ):
        (records / f"CMake{language}Compiler.cmake").write_text(
            f'set(CMAKE_{language}_COMPILER "{compiler}")\n'
            f'set(CMAKE_{language}_COMPILER_ID "{identity}")\n'
            f'set(CMAKE_{language}_COMPILER_VERSION "{version}")\n'
        )
    text = render_build_context(tmp_path)
    assert "GENERATIVEQC_CPU_LINALG_PROVIDER=auto" in text
    assert (
        f"Emitted CPU linear algebra provider: {'OpenBLAS' if provider == '1' else 'scalar'}"
        in text
    )
    assert "CMAKE_CXX_COMPILER_ID=GNU" in text
    assert "CMAKE_CXX_COMPILER_VERSION=13.3.0" in text
    assert "CMAKE_CUDA_COMPILER_VERSION=12.9.86" in text
    assert "must-not-be-printed" not in text


@pytest.mark.parametrize(
    "definitions", ["", "-DGENERATIVEQC_HAS_OPENBLAS=0 -DGENERATIVEQC_HAS_OPENBLAS=1"]
)
def test_incomplete_or_mixed_provider_evidence_is_not_guessed(
    tmp_path: Path, definitions: str
) -> None:
    (tmp_path / "build.ninja").write_text(definitions)
    text = render_build_context(tmp_path)
    assert "provider: unavailable or mixed" in text
    assert "CMake cache unavailable" in text
    assert "Configured CUDA compiler identity: unavailable" in text


def test_unreadable_build_context_is_non_gating(tmp_path: Path) -> None:
    (tmp_path / "CMakeCache.txt").write_bytes(b"\xff")
    (tmp_path / "build.ninja").mkdir()
    text = render_build_context(tmp_path)
    assert "CMake cache unavailable" in text
    assert "provider unavailable" in text

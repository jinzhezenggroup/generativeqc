"""Archived diff-context bytes must not be rewritten by whitespace hooks."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _trailing_whitespace_exclusion() -> str:
    lines = (ROOT / ".pre-commit-config.yaml").read_text().splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.strip() == "- id: trailing-whitespace"
    )
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("- id:"):
            break
        if stripped.startswith("exclude:"):
            return stripped.partition(":")[2].strip()
    raise AssertionError("trailing-whitespace hook must define an exclusion")


def test_trailing_whitespace_exemption_matches_patches_only() -> None:
    exclusion = re.compile(_trailing_whitespace_exclusion())
    assert exclusion.search("benchmarks/results/issue394-deferred/prototype.patch")
    assert exclusion.search("benchmarks/results/example/nested/source.patch")
    assert not exclusion.search("benchmarks/results/example/result.json")
    assert not exclusion.search("src/example.cpp")
    assert not exclusion.search("tests/python/example.py")

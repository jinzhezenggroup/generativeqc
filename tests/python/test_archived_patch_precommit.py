"""Archived diff-context bytes must not be rewritten by whitespace hooks."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_trailing_whitespace_exemption_matches_patches_only() -> None:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hook = next(
        hook
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "trailing-whitespace"
    )
    exclusion = re.compile(hook["exclude"])
    assert exclusion.search("benchmarks/results/issue394-deferred/prototype.patch")
    assert exclusion.search("benchmarks/results/example/nested/source.patch")
    assert not exclusion.search("benchmarks/results/example/result.json")
    assert not exclusion.search("src/example.cpp")
    assert not exclusion.search("tests/python/example.py")

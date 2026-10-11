"""Keep published compact-MP2 evidence bound to its actual artifact bytes."""

import hashlib
import json
from pathlib import Path

import pytest

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/mp2-factorized-first-20261011"
)
PROVENANCE = json.loads((EVIDENCE / "provenance.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name,expected", PROVENANCE["artifacts"].items())
def test_published_artifact_hash_matches_bytes(name: str, expected: str) -> None:
    # These are publication hashes. The tracker's historical compiled-source
    # identity is intentionally separate and must not be relabeled after formatters.
    actual = hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
    assert actual == expected, name

"""Storage replay reaches, and retains, historical dependency authentication."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from benchmarks._exact_fp64_evidence import load

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "benchmarks/results/cuda-ks-seed-products-20261101"


def test_seed_replay_authenticates_storage_before_historical_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = runpy.run_path(str(FAMILY / "report.py"))["main"]
    original_check = main.__globals__["checked_file"]
    _, files = load(FAMILY)
    original = json.loads(files["original-receipts.json"])
    authenticated = []

    class HistoricalDependencyReached(Exception):
        pass

    def checked(relative: str, expected: str) -> bytes:
        if relative == original["oracle_path"]:
            assert expected == original["oracle_sha256"]
            raise HistoricalDependencyReached
        raw = original_check(relative, expected)
        authenticated.append(relative)
        return raw

    monkeypatch.setitem(main.__globals__, "checked_file", checked)
    monkeypatch.setattr(sys, "argv", ["report.py", "--verify"])
    with pytest.raises(HistoricalDependencyReached):
        main()
    assert authenticated == [str((FAMILY / "fp64.xz").relative_to(ROOT))]


def test_seed_historical_dependency_rejects_changed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked = runpy.run_path(str(FAMILY / "report.py"))["checked_file"]
    _, files = load(FAMILY)
    original = json.loads(files["original-receipts.json"])
    path = tmp_path / original["oracle_path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed historical dependency")
    monkeypatch.setitem(checked.__globals__, "ROOT", tmp_path)
    with pytest.raises(AssertionError):
        checked(original["oracle_path"], original["oracle_sha256"])

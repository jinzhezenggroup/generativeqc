"""Archiving hash streams must preserve offline numerical and identity checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def reader() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/gfn2-current-source-20261114/reconcile.py"
    )
    spec = importlib.util.spec_from_file_location("gfn2_census_reconcile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_numerical_replay_retains_every_observation() -> None:
    module = reader()
    expected = json.loads((module.ROOT / "summary.json").read_text())
    actual = module.reconcile()
    assert {name: actual[name] for name in expected} == expected
    assert actual["whole_source_census_audited"] is False


def test_explicit_source_audit_rejects_wrong_original_bytes(tmp_path: Path) -> None:
    for name in ("cpu", "cuda"):
        (tmp_path / f"{name}-source.sha256.xz").write_bytes(b"not the archived input")
    with pytest.raises(ValueError, match="archive identity mismatch"):
        reader().reconcile(source_censuses=tmp_path)


def test_snapshot_corruption_is_not_hidden_by_numerical_replay(tmp_path: Path) -> None:
    module = reader()
    (tmp_path / "provenance.json").write_bytes(
        (module.ROOT / "provenance.json").read_bytes()
    )
    snapshot = json.loads((module.ROOT / "source-censuses-snapshot.json").read_text())
    snapshot["files"][0]["decoded_sha256"] = "0" * 64
    (tmp_path / "source-censuses-snapshot.json").write_text(json.dumps(snapshot))
    module.ROOT = tmp_path
    with pytest.raises(ValueError, match="source census snapshot identity mismatch"):
        module.reconcile()

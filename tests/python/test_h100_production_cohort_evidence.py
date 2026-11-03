"""Public replay rejects incomplete support cohorts and changed archive parts."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "h100_support_reader", ROOT / "benchmarks/h100_production_cohort.py"
)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Replay requires Python 3.12+")
def test_actual_supporting_phases_replay() -> None:
    result = evidence.replay()
    assert [row["cohort"] for row in result] == [
        "native-qualification-v1",
        "nsys-v1",
        "standard-guards-v1",
        "uks-fit-v1",
        "reference-six-v5",
    ]
    assert (
        result[2]["status"]
        == "NATIVE_GUARDS_FAILED_OR_INCOMPLETE_WITH_ALL_ORIGINALS_RETAINED"
    )


def _publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "support"
    shutil.copytree(evidence.DIRECTORY, target)
    monkeypatch.setattr(evidence, "DIRECTORY", target)
    return target


def test_support_cohort_binds_every_required_artifact() -> None:
    publication = evidence.checked_publication()
    assert len(publication["phases"]) == 5
    assert publication["expected_statuses"]["standard-guards-v1"] == (
        "NATIVE_GUARDS_FAILED_OR_INCOMPLETE_WITH_ALL_ORIGINALS_RETAINED"
    )


def test_support_cohort_rejects_missing_archive_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _publication(tmp_path, monkeypatch)
    (target / "nsys-v1-analysis.tar.xz.part01").unlink()
    with pytest.raises(ValueError, match="bytes differ"):
        evidence.checked_publication()


def test_support_cohort_rejects_changed_archive_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _publication(tmp_path, monkeypatch)
    path = target / "nsys-v1-analysis.tar.xz.part00"
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError, match="bytes differ"):
        evidence.checked_publication()


def test_support_cohort_rejects_unsafe_part_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _publication(tmp_path, monkeypatch)
    path = target / "cohort.json"
    publication = json.loads(path.read_text())
    publication["nsys_archive_parts"][0]["path"] = "/tmp/unrelated-input"
    path.write_text(json.dumps(publication))
    with pytest.raises(ValueError, match="part inventory differs"):
        evidence.checked_publication()

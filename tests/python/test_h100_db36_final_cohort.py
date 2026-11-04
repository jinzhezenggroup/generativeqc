"""Reject missing or changed observations in the frozen final publication."""

import json
import shutil
from pathlib import Path

import pytest

from benchmarks import h100_db36_final_cohort as evidence


def test_final_publication_retains_original_contract_and_actual_review() -> None:
    publication = evidence.checked_publication()
    assert publication["acceptance"]["returned_EF"] == 720
    assert publication["acceptance"]["physical_cases"] == 30
    assert publication["actual_result_audit"]["review_independence"] == "same-family"
    assert publication["actual_result_audit"]["acceptance_status"] == "provisional"


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "protocol", "part-order"])
def test_final_publication_rejects_changed_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    target = tmp_path / "final"
    shutil.copytree(evidence.DIRECTORY, target)
    monkeypatch.setattr(evidence, "DIRECTORY", target)
    if mutation == "missing":
        (target / "final-pairs-v1-completed.tar.xz.part01").unlink()
    elif mutation == "corrupt":
        path = target / "actual-result-audit.tar.xz"
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    else:
        path = target / "cohort.json"
        publication = json.loads(path.read_text())
        if mutation == "protocol":
            publication["original_final_protocol_sha256"] = "0" * 64
        else:
            publication["parts"].reverse()
        path.write_text(json.dumps(publication))
    with pytest.raises(ValueError):
        evidence.checked_publication()

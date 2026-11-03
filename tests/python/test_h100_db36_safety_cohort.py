"""Public safety evidence must preserve failed coverage and authenticated raw data."""

import json
import shutil
from pathlib import Path

import pytest

from benchmarks import h100_db36_safety_cohort as reader


def test_original_failed_race_and_zero_votes_remain_explicit() -> None:
    publication = reader.checked_publication()
    assert publication["original_failed_spatial_race_accepted_cases"] == 0
    assert publication["majority_accepted"] is False
    assert publication["clean_timing_votes"] == 0
    assert len(publication["proofs"]) == 7


@pytest.mark.parametrize("corruption", ["partial_race_promotion", "raw_vector_packet"])
def test_rejects_changed_coverage_or_raw_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    target = tmp_path / "observations"
    shutil.copytree(reader.DIRECTORY, target)
    monkeypatch.setattr(reader, "DIRECTORY", target)
    if corruption == "partial_race_promotion":
        path = target / "cohort.json"
        publication = json.loads(path.read_text())
        publication["original_failed_spatial_race_accepted_cases"] = 5
        path.write_text(json.dumps(publication))
    else:
        path = target / "public-direct-v1-completed.tar.xz"
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
    with pytest.raises(ValueError):
        reader.checked_publication()

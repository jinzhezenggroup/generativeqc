"""Prior-source data and intrusive profiles cannot become final timing votes."""

import json
import shutil
from pathlib import Path

import pytest

from benchmarks import h100_db36_profile_cohort as reader


def test_prior_source_and_zero_final_votes_remain_explicit() -> None:
    p = reader.checked_publication()
    assert p["prior_source_revision"] != p["source_revision"]
    assert p["majority_accepted"] is False
    assert p["clean_final_timing_votes"] == 0
    assert len(p["proofs"]) == 4


@pytest.mark.parametrize("mutation", ["source_transfer", "profile_bytes"])
def test_rejects_relabelled_or_corrupt_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    target = tmp_path / "profile"
    shutil.copytree(reader.DIRECTORY, target)
    monkeypatch.setattr(reader, "DIRECTORY", target)
    if mutation == "source_transfer":
        path = target / "cohort.json"
        p = json.loads(path.read_text())
        p["prior_source_revision"] = p["source_revision"]
        path.write_text(json.dumps(p))
    else:
        path = target / "nsys-v1-analysis.tar.xz"
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError):
        reader.checked_publication()

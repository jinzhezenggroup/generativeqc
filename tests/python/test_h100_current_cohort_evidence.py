"""The retained H100 cohort must replay offline and retain its scope boundary."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "h100_retained_reader", ROOT / "benchmarks/h100_current_cohort.py"
)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Filtered H100 packet replay requires Python 3.12+",
)
def test_actual_h100_build_df_and_screening_replay() -> None:
    result = reader.replay()
    assert result["df_tests"] == 44
    assert result["df_public_full_ef"] == 299
    assert result["screening_processes"] == 66
    assert result["primary_inputs"] == result["qualified_primary_inputs"] == 30
    assert result["exploratory_ratios_gt_1_05"] == 21
    assert result["complete_primary_geomean"] == pytest.approx(1.265755834772816)
    assert result["final_three_pair_acceptance"] is False


@pytest.mark.parametrize(
    "corruption", ["source_label", "unbound_reader", "archive_bytes"]
)
def test_publication_rejects_provenance_gaps(tmp_path: Path, corruption: str) -> None:
    shutil.copytree(reader.DIRECTORY, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "cohort.json"
    publication = json.loads(path.read_text())
    if corruption == "source_label":
        publication["source_revision"] = reader.REFERENCE
    elif corruption == "unbound_reader":
        publication["files"] = [
            entry
            for entry in publication["files"]
            if entry["path"] != "readers/replay_screening.py"
        ]
        with (tmp_path / "readers/replay_screening.py").open("a") as stream:
            stream.write("\n# unbound reader change\n")
    else:
        archive = tmp_path / "screening-v1-completed.tar.xz"
        data = bytearray(archive.read_bytes())
        data[len(data) // 2] ^= 1
        archive.write_bytes(data)
    path.write_text(json.dumps(publication))
    with pytest.raises(ValueError):
        reader.checked_publication(tmp_path)

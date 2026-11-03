"""The public final receiver keeps the full denominator and rejects damaged data."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from benchmarks import h100_final_cohort as evidence


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Replay requires Python 3.12+")
def test_actual_fixed30_full_raw_replay() -> None:
    final = evidence.support.replay(extension=evidence.extend)[-1]
    assert final["physical_cases"] == len(final["cases"]) == 30
    assert final["qualified_populations"] == final["attempted_populations"] == 180
    assert final["returned_EF"] == 720
    assert final["wins"] == 20
    assert final["complete_set_geomean"] == pytest.approx(1.2869162091880366)
    assert sum(not case["win"] for case in final["cases"]) == 10
    assert all(case["qualified"] for case in final["cases"])


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "order", "denominator"])
def test_final_packet_rejects_missing_changed_or_reordered_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    directory = tmp_path / "final"
    shutil.copytree(evidence.DIRECTORY, directory)
    monkeypatch.setattr(evidence, "DIRECTORY", directory)
    if mutation == "missing":
        (directory / "final-pairs-v3-completed.tar.xz.part01").unlink()
    elif mutation == "corrupt":
        path = directory / "final-pairs-v3-completed.tar.xz.part00"
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    else:
        path = directory / "cohort.json"
        publication = json.loads(path.read_text())
        if mutation == "order":
            publication["parts"].reverse()
        else:
            publication["original_final_protocol_sha256"] = "0" * 64
        path.write_text(json.dumps(publication))
    with pytest.raises(ValueError):
        evidence.checked_publication()

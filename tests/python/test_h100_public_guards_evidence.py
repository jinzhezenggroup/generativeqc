"""Public guard delivery retains failures and rejects incomplete scientific inputs."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from benchmarks import h100_public_guards as evidence


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Replay requires Python 3.12+")
def test_public_guard_raw_replay_preserves_failed_and_scoped_outcomes() -> None:
    rows = evidence.support.replay(extension=evidence.extend)
    indexed = {row["cohort"]: row for row in rows}
    assert indexed["public-budget-v5"]["returned_EF"] == 0
    assert indexed["public-budget-v7"]["returned_EF"] == 3
    assert indexed["public-budget-v7"]["scientific_acceptance"] is False
    assert indexed["facade-guards-v1"]["status"] == (
        "FACADE_GUARDS_FAILED_OR_INCOMPLETE_ALL_ORIGINALS_RETAINED"
    )
    corrected = indexed["facade-correction-v2"]
    assert (corrected["public_EF"], corrected["closed_controls"]) == (142, 84)
    assert corrected["native_heterogeneous_df_batch_qualified"] is False
    assert corrected["all_fast_trials_native_force_qualified"] is False


@pytest.mark.parametrize(
    "mutation", ["missing-reference", "corrupt-archive", "timing-vote"]
)
def test_public_guards_reject_missing_or_changed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    directory = tmp_path / "packet"
    shutil.copytree(evidence.DIRECTORY, directory)
    monkeypatch.setattr(evidence, "DIRECTORY", directory)
    if mutation == "missing-reference":
        (directory / "facade-guards-v1-completed.tar.xz").unlink()
    elif mutation == "corrupt-archive":
        path = directory / "notebook-public-guards-v1-completed.tar.xz"
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    else:
        path = directory / "cohort.json"
        publication = json.loads(path.read_text())
        publication["clean_timing_votes"] = 1
        path.write_text(json.dumps(publication))
    with pytest.raises(ValueError):
        evidence.checked_publication()

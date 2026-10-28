"""Replay CUDA storage evidence and reject incomplete package-load witnesses."""

import json
import runpy
from pathlib import Path

import pytest

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/cuda-image-storage-20261028"
)
REPLAY = runpy.run_path(str(BUNDLE / "reduce.py"))
REDUCE = REPLAY["reduce"]
LOAD = REPLAY["load_receipt"]


def receipt() -> dict:
    """Read a fresh receipt because replay decodes its samples in place."""
    return LOAD()


def test_complete_package_witnesses_replay_independent_of_row_order() -> None:
    data = receipt()
    for arm in ("baseline", "final", "default"):
        data["records"][f"remote/package-load-{arm}.json"]["rows"].reverse()
    result = REDUCE(data)
    assert result["campaigns"]["mainline-default"]["failures"] == []


@pytest.mark.parametrize("arm", ["baseline", "final", "default"])
def test_duplicate_package_witness_cannot_replace_missing_libraries(arm: str) -> None:
    data = receipt()
    rows = data["records"][f"remote/package-load-{arm}.json"]["rows"]
    rows[:] = [rows[0]] * len(rows)
    with pytest.raises(AssertionError):
        REDUCE(data)


def test_summary_retains_all_campaign_results() -> None:
    summary = json.loads((BUNDLE / "summary.json").read_text())
    result = REDUCE(receipt())
    assert summary["sizes"] == result["sizes"]
    for name, expected in summary["campaigns"].items():
        assert {key: result["campaigns"][name][key] for key in expected} == expected
    assert sum(c["calls"] for c in summary["campaigns"].values()) == 2075


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "0" * 64),
        ("git_blob_sha1", "0" * 40),
        ("bytes", 1),
        ("source_revision", "0" * 40),
        ("schema", "unsupported"),
    ],
)
def test_original_receipt_rejects_changed_identity_or_missing_history(
    tmp_path: Path, field: str, value: str | int
) -> None:
    manifest = json.loads((BUNDLE / "original-receipt.snapshot.json").read_text())
    if field in ("source_revision", "schema"):
        manifest[field] = value
    else:
        manifest["files"][0][field] = value
        if field == "bytes":
            manifest["total_bytes"] = value
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        LOAD(path)

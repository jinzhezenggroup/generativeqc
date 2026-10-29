"""Replay CUDA storage evidence and reject incomplete package-load witnesses."""

import json
import runpy
import shutil
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


def source_export(tmp_path: Path) -> Path:
    """Copy the real offline replay inputs without a repository or history."""
    root = tmp_path / "source"
    bundle = root / "benchmarks/results" / BUNDLE.name
    bundle.mkdir(parents=True)
    for name in (
        "reduce.py",
        "receipts.json.xz",
        "original-receipt.snapshot.json",
        "summary.json",
    ):
        shutil.copyfile(BUNDLE / name, bundle / name)
    tools = root / "tools"
    tools.mkdir()
    shutil.copyfile(
        BUNDLE.parents[2] / "tools/restore_retained_evidence.py",
        tools / "restore_retained_evidence.py",
    )
    return bundle


def test_real_source_export_replays_without_git_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = source_export(tmp_path)

    def unavailable(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline receipt must not invoke Git or a fetch")

    monkeypatch.setattr("subprocess.run", unavailable)
    reader = runpy.run_path(str(bundle / "reduce.py"))
    result = reader["reduce"](reader["load_receipt"]())
    assert sum(c["calls"] for c in result["campaigns"].values()) == 2075
    assert result["campaigns"]["compression"]["failures"]
    assert result["campaigns"]["mainline-default"]["failures"] == []


@pytest.mark.parametrize("damage", ["same-size-change", "truncated", "missing"])
def test_source_export_rejects_corrupt_or_missing_receipt(
    tmp_path: Path, damage: str
) -> None:
    bundle = source_export(tmp_path)
    path = bundle / "receipts.json.xz"
    raw = path.read_bytes()
    if damage == "same-size-change":
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    elif damage == "truncated":
        path.write_bytes(raw[:-1])
    else:
        path.unlink()
    reader = runpy.run_path(str(bundle / "reduce.py"))
    with pytest.raises((ValueError, FileNotFoundError)):
        reader["load_receipt"]()


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

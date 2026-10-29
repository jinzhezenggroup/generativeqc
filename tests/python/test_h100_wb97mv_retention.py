"""Keep live scientific replay offline and historical HOST recovery explicit."""

import base64
import hashlib
import json
import lzma
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/h100-direct-wb97mv-20261024"
READER = runpy.run_path(str(BUNDLE / "reader.py"))


def test_actual_scientific_replay_needs_no_history_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise AssertionError("ordinary scientific replay must remain offline")

    monkeypatch.setattr("subprocess.run", unavailable)
    result = READER["report"](BUNDLE)
    assert result["actual_complete_EF_returns"] == 98
    assert result["formal_EF_returns"] == 72
    assert result["not_run"] == 10
    assert result["formal_independent_processes"] == 18
    assert result["host_history"]["verification"] == "NOT_RUN_HISTORICAL_AUDIT"
    assert result["qualified_speedup"] is None


def host_fixture(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "restored"
    reference = tmp_path / "current"
    archive.mkdir()
    reference.mkdir()
    raw = b"HOST fixture only\n"
    digest = hashlib.sha256(raw).hexdigest()
    oracle = lzma.compress(b"synthetic reference bytes, no calculation")
    (reference / "independent-oracles.json.xz").write_bytes(oracle)
    history = {
        "schema": "generativeqc.h100-direct-host-history.v1",
        "objects": {digest: base64.b64encode(raw).decode()},
        "members": [{"path": "fixture.txt", "bytes": len(raw), "sha256": digest}],
        "standalone_original_compressed_files": [
            {
                "path": "independent-oracles.json.xz",
                "bytes": len(oracle),
                "sha256": hashlib.sha256(oracle).hexdigest(),
            }
        ],
    }
    (archive / "host-history.json.xz").write_bytes(
        lzma.compress(json.dumps(history).encode())
    )
    failed = {
        "probe_returncode": 1,
        "verified_control_originals": 27,
        "full_suite_qualification": "UNAVAILABLE synthetic fixture",
        "source_transport_sha256": "f" * 64,
    }
    (archive / "root-host-v2-failed-originals.json.xz").write_bytes(
        lzma.compress(json.dumps(failed).encode())
    )
    return archive, reference


def test_explicit_host_audit_uses_retained_reference_without_weakening_hashes(
    tmp_path: Path,
) -> None:
    archive, reference = host_fixture(tmp_path)
    result = READER["host_history"](archive, reference_root=reference)
    assert result["original_uncompressed_members"] == 1
    assert result["standalone_original_compressed_files"] == 1
    assert result["V2_probe_returncode"] == 1
    assert result["V2_full_suite"].startswith("UNAVAILABLE")


@pytest.mark.parametrize("damage", ["object", "duplicate", "reference", "missing"])
def test_explicit_host_audit_rejects_damaged_recovery(
    tmp_path: Path, damage: str
) -> None:
    archive, reference = host_fixture(tmp_path)
    path = archive / "host-history.json.xz"
    history = json.loads(lzma.decompress(path.read_bytes()))
    if damage == "object":
        key = next(iter(history["objects"]))
        history["objects"][key] = base64.b64encode(b"changed").decode()
    elif damage == "duplicate":
        history["members"].append(history["members"][0])
    elif damage == "reference":
        (reference / "independent-oracles.json.xz").write_bytes(b"changed")
    else:
        (archive / "root-host-v2-failed-originals.json.xz").unlink()
    path.write_bytes(lzma.compress(json.dumps(history).encode()))
    with pytest.raises((ValueError, FileNotFoundError)):
        READER["host_history"](archive, reference_root=reference)


def test_recorded_host_summary_does_not_accept_changed_snapshot(tmp_path: Path) -> None:
    for name in ("historical-host-summary.json", "historical-host-snapshot.json"):
        (tmp_path / name).write_bytes((BUNDLE / name).read_bytes())
    path = tmp_path / "historical-host-snapshot.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="historical HOST summary/recovery binding"):
        READER["recorded_host_history"](tmp_path)

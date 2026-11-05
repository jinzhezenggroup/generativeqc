"""Reject missing or changed observations in the frozen final publication."""

import json
import shutil
from pathlib import Path

import pytest

from benchmarks import h100_db36_final_cohort as evidence


@pytest.fixture
def packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Exercise storage with tiny payloads, without fetching historical science."""
    target = tmp_path / "final"
    shutil.copytree(evidence.DIRECTORY, target)
    monkeypatch.setattr(evidence, "DIRECTORY", target)
    path = target / "cohort.json"
    publication = json.loads(path.read_text())
    for part in publication["parts"]:
        raw = ("synthetic transport: " + part["path"]).encode()
        (target / part["path"]).write_bytes(raw)
        binding = {
            "path": part["path"],
            "bytes": len(raw),
            "sha256": evidence.profile.sha(raw),
        }
        part.update(binding)
        next(b for b in publication["files"] if b["path"] == part["path"]).update(
            binding
        )
    path.write_text(json.dumps(publication))

    def unavailable(path: Path, *, manifest: Path) -> bytes:
        raise ValueError("History transport is disabled in this synthetic test")

    monkeypatch.setattr(evidence, "read_evidence_bytes", unavailable)
    return target


def test_original_contract_and_actual_review_with_synthetic_transport(
    packet: Path,
) -> None:
    publication = evidence.checked_publication()
    assert publication["acceptance"]["returned_EF"] == 720
    assert publication["acceptance"]["physical_cases"] == 30
    assert publication["actual_result_audit"]["review_independence"] == "same-family"
    assert publication["actual_result_audit"]["acceptance_status"] == "provisional"


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "protocol", "part-order"])
def test_final_publication_rejects_changed_originals(
    packet: Path, mutation: str
) -> None:
    target = packet
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


def test_missing_declared_part_uses_canonical_path_and_adjacent_manifest(
    packet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "final-pairs-v1-completed.tar.xz.part00"
    raw = (packet / name).read_bytes()
    (packet / name).unlink()
    calls = []

    def recover(path: Path, *, manifest: Path) -> bytes:
        calls.append((path, manifest))
        return raw

    monkeypatch.setattr(evidence, "read_evidence_bytes", recover)
    evidence.checked_publication()
    assert calls == [
        (
            evidence.safety.ROOT
            / "benchmarks/results/h100-sm90-db36-final-20261103"
            / name,
            packet / "historical-snapshot.json",
        )
    ]


@pytest.mark.parametrize("mutation", ["corrupt", "symlink", "directory"])
def test_present_bad_part_cannot_be_replaced_with_history(
    packet: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path = packet / "final-pairs-v1-completed.tar.xz.part00"
    raw = path.read_bytes()
    path.unlink()
    if mutation == "corrupt":
        path.write_bytes(raw + b"changed")
    elif mutation == "directory":
        path.mkdir()
    else:
        backing = packet / "backing"
        backing.write_bytes(raw)
        path.symlink_to(backing)

    def forbidden(path: Path, *, manifest: Path) -> bytes:
        pytest.fail("Present invalid evidence must not fall back to history")

    monkeypatch.setattr(evidence, "read_evidence_bytes", forbidden)
    with pytest.raises(ValueError):
        evidence.checked_publication()

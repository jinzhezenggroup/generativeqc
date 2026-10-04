"""Preserve exact public force-phase receipts and their active publication bindings."""

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import decode_json, load_json

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "benchmarks/results/cc-force-phases-20261003"
SOURCE = "33700e6ccc43ea8a01c2a5a28661243c2494d838"
EXPECTED = {
    "evidence.json": {
        "original_path": "evidence.json",
        "stored_path": "evidence.original.json.gz",
        "encoding": "gzip",
        "original_bytes": 16892,
        "stored_bytes": 3017,
        "original_git_blob": "0cf7e23cc05b1bc41e86a19791e38cab6534ed01",
        "original_sha256": "bbe118d276ee1e6123610d453e6dc661526da28448b7a89f2aef16e3125236c5",
        "stored_sha256": "04fe8b6cfaccb9f465b2fa86fd9dad249bae691a44641dc854474698b6de0232",
    },
    "records.json": {
        "original_path": "records.json",
        "stored_path": "records.json.gz",
        "encoding": "gzip",
        "original_bytes": 44803,
        "stored_bytes": 7598,
        "original_git_blob": "cc94f145ac46309b7196ed8c738d65f3b606f837",
        "original_sha256": "ebc57d7fcc2c83f1bade524a3f6df1146910caf8dc99e783ab45e0bddd2e3aee",
        "stored_sha256": "a18bfc7affe91efa4c04c136a72682accd6aa71e278d81712d2c773df6a8fe41",
    },
    "source.patch": {
        "original_path": "source.patch",
        "stored_path": "source.patch.gz",
        "encoding": "gzip",
        "original_bytes": 13013,
        "stored_bytes": 3447,
        "original_git_blob": "f3f36cb651d9afe264afaa987665c38a2a9fdcdc",
        "original_sha256": "d256616ea67d057b84bffa451c799020bdb0a9b85a353a704e78c4cbd28bc91a",
        "stored_sha256": "727eb10b5271ad6a4aa2da390a48ebeaee160373bcf6610f58e38ef8e4d8031e",
    },
    "summary.json": {
        "original_path": "summary.json",
        "stored_path": "summary.json.gz",
        "encoding": "gzip",
        "original_bytes": 43808,
        "stored_bytes": 8274,
        "original_git_blob": "d2c147865f6a5e9c3f6427d9de252b7da0cad237",
        "original_sha256": "4514270397287374b83daeb9db9a0d6b3ed99c8e5eb4d882950871fb8ca4910e",
        "stored_sha256": "63bcbc51ae6837aa86be28d09c4a9772e2d469ed88e6a7ef9f4e12a98474d37f",
    },
}
ORIGINAL_PUBLICATION = {
    "archives": [],
    "decision": {
        "reason": "All twelve completed CPU/CUDA endpoint calls pass independent energy/triples/force/residual gates and trace work invariants. This is a diagnosis; no speedup is claimed.",
        "scope": "numerical",
        "status": "accepted",
    },
    "files": [
        {
            "bytes": 16892,
            "path": "evidence.json",
            "role": "evidence",
            "sha256": "bbe118d276ee1e6123610d453e6dc661526da28448b7a89f2aef16e3125236c5",
        },
        {
            "bytes": 43808,
            "path": "summary.json",
            "role": "summary",
            "sha256": "4514270397287374b83daeb9db9a0d6b3ed99c8e5eb4d882950871fb8ca4910e",
        },
        {
            "bytes": 44803,
            "path": "records.json",
            "role": "samples",
            "sha256": "ebc57d7fcc2c83f1bade524a3f6df1146910caf8dc99e783ab45e0bddd2e3aee",
        },
        {
            "bytes": 13013,
            "path": "source.patch",
            "role": "source-patch",
            "sha256": "d256616ea67d057b84bffa451c799020bdb0a9b85a353a704e78c4cbd28bc91a",
        },
    ],
    "reproduction": {
        "command": [
            "python",
            "benchmarks/ccsdt_prepared_endpoint.py",
            "--device",
            "cuda",
            "--atoms",
            "12",
            "--forces",
            "--output",
            ".artifacts/cc-force-phases-reproduction.json",
        ],
        "instructions": "Apply source.patch to revision; compile CUDA 12.9 sm_120 Release with explicit CXX/CUDA ccache launchers and checkout-root CCACHE_BASEDIR. Run via finite Slurm main allocation --gres=gpu:5090:1, preserve CUDA_VISIBLE_DEVICES, set GENERATIVEQC_LIBRARY/PYTHONPATH=python:., CPU thread limits=1, and GENERATIVEQC_DF_PROGRESS_TRACE to a fresh path. Also run --atoms 6 on CPU/CUDA. Traces are nested; use phase_interpretation from summary.json.",
    },
    "schema": "generativeqc.benchmark-publication.v1",
    "source": {"dirty": True, "revision": "2d2b649516886671b09f3de4cf8e217c37d199da"},
}


@pytest.mark.parametrize("name", EXPECTED)
def test_force_phase_receipt_preserves_original_bytes_and_readers(name: str) -> None:
    manifest = load_json(FAMILY / "storage.json")
    assert manifest["schema"] == "generativeqc.cc-evidence-storage.v1"
    assert manifest["storage_source_revision"] == SOURCE
    assert {r["original_path"] for r in manifest["records"]} == set(EXPECTED)
    assert len(manifest["records"]) == len(EXPECTED)
    row = next(r for r in manifest["records"] if r["original_path"] == name)
    assert row == EXPECTED[name]
    assert not (FAMILY / name).exists()
    path = FAMILY / row["stored_path"]
    packed = path.read_bytes()
    raw = gzip.decompress(packed)
    assert len(packed) == row["stored_bytes"]
    assert len(raw) == row["original_bytes"]
    assert hashlib.sha256(packed).hexdigest() == row["stored_sha256"]
    assert hashlib.sha256(raw).hexdigest() == row["original_sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["original_git_blob"]
    )
    assert packed == gzip.compress(raw, compresslevel=9, mtime=0)
    with gzip.open(path, "rb") as stream:
        assert stream.read() == raw
    if name.endswith(".json"):
        expected = json.loads(raw)
        assert load_json(path) == expected
        assert decode_json(packed) == expected
        assert decode_json(packed, path=path) == expected
    else:
        assert raw.startswith(b"diff --git ")


def test_active_envelope_only_rebinds_retained_attachments() -> None:
    original = load_json(FAMILY / "evidence.original.json.gz")
    expected = copy.deepcopy(original)
    for attachment in expected["attachments"]:
        row = EXPECTED.get(attachment["path"])
        if row is not None:
            attachment.update(path=row["stored_path"], sha256=row["stored_sha256"])
    assert load_json(FAMILY / "evidence.json.gz") == expected
    publication = load_json(FAMILY / "publication.json")
    files = {e["path"]: (FAMILY / e["path"]).read_bytes() for e in publication["files"]}
    validate_publication(publication, files)
    assert set(files) | {"publication.json"} == {p.name for p in FAMILY.iterdir()}
    for key in set(ORIGINAL_PUBLICATION) - {"files", "reproduction"}:
        assert publication[key] == ORIGINAL_PUBLICATION[key]
    assert (
        publication["reproduction"]["command"]
        == ORIGINAL_PUBLICATION["reproduction"]["command"]
    )
    assert publication["reproduction"]["instructions"] == (
        "Decode source.patch.gz byte-for-byte before applying the unchanged patch. "
        + ORIGINAL_PUBLICATION["reproduction"]["instructions"]
    )
    expected_publication = copy.deepcopy(ORIGINAL_PUBLICATION)
    for entry in expected_publication["files"]:
        if entry["role"] == "evidence":
            entry["path"] = "evidence.json.gz"
        elif entry["path"] in EXPECTED:
            entry["path"] = EXPECTED[entry["path"]]["stored_path"]
        raw = files[entry["path"]]
        entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    for name in ["evidence.original.json.gz", "storage.json"]:
        raw = files[name]
        expected_publication["files"].append(
            {
                "path": name,
                "role": "reproduction",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    assert publication["files"] == expected_publication["files"]

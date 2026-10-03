"""Preserve exact public orbit receipts and their active publication bindings."""

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import decode_json, load_json

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "benchmarks/results/cc-eri-orbits-20261003"
SOURCE = "913106492d0b39d11cc7a2363fb99dbcce3e7191"
EXPECTED = {
    "evidence.json": {
        "original_path": "evidence.json",
        "stored_path": "evidence.original.json.gz",
        "encoding": "gzip",
        "original_bytes": 20189,
        "stored_bytes": 2818,
        "original_git_blob": "b5d9b1f257b591829d41a439666a49fdcb6a3185",
        "original_sha256": "553a896e9fa3b29584cdead11ae5e41ba152d37aaf441864274b543a307ea590",
        "stored_sha256": "f75387c5746e036331c79e763d2ab8b5f13d459d7b17ad4e5ac2cc49b3d674f9",
    },
    "records.json": {
        "original_path": "records.json",
        "stored_path": "records.json.gz",
        "encoding": "gzip",
        "original_bytes": 51433,
        "stored_bytes": 8209,
        "original_git_blob": "6b2a87004d6b9b6f369ec76bace12b79a006984e",
        "original_sha256": "4bb509080f9fe06f812448e4cbfa7f0a535f53db314acd083dc9aa010e174a1e",
        "stored_sha256": "e31499a6b4b4049a561a7557a9d9ae2d711f4f5432452e05a992a04ec6ea686b",
    },
    "source.patch": {
        "original_path": "source.patch",
        "stored_path": "source.patch.gz",
        "encoding": "gzip",
        "original_bytes": 13233,
        "stored_bytes": 4420,
        "original_git_blob": "1df0c46b8233b56d1f68963e1d1d642d8b41cfb0",
        "original_sha256": "79770644bb524c89c631214267b27c6004ebffdfc3e02d549e1b0ea297778234",
        "stored_sha256": "e04c20d4003d549de2fb723d8d6a3ea3cffea3279c78f99eecc6fa194a725513",
    },
    "summary.json": {
        "original_path": "summary.json",
        "stored_path": "summary.json.gz",
        "encoding": "gzip",
        "original_bytes": 18455,
        "stored_bytes": 4317,
        "original_git_blob": "e334a802f3b794c9e6a2ecc9d3f9918d9ef72299",
        "original_sha256": "caa87f12f0bea4853f32bdd3dd11db0c13a2f54c54305b29522001f2337922fd",
        "stored_sha256": "88fef2efd0840d8ad0f5b6cf33d46c538ec3f9ee42b07971c4b9b4c3bfa89b24",
    },
}
ORIGINAL_PUBLICATION = {
    "archives": [],
    "decision": {
        "reason": "All retained calls satisfy independent energy/triples/force gates, work and resource invariants, and memcheck. Timing ratios are observational; no formal interleaved performance promotion is asserted.",
        "scope": "numerical",
        "status": "accepted",
    },
    "files": [
        {
            "bytes": 20189,
            "path": "evidence.json",
            "role": "evidence",
            "sha256": "553a896e9fa3b29584cdead11ae5e41ba152d37aaf441864274b543a307ea590",
        },
        {
            "bytes": 18455,
            "path": "summary.json",
            "role": "summary",
            "sha256": "caa87f12f0bea4853f32bdd3dd11db0c13a2f54c54305b29522001f2337922fd",
        },
        {
            "bytes": 51433,
            "path": "records.json",
            "role": "samples",
            "sha256": "4bb509080f9fe06f812448e4cbfa7f0a535f53db314acd083dc9aa010e174a1e",
        },
        {
            "bytes": 13233,
            "path": "source.patch",
            "role": "source-patch",
            "sha256": "79770644bb524c89c631214267b27c6004ebffdfc3e02d549e1b0ea297778234",
        },
    ],
    "reproduction": {
        "command": [
            "python",
            "benchmarks/ccsdt_prepared_endpoint.py",
            "--device",
            "cuda",
            "--atoms",
            "24",
            "--output",
            ".artifacts/cc-eri-orbits-reproduction.json",
        ],
        "instructions": "Apply source.patch to revision; build CUDA 12.9 sm_120 Release with explicit CXX/CUDA ccache launchers and checkout-root CCACHE_BASEDIR. Run through finite main-partition Slurm allocation preserving CUDA_VISIBLE_DEVICES. Set GENERATIVEQC_LIBRARY/PYTHONPATH=python:., all CPU thread limits=1. For 28 AOs use --atoms 12, optionally --forces. New GPU tests use the final code commit noted in summary.",
    },
    "schema": "generativeqc.benchmark-publication.v1",
    "source": {"dirty": True, "revision": "6fe3eb7daffb641e7efabe6ca8200e25ea6bffc9"},
}


@pytest.mark.parametrize("name", EXPECTED)
def test_orbit_receipt_preserves_original_bytes_and_readers(name: str) -> None:
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

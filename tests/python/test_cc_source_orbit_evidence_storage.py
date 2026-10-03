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
FAMILY = ROOT / "benchmarks/results/cc-source-orbits-20261003"
SOURCE = "2d2b649516886671b09f3de4cf8e217c37d199da"
EXPECTED = {
    "evidence.json": {
        "original_path": "evidence.json",
        "stored_path": "evidence.original.json.gz",
        "encoding": "gzip",
        "original_bytes": 24141,
        "stored_bytes": 3570,
        "original_git_blob": "72eae7b828439881faa944f03758c0fd73e5c407",
        "original_sha256": "b6f4da76cf9ccc8b393dc9533487fa6e4711404d3c6f5a004797c5f00f386237",
        "stored_sha256": "d9cccb685c022f2fd0abe9186015c20967e0a39d9b9aed775b60c2835dfeffc6",
    },
    "records.json": {
        "original_path": "records.json",
        "stored_path": "records.json.gz",
        "encoding": "gzip",
        "original_bytes": 51447,
        "stored_bytes": 8186,
        "original_git_blob": "243d1b849d09938d6f9d46a5a0e6e0fa136215d6",
        "original_sha256": "7a4d8570e295c6e1316ebf8ee702edd8ec1f8f43fa0e5bc16cf4a91b180030d8",
        "stored_sha256": "9a94e0d23f226147df6cede49688ec7a08c58e879286c16e62c3c4516282f531",
    },
    "source.patch": {
        "original_path": "source.patch",
        "stored_path": "source.patch.gz",
        "encoding": "gzip",
        "original_bytes": 18481,
        "stored_bytes": 5869,
        "original_git_blob": "2dd3e9181c7f80a1ed7b4bb24a49b90a12d3a074",
        "original_sha256": "419b5d859630919f2487f1a4a30e5899571eff733dec4be984aa770ce4fdf513",
        "stored_sha256": "3f160e3a790f383216b329ef7494f2a1376fb169a7986c3f299c596a316556a3",
    },
    "summary.json": {
        "original_path": "summary.json",
        "stored_path": "summary.json.gz",
        "encoding": "gzip",
        "original_bytes": 22610,
        "stored_bytes": 4488,
        "original_git_blob": "9817723446ffeb2e6ab1b0d3d82a1c372ca4aa71",
        "original_sha256": "0755cd8c3aa1f50f037509b5444a121c6ebbbfa2b6b5d1ef133d9c114822d645",
        "stored_sha256": "9a98e2e67fcfabe0a9c9dee7b2e210dd02108ab3370b19a2917d4f8ec578fc94",
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
            "bytes": 24141,
            "path": "evidence.json",
            "role": "evidence",
            "sha256": "b6f4da76cf9ccc8b393dc9533487fa6e4711404d3c6f5a004797c5f00f386237",
        },
        {
            "bytes": 22610,
            "path": "summary.json",
            "role": "summary",
            "sha256": "0755cd8c3aa1f50f037509b5444a121c6ebbbfa2b6b5d1ef133d9c114822d645",
        },
        {
            "bytes": 51447,
            "path": "records.json",
            "role": "samples",
            "sha256": "7a4d8570e295c6e1316ebf8ee702edd8ec1f8f43fa0e5bc16cf4a91b180030d8",
        },
        {
            "bytes": 18481,
            "path": "source.patch",
            "role": "source-patch",
            "sha256": "419b5d859630919f2487f1a4a30e5899571eff733dec4be984aa770ce4fdf513",
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
            ".artifacts/cc-source-orbits-reproduction.json",
        ],
        "instructions": "Apply source.patch to revision; build CUDA 12.9 sm_120 Release with explicit CXX/CUDA ccache launchers and checkout-root CCACHE_BASEDIR. Run through finite main-partition Slurm allocation preserving CUDA_VISIBLE_DEVICES. Set GENERATIVEQC_LIBRARY/PYTHONPATH=python:., all CPU thread limits=1. For 28 AOs use --atoms 12, optionally --forces. Use fock-provider-tests --eri-tiles-only for the focused native cases, including memcheck.",
    },
    "schema": "generativeqc.benchmark-publication.v1",
    "source": {"dirty": True, "revision": "913106492d0b39d11cc7a2363fb99dbcce3e7191"},
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

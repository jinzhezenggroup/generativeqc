"""Preserve the two approved RHF-reference receipts and existing JSON readers."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tools.generativeqc_validation.record import decode_json, load_json

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "benchmarks/results/cc-rhf-resident-20261003"
SOURCE = "6fe3eb7daffb641e7efabe6ca8200e25ea6bffc9"
EXPECTED = {
    "records.json": (
        88641,
        12318,
        "5b2352eb1a1fdfa4454b11ca1ec65704fa04cfad",
        "46118981281f997d1e69dcc900548a31cfd43312b2e76befc5ea31b32cad6839",
        "7cd09061364016fcdb44d08b1f9c685325819d920e04b6d74f73a6f9eaef3069",
    ),
    "summary.json": (
        19347,
        4754,
        "f73388e6b83f55498f8fb7438de89b6012161254",
        "73b627dbfcf444526082ac63198df19ebd61cb87c4e62aec6a9c787ff032d6ee",
        "a574feadc422b8f03e03b75cbac8ab3aacec6f9ea36988b39fa7e256e847816f",
    ),
}


@pytest.mark.parametrize("name", EXPECTED)
def test_rhf_receipt_preserves_original_bytes_and_readers(name: str) -> None:
    manifest = json.loads((FAMILY / "storage.json").read_text())
    assert manifest["schema"] == "generativeqc.cc-evidence-storage.v1"
    assert manifest["storage_source_revision"] == SOURCE
    rows = manifest["records"]
    assert len(rows) == len(EXPECTED)
    assert {row["original_path"] for row in rows} == set(EXPECTED)
    row = next(row for row in rows if row["original_path"] == name)
    original_bytes, stored_bytes, git_blob, original_sha, stored_sha = EXPECTED[name]
    assert row["stored_path"] == name + ".gz"
    assert row["encoding"] == "gzip"
    assert not (FAMILY / name).exists()
    path = FAMILY / row["stored_path"]
    packed = path.read_bytes()
    raw = gzip.decompress(packed)
    assert len(packed) == row["stored_bytes"] == stored_bytes
    assert len(raw) == row["original_bytes"] == original_bytes
    assert hashlib.sha256(packed).hexdigest() == row["stored_sha256"] == stored_sha
    assert hashlib.sha256(raw).hexdigest() == row["original_sha256"] == original_sha
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    assert blob == row["original_git_blob"] == git_blob
    with gzip.open(path, "rb") as stream:
        assert stream.read() == raw
    expected = json.loads(raw)
    assert load_json(path) == expected
    assert decode_json(packed) == expected
    assert decode_json(packed, path=path) == expected

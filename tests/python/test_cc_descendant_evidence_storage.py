"""Check CC publication storage without duplicating historical hash fixtures."""

import copy
import gzip
import hashlib
from pathlib import Path

import pytest

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import (
    decode_json,
    load_json,
    load_publication_record,
    load_record,
)

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = (
    "cc-derivative-batches-20261003",
    "cc-triples-response-cuda-20261003",
    "cc-lambda-preconditioner-20261003",
    "cc-triples-fock-response-20261003",
    "cc-curvature-eigensolver-20261003",
)


@pytest.mark.parametrize("family", FAMILIES)
def test_cc_descendant_publication_readers(family: str) -> None:
    directory = ROOT / "benchmarks/results" / family
    publication = load_json(directory / "publication.json")
    files = {p.name: p.read_bytes() for p in directory.iterdir()}
    files.pop("publication.json")
    validate_publication(publication, files)
    assert {entry["path"] for entry in publication["files"]} == set(files)
    for role, name in (("evidence", "evidence.json"), ("samples", "records.json")):
        path = next(e["path"] for e in publication["files"] if e["role"] == role)
        direct = load_record(directory / path)
        assert load_publication_record(directory, role=role) == direct
        assert load_publication_record(directory, role=role, name=name) == direct


@pytest.mark.parametrize("family", FAMILIES[:3])
def test_cc_descendant_original_receipts_and_attachment_only_rebinding(
    family: str,
) -> None:
    directory = ROOT / "benchmarks/results" / family
    storage = load_json(directory / "storage.json")
    assert storage["schema"] == "generativeqc.cc-evidence-storage.v1"
    rows = {r["original_path"]: r for r in storage["records"]}
    assert len(rows) == len(storage["records"])
    for name, row in rows.items():
        assert row["encoding"] == "gzip"
        assert not (directory / name).exists()
        packed = (directory / row["stored_path"]).read_bytes()
        original = gzip.decompress(packed)
        assert len(original) == row["original_bytes"]
        assert len(packed) == row["stored_bytes"]
        assert hashlib.sha256(original).hexdigest() == row["original_sha256"]
        assert hashlib.sha256(packed).hexdigest() == row["stored_sha256"]
        assert (
            hashlib.sha1(f"blob {len(original)}\0".encode() + original).hexdigest()
            == row["original_git_blob"]
        )
        with gzip.open(directory / row["stored_path"], "rb") as stream:
            assert stream.read() == original
        if name.endswith(".json"):
            assert load_json(directory / row["stored_path"]) == decode_json(original)
            assert decode_json(packed) == decode_json(original)
        else:
            assert original.startswith(b"diff --git ")
    expected = copy.deepcopy(load_json(directory / "evidence.original.json.gz"))
    for attachment in expected["attachments"]:
        row = rows.get(attachment["path"])
        if row:
            attachment.update(path=row["stored_path"], sha256=row["stored_sha256"])
            if "bytes" in attachment:
                attachment["bytes"] = row["stored_bytes"]
    assert load_publication_record(directory) == expected

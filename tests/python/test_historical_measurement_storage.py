"""Numerical tokens and bounded offline access survive historical consolidation."""

import gzip
import hashlib
import json
import lzma
from pathlib import Path

import pytest

from tools.generativeqc_validation.json_storage import audit_json, compact_json
from tools.inspect_retained_benchmarks import SCHEMA, load_measurements


def collection(directory: Path, payload: bytes) -> dict:
    """Bind synthetic records without requiring any historical commit or GPU."""
    data = lzma.compress(payload)
    (directory / "measurements.json.xz").write_bytes(data)
    manifest = {
        "schema": SCHEMA,
        "record_count": 1,
        "storage": {
            "path": "measurements.json.xz",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    }
    (directory / "measurements.storage.json").write_text(json.dumps(manifest))
    return manifest


def test_whitespace_compaction_preserves_strings_numbers_and_order() -> None:
    original = rb""" { "samples": [ -0.0, 1.2345678901234567890123456789, 1e-400, 9007199254740993 ],
      "message": "spaces  and \\\"quoted\\\" text\n\u03bc" } """
    expected = rb"""{"samples":[-0.0,1.2345678901234567890123456789,1e-400,9007199254740993],"message":"spaces  and \\\"quoted\\\" text\n\u03bc"}"""
    assert compact_json(original) == expected
    assert compact_json(expected) == expected


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":'])
def test_compaction_rejects_ambiguous_or_malformed_data(data: bytes) -> None:
    with pytest.raises(ValueError):
        compact_json(data)


def test_collection_extracts_original_numeric_tokens(tmp_path: Path) -> None:
    sample = b'{"seconds":[-0.0,1e-400,1.234567890123456789],"failed":true}'
    collection(tmp_path, b' \n { "pairs/run.json": ' + sample + b" }\n")
    assert load_measurements(tmp_path) == {"pairs/run.json": sample}


@pytest.mark.parametrize(
    "damage", ["hash", "count", "path", "member", "trailing", "duplicate"]
)
def test_invalid_collection_rejected(tmp_path: Path, damage: str) -> None:
    payload = b'{"run.json":{"energy":-1.0}}'
    if damage == "member":
        payload = b'{"../run.json":{}}'
    elif damage == "duplicate":
        payload = b'{"run.json":{},"run.json":{}}'
    manifest = collection(tmp_path, payload)
    if damage == "hash":
        manifest["storage"]["sha256"] = "0" * 64
    elif damage == "count":
        manifest["record_count"] = 2
    elif damage == "path":
        manifest["storage"]["path"] = "../measurements.json.xz"
    elif damage == "trailing":
        path = tmp_path / "measurements.json.xz"
        data = path.read_bytes() + lzma.compress(b"{}")
        path.write_bytes(data)
        manifest["storage"].update(
            bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )
    (tmp_path / "measurements.storage.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_measurements(tmp_path)


def test_format_audit_counts_stored_and_expanded_bytes_separately() -> None:
    plain = b'{ "samples": [ 1, 2, 3 ] }\n'
    prefix = "benchmarks/results/example/"
    blobs = {
        prefix + "plain.json": plain,
        prefix + "gzip.json.gz": gzip.compress(plain),
        prefix + "xz.json.xz": lzma.compress(plain),
    }
    report = audit_json({**blobs, "tests/reference_data/keep.json": plain})
    assert not report["errors"]
    assert report["totals"]["files"] == 3
    assert report["totals"]["stored_bytes"] == sum(map(len, blobs.values()))
    assert report["totals"]["expanded_bytes"] == 3 * len(plain)
    assert report["totals"]["compact_bytes"] == 3 * len(b'{"samples":[1,2,3]}')
    damaged = audit_json({prefix + "truncated.json.xz": lzma.compress(plain)[:-5]})
    assert damaged["errors"] and not damaged["totals"]


@pytest.mark.parametrize("damage", ["header", "crc"])
def test_gzip_error_preserves_valid_file_audit(damage: str) -> None:
    plain = b'{ "energy": -1.0 }\n'
    damaged = bytearray(gzip.compress(plain, mtime=0))
    if damage == "header":
        damaged[0] ^= 1
    else:
        damaged[-8] ^= 1
    prefix = "benchmarks/results/example/"
    good = {prefix + "good.json": plain}
    expected = audit_json(good)
    result = audit_json({prefix + "bad.json.gz": bytes(damaged), **good})
    assert result["totals"] == expected["totals"]
    assert result["formats"] == expected["formats"]
    assert len(result["errors"]) == 1
    assert result["errors"][0]["path"] == prefix + "bad.json.gz"
    assert result["errors"][0]["error"]

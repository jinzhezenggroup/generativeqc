"""Partitioned storage must preserve scientific values and reject corrupt parts."""

import gzip
import json
import typing
from pathlib import Path

import pytest

from tools.generativeqc_validation.record import (
    decode_json,
    decode_record,
    load_json,
    load_publication_record,
    load_record,
)
from tools.generativeqc_validation.retention import digest


@pytest.mark.parametrize("reader", [load_json, load_record])
def test_weighted_eri_gzip_preserves_original_endpoint_bytes(
    reader: typing.Any,
) -> None:
    """Canonical readers retain every byte and value of the historical record."""
    root = Path(__file__).resolve().parents[2]
    path = root / "benchmarks/results/weighted-eri-144/endpoints.json.gz"
    stored = path.read_bytes()
    assert len(stored) == 16945
    assert digest(stored) == (
        "8311ebc1a0dc3fba3d2991a8cf96cbd218725851593c819774e446abec65a882"
    )
    raw = gzip.decompress(stored)
    assert len(raw) == 177463
    assert digest(raw) == (
        "ccbf6dd63600d111015879194c1bae05cef12bad5e874cf8570af54dfebacdd7"
    )
    assert reader(path) == json.loads(raw)


def test_migrated_records_match_original_scientific_values() -> None:
    """Compare all reconstructed values with hashes taken before migration."""
    root = Path(__file__).resolve().parents[2]
    audit = json.loads(
        (root / "benchmarks/results/retention-size-limit/migration.json").read_text()
    )
    for entry in audit["records"]:
        path = root / entry["path"]
        inventory = path.parent / "publication.json"
        if inventory.exists():
            publication = json.loads(inventory.read_bytes())
            role = next(
                member["role"]
                for member in publication["files"]
                if member["path"] in {path.name, path.name + ".gz"}
            )
            restored = load_publication_record(path.parent, role=role, name=path.name)
        else:
            restored = load_record(path)
        canonical = json.dumps(
            restored, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        assert digest(canonical) == entry["canonical_sha256"]


def record(part: typing.Any) -> typing.Any:
    return {
        "schema": "example",
        "record_parts": {
            "timings": [
                {"path": "water.json", "bytes": len(part), "sha256": digest(part)}
            ]
        },
    }


def test_parts_restore_order_and_float_precision(tmp_path: typing.Any) -> None:
    values = [{"seconds": 0.12345678901234567}, {"seconds": 1e-14}]
    part = json.dumps(values).encode()
    main = json.dumps(record(part)).encode()
    assert decode_record(main, {"water.json": part}) == {
        "schema": "example",
        "timings": values,
    }
    (tmp_path / "evidence.json").write_bytes(main)
    (tmp_path / "water.json").write_bytes(part)
    assert load_record(tmp_path / "evidence.json")["timings"] == values


@pytest.mark.parametrize(
    "damage", ["bytes", "missing", "overwrite", "duplicate", "traversal", "type"]
)
def test_corrupt_parts_fail(damage: typing.Any) -> None:
    part = b"[1.25, 0.5]"
    main = record(part)
    files = {"water.json": part}
    entries = main["record_parts"]["timings"]
    if damage == "bytes":
        files["water.json"] += b" "
    elif damage == "missing":
        files.clear()
    elif damage == "overwrite":
        main["timings"] = []
    elif damage == "duplicate":
        entries *= 2
    elif damage == "traversal":
        entries[0]["path"] = "../water.json"
    else:
        part = b"{}"
        files["water.json"] = part
        entries[0].update(bytes=len(part), sha256=digest(part))
    with pytest.raises((ValueError, KeyError, TypeError)):
        decode_record(json.dumps(main).encode(), files)


def test_part_symlink_cannot_escape_directory(tmp_path: typing.Any) -> None:
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"[]")
    directory = tmp_path / "publication"
    directory.mkdir()
    (directory / "water.json").symlink_to(outside)
    (directory / "evidence.json").write_text(json.dumps(record(b"[]")))
    with pytest.raises(ValueError, match="escapes"):
        load_record(directory / "evidence.json")


def test_gzip_json_and_record_parts_restore_exact_values(tmp_path: typing.Any) -> None:
    values = [{"seconds": 0.12345678901234567}, {"seconds": 1e-14}]
    plain_part = json.dumps(values).encode()
    compressed_part = gzip.compress(plain_part, mtime=0)
    main = {
        "schema": "example",
        "record_parts": {
            "timings": [
                {
                    "path": "water.json.gz",
                    "bytes": len(compressed_part),
                    "sha256": digest(compressed_part),
                }
            ]
        },
    }
    compressed_main = gzip.compress(json.dumps(main).encode(), mtime=0)
    assert decode_json(compressed_main, path="evidence.json.gz") == main
    assert decode_record(
        compressed_main,
        {"water.json.gz": compressed_part},
        path="evidence.json.gz",
    ) == {"schema": "example", "timings": values}
    (tmp_path / "evidence.json.gz").write_bytes(compressed_main)
    (tmp_path / "water.json.gz").write_bytes(compressed_part)
    assert load_json(tmp_path / "evidence.json.gz") == main
    assert load_record(tmp_path / "evidence.json.gz")["timings"] == values

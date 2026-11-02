"""Exercise exact numerical/text recovery and fail-closed evidence decoding."""

import hashlib
import json
import lzma
import struct
from pathlib import Path

import pytest

from benchmarks._exact_fp64_evidence import load


def write_bundle(root: Path, bundle: dict, words: list[int]) -> None:
    """Construct a tiny independent byte-plane fixture with stream digests."""
    raw = b"".join(word.to_bytes(8, "little") for word in words)
    planes = b"".join(raw[lane::8] for lane in range(8))
    streams = {"receipts.json.xz": json.dumps(bundle).encode(), "fp64.xz": planes}
    digests = []
    for name, data in streams.items():
        packed = lzma.compress(data)
        (root / name).write_bytes(packed)
        digests.append(hashlib.sha256(packed).hexdigest() + "  " + name + "\n")
    (root / "sha256.txt").write_text("".join(digests))


@pytest.mark.parametrize("version", [1, 2, 3])
def test_exact_fp64_values_and_historical_versions(
    tmp_path: Path, version: int
) -> None:
    values = [-0.0, 5e-324, 1.2345678901234567, -1.0e300]
    text = json.dumps(values, indent=2) + "\n"
    row = {
        "path": "original.json",
        "anchor": "sample",
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "json": [{"$fp64": i if version == 1 else None} for i in range(len(values))],
    }
    schema = (
        "generativeqc.bounded-projection-evidence.v1"
        if version == 1
        else f"generativeqc.exact-fp64-evidence.v{version}"
    )
    write_bundle(
        tmp_path,
        {"schema": schema, "float_count": len(values), "files": [row]},
        [struct.unpack("<Q", struct.pack("<d", value))[0] for value in values],
    )
    _, files = load(tmp_path)
    assert files["original.json"] == text
    assert struct.pack("<4d", *json.loads(files["original.json"])) == struct.pack(
        "<4d", *values
    )


@pytest.mark.parametrize("formatting", ["compact", "compact-lf", "spaced"])
def test_nested_text_is_restored_before_original_hash(
    tmp_path: Path, formatting: str
) -> None:
    text = '{"failed": true, "reason": "retained original"}\n'
    outer = json.dumps({"receipt": text}, separators=(",", ":"))
    if formatting == "spaced":
        outer = json.dumps({"receipt": text})
    if formatting == "compact-lf":
        outer += "\n"
    bundle = {
        "schema": "generativeqc.exact-fp64-evidence.v3",
        "float_count": 0,
        "files": [
            {
                "path": "inner",
                "text": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            },
            {
                "path": "outer",
                "anchor": "container",
                "json_format": formatting,
                "json": {"receipt": {"$text": "inner"}},
                "sha256": hashlib.sha256(outer.encode()).hexdigest(),
            },
        ],
    }
    write_bundle(tmp_path, bundle, [])
    assert load(tmp_path)[1] == {"inner": text, "outer": outer}
    bundle["files"].append(
        {
            "path": "duplicate-observation",
            "text_ref": "inner",
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    )
    write_bundle(tmp_path, bundle, [])
    assert load(tmp_path)[1]["duplicate-observation"] == text
    # Retain the outer digest: even an independently rehashed inner replacement
    # must fail authentication of the historical container that embeds it.
    bundle["files"][0].update(
        text="changed", sha256=hashlib.sha256(b"changed").hexdigest()
    )
    write_bundle(tmp_path, bundle, [])
    with pytest.raises(ValueError, match="outer"):
        load(tmp_path)


@pytest.mark.parametrize("reference", ["outer", "absent", "../external"])
def test_text_references_cannot_recurse_or_read_external_files(
    tmp_path: Path, reference: str
) -> None:
    write_bundle(
        tmp_path,
        {
            "schema": "generativeqc.exact-fp64-evidence.v3",
            "float_count": 0,
            "files": [
                {
                    "path": "outer",
                    "anchor": "x",
                    "json": {"$text": reference},
                    "sha256": "",
                }
            ],
        },
        [],
    )
    with pytest.raises(ValueError, match="earlier authenticated"):
        load(tmp_path)


def test_missing_stream_digest_is_rejected(tmp_path: Path) -> None:
    write_bundle(
        tmp_path,
        {
            "schema": "generativeqc.exact-fp64-evidence.v3",
            "float_count": 0,
            "files": [],
        },
        [],
    )
    (tmp_path / "sha256.txt").write_text("")
    with pytest.raises(ValueError, match="missing stream digest"):
        load(tmp_path)

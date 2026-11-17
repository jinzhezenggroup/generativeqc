"""Authenticate compressed originals and independent exact-FP64 bundles."""

import gzip
import hashlib
import json
import lzma
import struct
from pathlib import Path

import pytest

from benchmarks._native_dft_force_evidence import load
from tools.generativeqc_validation.record import load_json
from tools.restore_retained_evidence import read_stored_json_bytes

ROOT = Path(__file__).resolve().parents[2]


def test_every_original_byte_remains_available_offline() -> None:
    """Transport compaction preserves observations, failures and input hashes."""
    index = json.loads(
        (ROOT / "benchmarks/results/evidence-storage-20261115/index.json").read_text()
    )
    for row in index["files"]:
        payload = (ROOT / row["payload"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["payload_sha256"]
        if row["storage"] == "gzip":
            raw = gzip.decompress(payload)
        elif row["storage"] == "xz":
            raw = lzma.decompress(payload)
        else:
            raw = read_stored_json_bytes(ROOT / row["path"])
        assert len(raw) == row["original_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["original_sha256"]
        source = row["payload"] if row["storage"] in {"gzip", "xz"} else row["path"]
        assert load_json(ROOT / source) == json.loads(raw)


def write_member(folder: Path, logical: str) -> str:
    """Create a v3 bundle with one exactly represented FP64 observation."""
    folder.mkdir()
    text = '{"value":1.25}'
    metadata = {
        "schema": "generativeqc.exact-fp64-evidence.v3",
        "float_count": 1,
        "files": [
            {
                "path": logical,
                "anchor": "sample",
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "json_format": "compact",
                "json": {"value": {"$fp64": None}},
            }
        ],
    }
    streams = {
        "receipts.json.xz": lzma.compress(json.dumps(metadata).encode()),
        "fp64.xz": lzma.compress(struct.pack("<d", 1.25)),
    }
    manifest = ""
    for name, data in streams.items():
        (folder / name).write_bytes(data)
        manifest += hashlib.sha256(data).hexdigest() + "  " + name + "\n"
    (folder / "sha256.txt").write_text(manifest)
    return hashlib.sha256(manifest.encode()).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    (
        None,
        "duplicate",
        "manifest",
        "escape",
        "link",
        "manifest-link",
        "receipts-link",
        "fp64-link",
        "index-link",
        "oversized-stream",
        "too-many-members",
    ),
)
def test_independent_bundles_fail_closed(tmp_path: Path, mutation: str | None) -> None:
    first = write_member(tmp_path / "first", "first.json")
    second = write_member(
        tmp_path / "second", "first.json" if mutation == "duplicate" else "second.json"
    )
    members = {"first": first, "second": second}
    if mutation == "manifest":
        members["second"] = "0" * 64
    elif mutation == "escape":
        members["../second"] = members.pop("second")
    elif mutation == "link":
        (tmp_path / "linked").symlink_to(tmp_path / "second", target_is_directory=True)
        members["linked"] = members.pop("second")
    elif mutation in {"manifest-link", "receipts-link", "fp64-link"}:
        filename = {
            "manifest-link": "sha256.txt",
            "receipts-link": "receipts.json.xz",
            "fp64-link": "fp64.xz",
        }[mutation]
        stream = tmp_path / "second" / filename
        external = tmp_path / filename
        stream.rename(external)
        stream.symlink_to(external)
    elif mutation == "oversized-stream":
        with (tmp_path / "second/fp64.xz").open("r+b") as stream:
            stream.truncate((1 << 20) + 1)
    elif mutation == "too-many-members":
        members.update({f"extra-{index}": first for index in range(63)})
    (tmp_path / "bundles.json").write_text(
        json.dumps(
            {
                "schema": "generativeqc.native-dft-force-evidence-set.v1",
                "bundles": members,
            }
        )
    )
    if mutation == "index-link":
        (tmp_path / "bundles.json").rename(tmp_path / "external-index.json")
        (tmp_path / "bundles.json").symlink_to(tmp_path / "external-index.json")
    if mutation is None:
        assert load(tmp_path)[1] == {
            "first.json": '{"value":1.25}',
            "second.json": '{"value":1.25}',
        }
    else:
        with pytest.raises(ValueError):
            load(tmp_path)

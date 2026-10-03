"""Keep the exact historical CG10 records and manifest through compression."""

import gzip
import hashlib
import json
from pathlib import Path

from tools.generativeqc_validation.record import load_json

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/posthf-147"
SOURCE_REVISION = "9041026afcc60209e26f8cd4fce1cb02c0c55cfb"
ORIGINAL_SHA256 = "6f7ad361e827ce83df319a0a21f957dd7d9d277571e4d5212b90af466a37591f"
STORED_SHA256 = "dcd4d68e4b5989ffc31177e2de8eefb3c909903cc5cebbd12da96cf0634eb2a8"
MANIFEST_SHA256 = "b128bc93cf41c179aa8c8a285b51be5bb87a6bd9788e5033446c177de9aee0d4"


def test_posthf_storage_preserves_original_bytes_and_historical_manifest() -> None:
    storage = json.loads((BUNDLE / "storage.json").read_text())
    assert storage["schema"] == "generativeqc.posthf-147-storage.v1"
    assert storage["source_revision"] == SOURCE_REVISION
    assert storage["records"] == [
        {
            "original_path": "evidence.json",
            "stored_path": "evidence.json.gz",
            "encoding": "gzip",
            "original_bytes": 291958,
            "original_sha256": ORIGINAL_SHA256,
            "stored_bytes": 24556,
            "stored_sha256": STORED_SHA256,
        }
    ]
    assert storage["original_manifest"] == {
        "path": "manifest.json",
        "bytes": 1981,
        "sha256": MANIFEST_SHA256,
    }
    manifest_bytes = (BUNDLE / "manifest.json").read_bytes()
    assert len(manifest_bytes) == 1981
    assert hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256
    manifest = json.loads(manifest_bytes)
    assert manifest["scientific_revision"] == (
        "bf85a4176b3c232c0f6cfd0715b324f07761e3e6"
    )
    assert len(manifest["sha256"]) == 20
    assert manifest["sha256"]["evidence.json"] == ORIGINAL_SHA256

    path = BUNDLE / "evidence.json.gz"
    packed = path.read_bytes()
    assert len(packed) == 24556
    assert hashlib.sha256(packed).hexdigest() == STORED_SHA256
    original = gzip.decompress(packed)
    assert len(original) == 291958
    assert hashlib.sha256(original).hexdigest() == ORIGINAL_SHA256
    assert not (BUNDLE / "evidence.json").exists()
    records = json.loads(original)
    assert isinstance(records, list)
    assert len(records) == 6
    assert load_json(path) == records

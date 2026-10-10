"""Historical scheduler evidence survives storage-only gzip compaction exactly."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tools.generativeqc_validation.record import load_json

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/tensor-resource-scheduler-508"
)
ORIGINAL_SHA256 = "39414917b758b0c50da615b231fb4736c2b95fd5d43b2c05e95ce69a40dc71aa"


def _restore(archive: bytes, receipt: dict) -> bytes:
    assert len(archive) == receipt["archive_bytes"]
    assert hashlib.sha256(archive).hexdigest() == receipt["archive_sha256"]
    restored = gzip.decompress(archive)
    assert len(restored) == receipt["original_bytes"] == 169698
    assert (
        hashlib.sha256(restored).hexdigest()
        == receipt["original_sha256"]
        == ORIGINAL_SHA256
    )
    return restored


def test_scheduler_qualification_restores_the_original_git_blob() -> None:
    receipt = json.loads((BUNDLE / "qualification-retention.json").read_bytes())
    assert receipt["schema"] == "generativeqc.lossless-evidence-compaction.v1"
    assert receipt["archive"] == "qualification.json.gz"
    assert receipt["original"] == "qualification.json"
    archive = (BUNDLE / receipt["archive"]).read_bytes()
    restored = _restore(archive, receipt)
    blob = b"blob " + str(len(restored)).encode() + b"\0" + restored
    assert hashlib.sha1(blob).hexdigest() == receipt["original_git_blob"]
    assert archive[4:8] == b"\0" * 4
    assert not (BUNDLE / receipt["original"]).exists()
    assert load_json(BUNDLE / receipt["archive"]) == json.loads(restored)


@pytest.mark.parametrize("mutation", ["truncate", "append", "decoded-drift"])
def test_scheduler_storage_rejects_corruption(mutation: str) -> None:
    receipt = json.loads((BUNDLE / "qualification-retention.json").read_bytes())
    archive = (BUNDLE / receipt["archive"]).read_bytes()
    if mutation == "truncate":
        archive = archive[:-1]
    elif mutation == "append":
        archive += b" "
    else:
        archive = gzip.compress(gzip.decompress(archive) + b"\n", mtime=0)
        receipt["archive_bytes"] = len(archive)
        receipt["archive_sha256"] = hashlib.sha256(archive).hexdigest()
    with pytest.raises(AssertionError):
        _restore(archive, receipt)

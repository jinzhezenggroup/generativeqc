"""Lossless historical receipts remain recoverable without changing their gates."""

import json
import lzma
import runpy
import shutil
from pathlib import Path

import pytest

DIRECTORY = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/xc-expressions-161"
)
MODULE = runpy.run_path(str(DIRECTORY / "restore.py"))


def test_original_members_restore_exactly(tmp_path: Path) -> None:
    records = MODULE["read_records"]()
    assert len(records) == 85
    output = tmp_path / "restored"
    assert MODULE["restore"](output) == 85
    assert {
        str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*.json")
    } == records
    with pytest.raises(FileExistsError):
        MODULE["restore"](output)
    assert (output / "summary.json").read_bytes() == records["summary.json"]


@pytest.mark.parametrize(
    "damage", ["payload", "missing", "extra", "trailing", "summary"]
)
def test_corrupt_receipts_fail_before_writing(tmp_path: Path, damage: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ("records.json.xz", "manifest.json", "summary.json"):
        shutil.copyfile(DIRECTORY / name, source / name)
    packed = source / "records.json.xz"
    texts = json.loads(lzma.decompress(packed.read_bytes()))
    if damage == "payload":
        texts[next(iter(texts))] += " "
    elif damage == "missing":
        texts.pop(next(iter(texts)))
    elif damage == "extra":
        texts["../escape.json"] = "{}"
    elif damage == "summary":
        (source / "summary.json").write_text("{}")
    packed.write_bytes(
        lzma.compress(json.dumps(texts).encode())
        + (b"tail" if damage == "trailing" else b"")
    )
    output = tmp_path / "restored"
    with pytest.raises(ValueError):
        MODULE["restore"](output, source)
    assert not output.exists()

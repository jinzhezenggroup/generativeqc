"""Offline scientific and corruption gates for native DFT default evidence."""

from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from pathlib import Path

import pytest

from benchmarks._exact_fp64_evidence import load
from benchmarks.dft_streaming_qualification import qualify_native

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "benchmarks/results/dft-shared-rys-streaming-20261108"


def test_native_default_qualification_is_scoped_and_preserves_raw_k_oracles() -> None:
    replay = runpy.run_path(str(FAMILY / "replay.py"))["replay"]
    result = replay()
    assert result["default_promotion_qualified"]
    assert not result["all_readme_leadership_established"]
    assert result["raw_k_matrix_predicates"] == {"a100": 144, "5090": 144}


@pytest.mark.parametrize("stream", ("receipts.json.xz", "fp64.xz"))
def test_evidence_rejects_changed_compressed_stream(
    tmp_path: Path, stream: str
) -> None:
    for name in ("receipts.json.xz", "fp64.xz", "sha256.txt"):
        shutil.copyfile(FAMILY / name, tmp_path / name)
    path = tmp_path / stream
    payload = path.read_bytes()
    path.write_bytes(bytes((payload[0] ^ 1,)) + payload[1:])
    with pytest.raises(ValueError):
        load(tmp_path)


def test_changed_initial_guess_work_does_not_qualify(tmp_path: Path) -> None:
    _, files = load(FAMILY)
    for name, text in files.items():
        if name.startswith("native-a100/"):
            (tmp_path / name.removeprefix("native-a100/")).write_text(text)
    path = tmp_path / "pbe0-2-candidate.json"
    record = json.loads(path.read_text())
    record["samples"][0]["initial_guess"]["preliminary_iterations"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="initial-guess"):
        qualify_native(tmp_path, "a100")


def test_storage_migration_authenticates_every_original_seed_receipt_byte() -> None:
    bundle, files = load(ROOT / "benchmarks/results/cuda-ks-seed-products-20261101")
    original = files["original-receipts.json"]
    assert (
        hashlib.sha256(original.encode()).hexdigest()
        == bundle["experiment"]["original_raw_sha256"]
    )
    assert len(json.loads(original)["records"]) == 139

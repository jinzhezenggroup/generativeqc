"""Exercise the retained scientific envelope, not source spelling or timings."""

import hashlib
import lzma
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools.generativeqc_validation import xc_diagonal_evidence


def test_complete_qualification_rechecks_prescribed_populations() -> None:
    result = xc_diagonal_evidence.summarize()
    assert result["qualified_calls"] + result["observational_calls"] == 192
    assert result["qualified_profile_calls"] == 8
    assert len(result["comparisons"]) == 8
    for population in result["populations"].values():
        assert population["maximum_energy_error"] <= 1e-8
        assert population["maximum_force_error"] <= 1e-7


def test_changed_force_bit_cannot_pass_projection_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = xc_diagonal_evidence.ROOT
    copied = tmp_path / original.name
    copied.mkdir()
    for name in ("forces.npz", "evidence.json.xz"):
        shutil.copy2(original / name, copied / name)
    shared = original.parent / "md-j-sparse-support-20261016"
    target = tmp_path / shared.name
    target.mkdir()
    for name in ("forces.npz", "evidence.json.xz"):
        shutil.copy2(shared / name, target / name)
    with np.load(copied / "forces.npz", allow_pickle=False) as archive:
        byteplanes = archive["xor_byteplanes"].copy()
    byteplanes[0, 0, 0] ^= 1
    np.savez_compressed(copied / "forces.npz", xor_byteplanes=byteplanes)
    monkeypatch.setattr(xc_diagonal_evidence, "ROOT", copied)
    with pytest.raises(AssertionError):
        xc_diagonal_evidence.summarize()


def test_archived_qualification_is_lossless() -> None:
    decoded = lzma.decompress(
        (xc_diagonal_evidence.ROOT / "qualification.md.xz").read_bytes()
    )
    assert len(decoded) == 9714
    assert hashlib.sha256(decoded).hexdigest() == (
        "84d39840db447dbbf4067f3652211d8c8e4a0c10146262dbf64588175c4a63d0"
    )


def test_current_manifest_covers_each_retained_member() -> None:
    root = xc_diagonal_evidence.ROOT
    entries = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        assert path not in entries
        entries[path] = digest
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != "SHA256SUMS"
    }
    assert set(entries) == actual


def test_evidence_recompression_preserves_exact_json() -> None:
    decoded = lzma.decompress(
        (xc_diagonal_evidence.ROOT / "evidence.json.xz").read_bytes()
    )
    assert len(decoded) == 685839
    assert hashlib.sha256(decoded).hexdigest() == (
        "4c6e19d0812c08a4030f0918f4f25f6c0540b3f2ea07fe3181c8957225f3b737"
    )

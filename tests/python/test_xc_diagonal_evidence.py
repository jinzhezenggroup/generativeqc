"""Exercise the retained scientific envelope, not source spelling or timings."""

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

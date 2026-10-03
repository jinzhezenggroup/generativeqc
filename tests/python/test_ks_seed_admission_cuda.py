"""Independent ensemble gates and atomic restore with the idle CUDA provider."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, _native
from test_dft_batch import restore_snapshots, warm_snapshot

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_GRID_CUDA_TEST") != "1",
    reason="requires an explicitly scheduled CUDA qualification",
)


@pytest.mark.parametrize("basis", ("sto-3g", "def2-svp"))
@pytest.mark.parametrize("method", ("lda-rks", "lda-uks"))
def test_idle_gpu_admission_preserves_ensemble_gates_and_atomicity(
    basis: str, method: str, monkeypatch: Any, tmp_path: Path
) -> None:
    """Exercise both native-small and provider eigensolvers without a seed repair.

    PySCF supplies an independent AO metric. Its inverse maps analytic
    orthonormal ensemble spectra into AO densities; no target SCF density is
    used to manufacture the valid/invalid occupation fixtures.
    """
    gto = pytest.importorskip("pyscf.gto")
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.4, 1.0)), ("H", (0.0, -1.4, 1.0))]
    mol = gto.M(atom=atoms, basis=basis, unit="Bohr", verbose=0)
    overlap = mol.intor("int1e_ovlp")
    values, vectors = np.linalg.eigh(overlap)
    x = (vectors / np.sqrt(values)) @ vectors.T
    n = len(values)
    spins = 2 if method.endswith("uks") else 1
    electron_count = 10 / spins
    spectrum = np.full(n, electron_count / n)
    valid = x @ np.diag(spectrum) @ x.T
    invalid_spectrum = spectrum.copy()
    invalid_spectrum[0] = -0.1
    invalid_spectrum[1] += spectrum[0] + 0.1
    invalid = x @ np.diag(invalid_spectrum) @ x.T
    # Both have the right charge; rejection must reach the occupation gate.
    assert np.einsum("ij,ji", valid, overlap) == pytest.approx(
        electron_count, abs=1e-10
    )
    assert np.einsum("ij,ji", invalid, overlap) == pytest.approx(
        electron_count, abs=1e-10
    )
    good = np.tile(valid.ravel(), spins)
    bad = np.tile(valid.ravel(), spins)
    bad[: n * n] = invalid.ravel()
    calc = Calculator(
        method=method,
        basis=basis,
        device="cuda",
        max_iterations=150,
        ks_options=KsOptions(grid=GridSpec(16, 8, 16)),
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
    )
    with calc.prepare_batch([atoms, atoms], warm_start=True) as batch:
        original = batch.execute(properties=("energy",), strict=True)
        saved = [warm_snapshot(batch, i) for i in range(2)]
        trace = tmp_path / "admission.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_HOST_TRACE", str(trace))
        proposal = (good, saved[0][1], saved[0][2])
        rejected = (bad, saved[1][1], saved[1][2])
        assert (
            restore_snapshots(batch, [proposal, rejected])
            == _native.STATUS_INVALID_ARGUMENT
        )
        for i in range(2):
            after = warm_snapshot(batch, i)
            assert np.array_equal(after[0], saved[i][0])
            assert np.array_equal(after[1], saved[i][1])
            assert after[2] == saved[i][2]
        # A rejected import must leave the live last-good resident density
        # usable even though the validator borrowed abandoned DIIS history.
        replay = batch.execute(properties=("energy",), strict=True)
        assert replay.energies == pytest.approx(original.energies, abs=1e-8)
        assert all(
            item.warm_start_used and not item.warm_start_fallback
            for item in replay.items
        )
        trace.unlink()
        assert restore_snapshots(batch, [proposal, None]) == _native.STATUS_SUCCESS
        assert np.array_equal(warm_snapshot(batch, 0)[0], good)
        assert np.array_equal(warm_snapshot(batch, 1)[0], saved[1][0])
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        regions = [region for record in records for region in record["regions"]]
        assert all(record["valid"] for record in records)
        assert (
            sum(region["name"] == "cuda_ks_seed_eigen" for region in regions)
            == spins + 1
        )
        assert not any(region["name"] == "reference_eigensolve" for region in regions)
        # Preserve the legacy <=1e-7 asymmetric-density contract. Its metric
        # spectrum deliberately keeps the common scalar fallback, not a repair.
        near_symmetric = good.copy()
        near_symmetric[1] += 1e-9
        near_symmetric[n] -= 1e-9
        assert (
            restore_snapshots(batch, [(near_symmetric, saved[0][1], saved[0][2]), None])
            == _native.STATUS_SUCCESS
        )
        assert np.array_equal(warm_snapshot(batch, 0)[0], near_symmetric)

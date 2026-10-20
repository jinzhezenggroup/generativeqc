"""HF warm imports preserve physical gates with bounded CUDA diagonalization."""

import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, _native
from test_dft_batch import restore_snapshots, warm_snapshot

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires an explicitly scheduled CUDA qualification",
)


@pytest.mark.parametrize("waters", (1, 4))
@pytest.mark.parametrize("method", ("rhf", "uhf"))
def test_idle_gpu_admission_preserves_ensemble_gates_and_atomicity(
    waters: int, method: str, monkeypatch: Any, tmp_path: Path
) -> None:
    """Exercise both native-small and provider eigensolvers without a seed repair.

    PySCF supplies an independent AO metric. Its inverse maps analytic
    orthonormal ensemble spectra into AO densities; no target SCF density is
    used to manufacture the valid/invalid occupation fixtures.
    """
    gto = pytest.importorskip("pyscf.gto")
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.4, 1.0)), ("H", (0.0, -1.4, 1.0))]
    atoms = [
        (element, (x + 8 * copy, y, z))
        for copy in range(waters)
        for element, (x, y, z) in atoms
    ]
    basis = "def2-svp"
    # Use the exact bundled exponents/contractions with independent PySCF
    # integrals, so the unchanged 1e-7 charge gate checks the actual model.
    pack = json.loads(
        files("generativeqc").joinpath("data/basis_pack.json").read_text()
    )
    elements = pack["bases"][basis]["elements"]
    oracle_basis = {
        symbol: [
            [
                shell["angular_momentum"],
                *[
                    [float(exponent), float(coefficient)]
                    for exponent, coefficient in zip(
                        shell["exponents"], shell["coefficients"], strict=True
                    )
                ],
            ]
            for shell in elements[number]
        ]
        for symbol, number in (("O", "8"), ("H", "1"))
    }
    mol = gto.M(atom=atoms, basis=oracle_basis, unit="Bohr", verbose=0)
    overlap = mol.intor("int1e_ovlp")
    values, vectors = np.linalg.eigh(overlap)
    x = (vectors / np.sqrt(values)) @ vectors.T
    n = len(values)
    spins = 2 if method == "uhf" else 1
    electron_count = 10 * waters / spins
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
        basis_representation="spherical",
        device="cuda",
        max_iterations=150,
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
        # Rejection leaves the live last-good density usable.
        replay = batch.execute(properties=("energy",), strict=True)
        assert replay.energies == pytest.approx(original.energies, abs=1e-8)
        assert all(
            item.warm_start_used and not item.warm_start_fallback
            for item in replay.items
        )
        # Replay performs a fresh SCF step, so preservation is relative to its
        # output, not the density saved before that step.
        neighbor = warm_snapshot(batch, 1)
        trace.unlink()
        _native.check(
            batch._library,
            restore_snapshots(batch, [proposal, None]),
            context=batch._context,
        )
        assert np.array_equal(warm_snapshot(batch, 0)[0], good)
        assert np.array_equal(warm_snapshot(batch, 1)[0], neighbor[0])
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        regions = [region for record in records for region in record["regions"]]
        assert all(record["valid"] for record in records)
        assert any(
            region["name"] == "cuda_hf_warm_density_validation" for region in regions
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

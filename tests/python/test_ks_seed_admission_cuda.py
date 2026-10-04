"""Independent ensemble gates and atomic restore with the idle CUDA provider."""

import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, _native
from generativeqc._ks_snapshot import NativeKsSnapshot
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
    # Use the exact bundled exponents/contractions with independent PySCF
    # integrals. PySCF's rounded STO-3G table differs enough to fail the
    # unchanged 1e-7 electron-count gate for the RKS ensemble.
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
        basis_representation="spherical",
        device="cuda",
        max_iterations=150,
        ks_options=KsOptions(grid=GridSpec(16, 8, 16)),
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
    )
    with calc.prepare_batch([atoms, atoms], warm_start=True) as batch:
        original = batch.execute(properties=("energy",), strict=True)
        saved = [warm_snapshot(batch, i) for i in range(2)]
        snapshot = NativeKsSnapshot(batch, 0)
        try:
            before_integrals = snapshot.cuda_integral_derivatives(
                len(atoms), 256 << 20, range_exchange=False
            )
            assert before_integrals is not None
        except Exception:
            snapshot.close()
            raise
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
        try:
            snapshot.check_current()
            after_integrals = snapshot.cuda_integral_derivatives(
                len(atoms), 256 << 20, range_exchange=False
            )
            assert after_integrals is not None
            # The rejected admission must preserve the already staged resident
            # stationary weights, including the UKS total density in tmp2.
            np.testing.assert_allclose(
                after_integrals[0], before_integrals[0], atol=1e-11, rtol=0
            )
        finally:
            snapshot.close()
        # A rejected import must leave the live last-good resident density
        # usable even though the validator borrowed abandoned DIIS history.
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

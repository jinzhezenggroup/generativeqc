"""Complete native meta-GGA force ownership, independent oracles and replay."""

from __future__ import annotations

import ctypes as ct
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from test_nonlocal_grid_endpoints import ATOMS, PRIMARY, _cpu_reference

WATER = (("O", (0.1, -0.1, 0.0)), ("H", (0.1, 0.2, 1.7)), ("H", (1.6, -0.2, -0.5)))


def require_device(device: str) -> None:
    if device == "cuda":
        if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
            pytest.skip("requires an allocated CUDA device")
        assert os.environ.get("SLURM_JOB_ID"), "GPU qualification must run through srun"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("spin,atoms", [(0, ATOMS), (1, ATOMS), (0, WATER), (1, WATER)])
@pytest.mark.parametrize("secondary", [PRIMARY, GridSpec(6, 4, 8), GridSpec(10, 4, 8)])
def test_complete_wb97mv_native_force_matches_independent_oracle(
    device: str,
    spin: int,
    atoms: Any,
    secondary: GridSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PySCF supplies its own integrals, XC, VV10 and moving-grid response."""
    require_device(device)
    pytest.importorskip("pyscf")
    from generativeqc import batch as batch_module

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("complete native force escaped to the Python derivative consumer")

    monkeypatch.setattr(batch_module.PreparedBatch, "_public_dft_cpu_force", forbidden)
    monkeypatch.setattr(batch_module.PreparedBatch, "_public_dft_cuda_force", forbidden)
    calculator = Calculator(
        method="wb97m-v-uks" if spin else "wb97m-v-rks",
        basis="sto-3g",
        device=device,
        basis_representation="spherical",
        ks_options=KsOptions(grid=PRIMARY, nonlocal_grid=secondary, tile_points=64),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )
    with calculator.prepare_batch(
        [atoms], charges=[spin], multiplicities=[spin + 1], warm_start=False
    ) as batch:
        assert batch._native_dft_force_eligible()
        result = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        energy = batch.execute(strict=True, properties=("energy",)).items[0]
        assert energy.forces is None
        assert energy.energy == pytest.approx(result.energy, abs=2e-10)
        failed = batch.execute(
            [np.full((len(atoms), 3), np.nan)], properties=("energy", "forces")
        )
        assert not failed.items[0].succeeded and failed.items[0].forces is None
        replay = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        np.testing.assert_allclose(replay.forces, result.forces, atol=1e-9, rtol=0)
    # Load the pinned *input* coefficients. PySCF's named STO-3G table rounds
    # them differently; using it would compare two different Hamiltonians.
    from pyscf.data.elements import charge

    pack = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "python/generativeqc/data/basis_pack.json"
        ).read_text()
    )
    basis = {
        symbol: [
            [
                shell["angular_momentum"],
                *[
                    (float(exponent), float(coefficient))
                    for exponent, coefficient in zip(
                        shell["exponents"], shell["coefficients"], strict=True
                    )
                ],
            ]
            for shell in pack["bases"]["sto-3g"]["elements"][str(charge(symbol))]
        ]
        for symbol, _ in atoms
    }
    reference_energy, reference_forces = _cpu_reference(
        spin, secondary, atoms=atoms, basis=basis
    )
    assert result.energy == pytest.approx(reference_energy, abs=2e-8)
    np.testing.assert_allclose(result.forces, reference_forces, atol=3e-7, rtol=0)
    np.testing.assert_allclose(np.sum(result.forces, axis=0), 0, atol=1e-10, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("spin", [0, 1])
@pytest.mark.parametrize("xc_level", [0, 1])
def test_native_wb97mv_adjusted_atom_grids(
    device: str, spin: int, xc_level: int
) -> None:
    """Exercise V4 setup and topology switches with independent PySCF grids."""
    require_device(device)
    from test_treutler_endpoints import reference

    calculator = Calculator(
        method="wb97m-v-uks" if spin else "wb97m-v-rks",
        basis="sto-3g",
        basis_representation="spherical",
        device=device,
        ks_options=KsOptions(
            grid=GridSpec.treutler(xc_level), nonlocal_grid=GridSpec.treutler(0)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )
    with calculator.prepare_batch(
        [ATOMS], charges=[spin], multiplicities=[spin + 1]
    ) as batch:
        assert batch._native_dft_force_eligible()
        actual = batch.execute(strict=True, properties=("energy", "forces")).items[0]
    # Match the documented molecular rho >= 1e-10 pair domain. PySCF's
    # unmodified CPU default drops a larger tail at 1e-8 on Treutler grids.
    from tools._pyscf_vv10_domain import vv10_density_domain

    with vv10_density_domain(1e-10):
        energy, forces = reference(
            calculator,
            fitted=False,
            functional="WB97M_V",
            spin=spin,
            nonlocal_level=0,
            grid_level=xc_level,
            atom_values=list(ATOMS),
        )
    assert actual.energy == pytest.approx(energy, abs=2e-8)
    np.testing.assert_allclose(actual.forces, forces, atol=3e-7, rtol=0)


def test_cuda_nonlocal_capacity_matches_prepared_grids() -> None:
    """Descriptor sizing cannot substitute XC extent for a distinct VV10 grid."""
    require_device("cuda")
    from generativeqc import _native

    for secondary in (GridSpec(10, 4, 8), GridSpec.treutler(0), GridSpec(160, 4, 8)):
        calculator = Calculator(
            method="wb97m-v-rks",
            basis="sto-3g",
            device="cuda",
            ks_options=KsOptions(grid=PRIMARY, nonlocal_grid=secondary, tile_points=64),
        )
        library = calculator._library
        descriptor = calculator._method_descriptor(systems=[ATOMS])
        shape = (ct.c_uint64 * 7)(2, 2, 6, 512, 0, 0, 0)
        output = (ct.c_uint64 * 9)(*[991] * 9)
        query = library.generativeqc_resource_ks_stationary_cuda_v1
        query.argtypes = [
            ct.POINTER(_native.MethodDescriptor),
            ct.POINTER(ct.c_uint64),
            ct.c_size_t,
            ct.POINTER(ct.c_uint64),
            ct.c_size_t,
        ]
        query.restype = ct.c_int
        status = query(ct.byref(descriptor), shape, 7, output, 9)
        if secondary == GridSpec.treutler(0):
            # Ragged element-specific grids require prepared topology; large
            # rectangular point grids instead use the checked shape polynomial.
            assert status == _native.STATUS_NOT_IMPLEMENTED
            assert list(output) == [991] * 9
            continue
        assert status == _native.STATUS_SUCCESS
        if secondary == GridSpec(160, 4, 8):
            assert 2 * 160 * 4 * 8 > 8192
        else:
            # Old 8-atom/64-AO admission is not a real executor capacity bound.
            wide = (ct.c_uint64 * 7)(12, 96, 192, 12 * 8 * 4 * 8, 1, 0, 0)
            wide_capacity = (ct.c_uint64 * 9)(*[991] * 9)
            assert (
                query(ct.byref(descriptor), wide, 7, wide_capacity, 9)
                == _native.STATUS_SUCCESS
            )
            assert list(wide_capacity) != [991] * 9
        prepared_query = library.generativeqc_ks_batch_stationary_cuda_capacity_v1
        prepared_query.argtypes = [
            ct.c_void_p,
            ct.c_uint32,
            ct.POINTER(ct.c_uint64),
            ct.c_size_t,
        ]
        prepared_query.restype = ct.c_int
        prepared = (ct.c_uint64 * 9)()
        with calculator.prepare_batch([ATOMS]) as batch:
            assert (
                prepared_query(batch._batch, 0, prepared, 9) == _native.STATUS_SUCCESS
            )
            assert list(prepared) == list(output)

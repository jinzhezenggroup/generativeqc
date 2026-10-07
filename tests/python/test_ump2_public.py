"""Source-matched native Calculator UMP2 energy and independent molecular gates."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, _native, method_capabilities
from generativeqc.fock import FockBuildSpec, FockPlan
from generativeqc_compiler.dft import NativeAO

ROOT = Path(__file__).resolve().parents[2]


def test_ump2_manifest_declares_distinct_energy_only_native_method() -> None:
    methods = json.loads((ROOT / "manifests/public_methods.json").read_text())[
        "methods"
    ]
    row = next(method for method in methods if method["name"] == "ump2")
    assert row == {
        "name": "ump2",
        "symbol": "UMP2",
        "abi_id": 20,
        "family": "perturbation",
        "provider": "ump2",
        "properties": ["energy"],
        "supports_batch": True,
    }
    capabilities = method_capabilities("ump2")
    assert capabilities.available
    assert capabilities.supported_properties == frozenset({"energy"})
    assert capabilities.supports_batch


def test_native_ump2_cpu_header_matches_compiler_equations() -> None:
    import re

    from tools.generate_mp2_native import ump2_cpu_header

    actual = (ROOT / "src/posthf/ump2_cpu_generated.hpp").read_text()
    expected = ump2_cpu_header()

    def tokens(value: str) -> str:
        return "".join(re.sub(r"//[^\n]*", "", value).split())

    assert tokens(actual) == tokens(expected)


def test_public_ump2_li_matches_independent_pyscf_214() -> None:
    pyscf = pytest.importorskip("pyscf")
    assert pyscf.__version__ == "2.14.0"
    from pyscf import gto, mp, scf

    atoms = [("Li", (0.0, 0.0, 0.0))]
    # PySCF's stock STO-3G Li table rounds exponents differently from this
    # repository's pinned table. Match basis source bytes, then use PySCF only
    # as an independent SCF/UMP2 numerical oracle.
    with NativeAO(atoms, basis="sto-3g", multiplicity=2) as native_basis:
        matched_basis = {
            "Li": [
                [
                    shell.angular_momentum,
                    *[
                        [primitive.exponent, primitive.coefficient]
                        for primitive in shell.primitives
                    ],
                ]
                for shell in native_basis.shells
            ]
        }
    mol = gto.M(
        atom=atoms, basis=matched_basis, spin=1, unit="Bohr", cart=True, verbose=0
    )
    mf = scf.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.kernel()
    assert mf.converged
    oracle = mp.UMP2(mf).run()

    calc = Calculator(method="ump2", basis="sto-3g", device="cpu", initial_guess=None)
    result = calc.singlepoint(atoms, multiplicity=2)
    assert result.converged and result.forces is None
    assert result.executed_backend == "cpu_reference"
    assert abs(result.correlation.reference_energy - mf.e_tot) <= 1e-8
    assert abs(result.energy - oracle.e_tot) <= 1e-8
    assert abs(result.correlation.opposite_spin_energy) > 1e-8
    assert 0 <= result.correlation.reference_residual <= 1e-8
    assert result.correlation.numeric_capacity_bytes <= 256 << 20
    assert result.correlation.equation_hash


def test_public_ump2_restricted_limit_matches_public_mp2() -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    options = {"basis": "sto-3g", "device": "cpu", "initial_guess": None}
    unrestricted = Calculator(method="ump2", **options).singlepoint(atoms)
    restricted = Calculator(method="mp2", **options).singlepoint(
        atoms, properties=("energy",)
    )
    assert unrestricted.converged and restricted.converged
    assert (
        abs(
            unrestricted.correlation.reference_energy
            - restricted.correlation.reference_energy
        )
        <= 1e-9
    )
    assert abs(unrestricted.energy - restricted.energy) <= 1e-9
    assert (
        abs(
            unrestricted.correlation.opposite_spin_energy
            - restricted.correlation.opposite_spin_energy
        )
        <= 1e-9
    )
    assert (
        abs(
            unrestricted.correlation.same_spin_energy
            - restricted.correlation.same_spin_energy
        )
        <= 1e-9
    )


def test_public_native_ump2_broken_symmetry_from_validated_uhf_seed() -> None:
    """The public HF warm-state ABI selects a real native UHF determinant."""
    pyscf = pytest.importorskip("pyscf")
    assert pyscf.__version__ == "2.14.0"
    from pyscf import gto, mp, scf

    atoms = [("H", (0.0, 0.0, -2.0)), ("H", (0.0, 0.0, 2.0))]
    localized = np.zeros((2, 2, 2), dtype=np.float64)
    localized[0, 0, 0] = localized[1, 1, 1] = 1.0
    with (
        NativeAO(atoms, basis="sto-3g", multiplicity=1) as basis,
        FockPlan(
            basis,
            FockBuildSpec.hf("unrestricted", derivative_order=0),
            screening_tolerance=0.0,
        ) as source,
    ):
        native_hf = source.solve(
            initial_density=localized,
            compute_forces=False,
            max_iterations=200,
            energy_tolerance=1e-11,
            density_tolerance=1e-10,
        )
    assert native_hf.initial_density_used
    assert np.linalg.norm(native_hf.density[0] - native_hf.density[1]) > 0.1

    mol = gto.M(atom=atoms, basis="sto-3g", spin=0, unit="Bohr", cart=True, verbose=0)
    independent_hf = scf.UHF(mol)
    independent_hf.conv_tol = 1e-13
    independent_hf.conv_tol_grad = 1e-11
    independent_hf.max_cycle = 200
    independent_hf.kernel(dm0=localized)
    assert independent_hf.converged
    assert np.linalg.norm(np.diff(independent_hf.make_rdm1(), axis=0)) > 0.1
    oracle = mp.UMP2(independent_hf).run()

    calc = Calculator(
        method="ump2",
        basis="sto-3g",
        device="cpu",
        initial_guess=None,
        max_iterations=200,
    )
    with calc.prepare_batch([atoms], multiplicities=[1], warm_start=True) as batch:
        density = np.ascontiguousarray(native_hf.density, dtype=np.float64)
        coordinates = np.ascontiguousarray(
            [coordinate for _, position in atoms for coordinate in position],
            dtype=np.float64,
        )
        # These diagnostics come from the native source UHF solve above. The
        # imported density remains only a proposal for this fresh UMP2 solve.
        state = _native.HfWarmState(
            struct_size=ctypes.sizeof(_native.HfWarmState),
            abi_version=_native.ABI_VERSION,
            density=density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            density_count=density.size,
            coordinates=coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            coordinate_count=coordinates.size,
            energy=native_hf.energy,
            energy_change=native_hf.energy_change,
            density_rms=native_hf.density_rms,
            iterations=native_hf.iterations,
            present=1,
        )
        status = batch._library.generativeqc_batch_restore_hf_warm_states(
            batch._batch, ctypes.byref(state), 1
        )
        _native.check(batch._library, status, context=batch._context)
        rejected_density = np.zeros_like(density)
        rejected = _native.HfWarmState(
            struct_size=state.struct_size,
            abi_version=state.abi_version,
            density=rejected_density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            density_count=state.density_count,
            coordinates=state.coordinates,
            coordinate_count=state.coordinate_count,
            energy=state.energy,
            energy_change=state.energy_change,
            density_rms=state.density_rms,
            iterations=state.iterations,
            present=1,
        )
        assert (
            batch._library.generativeqc_batch_restore_hf_warm_states(
                batch._batch, ctypes.byref(rejected), 1
            )
            != _native.STATUS_SUCCESS
        )
        item = batch.execute(strict=True).items[0]
    assert item.succeeded and item.warm_start_used and not item.warm_start_fallback
    assert abs(item.correlation.reference_energy - independent_hf.e_tot) <= 1e-8
    assert abs(item.energy - oracle.e_tot) <= 1e-8
    assert abs(item.correlation.opposite_spin_energy) > 1e-8
    assert 0 <= item.correlation.reference_residual <= 1e-8


def test_ump2_batch_replaces_geometry_source_and_clears_failed_diagnostic() -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    moved = [("H", (0.0, 0.0, -0.9)), ("H", (0.0, 0.0, 0.9))]
    calc = Calculator(method="ump2", basis="sto-3g", device="cpu", initial_guess=None)
    with calc.prepare_batch([atoms], warm_start=False) as batch:
        first = batch.execute(strict=True).items[0]
        failed = batch.execute(
            coordinates=[np.array([[np.nan, 0, -0.9], [0, 0, 0.9]])]
        ).items[0]
        assert not failed.succeeded and failed.correlation is None
        replay = batch.execute(
            coordinates=[np.array([atom[1] for atom in moved])], strict=True
        ).items[0]
    fresh = calc.singlepoint(moved)
    assert replay.succeeded and abs(replay.energy - first.energy) > 1e-4
    assert abs(replay.energy - fresh.energy) <= 1e-10


def test_public_ump2_unsupported_requests_fail_closed() -> None:
    atoms = [("Li", (0.0, 0.0, 0.0))]
    calc = Calculator(method="ump2", basis="sto-3g", device="cpu", initial_guess=None)
    with pytest.raises(ValueError, match="does not support properties"):
        calc.singlepoint(atoms, multiplicity=2, properties=("energy", "forces"))
    with pytest.raises(ValueError, match="fp64"):
        Calculator(method="ump2", precision="auto")
    with pytest.raises(NotImplementedError, match="UMP2 frozen-core"):
        Calculator(method="ump2", ccsd_frozen_core=1)
    with pytest.raises(Exception, match="RI/DF"):
        Calculator(
            method="ump2", density_fitting="cpu", initial_guess=None
        ).singlepoint(atoms, multiplicity=2)
    with pytest.raises(Exception, match="numeric memory budget"):
        Calculator(
            method="ump2", correlation_memory_budget_bytes=1, initial_guess=None
        ).singlepoint(atoms, multiplicity=2)
    with pytest.raises(Exception, match="near-zero UMP2 denominator"):
        Calculator(
            method="ump2", mp2_denominator_threshold=100.0, initial_guess=None
        ).singlepoint(atoms, multiplicity=2)

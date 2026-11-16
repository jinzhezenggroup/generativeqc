"""Qualify complete native through-f full-range forces, not Ni Hessian support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, Primitive, Shell, import_bse
from generativeqc._dft_gradient import StationaryKsState
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.method import resolve_method
from test_dft_complete_cpu import independent_global_hybrid_gradient
from test_integral_autodiff_second import require_native_library  # noqa: F401

if TYPE_CHECKING:
    from typing import Any

ATOMS = [("H", (0, 0.1, -1.6)), ("H", (0.3, -0.2, 0)), ("H", (-0.1, 0.4, 1.6))]


@pytest.mark.parametrize("representation", ("spherical", "cartesian"))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_native_full_range_f_shell_force_matches_independent_complete_endpoint(
    representation: str, spin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Match independently converged D, E and moving-grid gradients in both spins."""
    pytest.importorskip("pyscf")
    atoms = ATOMS if spin == "polarized" else ATOMS + [("H", (1.4, -0.2, 2.9))]
    multiplicity = 2 if spin == "polarized" else 1
    basis = (
        Shell(0, 0, (Primitive(0.8, 1.0),)),
        Shell(0, 3, (Primitive(0.6, 1.0),)),
    ) + tuple(Shell(slot, 0, (Primitive(0.8, 1.0),)) for slot in range(1, len(atoms)))
    method = resolve_method("B3LYP", spin=spin)
    functional = method.primitives[0].functional
    code = "+".join(f"{float(weight)}*{name}" for name, weight in functional.components)
    code += f"+{float(method.primitives[1].coefficient)}*HF"
    calculator = Calculator(
        method=method,
        device="cpu",
        basis=basis,
        basis_representation=representation,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    try:
        with calculator.prepare_batch([atoms], multiplicities=[multiplicity]) as batch:
            assert batch._native_dft_force_eligible()

            def forbidden_python_force(*_args: Any, **_kwargs: Any) -> None:
                pytest.fail("native f-shell forces invoked the Python diagnostic")

            monkeypatch.setattr(batch, "_public_dft_cpu_force", forbidden_python_force)
            point = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            assert point.converged
            with NativeAO(
                atoms,
                basis=basis,
                representation=representation,
                multiplicity=multiplicity,
            ) as ao:
                state = StationaryKsState.from_native(batch, ao)
                try:
                    energy, gradient, reference = independent_global_hybrid_gradient(
                        ao,
                        state,
                        "full-range-uks" if spin == "polarized" else "full-range-rks",
                        xc=code,
                        return_mean_field=True,
                    )
                    assert reference.converged
                    assert (
                        np.linalg.norm(
                            reference.get_grad(reference.mo_coeff, reference.mo_occ)
                        )
                        < 2e-9
                    )
                    oracle_density = reference.make_rdm1()
                    if representation == "cartesian":
                        scale = np.sqrt(np.diag(reference.get_ovlp()))
                        oracle_density = (
                            scale[:, None] * oracle_density * scale[None, :]
                        )
                    if spin == "unpolarized":
                        oracle_density = oracle_density[None, :, :]
                    np.testing.assert_allclose(
                        state.density, oracle_density, atol=2e-8, rtol=0
                    )
                    assert point.energy == pytest.approx(energy, abs=2e-10)
                    np.testing.assert_allclose(
                        point.forces, -gradient, atol=2e-8, rtol=0
                    )
                finally:
                    state._source.close()
    finally:
        calculator.clear_cache()


@pytest.mark.ci_nightly
def test_native_ni_doublet_default_grid_force_matches_independent_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qualify one asymmetric Ni state/force, without granting UKS curvature."""
    pytest.importorskip("pyscf")
    directory = Path(__file__).resolve().parent.parent / "data/global_hybrid"
    manifest = json.loads((directory / "manifest.json").read_text())
    basis = import_bse(
        directory / manifest["file"],
        source=manifest["source_repository"],
        source_version=manifest["source_commit"],
        license=manifest["license"],
        representation="spherical",
    )
    assert basis.provenance.checksum == manifest["sha256"]
    atoms = [
        ("Ni", (0.01, -0.02, 0.03)),
        ("H", (1.8, 0.1, 0.03)),
        ("H", (-0.5, 2.4, 0.3)),
        ("H", (-1.7, -0.8, 1.1)),
    ]
    method = resolve_method("B3LYP", spin="polarized")
    functional = method.primitives[0].functional
    code = "+".join(f"{float(weight)}*{name}" for name, weight in functional.components)
    code += f"+{float(method.primitives[1].coefficient)}*HF"
    calculator = Calculator(
        method=method,
        device="cpu",
        basis=basis,
        basis_representation="spherical",
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=600,
    )
    try:
        with calculator.prepare_batch(
            [atoms], multiplicities=[2], warm_start=False
        ) as batch:
            assert batch._native_dft_force_eligible()

            def forbidden_python_force(*_args: Any, **_kwargs: Any) -> None:
                pytest.fail("Ni forces invoked the bounded Python diagnostic")

            monkeypatch.setattr(batch, "_public_dft_cpu_force", forbidden_python_force)
            point = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            assert point.converged
            with NativeAO(
                atoms, basis=basis, representation="spherical", multiplicity=2
            ) as ao:
                state = StationaryKsState.from_native(batch, ao)
                try:
                    energy, gradient, reference = independent_global_hybrid_gradient(
                        ao,
                        state,
                        "full-range-uks",
                        xc=code,
                        initial_guess="atom",
                        return_mean_field=True,
                    )
                    assert reference.converged
                    assert (
                        np.linalg.norm(
                            reference.get_grad(reference.mo_coeff, reference.mo_occ)
                        )
                        < 2e-9
                    )
                    np.testing.assert_allclose(
                        state.density, reference.make_rdm1(), atol=2e-7, rtol=0
                    )
                    np.testing.assert_allclose(
                        np.einsum("sij,ji->s", state.density, state.overlap),
                        [16, 15],
                        atol=2e-11,
                        rtol=0,
                    )
                    assert point.energy == pytest.approx(energy, abs=2e-8)
                    np.testing.assert_allclose(
                        point.forces, -gradient, atol=2e-8, rtol=0
                    )
                finally:
                    state._source.close()
    finally:
        calculator.clear_cache()

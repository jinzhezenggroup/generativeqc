"""Original #3000/#3001 inputs, complete native endpoints and independent oracles."""

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, basis_capability, load_basis
from generativeqc.batch import PreparedBatch
from generativeqc.calculator import Atom
from generativeqc_compiler.common.resources import ResourceBudget
from generativeqc_compiler.method import (
    D3Spec,
    DispersionCorrectionPrimitive,
    resolve_method,
)

ASSETS = Path(__file__).resolve().parents[1] / "reference_data" / "ks-se-br-3000"
CASES = json.loads((ASSETS / "cases.json").read_text())["cases"]
PROTOCOL = json.loads((ASSETS / "protocol.json").read_text())


def calculator(route: str) -> Calculator:
    """Keep full fitting spaces, native cold initialization and the reported controls."""
    method = resolve_method("B3LYP", spin="polarized")
    method = dataclasses.replace(
        method,
        identifier="B3LYP-D3(BJ)",
        primitives=method.primitives
        + (DispersionCorrectionPrimitive(D3Spec(**PROTOCOL["dispersion"])),),
    )
    fitting = (
        {
            "auxiliary_basis": load_basis(ASSETS / "auxbasis.json"),
            "density_fitting_relative_threshold": PROTOCOL["df_relative_threshold"],
            "density_fitting_memory_budget_bytes": PROTOCOL["df_memory_budget_bytes"],
        }
        if route == "df"
        else {}
    )
    return Calculator(
        method=method,
        basis=load_basis(ASSETS / "basis.json"),
        basis_representation="spherical",
        device="cuda",
        precision="fp64",
        initial_guess=PROTOCOL["scf"]["initial_guess"],
        max_iterations=200,
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        screening_tolerance=1e-12,
        diis_history=8,
        ks_options=KsOptions(grid=GridSpec.treutler(3)),
        density_fitting="cuda" if route == "df" else "none",
        resource_budget=(
            None
            if route == "df"
            else ResourceBudget(
                host_bytes=PROTOCOL["host_memory_budget_bytes"],
                device_bytes=PROTOCOL["native_memory_budget_bytes"],
            )
        ),
        **fitting,
    )


def test_original_spaces_and_role_specific_admission() -> None:
    """The new source contract must not promote orbital h or unrelated consumers."""
    orbital = load_basis(ASSETS / "basis.json")
    auxiliary = load_basis(ASSETS / "auxbasis.json")
    for case in CASES:
        atoms = [
            Atom.from_value(atom)
            for atom in zip(case["symbols"], case["coordinates_angstrom"], strict=True)
        ]
        assert (
            sum(2 * shell.angular_momentum + 1 for shell in orbital.shells_for(atoms))
            == case["nao"]
        )
        auxiliary_shells = auxiliary.shells_for(
            atoms,
            backend="cuda",
            operator="df_metric",
            role="auxiliary",
            consumer="scf_df",
        )
        assert max(shell.angular_momentum for shell in auxiliary_shells) == 5
        for operator in ("df_metric", "df_three_center"):
            for order in (0, 1):
                request = {
                    "backend": "cuda",
                    "operator": operator,
                    "derivative_order": order,
                    "role": "auxiliary",
                }
                assert basis_capability(auxiliary, atoms, consumer="scf_df", **request)[
                    "eligible"
                ]
                assert not basis_capability(auxiliary, atoms, **request)["eligible"]
                assert not basis_capability(
                    auxiliary, atoms, consumer="correlation_df", **request
                )["eligible"]
        assert not basis_capability(auxiliary, atoms, backend="cuda", operator="eri")[
            "eligible"
        ]


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_SE_BR_KS_CUDA_TEST") != "1",
    reason="explicit finite Slurm complete-endpoint qualification",
)
@pytest.mark.parametrize("route", ("direct", "df"))
@pytest.mark.parametrize("case", CASES, ids=[case["task_id"] for case in CASES])
def test_original_native_energy_force_endpoints(
    route: str,
    case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
) -> None:
    """No reference seed enters production; the oracle reconverges a matched root."""
    assert os.environ.get("SLURM_JOB_ID"), "allocate GPU through Slurm"
    record_property("slurm_job_id", os.environ["SLURM_JOB_ID"])
    record_property("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    pytest.importorskip("pyscf")
    pytest.importorskip("dftd3")
    from dftd3.interface import DispersionModel, RationalDampingParam
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_df_ks_convergence_cuda import _matched_reference_seed
    from test_dft_complete_cpu import independent_global_hybrid_gradient

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("native endpoint used a Python scientific force fallback")

    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", forbidden)
    monkeypatch.setattr(PreparedBatch, "_public_dft_cuda_force", forbidden)
    coordinates = (
        np.asarray(case["coordinates_angstrom"]) / PROTOCOL["units"]["bohr_in_angstrom"]
    )
    atoms = list(zip(case["symbols"], coordinates, strict=True))
    preparation_started = time.monotonic()
    calc = calculator(route)
    try:
        with calc.prepare_batch(
            [atoms], charges=[case["charge"]], multiplicities=[case["multiplicity"]]
        ) as batch:
            record_property(
                "preparation_seconds", time.monotonic() - preparation_started
            )
            results = []
            for repeat in range(2):
                started = time.monotonic()
                result = batch.execute(properties=("energy", "forces")).items[0]
                record_property(
                    f"endpoint_seconds_{repeat}", time.monotonic() - started
                )
                detail = batch._library.generativeqc_context_get_last_detail(
                    batch._context
                )
                assert result.succeeded and result.converged, (
                    result.status_message,
                    detail,
                )
                assert result.executed_backend == "cuda" and result.iterations <= 200
                assert (
                    result.dispersion.backend == "cuda"
                    and np.isfinite(result.forces).all()
                )
                assert result.warm_start_used == bool(repeat)
                record_property(f"iterations_{repeat}", result.iterations)
                record_property(f"fock_builds_{repeat}", result.fock_builds)
                record_property(f"total_energy_hartree_{repeat}", result.energy)
                results.append(result)
            result = results[-1]
            with NativeAO(
                atoms,
                basis=load_basis(ASSETS / "basis.json"),
                representation="spherical",
                charge=case["charge"],
                multiplicity=case["multiplicity"],
            ) as basis:
                state = StationaryKsState.from_native(batch, basis)
                initial_density, transform = _matched_reference_seed(basis, state)
                energy, gradient, reference = independent_global_hybrid_gradient(
                    basis,
                    state,
                    "b3lyp-uks",
                    xc=PROTOCOL["electronic_xc"],
                    density_fitting=route == "df",
                    auxiliary_shells=load_basis(ASSETS / "auxbasis.json").shells_for(
                        basis.atoms,
                        backend="cuda",
                        operator="df_metric",
                        role="auxiliary",
                        consumer="scf_df",
                    ),
                    initial_density=initial_density,
                    scf_gradient_tolerance=3e-7,
                    return_mean_field=True,
                )
                independent_fock = np.asarray(
                    reference.get_fock(
                        dm=initial_density, vhf=reference.get_veff(dm=initial_density)
                    )
                ).reshape(state.fock.shape)
                inverse = np.linalg.inv(transform)
                fock_error = residual_error = 0.0
                for spin, block in enumerate(independent_fock):
                    native_fock = inverse.T @ block @ inverse
                    np.testing.assert_allclose(
                        native_fock, state.fock[spin], atol=5e-9, rtol=0
                    )
                    commutator = (
                        native_fock @ state.density[spin] @ state.overlap
                        - state.overlap @ state.density[spin] @ native_fock
                    )
                    assert np.max(np.abs(commutator)) < 1e-9
                    fock_error = max(
                        fock_error,
                        float(np.max(np.abs(native_fock - state.fock[spin]))),
                    )
                    residual_error = max(
                        residual_error, float(np.max(np.abs(commutator)))
                    )
                record_property("independent_fock_error", fock_error)
                record_property("independent_residual_ao_max", residual_error)
                dispersion = DispersionModel(
                    np.asarray(case["atomic_numbers"], dtype=np.int32), coordinates
                )
                dispersion.set_realspace_cutoff(1e5, 1e5, 1e5)
                damping = {
                    key: PROTOCOL["dispersion"][key]
                    for key in ("s6", "s8", "a1", "a2", "s9")
                }
                correction = dispersion.get_dispersion(
                    RationalDampingParam(**damping), grad=True
                )
                energy_error = abs(result.energy - energy - float(correction["energy"]))
                force_error = float(
                    np.max(np.abs(result.forces + gradient + correction["gradient"]))
                )
                record_property("energy_error_hartree", energy_error)
                record_property("force_error_hartree_bohr", force_error)
                record_property(
                    "reference_total_energy_hartree",
                    energy + float(correction["energy"]),
                )
                assert energy_error < 3e-7 and force_error < 1e-6
            np.testing.assert_allclose(
                results[0].forces, result.forces, atol=1e-7, rtol=0
            )
            assert abs(results[0].energy - result.energy) < 1e-9
            atom = len(atoms) - 1
            axis = int(np.argmax(np.abs(result.forces[atom])))
            for step in (2e-4, 1e-4):
                energies = []
                for sign in (1, -1):
                    moved = coordinates.copy()
                    moved[atom, axis] += sign * step
                    displaced = batch.execute(
                        coordinates=[moved], properties=("energy",), strict=True
                    ).items[0]
                    assert displaced.succeeded and displaced.converged
                    energies.append(displaced.energy)
                error = abs(
                    -(energies[0] - energies[1]) / (2 * step)
                    - result.forces[atom, axis]
                )
                record_property(f"finite_difference_error_{step}", error)
                assert error < 3e-6
    finally:
        calc.clear_cache()

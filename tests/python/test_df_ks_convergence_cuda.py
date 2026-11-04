"""Source-pinned #2984 stationary endpoints, with independent matched-root gates."""

import dataclasses
import json
import os
import time
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, load_basis
from generativeqc.batch import PreparedBatch
from generativeqc_compiler.method import (
    D3Spec,
    DispersionCorrectionPrimitive,
    resolve_method,
)

ASSETS = Path(__file__).resolve().parents[1] / "reference_data" / "ks-scf-2984"
CASES = json.loads((ASSETS / "cases.json").read_text())["cases"]
XC = json.loads((ASSETS / "provenance.json").read_text())["electronic_xc"]
BOHR_ANGSTROM = 0.529177210903
D3 = {"s6": 1.0, "s8": 1.9889, "a1": 0.3981, "a2": 4.4211, "s9": 0.0}


def test_issue_inputs_keep_the_requested_all_electron_spaces() -> None:
    """Retained inputs, not production function bodies, carry source identities."""
    orbital = load_basis(ASSETS / "basis.json")
    auxiliary = load_basis(ASSETS / "auxbasis.json")
    assert set(orbital.by_element) == {7, 8, 9, 13, 38}
    assert set(auxiliary.by_element) == {7, 8, 9, 13}
    assert orbital.by_element[38].ecp_core_electrons == 28
    for case in CASES:
        assert case["ecp_core_electrons"] == 0
        assert case["multiplicity"] in (1, 2)
        assert all(
            orbital.by_element[number].ecp_core_electrons == 0
            for number in case["atomic_numbers"]
        )


def _calculator(case: typing.Any) -> Calculator:
    """Keep the reported XC, integer Q/M, DF conditioning and strict SCF controls."""
    method = resolve_method(
        "B3LYP", spin="unpolarized" if case["multiplicity"] == 1 else "polarized"
    )
    method = dataclasses.replace(
        method,
        identifier="B3LYP-D3(BJ)",
        primitives=method.primitives
        + (
            DispersionCorrectionPrimitive(
                D3Spec(
                    **D3,
                    table_sha256="9ff932ea598f690c1fb599a67762060ba1907102d5ec132164f2a7e8886cd22e",
                    radii_sha256="92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111",
                )
            ),
        ),
    )
    return Calculator(
        method=method,
        basis=load_basis(ASSETS / "basis.json"),
        basis_representation="spherical",
        device="cuda",
        precision="fp64",
        initial_guess="auto",
        max_iterations=200,
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
        screening_tolerance=1e-12,
        diis_history=8,
        ks_options=KsOptions(grid=GridSpec.treutler(3)),
        density_fitting="cuda",
        auxiliary_basis=load_basis(ASSETS / "auxbasis.json"),
        density_fitting_relative_threshold=1e-10,
        density_fitting_memory_budget_bytes=4 << 30,
    )


def _matched_reference_seed(basis: typing.Any, state: typing.Any) -> typing.Any:
    """Transform native D, not its energy/Fock, into independent libcint AOs.

    O2+ has distinct stationary roots under the two cold guesses. Seeding the
    independent solver establishes a same-root comparison without prescribing
    a PySCF density to production. Samples surround each atom, including the
    untranslated AlF2 coordinates far from the origin.
    """
    from pyscf import gto
    from pyscf.data.elements import ELEMENTS

    labels = [
        f"{ELEMENTS[atom.atomic_number]}{index}"
        for index, atom in enumerate(basis.atoms)
    ]
    shells = {label: [] for label in labels}
    for shell in basis.shells:
        shells[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[
                    (primitive.exponent, primitive.coefficient)
                    for primitive in shell.primitives
                ],
            ]
        )
    molecule = gto.M(
        atom=[
            (label, atom.position)
            for label, atom in zip(labels, basis.atoms, strict=True)
        ],
        basis=shells,
        unit="Bohr",
        cart=False,
        charge=basis.charge,
        spin=basis.multiplicity - 1,
        verbose=0,
    )
    rng = np.random.default_rng(2984)
    samples = np.concatenate(
        [
            np.asarray(atom.position) + rng.normal(size=(basis.nao + 20, 3))
            for atom in basis.atoms
        ]
    )
    native_ao = basis.evaluate(samples, order=0)[0]
    reference_ao = molecule.eval_gto("GTOval_sph", samples)
    transform, _, rank, _ = np.linalg.lstsq(native_ao, reference_ao, rcond=1e-13)
    assert rank == basis.nao
    np.testing.assert_allclose(
        native_ao @ transform, reference_ao, atol=2e-11, rtol=2e-11
    )
    inverse = np.linalg.inv(transform)
    density = np.asarray([inverse @ spin @ inverse.T for spin in state.density])
    return (density[0] if basis.multiplicity == 1 else density), transform


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_KS_CONVERGENCE_CUDA_TEST") != "1",
    reason="explicit Slurm CUDA stationary-endpoint qualification",
)
@pytest.mark.parametrize("case", CASES, ids=[case["alias"] for case in CASES])
def test_source_pinned_df_b3lyp_stationary_endpoints(
    case: typing.Any, monkeypatch: pytest.MonkeyPatch, record_property: typing.Any
) -> None:
    """Cold/warm complete CUDA E/F, same-root oracle and two-step reconvergence."""
    pytest.importorskip("pyscf", reason="independent SCF and moving-grid force oracle")
    pytest.importorskip("dftd3", reason="independent complete D3(BJ) endpoint")
    from dftd3.interface import DispersionModel, RationalDampingParam
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import independent_global_hybrid_gradient

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("CUDA E/F called a Python scientific force fallback")

    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", forbidden)
    monkeypatch.setattr(PreparedBatch, "_public_dft_cuda_force", forbidden)
    coordinates = np.asarray(case["coordinates_angstrom"]) / BOHR_ANGSTROM
    atoms = list(zip(case["symbols"], coordinates, strict=True))
    started = time.monotonic()
    calc = _calculator(case)
    try:
        with calc.prepare_batch(
            [atoms], charges=[case["charge"]], multiplicities=[case["multiplicity"]]
        ) as batch:
            assert batch._native_dft_force_eligible()
            setup_seconds = time.monotonic() - started
            results = []
            endpoint_seconds = []
            for repeat in range(3):
                started = time.monotonic()
                result = batch.execute(properties=("energy", "forces")).items[0]
                endpoint_seconds.append(time.monotonic() - started)
                detail = batch._library.generativeqc_context_get_last_detail(
                    batch._context
                )
                assert result.succeeded and result.converged, (
                    result.status_message,
                    detail,
                )
                assert result.executed_backend == "cuda"
                assert result.dispersion.backend == "cuda"
                assert np.isfinite(result.forces).all()
                assert result.iterations <= 200
                results.append(result)
            if case["multiplicity"] == 1:
                assert results[0].initial_guess["outcome"] == "used"
                assert results[0].initial_guess["target_attempts"] == 1
            assert not results[0].warm_start_used
            assert all(result.warm_start_used for result in results[1:])
            record_property("setup_seconds", setup_seconds)
            record_property("endpoint_seconds", json.dumps(endpoint_seconds))
            record_property(
                "iterations", json.dumps([result.iterations for result in results])
            )
            record_property(
                "fock_builds",
                json.dumps([result.ks_diagnostic.fock_builds for result in results]),
            )
            record_property(
                "preliminary_guess",
                json.dumps([result.initial_guess for result in results]),
            )
            for result in results[1:]:
                assert result.energy == pytest.approx(results[0].energy, abs=2e-9)
                np.testing.assert_allclose(
                    result.forces, results[0].forces, atol=2e-7, rtol=0
                )
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
                auxiliary_shells = load_basis(ASSETS / "auxbasis.json").shells_for(
                    basis.atoms
                )
                started = time.monotonic()
                energy, gradient, reference = independent_global_hybrid_gradient(
                    basis,
                    state,
                    "b3lyp-rks" if case["multiplicity"] == 1 else "b3lyp-uks",
                    xc=XC,
                    density_fitting=True,
                    auxiliary_shells=auxiliary_shells,
                    initial_density=initial_density,
                    scf_gradient_tolerance=3e-7,
                    return_mean_field=True,
                )
                record_property(
                    "reference_endpoint_seconds", time.monotonic() - started
                )
                independent_fock = np.asarray(
                    reference.get_fock(
                        dm=initial_density,
                        vhf=reference.get_veff(dm=initial_density),
                    )
                ).reshape(state.fock.shape)
                inverse = np.linalg.inv(transform)
                physical_errors = []
                independent_residuals = []
                for spin, block in enumerate(independent_fock):
                    native_fock = inverse.T @ block @ inverse
                    physical_errors.append(
                        float(np.max(np.abs(native_fock - state.fock[spin])))
                    )
                    commutator = (
                        native_fock @ state.density[spin] @ state.overlap
                        - state.overlap @ state.density[spin] @ native_fock
                    )
                    independent_residuals.append(float(np.max(np.abs(commutator))))
                    assert independent_residuals[-1] < 1e-9
                assert max(physical_errors) < 5e-10
                record_property(
                    "independent_physical_fock_ao_max", max(physical_errors)
                )
                record_property(
                    "independent_stationarity_ao_max", max(independent_residuals)
                )
                reference_density = np.asarray(reference.make_rdm1()).reshape(
                    state.density.shape
                )
                for native_spin, reference_spin in zip(
                    state.density, reference_density, strict=True
                ):
                    np.testing.assert_allclose(
                        native_spin,
                        transform @ reference_spin @ transform.T,
                        atol=2e-5,
                        rtol=0,
                    )
                dispersion = DispersionModel(
                    np.asarray(case["atomic_numbers"], dtype=np.int32), coordinates
                )
                dispersion.set_realspace_cutoff(1e5, 1e5, 1e5)
                correction = dispersion.get_dispersion(
                    RationalDampingParam(**D3), grad=True
                )
                reference_energy = energy + float(correction["energy"])
                reference_force = -gradient - np.asarray(correction["gradient"])
                record_property(
                    "energy_error_hartree", abs(result.energy - reference_energy)
                )
                record_property(
                    "force_error_hartree_bohr",
                    float(np.max(np.abs(result.forces - reference_force))),
                )
                assert result.energy == pytest.approx(reference_energy, abs=2e-8)
                np.testing.assert_allclose(
                    result.forces, reference_force, atol=2e-6, rtol=0
                )
            np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-8, rtol=0)
            atom = len(atoms) - 1
            axis = int(np.argmax(np.abs(result.forces[atom])))
            errors = []
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
                finite_difference_force = -(energies[0] - energies[1]) / (2 * step)
                errors.append(abs(finite_difference_force - result.forces[atom, axis]))
                assert errors[-1] < 2e-5
            record_property("finite_difference_errors", json.dumps(errors))
    finally:
        calc.clear_cache()

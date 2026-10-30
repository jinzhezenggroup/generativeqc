"""Public shared-owner CUDA forces against independently reconverged Libxc.

These opt-in tests require the selected static force package and finite Slurm.
The oracle runs after each production endpoint, on its exported moving grid;
neither PySCF nor Python scientific consumers may contribute to the result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, ResourceBudget
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc.automatic_artifacts import read_automatic_artifact_diagnostic
from generativeqc.ks_diagnostics import read_ks_transport_diagnostic
from generativeqc_compiler.common.provenance import atomic_json, file_hash
from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks
from generativeqc_compiler.xc.libxc_bulk_capabilities import functional_capability
from test_dft_complete_cuda import no_cpu_derivatives
from test_libxc_automatic_cuda_ks import ATOMS, MOVED, _physical_state, _reference
from test_libxc_automatic_point_native import NAMES
from test_stationary_aot_cuda import _forbid_aot_generation

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_AUTOMATIC_CUDA_FORCE_TEST") != "1",
    reason="requires selected native CUDA force artifacts and a finite Slurm job",
)
GRID = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
PROPERTIES = ("energy", "forces")


def _oracle_basis() -> dict[str, Any]:
    """Read the pinned BSE inputs independently of the native basis compiler."""
    data = json.loads(
        (Path(__file__).parents[1] / "data/external_basis/sto-3g-ho.json").read_text()
    )
    return {
        symbol: [
            [
                angular,
                *[
                    [float(exponent), float(coefficient)]
                    for exponent, coefficient in zip(
                        shell["exponents"], coefficients, strict=True
                    )
                ],
            ]
            for shell in data["elements"][number]["electron_shells"]
            for angular, coefficients in zip(
                shell["angular_momentum"], shell["coefficients"], strict=True
            )
        ]
        for symbol, number in (("H", "1"), ("O", "8"))
    }


def _boundary_counts(
    atoms: Any, state: Any, charge: int, multiplicity: int
) -> dict[str, int]:
    """Measure physical tau boundaries independently on the exported native grid."""
    from pyscf import dft, gto

    molecule = gto.M(
        atom=atoms,
        basis=_oracle_basis(),
        unit="Bohr",
        cart=True,
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    ao = dft.numint.eval_ao(molecule, state["points"], deriv=1)
    near_iso = low_density = 0
    for density in state["density"]:
        features = dft.numint.eval_rho(
            molecule, ao, density, xctype="MGGA", with_lapl=False
        )
        rho, gradient, tau = features[0], features[1:4], features[4]
        low_density += int(np.count_nonzero((rho > 0) & (rho < 1e-12)))
        populated = rho > 1e-8
        tau_w = np.sum(gradient[:, populated] ** 2, axis=0) / (8 * rho[populated])
        near_iso += int(
            np.count_nonzero(
                np.abs(tau[populated] - tau_w)
                < 1e-10 * np.maximum(tau[populated], 1e-12)
            )
        )
    return {
        "positive_low_density_points": low_density,
        "near_iso_orbital_points": near_iso,
    }


def _calculator(name: str, spins: int, **overrides: Any) -> Calculator:
    """Freeze the scientific controls; the Hcore seed is independent of PySCF."""
    controls: dict[str, Any] = {
        "method": f"libxc-{'rks' if spins == 1 else 'uks'}:{name}",
        "basis": "sto-3g",
        "device": "cuda",
        "precision": "fp64",
        "initial_guess": None,
        "max_iterations": 300,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-12,
        "screening_tolerance": 1e-12,
        "ks_options": KsOptions(grid=GRID, tile_points=37),
    }
    controls.update(overrides)
    return Calculator(**controls)


def _lifecycle(
    name: str,
    spins: int,
    monkeypatch: pytest.MonkeyPatch,
    *,
    full_vector: bool,
    replay: bool,
    atoms: Any = ATOMS,
    moved: Any = MOVED,
    method_ir: bool = False,
    tag: str = "h4",
    charge: int | None = None,
    multiplicity: int | None = None,
) -> None:
    """Qualify host-return endpoints, plus native and independent energy FDs."""
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    started = perf_counter()
    charge = spins - 1 if charge is None else charge
    multiplicity = spins if multiplicity is None else multiplicity
    overrides = {}
    if method_ir:
        overrides["method"] = resolve_bulk_ks(
            name, spin="unpolarized" if spins == 1 else "polarized", backend="cuda"
        ).method
    probe = _calculator(name, spins, **overrides)
    assert probe.capabilities.supported_properties == frozenset(PROPERTIES)
    assert not probe.capabilities.supported_second_order
    capacity = probe.estimate_resources(
        [atoms], charges=[charge], multiplicities=[multiplicity], properties=PROPERTIES
    ).require_feasible()
    calculator = _calculator(
        name,
        spins,
        **overrides,
        resource_budget=ResourceBudget(
            host_bytes=capacity.peak_bytes["host"],
            device_bytes=capacity.peak_bytes["device"],
        ),
    )
    configured_seconds = perf_counter() - started
    directions = np.eye(3 * len(atoms)).reshape(-1, len(atoms), 3)
    if not full_vector:
        direction = np.zeros((len(atoms), 3))
        direction[-1] = (0.2, -0.3, 0.4)
        directions = (direction,)
    phases = (
        (("cold", atoms), ("warm", atoms), ("moved", moved))
        if replay
        else (("cold", atoms),)
    )
    rows = []
    started = perf_counter()
    with calculator.prepare_batch(
        [atoms],
        charges=[charge],
        multiplicities=[multiplicity],
        warm_start=True,
    ) as batch:
        prepared_seconds = perf_counter() - started
        previous = None
        for phase, geometry in phases:
            started = perf_counter()
            with monkeypatch.context() as guard, no_cpu_derivatives():
                _forbid_aot_generation(guard, batch)
                # Every force must pass native per-item capability. A Python
                # stationary implementation is not a fallback for this product.
                assert batch._native_dft_force_eligible(0)
                result = batch.execute(
                    coordinates=([position for _, position in geometry],),
                    properties=PROPERTIES,
                    strict=True,
                ).items[0]
            seconds = perf_counter() - started
            assert result.succeeded and result.converged
            assert result.executed_backend == "cuda"
            assert result.ks_diagnostic.physical_residual_max < 2e-11
            forces = np.asarray(result.forces)
            receipt = read_automatic_artifact_diagnostic(batch)
            assert receipt["force_completed"] and receipt["fallback_reason"] == "none"
            assert (
                receipt["point_artifact"]["sha256"]
                == receipt["force_artifact"]["sha256"]
            )
            assert receipt["spin_blocks"] == spins and receipt["source_count"] == 7
            assert receipt["capacity"]["minimum_device_bytes"] > 0
            assert receipt["stationary_work_v1"][3] > 0
            if rows:
                for key in (
                    "point_artifact",
                    "force_artifact",
                    "point_source_identity",
                    "force_source_identity",
                    "force_plan_identity",
                ):
                    assert receipt[key] == rows[0]["artifact"][key]
            assert forces.shape == (len(geometry), 3) and np.isfinite(forces).all()
            np.testing.assert_allclose(forces.sum(axis=0), 0, rtol=0, atol=2e-8)
            transport = read_ks_transport_diagnostic(
                batch._library, batch._batch, index=0
            )
            if phase == "cold":
                assert transport.matrix_d2h_bytes == 0
            if previous is not None:
                with pytest.raises(ValueError, match="stale"):
                    previous.check_current()
                previous.close()
            previous = NativeKsSnapshot(batch, 0)
            state = _physical_state(previous)
            boundary = None
            if name.startswith("MGGA_") and spins == 2 and tag == "h4":
                boundary = _boundary_counts(geometry, state, charge, multiplicity)
                assert boundary["positive_low_density_points"] > 0
                assert boundary["near_iso_orbital_points"] > 0
            reference, fixed, density, fock, directional_error = _reference(
                name,
                spins,
                geometry,
                state,
                basis=_oracle_basis(),
                charge=charge,
                multiplicity=multiplicity,
            )
            np.testing.assert_allclose(result.energy, reference, atol=3e-9, rtol=0)
            np.testing.assert_allclose(result.energy, fixed, atol=3e-10, rtol=0)
            np.testing.assert_allclose(
                state["density"],
                density.reshape(state["density"].shape),
                atol=2e-7,
                rtol=0,
            )
            np.testing.assert_allclose(
                state["fock"], fock.reshape(state["fock"].shape), atol=3e-9, rtol=0
            )
            assert directional_error < 5e-8
            native_fd, independent_fd, analytic = [], [], []
            for direction in directions:
                stencil = []
                for sign in (-1, 1):
                    displaced = [
                        (symbol, tuple(np.asarray(position) + sign * 1e-4 * shift))
                        for (symbol, position), shift in zip(
                            geometry, direction, strict=True
                        )
                    ]
                    # Restore the native center before each independently
                    # reconverged stencil; never seed with an oracle density.
                    restored = batch.execute(
                        coordinates=([p for _, p in geometry],),
                        properties=("energy",),
                        strict=True,
                    ).items[0]
                    assert restored.converged
                    energy_receipt = read_automatic_artifact_diagnostic(batch)
                    assert not energy_receipt["force_completed"]
                    assert energy_receipt["force_artifact"] == receipt["force_artifact"]
                    assert not any(energy_receipt["stationary_work_v1"])
                    endpoint = batch.execute(
                        coordinates=([p for _, p in displaced],),
                        properties=("energy",),
                        strict=True,
                    ).items[0]
                    assert endpoint.converged
                    snapshot = NativeKsSnapshot(batch, 0)
                    try:
                        expected = _reference(
                            name,
                            spins,
                            displaced,
                            _physical_state(snapshot),
                            basis=_oracle_basis(),
                            charge=charge,
                            multiplicity=multiplicity,
                        )[0]
                    finally:
                        snapshot.close()
                    np.testing.assert_allclose(
                        endpoint.energy, expected, atol=3e-9, rtol=0
                    )
                    stencil.append((endpoint.energy, expected))
                numerical = -(np.asarray(stencil[1]) - np.asarray(stencil[0])) / 2e-4
                contracted = float(np.sum(forces * direction))
                np.testing.assert_allclose(numerical, contracted, atol=3e-5, rtol=0)
                native_fd.append(float(numerical[0]))
                independent_fd.append(float(numerical[1]))
                analytic.append(contracted)
            rows.append(
                {
                    "phase": phase,
                    # Retain the measured gates, not just the test's pass flag:
                    # the offline catalog must recheck endpoint coverage.
                    "coordinates": [list(position) for _, position in geometry],
                    "energy": float(result.energy),
                    "reference_energy": float(reference),
                    "fixed_density_energy": float(fixed),
                    "density_error": float(
                        np.max(
                            np.abs(
                                state["density"]
                                - density.reshape(state["density"].shape)
                            )
                        )
                    ),
                    "fock_error": float(
                        np.max(
                            np.abs(state["fock"] - fock.reshape(state["fock"].shape))
                        )
                    ),
                    "directional_error": float(directional_error),
                    "converged": bool(result.converged),
                    "executed_backend": result.executed_backend,
                    "artifact": receipt,
                    "independent_boundary_counts": boundary,
                    "endpoint_seconds": seconds,
                    "complete_endpoint_seconds": seconds
                    + (configured_seconds + prepared_seconds if phase == "cold" else 0),
                    "forces": forces.tolist(),
                    "native_fd": native_fd,
                    "independent_fd": independent_fd,
                    "analytic_directions": analytic,
                    "transport": transport.to_payload(),
                    "scf": result.ks_diagnostic.to_payload(),
                    "semantic_work": result.precision,
                }
            )
            # The next warm force observes an ordinary center replay, not a
            # cached result or the final displaced stencil.
            assert (
                batch.execute(
                    coordinates=([p for _, p in geometry],),
                    properties=("energy",),
                    strict=True,
                )
                .items[0]
                .converged
            )
        if previous is not None:
            previous.close()
    if directory := os.environ.get("GENERATIVEQC_AUTOMATIC_CUDA_FORCE_EVIDENCE_DIR"):
        atomic_json(
            Path(directory)
            / f"{name}-{spins}-{full_vector}-{replay}-{method_ir}-{tag}.json",
            {
                "schema": "generativeqc.native-automatic-cuda-forces.v1",
                "name": name,
                "subject_identity": functional_capability(name).identity,
                "context": {
                    "precision": "fp64",
                    "basis": "sto-3g",
                    "basis_max_l": 0 if tag == "h4" else 1,
                    "integrals": "direct",
                    "ecp": False,
                    "grid": [24, 8, 16],
                    "partition_iterations": 3,
                    "tile_points": 37,
                    "fixture": tag,
                    "full_vector": full_vector,
                    "reference": "PySCF-2.14.0/Libxc-7.0.0",
                },
                "spins": spins,
                "charge": charge,
                "multiplicity": multiplicity,
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "native_library_sha256": file_hash(
                    Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
                ),
                "rows": rows,
            },
        )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("spins", (1, 2))
def test_native_automatic_cuda_force_full_vector(
    name: str, spins: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lifecycle(name, spins, monkeypatch, full_vector=True, replay=False)


@pytest.mark.parametrize("name", ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_X_LTA"))
@pytest.mark.parametrize("spins", (1, 2))
def test_native_automatic_cuda_force_replay(
    name: str, spins: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lifecycle(name, spins, monkeypatch, full_vector=False, replay=True, method_ir=True)


@pytest.mark.parametrize("name", ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_C_TPSS"))
@pytest.mark.parametrize("spins", (1, 2))
def test_native_automatic_cuda_force_sp_replay(
    name: str, spins: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-axis-aligned p AOs exercise AO/partition/tau response after a move.

    This fixture is neutral in both layouts. Open-shell H4 above supplies the
    independent spin-polarized force gate; correlation-only H2O+ from Hcore can
    fail SCF before any force is authorized.
    """
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.2, 1.5, 1.1)), ("H", (-0.3, -1.4, 1.0))]
    moved = [*atoms[:2], ("H", (-0.27, -1.4, 1.0))]
    _lifecycle(
        name,
        spins,
        monkeypatch,
        full_vector=False,
        replay=True,
        atoms=atoms,
        moved=moved,
        tag="water",
        charge=0,
        multiplicity=1,
    )


@pytest.mark.parametrize("spins", (1, 2))
def test_native_automatic_force_minimum_and_below_budget(spins: int) -> None:
    """The bounded geometry fallback must fit exactly and reject one byte less."""
    from dataclasses import replace

    from generativeqc.resources_planning import plan_resources

    name = "GGA_X_APBE"
    capacity = (
        _calculator(name, spins)
        .estimate_resources(
            [ATOMS], charges=[spins - 1], multiplicities=[spins], properties=PROPERTIES
        )
        .require_feasible()
    )
    request = capacity.requests[0]
    minimum = plan_resources(
        (replace(request, candidates=(request.candidates[-1],)),), ResourceBudget()
    ).require_feasible()
    limits = {
        "host_bytes": minimum.peak_bytes["host"],
        "device_bytes": minimum.peak_bytes["device"],
    }
    bounded = _calculator(name, spins, resource_budget=ResourceBudget(**limits))
    with bounded.prepare_batch(
        [ATOMS], charges=[spins - 1], multiplicities=[spins], warm_start=True
    ) as batch:
        first = batch.execute(properties=PROPERTIES, strict=True).items[0]
        second = batch.execute(properties=PROPERTIES, strict=True).items[0]
        np.testing.assert_allclose(first.forces, second.forces, atol=2e-9, rtol=0)
        receipt = read_automatic_artifact_diagnostic(batch)
        assert receipt["force_completed"]
        assert receipt["fallback_reason"] == "minimum-workspace"
    limits["device_bytes"] -= 1
    rejected = _calculator(name, spins, resource_budget=ResourceBudget(**limits))
    assert (
        rejected.estimate_resources(
            [ATOMS], charges=[spins - 1], multiplicities=[spins], properties=PROPERTIES
        ).status
        == "infeasible"
    )


def test_native_automatic_force_ragged_failure_and_recovery() -> None:
    """A failed neighbor cannot publish stale work or poison last-good replay."""
    water = [("O", (0.0, 0.0, 0.0)), ("H", (0.2, 1.5, 1.1)), ("H", (-0.3, -1.4, 1.0))]
    calculator = _calculator("LDA_C_VWN_4", 1)
    with calculator.prepare_batch([ATOMS, water], warm_start=True) as batch:
        baseline = batch.execute(properties=PROPERTIES, strict=True)
        invalid = np.asarray([point for _, point in water])
        invalid[0, 0] = np.nan
        failed = batch.execute(
            coordinates=([point for _, point in ATOMS], invalid), properties=PROPERTIES
        )
        assert failed.items[0].succeeded and not failed.items[1].succeeded
        assert failed.items[1].forces is None
        assert read_automatic_artifact_diagnostic(batch, 0)["force_completed"]
        with pytest.raises((RuntimeError, ValueError)):
            read_automatic_artifact_diagnostic(batch, 1)
        recovered = batch.execute(properties=PROPERTIES, strict=True)
        for first, last in zip(baseline.items, recovered.items, strict=True):
            np.testing.assert_allclose(first.forces, last.forces, atol=2e-9, rtol=0)
        assert all(
            read_automatic_artifact_diagnostic(batch, i)["force_completed"]
            for i in (0, 1)
        )
    with pytest.raises(RuntimeError, match="closed"):
        read_automatic_artifact_diagnostic(batch)


def test_native_automatic_nonconvergence_has_no_force_receipt() -> None:
    """An unfinished SCF publishes neither forces nor a warm seed/force receipt."""
    calculator = _calculator("MGGA_X_LTA", 1, max_iterations=1)
    with calculator.prepare_batch([ATOMS], warm_start=True) as batch:
        for _ in range(2):
            result = batch.execute(properties=PROPERTIES).items[0]
            assert not result.converged and not result.succeeded
            assert result.forces is None and not result.warm_start_used
            with pytest.raises((RuntimeError, ValueError)):
                read_automatic_artifact_diagnostic(batch)

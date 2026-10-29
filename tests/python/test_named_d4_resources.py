"""Native D4 correction inventory and complete budgeted KS ownership gates."""

from __future__ import annotations

import ctypes as ct
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    D4CorrectionBatch,
    GridSpec,
    ResourceBudget,
    _native,
    dispersion,
    resources_d4,
)
from generativeqc.ks import KsOptions
from generativeqc_compiler.method import resolve_method


@pytest.fixture(scope="module")
def library() -> Any:
    """Missing source-built inventory is a qualification failure."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_ks_d4_resource_query_v1")
    return loaded


def systems() -> list[Any]:
    return [
        [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
        [("O", (0, 0, 0)), ("H", (1.8, 0, 0)), ("H", (-0.45, 1.75, 0))],
    ]


def calculator(device: str = "cpu", **kwargs: Any) -> Calculator:
    return Calculator(
        method=resolve_method("PBE-D4(BJ-EEQ-ATM)"),
        device=device,
        ks_options=KsOptions(grid=GridSpec(12, 6, 12)),
        initial_guess=None,
        **kwargs,
    )


@pytest.mark.parametrize("counts", [(1,), (2, 3, 7), (256, 1)])
def test_query_matches_independent_prepared_cpu_owners(
    library: Any, counts: tuple[int, ...]
) -> None:
    rows = []
    for atoms in counts:
        xyz = np.zeros((atoms, 3))
        xyz[:, 0] = np.arange(atoms) * 2.0
        with D4CorrectionBatch(
            "PBE-D4(BJ-EEQ-ATM)", [(np.ones(atoms, dtype=np.int32), xyz, 0.0)]
        ) as owner:
            rows.append(owner.diagnostic())
    actual = resources_d4._ks_d4_inventory(counts, "cpu", library)
    for key in resources_d4._KEYS:
        assert actual[key] == sum(getattr(row, key) for row in rows)
    totals = [
        row.plan_host_bytes + row.execution_host_bytes + row.device_bytes
        for row in rows
    ]
    assert actual["retained_bytes"] == sum(totals)
    assert actual["minimum_maximum_bytes"] == len(counts) * max(totals)


@pytest.mark.parametrize("counts", [(1,), (2, 3, 7), (256, 1), (1,) * 4100])
def test_cuda_shape_query_requires_no_device(
    library: Any, counts: tuple[int, ...]
) -> None:
    """The CPU-only library must still size the bounded CUDA worker storage."""
    result = resources_d4._ks_d4_inventory(counts, "cuda", library)
    assert (
        result["device_bytes"] >= result["workspace_bytes"] + result["table_bytes"] > 0
    )
    single = [
        resources_d4._standalone_d4_inventory((n,), "cuda", library, 1 << 32)
        for n in counts
    ]
    for key in resources_d4._KEYS:
        assert result[key] == sum(row[key] for row in single)
    assert result["minimum_maximum_bytes"] == len(counts) * max(
        row["retained_bytes"] for row in single
    )


@pytest.mark.parametrize("backend,counts", [(2, [1]), (0, [0]), (1, [257])])
def test_invalid_public_query_is_transactional(
    library: Any, backend: int, counts: list[int]
) -> None:
    query = library.generativeqc_ks_d4_resource_query_v1
    query.argtypes = [
        ct.c_int32,
        ct.POINTER(ct.c_uint32),
        ct.c_uint32,
        ct.POINTER(_native.D4RuntimeDiagnostic),
    ]
    query.restype = ct.c_int
    output = _native.D4RuntimeDiagnostic(
        ct.sizeof(_native.D4RuntimeDiagnostic), _native.ABI_VERSION
    )
    output.device_bytes = 1234567
    before = bytes(output)
    assert (
        query(
            backend, (ct.c_uint32 * len(counts))(*counts), len(counts), ct.byref(output)
        )
        == _native.STATUS_INVALID_ARGUMENT
    )
    assert bytes(output) == before


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_budgeted_complete_d4_replay_failure_and_recovery(
    library: Any, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if device == "cuda":
        if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
            pytest.skip("requires a finite Slurm CUDA allocation")
        assert os.environ.get("SLURM_JOB_ID")
    calc = calculator(device)
    expected = [
        calc.singlepoint(atoms, properties=("energy", "forces")) for atoms in systems()
    ]
    plan = calc.estimate_resources(
        systems(), budget=ResourceBudget(host_bytes=2 << 30, device_bytes=2 << 30)
    )
    request = next(r for r in plan.requests if r.name == "d4")
    assert (
        json.loads(request.identity.schedule)["inventory_owner"]
        == "native-ks-d4-layout-v1"
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("budgeted native D4 endpoint created a Python correction owner")

    monkeypatch.setattr(dispersion, "D4CorrectionBatch", forbidden)
    with calc.prepare_batch(systems(), resource_plan=plan) as batch:
        assert batch._native_dispersion is not None
        if device == "cuda" and hasattr(
            library, "generativeqc_resource_ks_d4_optional_cuda_v1"
        ):
            # The optional candidate must actually select a native complete
            # promise. A larger declared budget alone is not an execution test.
            observation = batch._resource_ledger.native_observation()["device_ledger"]
            assert observation["admission"]["mode"] == "numeric-cap"
            assert observation["admission"]["capacity_reserved_bytes"] > 0
            assert observation["rejected_allocations"] == 0
        diagnostic = batch.dispersion_diagnostic
        inventory = resources_d4._ks_d4_inventory((2, 3), device, library)
        for key in resources_d4._KEYS:
            assert getattr(diagnostic, key) == inventory[key]
        for _ in range(2):
            result = batch.execute(properties=("energy", "forces"), strict=True)
            for actual, reference in zip(result.items, expected, strict=True):
                assert actual.energy == pytest.approx(reference.energy, abs=2e-8)
                np.testing.assert_allclose(
                    actual.forces, reference.forces, atol=2e-7, rtol=0
                )
        moved = np.array([p for _, p in systems()[0]], dtype=float)
        moved[1, 0] += 0.15
        moved_result = batch.execute(
            [moved, None], properties=("energy", "forces"), strict=True
        )
        assert abs(moved_result.items[0].energy - expected[0].energy) > 1e-6
        failed = batch.execute(
            [np.full((2, 3), np.nan), None], properties=("energy", "forces")
        )
        assert not failed.items[0].succeeded and failed.items[0].dispersion is None
        assert failed.items[1].succeeded
        recovered = batch.execute(
            [moved, None], properties=("energy", "forces"), strict=True
        )
        assert recovered.items[0].energy == pytest.approx(
            moved_result.items[0].energy, abs=2e-8
        )
        assert all(
            item.forces is None and item.dispersion.gradient is None
            for item in batch.execute(properties=("energy",), strict=True).items
        )
        if device == "cuda":
            assert (
                batch._resource_ledger.native_observation()["device_ledger"][
                    "live_bytes"
                ]
                >= inventory["device_bytes"]
            )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_equal_share_correction_budget_boundary(library: Any, device: str) -> None:
    if device == "cuda":
        if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
            pytest.skip("requires a finite Slurm CUDA allocation")
        assert os.environ.get("SLURM_JOB_ID")
    inventory = resources_d4._ks_d4_inventory((2, 3), device, library)
    required = inventory["minimum_maximum_bytes"]
    assert required > inventory["retained_bytes"]
    for delta in (-1, 0):
        calc = calculator(device, dispersion_memory_budget_bytes=required + delta)
        plan = calc.estimate_resources(systems())
        assert (plan.status == "feasible") == (delta == 0)
        if delta == 0:
            with calc.prepare_batch(systems(), resource_plan=plan) as batch:
                assert all(
                    item.succeeded
                    for item in batch.execute(properties=("energy",)).items
                )
        else:
            with pytest.raises(RuntimeError):
                calc.prepare_batch(systems())


def test_missing_inventory_refuses_global_plan_but_preserves_ordinary_d4(
    library: Any,
) -> None:
    class Legacy:
        def __getattr__(self, name: str) -> Any:
            if name == "generativeqc_ks_d4_resource_query_v1":
                raise AttributeError(name)
            return getattr(library, name)

    calc = calculator()
    calc._library = Legacy()
    with pytest.raises(NotImplementedError, match="native KS correction capacity"):
        calc.estimate_resources(systems())
    with calc.prepare_batch(systems()) as batch:
        assert all(
            item.succeeded for item in batch.execute(properties=("energy",)).items
        )


def test_native_preparation_refusal_does_not_retry_corrections(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calc = calculator()
    plan = calc.estimate_resources(systems())

    def refuse(*args: Any) -> int:
        return _native.STATUS_OUT_OF_MEMORY

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("selected native refusal retried Python D4")

    monkeypatch.setattr(calc._library, "generativeqc_batch_prepare_named_ks_v1", refuse)
    monkeypatch.setattr(dispersion, "D4CorrectionBatch", forbidden)
    with pytest.raises(MemoryError):
        calc.prepare_batch(systems(), resource_plan=plan)


def test_composite_gcp_remains_outside_global_inventory() -> None:
    with pytest.raises(NotImplementedError, match="gCP"):
        Calculator(
            method=resolve_method("R2SCAN-3c"),
            resource_budget=ResourceBudget(host_bytes=1 << 30),
        )


@pytest.mark.parametrize("optional_protocol", [True, False])
def test_larger_cuda_d4_fleet_keeps_bounded_optional_fallback(
    library: Any,
    optional_protocol: bool,
) -> None:
    """Exercise a larger grid and provider owner, plus the old-image fallback."""
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires a finite Slurm CUDA allocation")
    assert os.environ.get("SLURM_JOB_ID")
    waters = [
        (symbol, (xyz[0] + 8 * copy, xyz[1], xyz[2]))
        for copy in range(4)
        for symbol, xyz in systems()[1]
    ]
    fleet = [systems()[0], waters]
    calc = calculator(
        "cuda", basis="def2-svp", energy_tolerance=1e-12, density_tolerance=1e-10
    )
    with calc.prepare_batch(fleet) as baseline:
        reference = baseline.execute(properties=("energy", "forces"), strict=True)

    class WithoutOptionalProtocol:
        def __getattr__(self, name: str) -> Any:
            if name == "generativeqc_resource_ks_d4_optional_cuda_v1":
                raise AttributeError(name)
            return getattr(library, name)

    if not optional_protocol:
        calc._library = WithoutOptionalProtocol()
    # Four d-shell waters have a declared host setup bound above 2 GiB.
    # Keep the original endpoint campaign's smaller-fixture caps unchanged.
    plan = calc.estimate_resources(
        fleet, budget=ResourceBudget(host_bytes=4 << 30, device_bytes=2 << 30)
    )
    with calc.prepare_batch(fleet, resource_plan=plan) as batch:
        for _ in range(2):
            result = batch.execute(properties=("energy", "forces"), strict=True)
            for actual, expected in zip(result.items, reference.items, strict=True):
                assert actual.energy == pytest.approx(expected.energy, abs=2e-8)
                np.testing.assert_allclose(
                    actual.forces, expected.forces, atol=2e-7, rtol=0
                )
        observation = batch._resource_ledger.native_observation()["device_ledger"]
        assert observation["rejected_allocations"] == 0
        assert observation["admission"]["mode"] == (
            "numeric-cap" if optional_protocol else "selected-inventory"
        )
        assert (
            observation["admission"]["capacity_reserved_bytes"] > 0
        ) == optional_protocol
        assert (
            observation["admission"]["combined_peak_bytes"] <= plan.peak_bytes["device"]
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_global_exact_and_below_capacity_gate(library: Any, device: str) -> None:
    from generativeqc.resources_planning import plan_resources

    if device == "cuda":
        if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
            pytest.skip("requires a finite Slurm CUDA allocation")
        assert os.environ.get("SLURM_JOB_ID")
    from itertools import product

    from generativeqc import ResourcePlan

    calc = calculator(device)
    full = calc.estimate_resources(systems())
    space = "device" if device == "cuda" else "host"
    # CUDA may drop optional electronic caches below the preferred plan's
    # peak. Independently enumerate the declared alternatives to test the
    # mandatory floor, without treating that valid fallback as a failure.
    required = min(
        ResourcePlan(
            ResourceBudget(),
            full.requests,
            tuple(
                (request.name, candidate.name)
                for request, candidate in zip(full.requests, combination, strict=True)
            ),
            "feasible",
        ).peak_bytes[space]
        for combination in product(*(request.candidates for request in full.requests))
    )
    for delta in (-1, 0):
        plan = plan_resources(
            full.requests,
            ResourceBudget(
                **{
                    "device_bytes" if device == "cuda" else "host_bytes": required
                    + delta
                }
            ),
            library=calc._library,
        )
        assert (plan.status == "feasible") == (delta == 0)
        if delta == 0:
            with calc.prepare_batch(systems(), resource_plan=plan) as batch:
                if device == "cuda":
                    observation = batch._resource_ledger.native_observation()[
                        "device_ledger"
                    ]
                    assert observation["admission"]["mode"] == "selected-inventory"
                assert all(
                    item.succeeded
                    for item in batch.execute(properties=("energy",)).items
                )


def test_cpu_compatibility_gradient_owner_has_reserved_capacity(library: Any) -> None:
    """A native energy owner cannot hide the lifetime of an optional force peer."""
    calc = calculator()
    plan = calc.estimate_resources(systems())
    correction = next(r for r in plan.requests if r.name == "d4")
    allowance = next(
        row.bytes
        for row in correction.candidates[0].estimates
        if row.name == "D4 CPU compatibility force owner allowance"
    )
    with calc.prepare_batch(systems(), resource_plan=plan) as batch:
        full = batch.execute(properties=("energy", "forces"), strict=True)
        with batch._native_dispersion.compatibility_owner() as compatibility:
            diagnostic = compatibility.diagnostic()
            assert (
                diagnostic.plan_host_bytes + diagnostic.execution_host_bytes
                == allowance
            )
            components = compatibility.execute()
            for item, component in zip(full.items, components, strict=True):
                assert item.dispersion.energy == pytest.approx(
                    component.energy, abs=2e-14
                )
                np.testing.assert_allclose(
                    item.dispersion.gradient, component.gradient, atol=3e-13, rtol=0
                )


@pytest.mark.parametrize("properties", [("energy",), ("energy", "forces")])
def test_resource_budget_constructor_uses_native_d4(
    library: Any, properties: tuple[str, ...]
) -> None:
    """Scalar preparation must retain the complete correction budget boundary."""
    from itertools import product

    from generativeqc import ResourcePlan

    atoms = systems()[0]
    plain = calculator()
    reference = plain.singlepoint(atoms, properties=properties)
    full = plain.estimate_resources([atoms], properties=properties)
    required = min(
        ResourcePlan(
            ResourceBudget(),
            full.requests,
            tuple(
                (request.name, candidate.name)
                for request, candidate in zip(full.requests, combination, strict=True)
            ),
            "feasible",
        ).peak_bytes["host"]
        for combination in product(*(request.candidates for request in full.requests))
    )
    for delta in (-1, 0):
        calc = calculator(resource_budget=ResourceBudget(host_bytes=required + delta))
        if delta < 0:
            with pytest.raises(MemoryError, match="no supported plan fits"):
                calc.singlepoint(atoms, properties=properties)
            continue
        result = calc.singlepoint(atoms, properties=properties)
        assert result.converged and result.dispersion is not None
        assert result.energy == pytest.approx(reference.energy, abs=2e-9)
        if "forces" in properties:
            np.testing.assert_allclose(result.forces, reference.forces, atol=2e-8)
        else:
            assert result.forces is None and result.dispersion.gradient is None


def test_standalone_cuda_budget_preserves_bounded_worker_selection(
    library: Any,
) -> None:
    counts = (2, 3, 7)
    full = resources_d4._standalone_d4_inventory(counts, "cuda", library, 1 << 30)
    assert full["workspace_slots"] == 3
    smaller = resources_d4._standalone_d4_inventory(
        counts, "cuda", library, full["retained_bytes"] - 1
    )
    assert smaller["workspace_slots"] == 2
    assert smaller["retained_bytes"] < full["retained_bytes"]
    exact = resources_d4._standalone_d4_inventory(
        counts, "cuda", library, smaller["retained_bytes"]
    )
    assert exact == smaller
    minimum = resources_d4._standalone_d4_inventory(
        counts, "cuda", library, smaller["retained_bytes"] - 1
    )
    assert minimum["workspace_slots"] == 1
    with pytest.raises(ValueError, match="rejected"):
        resources_d4._standalone_d4_inventory(
            counts, "cuda", library, minimum["retained_bytes"] - 1
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_budgeted_endpoint_against_independent_pyscf_and_d4(
    library: Any, device: str
) -> None:
    """Compare complete E/F to PySCF differences plus pinned dftd4 EEQ data.

    Quadrature is the declared moving discrete grid; the independent engine
    owns SCF, integrals and XC. No production derivative participates in its
    finite-difference electronic force. This is not a grid-limit claim.
    """
    if device == "cuda":
        if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
            pytest.skip("requires a Slurm-allocated CUDA device")
        assert os.environ.get("SLURM_JOB_ID")
    pyscf = pytest.importorskip("pyscf")
    from generativeqc import Atom
    from generativeqc_compiler.dft.grid import MolecularGrid
    from pyscf import dft, gto
    from pyscf.data.elements import ELEMENTS

    pyscf.lib.num_threads(1)
    # The electronic reference uses a stable water geometry; the standalone
    # CNOH correction fixture remains covered by the existing D4 oracle tests.
    case = json.loads(
        (
            Path(__file__).parents[1] / "data/d4/pbe_water_resource_reference.json"
        ).read_text()
    )
    atoms = tuple(
        Atom(z, tuple(x))
        for z, x in zip(case["numbers"], case["positions"], strict=True)
    )
    calc = calculator(
        device, resource_budget=ResourceBudget(host_bytes=2 << 30, device_bytes=2 << 30)
    )
    result = calc.singlepoint(atoms, properties=("energy", "forces"))
    labels = [
        f"{ELEMENTS[atom.atomic_number]}{index}" for index, atom in enumerate(atoms)
    ]
    basis: dict[str, Any] = {label: [] for label in labels}
    for shell in calc._shells_for_atoms(atoms):
        basis[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )

    def energy(geometry: Any) -> float:
        molecule = gto.M(
            atom=list(zip(labels, geometry, strict=True)),
            basis=basis,
            unit="Bohr",
            cart=True,
            verbose=0,
        )
        moved = tuple(
            Atom(atom.atomic_number, tuple(x))
            for atom, x in zip(atoms, geometry, strict=True)
        )
        grid = MolecularGrid(moved, spec=calc.ks_options.grid).explicit()
        reference = dft.RKS(molecule)
        reference.xc = "PBE"
        reference.grids.coords = np.asarray(grid.points)
        reference.grids.weights = np.asarray(grid.weights)
        reference.small_rho_cutoff = 0
        reference.conv_tol, reference.conv_tol_grad = 1e-13, 1e-10
        reference.kernel()
        assert reference.converged
        return float(reference.e_tot)

    xyz = np.asarray(case["positions"], dtype=float)
    assert result.energy == pytest.approx(energy(xyz) + case["energy"], abs=2e-8)
    step = 2e-4
    forces = np.empty_like(xyz)
    for index in np.ndindex(xyz.shape):
        plus, minus = xyz.copy(), xyz.copy()
        plus[index] += step
        minus[index] -= step
        forces[index] = -(energy(plus) - energy(minus)) / (2 * step)
    forces -= np.asarray(case["gradient"])
    np.testing.assert_allclose(result.forces, forces, atol=3e-6, rtol=0)
    assert result.dispersion.energy == pytest.approx(case["energy"], abs=2e-14)
    np.testing.assert_allclose(
        result.dispersion.gradient, case["gradient"], atol=3e-13, rtol=0
    )

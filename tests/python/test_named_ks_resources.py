"""Complete native corrected endpoints share their actual resource ownership."""

import ctypes as ct
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    ResourceBudget,
    _native,
    dispersion,
    resources_d3,
)
from generativeqc.ks import KsOptions
from generativeqc_compiler.method import resolve_method


@pytest.fixture(scope="module")
def library() -> Any:
    """Require the source-built query; missing ABI is a qualification failure."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_ks_d3_resource_query_v1")
    return loaded


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("counts", [(1,), (2, 3, 7), (4096, 1), (1,) * 4100])
def test_fleet_capacity_matches_independent_single_owner_layouts(
    library: Any, backend: str, counts: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shape-only CUDA metadata requires no GPU or driver initialization."""
    items = [resources_d3._portable_d3_inventory((n,), backend) for n in counts]
    expected = {key: sum(item[key] for item in items) for key in items[0]}
    expected["minimum_maximum_bytes"] = len(counts) * max(
        item["retained_bytes"] for item in items
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("selected KS inventory delegated sizing to the standalone owner")

    monkeypatch.setattr(resources_d3, "_portable_d3_inventory", forbidden)
    monkeypatch.setattr(resources_d3, "_d3_inventory", forbidden)
    assert resources_d3._ks_d3_inventory(counts, backend, library) == expected


@pytest.mark.parametrize("backend,counts", [(2, [1]), (0, [0]), (1, [4097])])
def test_invalid_query_leaves_receipt_unchanged(
    library: Any, backend: int, counts: list[int]
) -> None:
    query = library.generativeqc_ks_d3_resource_query_v1
    query.argtypes = [
        ct.c_int32,
        ct.POINTER(ct.c_uint32),
        ct.c_uint32,
        ct.POINTER(_native.D3RuntimeDiagnostic),
    ]
    query.restype = ct.c_int
    out = _native.D3RuntimeDiagnostic(
        ct.sizeof(_native.D3RuntimeDiagnostic), _native.ABI_VERSION
    )
    out.device_bytes = 1234567
    before = bytes(out)
    assert (
        query(backend, (ct.c_uint32 * len(counts))(*counts), len(counts), ct.byref(out))
        == _native.STATUS_INVALID_ARGUMENT
    )
    assert bytes(out) == before


def _systems() -> list[Any]:
    return [
        [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
        [("O", (0, 0, 0)), ("H", (1.8, 0, 0)), ("H", (-0.45, 1.75, 0))],
    ]


def _calculator(device: str = "cpu", **kwargs: Any) -> Calculator:
    return Calculator(
        method=resolve_method("PBE-D3(BJ)"),
        device=device,
        ks_options=KsOptions(grid=GridSpec(12, 6, 12)),
        initial_guess=None,
        **kwargs,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_budgeted_complete_program_replay_failure_and_recovery(
    library: Any, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if device == "cuda":
        if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
            pytest.skip("requires a Slurm-allocated CUDA device")
        assert os.environ.get("SLURM_JOB_ID")
    systems = _systems()
    calculator = _calculator(device)
    # The existing complete unbudgeted endpoint is the ownership parity oracle;
    # independent D3/scientific reference gates run separately.
    expected = [
        calculator.singlepoint(atoms, properties=("energy", "forces"))
        for atoms in systems
    ]
    plan = calculator.estimate_resources(
        systems, budget=ResourceBudget(host_bytes=2 << 30, device_bytes=2 << 30)
    )
    correction = next(r for r in plan.requests if r.name == "d3")
    assert (
        json.loads(correction.identity.schedule)["inventory_owner"]
        == "native-ks-d3-layout-v1"
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("budgeted complete endpoint created a Python correction owner")

    monkeypatch.setattr(dispersion, "D3CorrectionBatch", forbidden)
    with calculator.prepare_batch(systems, resource_plan=plan) as batch:
        assert batch._native_dispersion is not None
        diagnostic = batch.dispersion_diagnostic
        inventory = resources_d3._ks_d3_inventory((2, 3), device, library)
        for key in resources_d3._D3_RESOURCE_KEYS[:-1]:
            assert getattr(diagnostic, key) == inventory[key]
        for _ in range(2):
            result = batch.execute(properties=("energy", "forces"), strict=True)
            for item, reference in zip(result.items, expected, strict=True):
                assert item.energy == pytest.approx(reference.energy, abs=2e-8)
                np.testing.assert_allclose(
                    item.forces, reference.forces, atol=2e-7, rtol=0
                )
        moved = np.array([position for _, position in systems[0]], dtype=float)
        moved[1, 0] += 0.15
        result = batch.execute(
            [moved, None], properties=("energy", "forces"), strict=True
        )
        moved_energy = result.items[0].energy
        assert abs(moved_energy - expected[0].energy) > 1e-6
        failed = batch.execute(
            [np.full((2, 3), np.nan), None], properties=("energy", "forces")
        )
        assert not failed.items[0].succeeded and failed.items[0].dispersion is None
        assert failed.items[1].succeeded
        recovered = batch.execute(
            [moved, None], properties=("energy", "forces"), strict=True
        )
        assert recovered.items[0].energy == pytest.approx(moved_energy, abs=2e-8)
        energy_only = batch.execute(properties=("energy",), strict=True)
        assert all(
            item.forces is None and item.dispersion.gradient is None
            for item in energy_only.items
        )
        if device == "cuda":
            receipt = batch._resource_ledger.native_observation()["device_ledger"]
            assert receipt["live_bytes"] >= inventory["device_bytes"]


def test_equal_share_correction_cap_and_global_capacity(library: Any) -> None:
    inventory = resources_d3._ks_d3_inventory((2, 3), "cpu", library)
    required = inventory["minimum_maximum_bytes"]
    assert required > inventory["retained_bytes"]
    for delta in (-1, 0):
        calculator = _calculator(dispersion_memory_budget_bytes=required + delta)
        plan = calculator.estimate_resources(_systems())
        assert (plan.status == "feasible") == (delta == 0)
        if delta == 0:
            with calculator.prepare_batch(_systems(), resource_plan=plan) as batch:
                assert all(
                    item.succeeded
                    for item in batch.execute(properties=("energy",)).items
                )
        else:
            with pytest.raises(RuntimeError):
                calculator.prepare_batch(_systems())


def test_missing_inventory_retains_the_existing_standalone_route(library: Any) -> None:
    """Compatibility is selected before planning, never after native rejection."""

    class Legacy:
        def __getattr__(self, name: str) -> Any:
            if name == "generativeqc_ks_d3_resource_query_v1":
                raise AttributeError(name)
            return getattr(library, name)

    calculator = _calculator()
    calculator._library = Legacy()
    plan = calculator.estimate_resources(_systems())
    correction = next(r for r in plan.requests if r.name == "d3")
    assert (
        json.loads(correction.identity.schedule)["inventory_owner"]
        == "native-d3-layout-v1"
    )
    with calculator.prepare_batch(_systems(), resource_plan=plan) as batch:
        assert batch._native_dispersion is None
        assert batch._dispersion_batch is not None
        assert all(
            item.succeeded
            for item in batch.execute(properties=("energy", "forces")).items
        )


def test_selected_preparation_failure_never_retries_external_corrections(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calculator = _calculator()
    plan = calculator.estimate_resources(_systems())

    def refuse(*args: Any) -> int:
        return _native.STATUS_OUT_OF_MEMORY

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("selected native refusal retried Python correction execution")

    monkeypatch.setattr(
        calculator._library, "generativeqc_batch_prepare_named_ks_v1", refuse
    )
    monkeypatch.setattr(dispersion, "D3CorrectionBatch", forbidden)
    with pytest.raises(MemoryError):
        calculator.prepare_batch(_systems(), resource_plan=plan)


@pytest.mark.parametrize("method", ["PBE-D3(0)", "PBE-D3(BJ)-ATM", "PBE0-D3(BJ)"])
@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_resource_program_preserves_named_spin_and_correction_variants(
    library: Any, method: str, spin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared native catalog, not a frontend name list, selects the science."""
    calculator = Calculator(
        method=resolve_method(method, spin=spin),
        ks_options=KsOptions(grid=GridSpec(12, 6, 12)),
        initial_guess=None,
    )
    systems = _systems()
    charges = [0, 0] if spin == "unpolarized" else [1, 1]
    multiplicities = [1, 1] if spin == "unpolarized" else [2, 2]
    arguments = {"charges": charges, "multiplicities": multiplicities}
    with calculator.prepare_batch(systems, **arguments) as batch:
        expected = batch.execute(properties=("energy", "forces"), strict=True)
    plan = calculator.estimate_resources(systems, **arguments)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("budgeted corrected program used Python correction execution")

    monkeypatch.setattr(dispersion, "D3CorrectionBatch", forbidden)
    with calculator.prepare_batch(systems, resource_plan=plan, **arguments) as batch:
        observed = batch.execute(properties=("energy", "forces"), strict=True)
    for actual, reference in zip(observed.items, expected.items, strict=True):
        assert actual.energy == pytest.approx(reference.energy, abs=2e-11)
        np.testing.assert_allclose(actual.forces, reference.forces, atol=3e-10, rtol=0)
        assert (
            actual.dispersion.correction_identity
            == reference.dispersion.correction_identity
        )


def test_global_exact_and_below_capacity_gate(library: Any) -> None:
    from generativeqc.resources_planning import plan_resources

    calculator = _calculator()
    full = calculator.estimate_resources(_systems())
    for delta in (-1, 0):
        plan = plan_resources(
            full.requests,
            ResourceBudget(host_bytes=full.peak_bytes["host"] + delta),
            library=calculator._library,
        )
        assert (plan.status == "feasible") == (delta == 0)
        if delta == 0:
            with calculator.prepare_batch(_systems(), resource_plan=plan) as batch:
                assert all(
                    item.succeeded
                    for item in batch.execute(properties=("energy",)).items
                )


def test_cpu_compatibility_gradient_owner_has_reserved_capacity(library: Any) -> None:
    """A native energy owner cannot hide the lifetime of an optional force peer."""
    calculator = _calculator()
    plan = calculator.estimate_resources(_systems())
    correction = next(r for r in plan.requests if r.name == "d3")
    allowance = next(
        row.bytes
        for row in correction.candidates[0].estimates
        if row.name == "D3 CPU compatibility force owner allowance"
    )
    with calculator.prepare_batch(_systems(), resource_plan=plan) as batch:
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_budgeted_endpoint_against_independent_pyscf_and_d3(
    library: Any, device: str
) -> None:
    """Compare complete E/F to PySCF differences plus pinned simple-dftd3 data.

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
    cases = json.loads(
        (Path(__file__).parents[1] / "data/d3_bj_reference.json").read_text()
    )["fixtures"]
    case = next(
        case for case in cases if case["name"] == "pbe-bj-two-body/water-asymmetric"
    )
    atoms = tuple(
        Atom(z, tuple(x))
        for z, x in zip(case["numbers"], case["positions"], strict=True)
    )
    calculator = _calculator(
        device, resource_budget=ResourceBudget(host_bytes=2 << 30, device_bytes=2 << 30)
    )
    result = calculator.singlepoint(atoms, properties=("energy", "forces"))
    labels = [
        f"{ELEMENTS[atom.atomic_number]}{index}" for index, atom in enumerate(atoms)
    ]
    basis: dict[str, Any] = {label: [] for label in labels}
    for shell in calculator._shells_for_atoms(atoms):
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
        grid = MolecularGrid(moved, spec=calculator.ks_options.grid).explicit()
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

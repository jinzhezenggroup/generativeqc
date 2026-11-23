"""Public DF-KS energy endpoints against independently converged PySCF states."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from generativeqc import (
    Atom,
    Calculator,
    HostResources,
    KsOptions,
    ResourceAllocationError,
)
from generativeqc_compiler.dft.grid import GridSpec, MolecularGrid


@pytest.fixture(params=("cpu", "cuda"))
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "cuda" and os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an allocated native CUDA library/device")
    return request.param


def calculator(device: str, method: str = "pbe-rks", **kwargs: Any) -> Calculator:
    """Use tight physical convergence and an explicitly selected auxiliary basis."""
    if method.startswith(("pbe0", "b3lyp")):
        kwargs.setdefault("ks_options", KsOptions(grid=GridSpec()))
    return Calculator(
        method=method,
        basis="sto-3g",
        device=device,
        density_fitting="auto",
        auxiliary_basis="def2-svp",
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        **kwargs,
    )


WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]


def test_cpu_df_shared_host_leases() -> None:
    """Independent Python contexts share native retained and scratch capacity."""
    h2 = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calc = Calculator(
        method="pbe-rks",
        device="cpu",
        density_fitting="cpu",
        initial_guess=None,
        ks_options=KsOptions(grid=GridSpec.treutler(0)),
    )
    energy = ("energy",)
    forces = ("energy", "forces")
    energy_plan = calc.estimate_resources([h2], properties=energy).require_feasible()
    force_plan = calc.estimate_resources([WATER], properties=forces).require_feasible()
    resident = [plan.resident_bytes["host"] for plan in (energy_plan, force_plan)]
    workspace = [
        plan.peak_bytes["host"] - held
        for plan, held in zip((energy_plan, force_plan), resident, strict=True)
    ]
    assert min(resident) > 0 and max(workspace) > 0
    capacity = sum(resident) + max(workspace)
    h2_options = {"resource_plan": energy_plan, "ks_export": False, "warm_start": False}
    water_options = {"resource_plan": force_plan, "warm_start": False}
    with HostResources(capacity - 1) as short:
        with calc.prepare_batch([h2], host_resources=short, **h2_options):
            busy = short.snapshot()
            assert (busy.retained_bytes, busy.workspace_bytes, busy.owner_count) == (
                resident[0],
                workspace[0],
                1,
            )
            with pytest.raises(ResourceAllocationError):
                calc.prepare_batch([WATER], host_resources=short, **water_options)
            assert short.snapshot() == busy
        assert short.snapshot().owner_count == 0
        with calc.prepare_batch([WATER], host_resources=short, **water_options):
            assert short.snapshot().owner_count == 1

    with calc.prepare_batch([WATER], **water_options) as reference:
        expected = reference.execute(properties=forces, strict=True).items[0]
    with (
        HostResources(capacity) as pool,
        calc.prepare_batch([h2], host_resources=pool, **h2_options) as first,
        calc.prepare_batch([WATER], host_resources=pool, **water_options) as second,
    ):
        snapshot = pool.snapshot()
        assert (
            snapshot.maximum_bytes,
            snapshot.retained_bytes,
            snapshot.workspace_bytes,
        ) == (capacity, sum(resident), max(workspace))
        assert snapshot.owner_count == 2
        pool.close()
        with ThreadPoolExecutor(max_workers=2) as workers:
            h2_run = workers.submit(first.execute, properties=energy, strict=True)
            water_run = workers.submit(second.execute, properties=forces, strict=True)
            assert h2_run.result().items[0].converged
            actual = water_run.result().items[0]
        assert (actual.energy, actual.iterations, actual.fock_builds) == (
            expected.energy,
            expected.iterations,
            expected.fock_builds,
        )
        np.testing.assert_array_equal(actual.forces, expected.forces)


@pytest.mark.parametrize("explicit_auxiliary", [True, False])
def test_cpu_df_energy_host_budget(
    explicit_auxiliary: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind a single fitted owner or a heterogeneous fleet to its exact cap."""
    from generativeqc_compiler.common.resources import ResourceBudget

    calc = Calculator(
        method="pbe-rks",
        device="cpu",
        basis="def2-svp" if explicit_auxiliary else "sto-3g",
        basis_representation="spherical" if explicit_auxiliary else "cartesian",
        density_fitting="cpu",
        auxiliary_basis="def2-tzvp" if explicit_auxiliary else None,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        initial_guess=None,
    )
    systems = (
        [WATER]
        if explicit_auxiliary
        else [WATER, [("H", (0, 0, 0)), ("H", (0, 0, 1.4))]]
    )
    properties = ("energy",)
    quote = calc.estimate_resources(systems, properties=properties).require_feasible()
    peak = quote.peak_bytes["host"]
    plans = [
        calc.estimate_resources(
            systems,
            properties=properties,
            budget=ResourceBudget(host_bytes=peak + delta),
        )
        for delta in (0, -1)
    ]
    exact, short = plans
    assert exact.status == "feasible" and short.status == "infeasible"
    with calc.prepare_batch(systems, ks_export=False) as baseline:
        expected = baseline.execute(properties=properties, strict=True).items
    with calc.prepare_batch(systems, ks_export=False, resource_plan=exact) as batch:
        assert batch.basis_metadata == tuple(
            calc.basis_metadata(atoms) for atoms in systems
        )
        actual = batch.execute(properties=properties, strict=True).items
        with pytest.raises(NotImplementedError, match="energy only"):
            batch.execute(properties=("energy", "forces"))
        with monkeypatch.context() as environment:
            environment.setenv("GENERATIVEQC_CPU_DF_MATERIALIZED_DERIVATIVES", "1")
            with pytest.raises(
                NotImplementedError, match="materialized DF derivatives"
            ):
                batch.execute(properties=properties)
    for atoms, before, after in zip(systems, expected, actual, strict=True):
        assert after.fock_builds == before.fock_builds and after.fock_builds > 0
        assert after.energy == pytest.approx(before.energy, abs=1e-12)
        assert after.energy == pytest.approx(
            pyscf_energy(calc, atoms, 1, "PBE"), abs=1e-8
        )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("an infeasible DF plan reached native system creation")

    monkeypatch.setattr(calc, "_create_native_system", forbidden)
    with pytest.raises(MemoryError):
        calc.prepare_batch(systems, ks_export=False, resource_plan=short)
    (request,) = exact.requests
    controls = json.loads(request.identity.schedule)
    controls["cpu_df_named_program"] = "pbe0-rks"
    substituted = replace(
        request,
        identity=replace(
            request.identity, schedule=json.dumps(controls, sort_keys=True)
        ),
    )
    with pytest.raises(ValueError, match="inputs differ from the global resource plan"):
        calc.prepare_batch(
            systems,
            ks_export=False,
            resource_plan=replace(exact, requests=(substituted,)),
        )


def pyscf_energy(
    calc: Calculator,
    raw_atoms: Any,
    multiplicity: int,
    functional: str,
    *,
    max_grid_points: int = 200_000,
) -> float:
    """Copy orbital/auxiliary primitives and quadrature; only the solvers differ."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    pyscf.lib.num_threads(1)
    atoms = tuple(Atom.from_value(a) for a in raw_atoms)
    labels = [f"{symbol}{i}" for i, (symbol, _) in enumerate(raw_atoms)]

    def basis_dict(selected: Any) -> dict:
        result = {label: [] for label in labels}
        for shell in calc._shells_for_atoms(atoms, selected):
            result[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *[(p.exponent, p.coefficient) for p in shell.primitives],
                ]
            )
        return result

    mol = gto.M(
        atom=[
            (label, atom.position) for label, atom in zip(labels, atoms, strict=True)
        ],
        basis=basis_dict(calc._basis),
        unit="Bohr",
        cart=calc._representation_name == "cartesian",
        spin=multiplicity - 1,
        verbose=0,
    )
    mf = (dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)).density_fit(
        auxbasis=basis_dict(calc._auxiliary_basis)
    )
    grid = MolecularGrid(
        atoms, spec=calc.ks_options.grid, multiplicity=multiplicity
    ).explicit(max_points=max_grid_points)
    mf.xc = functional
    mf.grids.coords = np.asarray(grid.points)
    mf.grids.weights = np.asarray(grid.weights)
    mf.small_rho_cutoff = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf.e_tot


def pyscf_energy_gradient(
    calc: Calculator,
    state: Any,
    raw_atoms: Any,
    multiplicity: int,
    functional: str,
    *,
    charge: int = 0,
) -> tuple[float, np.ndarray]:
    """Independent DF-SCF plus auxiliary/metric and moving-grid response."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    pyscf.lib.num_threads(1)
    atoms = tuple(Atom.from_value(a) for a in raw_atoms)
    labels = [f"{symbol}{i}" for i, (symbol, _) in enumerate(raw_atoms)]

    def basis_dict(selected: Any) -> dict:
        result = {label: [] for label in labels}
        for shell in calc._shells_for_atoms(atoms, selected):
            result[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *[(p.exponent, p.coefficient) for p in shell.primitives],
                ]
            )
        return result

    mol = gto.M(
        atom=[
            (label, atom.position) for label, atom in zip(labels, atoms, strict=True)
        ],
        basis=basis_dict(calc._basis),
        unit="Bohr",
        cart=calc._representation_name == "cartesian",
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    mf = (dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)).density_fit(
        auxbasis=basis_dict(calc._auxiliary_basis)
    )
    points = np.asarray(state.grid.points)
    weights = np.asarray(state.grid.weights)
    owners = np.asarray(state.grid.owners)
    atomic_weights = np.asarray(state._source.atomic_weights)
    mf.xc = functional
    # Set partition inputs before explicit arrays: PySCF invalidates its grid
    # when radii_adjust or atomic_radii changes.
    spec = state._source.grid_spec
    if spec.version == 4:
        from pyscf.dft import radi

        radii = np.array(radi.BRAGG_RADII, copy=True)
        for z, radius in spec.element_radii:
            radii[z] = radius
        mf.grids.atomic_radii = radii
        mf.grids.radii_adjust = radi.treutler_atomic_radii_adjust
    else:
        mf.grids.radii_adjust = None
    mf.grids.coords = points
    mf.grids.weights = weights
    atomic_grid = {
        mol.atom_symbol(a): (
            points[owners == a] - mol.atom_coord(a),
            atomic_weights[owners == a],
        )
        for a in range(mol.natm)
    }
    mf.grids.gen_atomic_grids = lambda *args, **kwargs: atomic_grid
    mf.small_rho_cutoff = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    np.testing.assert_array_equal(mf.grids.coords, points)
    np.testing.assert_array_equal(mf.grids.weights, weights)
    gradient = mf.nuc_grad_method()
    gradient.grid_response = True
    return mf.e_tot, np.asarray(gradient.kernel())


@pytest.mark.parametrize("unrestricted", (False, True))
def test_cpu_df_force_host_budget(
    unrestricted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact force plan publishes the same native result and work as baseline."""
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.common.resources import ResourceBudget
    from generativeqc_compiler.dft import NativeAO

    method = "pbe-uks" if unrestricted else "pbe-rks"
    atoms = (
        [
            ("O", (0.0, 0.0, 0.12)),
            ("H", (1.45, 0.04, -1.02)),
            ("H", (-1.37, -0.11, -1.07)),
        ]
        if unrestricted
        else WATER
    )
    charge = 1 if unrestricted else 0
    multiplicity = 2 if unrestricted else 1
    basis_name = "sto-3g" if unrestricted else "def2-svp"
    calc = Calculator(
        method=method,
        device="cpu",
        basis=basis_name,
        basis_representation="spherical",
        auxiliary_basis=None if unrestricted else "def2-tzvp",
        density_fitting="cpu",
        initial_guess=None,
        ks_options=KsOptions(grid=GridSpec.treutler(0)),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    properties = ("energy", "forces")
    options = {
        "charges": [charge],
        "multiplicities": [multiplicity],
    }
    peak = (
        calc.estimate_resources(
            [atoms],
            charges=[charge],
            multiplicities=[multiplicity],
            properties=properties,
        )
        .require_feasible()
        .peak_bytes["host"]
    )
    exact, short = (
        calc.estimate_resources(
            [atoms],
            charges=[charge],
            multiplicities=[multiplicity],
            properties=properties,
            budget=ResourceBudget(host_bytes=peak + delta),
        )
        for delta in (0, -1)
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail(
            "force plan attempted compatibility derivatives or early allocation"
        )

    with calc.prepare_batch([atoms], **options) as batch:
        expected = batch.execute(strict=True, properties=properties).items[0]
    with (
        calc.prepare_batch([atoms], resource_plan=exact, **options) as batch,
        NativeAO(
            atoms,
            basis=basis_name,
            representation="spherical",
            charge=charge,
            multiplicity=multiplicity,
        ) as basis,
    ):
        monkeypatch.setattr(batch, "_public_dft_cpu_force", forbidden)
        actual = batch.execute(strict=True, properties=properties).items[0]
        state = StationaryKsState.from_native(batch, basis, allow_reference_domain=True)
        try:
            energy, gradient = pyscf_energy_gradient(
                calc, state, atoms, multiplicity, "PBE", charge=charge
            )
        finally:
            state._source.close()
        assert (actual.energy, actual.iterations, actual.fock_builds) == (
            expected.energy,
            expected.iterations,
            expected.fock_builds,
        )
        np.testing.assert_array_equal(actual.forces, expected.forces)
        assert actual.energy == pytest.approx(energy, abs=1e-8)
        np.testing.assert_allclose(actual.forces, -gradient, atol=3e-7, rtol=0)
        with monkeypatch.context() as environment:
            environment.setenv("GENERATIVEQC_CPU_DF_MATERIALIZED_DERIVATIVES", "1")
            with pytest.raises(
                NotImplementedError, match="materialized DF derivatives"
            ):
                batch.execute(properties=properties)
        moved = list(atoms)
        element, (x_coord, y_coord, z_coord) = moved[-1]
        moved[-1] = (element, (x_coord, y_coord, z_coord + 0.003))
        updated = batch.execute(
            coordinates=[[coords for _, coords in moved]],
            strict=True,
            properties=properties,
        ).items[0]
        assert updated.warm_start_used
        assert batch.resource_plan == exact
    moved_reference = calc.singlepoint(
        moved, charge=charge, multiplicity=multiplicity, properties=properties
    )
    assert updated.energy == pytest.approx(moved_reference.energy, abs=1e-9)
    np.testing.assert_allclose(
        updated.forces, moved_reference.forces, atol=2e-7, rtol=0
    )
    monkeypatch.setattr(calc, "_create_native_system", forbidden)
    with pytest.raises(MemoryError):
        calc.prepare_batch([atoms], resource_plan=short, **options)
    with pytest.raises(ValueError, match="ks_export=True"):
        calc.prepare_batch([atoms], ks_export=False, resource_plan=exact, **options)


@pytest.mark.parametrize(
    "method,functional,atoms,multiplicity",
    [
        ("lda-rks", "LDA_X,LDA_C_PW", WATER, 1),
        ("pbe-rks", "PBE", WATER, 1),
        ("r2scan-rks", "R2SCAN", WATER, 1),
        ("pbe-uks", "PBE", [("Li", (0.0, 0.0, 0.0))], 2),
    ],
)
def test_df_semilocal_matches_independent_reference(
    device: str, method: str, functional: str, atoms: Any, multiplicity: int
) -> None:
    calc = calculator(device, method)
    result = calc.singlepoint(atoms, multiplicity=multiplicity, properties=("energy",))
    assert result.converged and result.forces is None
    assert result.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    assert result.physical_residual_rms < 1e-9
    assert result.energy == pytest.approx(
        pyscf_energy(calc, atoms, multiplicity, functional), abs=1e-8
    )


@pytest.mark.parametrize(
    "method,functional,atoms,multiplicity,has_exchange",
    [
        ("pbe-rks", "PBE", WATER, 1, False),
        ("r2scan-rks", "R2SCAN", WATER, 1, False),
        ("pbe-uks", "PBE", [("Li", (0.0, 0.0, 0.0))], 2, False),
        ("pbe0-rks", "PBE0", WATER, 1, True),
    ],
)
def test_df_force_matches_independent_response(
    device: str,
    method: str,
    functional: str,
    atoms: Any,
    multiplicity: int,
    has_exchange: bool,
) -> None:
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO

    calc = calculator(device, method)
    assert "forces" in calc.capabilities.supported_properties
    with calc.prepare_batch(
        [atoms], multiplicities=[multiplicity], warm_start=True
    ) as batch:
        public = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        with NativeAO(
            atoms,
            basis=calc._basis,
            representation=calc._representation_name,
            multiplicity=multiplicity,
        ) as basis:
            state = StationaryKsState.from_native(
                batch, basis, allow_reference_domain=True
            )
            try:
                coulomb, exchange, threshold = state._source.fock_provider_proof()
                assert coulomb == "density-fitted"
                assert exchange == ("density-fitted" if has_exchange else None)
                assert threshold == calc._density_fitting_relative_threshold
                reference_energy, reference_gradient = pyscf_energy_gradient(
                    calc, state, atoms, multiplicity, functional
                )
            finally:
                state._source.close()
        assert public.executed_backend == (
            "cuda" if device == "cuda" else "cpu_reference"
        )
        assert public.converged
        assert public.energy == pytest.approx(reference_energy, abs=1e-8)
        np.testing.assert_allclose(
            public.forces, -reference_gradient, atol=3e-7, rtol=0
        )
        np.testing.assert_allclose(public.forces.sum(axis=0), 0, atol=2e-9, rtol=0)
        replay = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        np.testing.assert_allclose(replay.forces, public.forces, atol=2e-8, rtol=0)


@pytest.mark.parametrize(
    "method,functional", [("pbe0-rks", "PBE0"), ("b3lyp-rks", "B3LYP")]
)
def test_df_cpu_global_hybrid_matches_independent_reference(
    method: str, functional: str
) -> None:
    calc = calculator("cpu", method)
    result = calc.singlepoint(WATER, properties=("energy",))
    assert result.energy == pytest.approx(
        pyscf_energy(calc, WATER, 1, functional), abs=1e-8
    )


def test_df_batch_warm_replay_rebinds_auxiliary_centers(device: str) -> None:
    calc = calculator(device)
    moved = [
        (symbol, (x, y, z + (0.04 if i == 1 else 0.0)))
        for i, (symbol, (x, y, z)) in enumerate(WATER)
    ]
    expected = calc.singlepoint(moved, properties=("energy",)).energy
    with calc.prepare_batch([WATER, WATER], warm_start=True) as batch:
        cold = batch.execute(strict=True, properties=("energy",))
        from generativeqc._ks_snapshot import NativeKsSnapshot

        if device == "cuda":
            transport_before_snapshot = tuple(batch.ks_transport_diagnostics)
            assert all(item.matrix_d2h_bytes == 0 for item in transport_before_snapshot)
            snapshot = NativeKsSnapshot(batch, 0)
            try:
                coulomb, exchange, threshold = snapshot.fock_provider_proof()
                assert coulomb == "density-fitted" and exchange is None
                assert threshold == calc._density_fitting_relative_threshold
            finally:
                snapshot.close()
            transport_after_snapshot = tuple(batch.ks_transport_diagnostics)
            matrix_bytes = 7 * 7 * np.dtype(np.float64).itemsize
            assert [
                after.matrix_d2h_bytes - before.matrix_d2h_bytes
                for before, after in zip(
                    transport_before_snapshot, transport_after_snapshot, strict=True
                )
            ] == [3 * matrix_bytes, 0]
        else:
            snapshot = NativeKsSnapshot(batch, 0)
            try:
                coulomb, exchange, threshold = snapshot.fock_provider_proof()
                assert coulomb == "density-fitted" and exchange is None
                assert threshold == calc._density_fitting_relative_threshold
            finally:
                snapshot.close()
        transport_before_warm = (
            tuple(batch.ks_transport_diagnostics) if device == "cuda" else None
        )
        warm = batch.execute(strict=True, properties=("energy",))
        if device == "cuda":
            # NativeKsSnapshot is an explicit final-state export and its matrices
            # remain charged to the cumulative legacy matrix_d2h_bytes counter.
            # The warm SCF replay itself must add no further matrix D2H.
            transport_after_warm = tuple(batch.ks_transport_diagnostics)
            assert transport_before_warm is not None
            assert all(
                after.matrix_d2h_bytes == before.matrix_d2h_bytes
                for before, after in zip(
                    transport_before_warm, transport_after_warm, strict=True
                )
            )
        updated = batch.execute(
            coordinates=[None, [xyz for _, xyz in moved]],
            strict=True,
            properties=("energy",),
        )
        assert warm.energies == pytest.approx(cold.energies, abs=1e-9)
        assert all(item.warm_start_used for item in warm.items)
        assert updated.items[1].energy == pytest.approx(expected, abs=1e-9)
        if device == "cuda":
            metrics = batch.last_density_fitting_metric_diagnostics()
            assert [item.system_index for item in metrics] == [0, 1]
            assert [item.bucket_id for item in metrics] == [0, 1]
            assert all(
                item.effective_rank > 0 and item.device_resident_bytes > 0
                for item in metrics
            )
            transport = batch.ks_transport_diagnostics
            # Rebuilding only the moved item's owner exports its last-good RKS
            # density once; WATER/STO-3G has seven orbital basis functions.
            density_bytes = 7 * 7 * np.dtype(np.float64).itemsize
            assert [
                after.matrix_d2h_bytes - before.matrix_d2h_bytes
                for before, after in zip(transport_after_warm, transport, strict=True)
            ] == [0, density_bytes]


def test_df_rejects_unqualified_precision_and_resource_consumers(
    device: str,
) -> None:
    calc = calculator(device)
    assert "forces" in calc.capabilities.supported_properties
    with pytest.raises(NotImplementedError, match="resource plans"):
        if device == "cpu":
            calc.estimate_resources(
                [WATER], properties=("energy", "forces")
            ).require_feasible()
        else:
            calc.estimate_resources([WATER])
    if device == "cuda":
        with pytest.raises(NotImplementedError, match="fp64"):
            calculator(device, method="pbe0-rks", precision="auto")
        with pytest.raises(ValueError, match="backend must match"):
            Calculator(method="pbe-rks", device=device, density_fitting="cpu")


def test_cpu_df_energy_inventory_refuses_materialized_derivatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENERATIVEQC_CPU_DF_MATERIALIZED_DERIVATIVES", "1")
    with pytest.raises(NotImplementedError, match="materialized DF derivatives"):
        calculator("cpu", initial_guess=None).estimate_resources(
            [WATER], properties=("energy",)
        ).require_feasible()


def test_df_cuda_provider_budget_is_not_silently_enlarged(device: str) -> None:
    if device != "cuda":
        pytest.skip("DF device allocation budget")
    with pytest.raises(
        (MemoryError, RuntimeError), match="budget|memory|fit|workspace"
    ):
        calculator(device, density_fitting_memory_budget_bytes=1).singlepoint(WATER)


def test_df_default_auxiliary_ragged_batch(device: str) -> None:
    calc = Calculator(
        method="pbe-rks",
        basis="sto-3g",
        device=device,
        density_fitting="auto",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    systems = [
        [("He", (0.0, 0.0, 0.0))],
        [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    ]
    expected = [pyscf_energy(calc, atoms, 1, "PBE") for atoms in systems]
    with calc.prepare_batch(systems) as batch:
        result = batch.execute(strict=True, properties=("energy",))
        assert result.energies == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize(
    "method,functional,atoms,multiplicity",
    [
        ("pbe-rks", "PBE", WATER, 1),
        ("r2scan-rks", "R2SCAN", WATER, 1),
        ("pbe-uks", "PBE", [("Li", (0.0, 0.0, 0.0))], 2),
    ],
)
def test_df_cuda_semilocal_force_matches_independent_response(
    method: str, functional: str, atoms: Any, multiplicity: int
) -> None:
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an allocated native CUDA library/device")
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO

    calc = calculator("cuda", method)
    assert "forces" in calc.capabilities.supported_properties
    with calc.prepare_batch(
        [atoms], multiplicities=[multiplicity], warm_start=True
    ) as batch:
        public = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        with NativeAO(
            atoms,
            basis=calc._basis,
            representation=calc._representation_name,
            multiplicity=multiplicity,
        ) as basis:
            state = StationaryKsState.from_native(
                batch, basis, allow_reference_domain=True
            )
            try:
                coulomb, exchange, threshold = state._source.fock_provider_proof()
                assert coulomb == "density-fitted" and exchange is None
                assert threshold == calc._density_fitting_relative_threshold
                reference_energy, reference_gradient = pyscf_energy_gradient(
                    calc, state, atoms, multiplicity, functional
                )
            finally:
                state._source.close()
        assert public.executed_backend == "cuda" and public.converged
        assert public.energy == pytest.approx(reference_energy, abs=1e-8)
        np.testing.assert_allclose(
            public.forces, -reference_gradient, atol=3e-7, rtol=0
        )
        np.testing.assert_allclose(public.forces.sum(axis=0), 0, atol=2e-9, rtol=0)
        replay = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        np.testing.assert_allclose(replay.forces, public.forces, atol=2e-8, rtol=0)


def test_df_cuda_force_rebinds_auxiliary_response_on_moved_geometry() -> None:
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an allocated native CUDA library/device")
    calc = calculator("cuda", "pbe-rks")
    moved = np.asarray([xyz for _, xyz in WATER], dtype=np.float64)
    moved[1] += (0.02, -0.01, 0.03)
    moved_atoms = [
        (atom[0], tuple(position)) for atom, position in zip(WATER, moved, strict=True)
    ]
    with calc.prepare_batch([WATER], warm_start=True) as batch:
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        replay = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        changed = batch.execute(
            coordinates=(moved,), strict=True, properties=("energy", "forces")
        ).items[0]
    fresh = calc.singlepoint(moved_atoms, properties=("energy", "forces"))
    np.testing.assert_allclose(replay.forces, cold.forces, atol=2e-8, rtol=0)
    assert changed.energy == pytest.approx(fresh.energy, abs=1e-9)
    np.testing.assert_allclose(changed.forces, fresh.forces, atol=3e-8, rtol=0)

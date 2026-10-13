"""Native CUDA RHF Hückel admission against independent CPU PySCF endpoints.

These are real-device tests, not host control-flow doubles. They require an
explicit finite Slurm allocation; neither GPU visibility nor backend is changed.
"""

from __future__ import annotations

import ctypes
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    InitialGuessSpec,
    Primitive,
    ResourceBudget,
    Shell,
    _native,
)
from generativeqc.checkpoint import _MAX_MANIFEST, _descriptor, _pointer

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_HUCKEL_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)

WATER = [(8, (0.0, 0.0, 0.0)), (1, (1.43, 0.0, 1.11)), (1, (-1.43, 0.0, 1.11))]
H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]


@pytest.fixture(scope="module", autouse=True)
def scheduler_visibility() -> None:
    """Fail before creating a GPU context if the scheduling contract is missing."""
    assert os.environ.get("SLURM_JOB_ID")
    assert os.environ.get("CUDA_VISIBLE_DEVICES")


def calculator(**options: object) -> Calculator:
    """Keep physical target controls identical between core and seeded arms."""
    settings = {
        "method": "rhf",
        "basis": "sto-3g",
        "device": "cuda",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
        "max_iterations": 100,
        "initial_guess": InitialGuessSpec("huckel"),
    }
    settings.update(options)
    return Calculator(**settings)


def final_density(batch: object) -> np.ndarray:
    """Read the ordinary retained target density only after endpoint execution."""
    state = _descriptor()
    query = batch._library.generativeqc_batch_get_hf_warm_state
    _native.check(batch._library, query(batch._batch, 0, ctypes.byref(state)))
    assert state.present and state.density_count > 0
    density = np.empty(state.density_count, dtype=np.float64)
    coordinates = np.empty(state.coordinate_count, dtype=np.float64)
    state.density, state.coordinates = _pointer(density), _pointer(coordinates)
    _native.check(batch._library, query(batch._batch, 0, ctypes.byref(state)))
    return density


def target_fock_builds(result: object) -> int:
    """Read the certified device census, not the legacy HF scalar API field."""
    work = result.precision
    assert work is not None and work["operator_work_counters_valid"]
    return sum(
        work[name]
        for name in (
            "mixed_stage_fock_builds",
            "strict_stage_fock_builds",
            "post_scf_fock_builds",
        )
    )


@pytest.mark.parametrize(
    "basis,representation",
    [("sto-3g", "cartesian"), ("def2-svp", "spherical"), ("def2-svp", "cartesian")],
)
@pytest.mark.parametrize("properties", [("energy",), ("energy", "forces")])
def test_cuda_hf_endpoint_matches_independent_oracle(
    basis: str, representation: str, properties: tuple[str, ...]
) -> None:
    pyscf = pytest.importorskip("pyscf")
    molecule = pyscf.gto.M(
        atom=WATER,
        basis=basis,
        unit="Bohr",
        cart=representation == "cartesian",
        verbose=0,
    )
    oracle = pyscf.scf.RHF(molecule)
    oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-13, 1e-10, 100
    energy = oracle.kernel()
    assert oracle.converged
    oracle_overlap = oracle.get_ovlp()
    # GenerativeQC normalizes every contracted Cartesian component. PySCF
    # retains a shared radial normalization for higher-angular components.
    normalization = np.sqrt(np.diag(oracle_overlap))
    metric_scale = normalization[:, None] * normalization[None, :]
    expected_density = oracle.make_rdm1() * metric_scale
    expected_forces = (
        -oracle.nuc_grad_method().kernel() if "forces" in properties else None
    )
    # Packaged basis tables can be rounded relative to PySCF. Share exact input
    # coefficients rather than relaxing the independent numerical gates.
    native_basis = tuple(
        Shell(
            atom_index,
            shell[0],
            tuple(Primitive(row[0], row[1]) for row in shell[1:]),
        )
        for atom_index in range(molecule.natm)
        for shell in pyscf.gto.basis.load(basis, molecule.atom_symbol(atom_index))
    )
    calc = calculator(basis=native_basis, basis_representation=representation)
    with calc.prepare_batch([WATER]) as batch:
        actual = batch.execute(strict=True, properties=properties).items[0]
        density = final_density(batch).reshape(expected_density.shape)
    assert actual.converged and actual.initial_guess["outcome"] == "used"
    assert actual.initial_guess["kind"] == "huckel"
    assert actual.initial_guess["target_attempts"] == 1
    assert actual.initial_guess["work_counters_complete"]
    assert actual.initial_guess["preliminary_fock_builds"] > 0
    assert actual.energy == pytest.approx(energy, abs=1e-8)
    assert target_fock_builds(actual) >= actual.iterations
    if actual.physical_residual_rms is not None:
        assert 0 <= actual.physical_residual_rms <= 1e-8
    np.testing.assert_allclose(density, expected_density, atol=1e-8, rtol=0)
    overlap = oracle_overlap / metric_scale
    fock = oracle.get_fock(dm=density / metric_scale) / metric_scale
    residual = fock @ density @ overlap - overlap @ density @ fock
    # CUDA HF's legacy scalar report can omit this field. The independent
    # physical operator still gates every exported final target density.
    assert np.sqrt(np.mean(residual**2)) <= 1e-8
    if expected_forces is not None:
        np.testing.assert_allclose(actual.forces, expected_forces, atol=1e-7, rtol=0)
    single = calc.singlepoint(WATER, properties=properties)
    assert single.converged and single.initial_guess["outcome"] == "used"
    assert single.energy == pytest.approx(energy, abs=1e-8)
    if expected_forces is not None:
        np.testing.assert_allclose(single.forces, expected_forces, atol=1e-7, rtol=0)


@pytest.mark.parametrize("basis", ["sto-3g", "def2-svp"])
def test_imported_warm_and_cold_neighbors_keep_precedence(
    tmp_path: Path, basis: str
) -> None:
    moved = [WATER[0], (1, (1.47, 0.0, 1.11)), WATER[2]]
    checkpoint = tmp_path / "core.bin"
    with calculator(basis=basis, initial_guess=None).prepare_batch(
        [WATER, WATER]
    ) as source:
        source.execute(strict=True, properties=("energy", "forces"))
        source.save_checkpoint(checkpoint)
    with calculator(basis=basis).prepare_batch([WATER, moved]) as batch:
        batch.load_checkpoint(checkpoint, strict=False)
        results = batch.execute(strict=True, properties=("energy", "forces")).items
        assert results[0].warm_start_used and not results[1].warm_start_used
        assert results[0].initial_guess["outcome"] == "existing_density"
        assert results[0].initial_guess["preliminary_fock_builds"] == 0
        assert results[1].initial_guess["outcome"] == "used"
        warm = batch.execute(strict=True, properties=("energy", "forces")).items
        assert all(item.initial_guess["outcome"] == "existing_density" for item in warm)
        assert all(item.initial_guess["preliminary_fock_builds"] == 0 for item in warm)
        batch.clear_warm_starts()
        cold = batch.execute(strict=True, properties=("energy", "forces")).items
        assert all(item.initial_guess["outcome"] == "used" for item in cold)


@pytest.mark.parametrize("properties", [("energy",), ("energy", "forces")])
@pytest.mark.parametrize(
    "policy,outcome",
    [
        (InitialGuessSpec("huckel", maximum_numeric_bytes=1), "budget_skipped"),
        (InitialGuessSpec("huckel", max_iterations=1), "preparation_failed"),
    ],
)
def test_bounded_preparation_fallback_preserves_core_endpoint(
    properties: tuple[str, ...], policy: InitialGuessSpec, outcome: str
) -> None:
    actual = calculator(initial_guess=policy).singlepoint(WATER, properties=properties)
    core = calculator(initial_guess=None).singlepoint(WATER, properties=properties)
    assert actual.converged and actual.initial_guess["outcome"] == outcome
    assert actual.initial_guess["target_attempts"] == 1
    assert actual.energy == pytest.approx(core.energy, abs=1e-10)
    if "forces" in properties:
        np.testing.assert_allclose(actual.forces, core.forces, atol=1e-9, rtol=0)


@pytest.mark.parametrize("properties", [("energy",), ("energy", "forces")])
def test_seeded_target_has_only_one_core_retry(properties: tuple[str, ...]) -> None:
    with calculator(max_iterations=1).prepare_batch([WATER], warm_start=False) as batch:
        actual = batch.execute(strict=False, properties=properties).items[0]
    assert not actual.converged
    assert actual.initial_guess["outcome"] == "target_retried"
    assert actual.initial_guess["target_attempts"] == 2
    assert actual.initial_guess["discarded_target_fock_builds"] > 0


def test_small_cuda_global_budget_accounts_for_preparation() -> None:
    systems = [WATER, H2, WATER]
    reference = calculator()
    plan = reference.estimate_resources(systems).require_feasible()
    budget = ResourceBudget(
        device_bytes=plan.peak_bytes["device"], host_bytes=plan.peak_bytes["host"]
    )
    with calculator(resource_budget=budget).prepare_batch(systems) as batch:
        cold = batch.execute(strict=True, properties=("energy", "forces")).items
        assert all(item.initial_guess["outcome"] == "used" for item in cold)
        assert cold[1].initial_guess["preliminary_fock_builds"] == 0
        warm = batch.execute(strict=True, properties=("energy", "forces")).items
        assert all(item.initial_guess["outcome"] == "existing_density" for item in warm)
        ledger = batch.resource_diagnostics["observation"]["device_ledger"]
        assert ledger["peak_bytes"] <= ledger["limit_bytes"]
        assert ledger["rejected_allocations"] == 0


def test_warm_failure_uses_one_budgeted_core_retry(tmp_path: Path) -> None:
    checkpoint = tmp_path / "warm.bin"
    with calculator(initial_guess=None).prepare_batch([WATER]) as source:
        source.execute(strict=True, properties=("energy", "forces"))
        source.save_checkpoint(checkpoint)
    target = calculator(max_iterations=1)
    plan = target.estimate_resources([WATER]).require_feasible()
    # Checkpoint staging has an independent host contract. Keep the device
    # budget exact so the retry cannot hide a second live target arena.
    density_bytes = 7 * 7 * np.dtype(np.float64).itemsize
    coordinate_bytes = len(WATER) * 3 * np.dtype(np.float64).itemsize
    staging_bytes = (
        4 * (density_bytes + coordinate_bytes) + 24 * density_bytes + _MAX_MANIFEST
    )
    budget = ResourceBudget(
        device_bytes=plan.peak_bytes["device"],
        host_bytes=plan.peak_bytes["host"] + staging_bytes,
    )
    with calculator(max_iterations=1, resource_budget=budget).prepare_batch(
        [WATER]
    ) as batch:
        batch.load_checkpoint(checkpoint, allow_warm=True)
        actual = batch.execute(strict=False, properties=("energy", "forces")).items[0]
        assert actual.warm_start_used and actual.warm_start_fallback
        assert not actual.converged
        assert actual.initial_guess["outcome"] == "existing_density"
        assert actual.initial_guess["target_attempts"] == 2
        assert actual.initial_guess["preliminary_fock_builds"] == 0
        assert actual.initial_guess["discarded_target_fock_builds"] > 0
        assert target_fock_builds(actual) > 0
        ledger = batch.resource_diagnostics["observation"]["device_ledger"]
        assert ledger["rejected_allocations"] == 0


def test_cuda_hf_default_and_unqualified_domains_stay_guarded() -> None:
    assert calculator(initial_guess="auto").initial_guess == InitialGuessSpec("huckel")
    with pytest.raises(NotImplementedError):
        calculator(initial_guess=InitialGuessSpec("minao"))
    with pytest.raises(NotImplementedError):
        calculator(method="uhf")

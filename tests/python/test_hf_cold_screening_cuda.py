"""Opt-in large-owner qualification for screened cold HF and bounded pages.

These complete endpoints require a finite Slurm allocation. GPU4PySCF supplies
an independent target-operator energy/force oracle; it never seeds native SCF.
"""

from __future__ import annotations

import ctypes
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_HF_COLD_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="explicit large HF cold qualification under finite Slurm",
)


@pytest.fixture(scope="module")
def independent_targets() -> tuple[Any, Any, Any]:
    """Converge each original/displaced target without a native density."""
    from gpu4pyscf import scf
    from pyscf import gto, lib

    from benchmarks.readme_hf_scaling import scaling_cases

    lib.num_threads(1)
    atoms = scaling_cases()["water-96"].atoms
    moved = [(symbol, list(position)) for symbol, position in atoms]
    moved[1][1][0] += 0.01
    references = []
    for geometry in (atoms, moved):
        mol = gto.M(atom=geometry, basis="def2-svp", unit="Bohr", verbose=0)
        solver = scf.RHF(mol)
        # The extensive ~2400-Eh energy fluctuates by a few 1e-12 Eh on
        # GPU4; the force-residual gate, not an unattainable energy decrement,
        # controls this independent reference. Observable gates remain fixed.
        solver.conv_tol, solver.conv_tol_grad = 1e-10, 1e-8
        solver.kernel()
        assert solver.converged
        gradient = solver.nuc_grad_method().kernel()
        gradient = gradient.get() if hasattr(gradient, "get") else np.asarray(gradient)
        references.append((float(solver.e_tot), -gradient))
    if output := os.environ.get("GENERATIVEQC_HF_COLD_ORACLE_RECEIPT"):
        Path(output).write_text(
            json.dumps(
                [
                    {"energy": energy, "forces": forces.tolist()}
                    for energy, forces in references
                ],
                indent=2,
            )
            + "\n"
        )
    return atoms, moved, references


def calculator(**options: Any) -> Any:
    """Qualify the default algorithm at a tight independent force target.

    Default-tolerance endpoint parity is separately retained by the #2786
    campaign. These controls demand a tighter stationary solution without
    changing the screened proposal or any observable acceptance threshold.
    """
    from generativeqc import Calculator

    return Calculator(
        method="rhf",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        precision="fp64",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        **options,
    )


def check_target(item: Any, target: Any, *, forces: bool = True) -> None:
    """Require independent observables and a complete, honest FP64 work total."""
    assert item.converged and item.executed_backend == "cuda"
    assert abs(item.energy - target[0]) < 1e-8
    if forces:
        np.testing.assert_allclose(item.forces, target[1], atol=1e-7, rtol=0)
    work = item.precision
    assert work["operator_work_counters_valid"]
    assert work["effective_bits"] == 64 and work["mixed_stage_fock_builds"] == 0
    assert (
        item.fock_builds
        == work["strict_stage_fock_builds"] + work["post_scf_fock_builds"]
    )
    assert work["strict_stage_fock_builds"] == item.iterations


def test_cold_frozen_moved_and_imported_owners(
    independent_targets: Any, tmp_path: Path
) -> None:
    """A provisional graph must never enter warm, moved or imported execution."""
    atoms, moved, targets = independent_targets
    calc = calculator()
    try:
        with calc.prepare_batch([atoms], warm_start=True) as owner:
            cold = owner.execute(strict=True).items[0]
            check_target(cold, targets[0])
            assert 0 < cold.precision["refinement_iterations"] < cold.iterations
            checkpoint = tmp_path / "cold.npz"
            owner.save_checkpoint(checkpoint)
            owner.set_warm_start_updates(False)
            for _ in range(2):
                warm = owner.execute(strict=True).items[0]
                check_target(warm, targets[0])
                assert warm.warm_start_used
                assert warm.precision["refinement_iterations"] == 0
            shifted = owner.execute(
                [[position for _, position in moved]], strict=True
            ).items[0]
            check_target(shifted, targets[1])
            assert shifted.precision["refinement_iterations"] == 0
            restored = owner.execute(
                [[position for _, position in atoms]], strict=True
            ).items[0]
            check_target(restored, targets[0])
        with calc.prepare_batch([atoms], warm_start=True) as imported:
            imported.load_checkpoint(checkpoint)
            item = imported.execute(strict=True).items[0]
            check_target(item, targets[0])
            assert item.warm_start_used and item.precision["refinement_iterations"] == 0
    finally:
        calc.clear_cache()


@contextmanager
def device_allowance(library: Any, limit: int) -> Any:
    """Inject a real ledger ceiling without exhausting physical device memory."""
    create = library.generativeqc_resource_ledger_create_v1
    create.argtypes, create.restype = [ctypes.c_size_t, ctypes.c_int], ctypes.c_void_p
    destroy = library.generativeqc_resource_ledger_destroy_v1
    destroy.argtypes = [ctypes.c_void_p]
    begin = library.generativeqc_resource_tracking_begin_v1
    begin.argtypes = [ctypes.c_uint]
    bind = library.generativeqc_resource_ledger_bind_v1
    bind.argtypes = [ctypes.c_void_p]
    read = library.generativeqc_resource_ledger_read_v1
    read.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    end = library.generativeqc_resource_tracking_end_v1
    end.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 2
    handle = create(limit, 0)
    assert handle
    assert begin(1) == 0
    try:
        assert bind(handle) == 0

        def snapshot() -> tuple[int, ...]:
            values = (ctypes.c_uint64 * 4)()
            assert read(handle, values) == 0
            return tuple(values)

        yield snapshot
    finally:
        peak, samples = ctypes.c_uint64(), ctypes.c_uint64()
        assert end(ctypes.byref(peak), ctypes.byref(samples)) == 0
        destroy(handle)


def test_page_refusal_retries_bounded_owner_and_releases(
    independent_targets: Any,
) -> None:
    """A 4-GiB ceiling refuses the optional 6-GiB page but admits the old page."""
    atoms, _, targets = independent_targets
    calc = calculator()
    limit = 4 << 30
    try:
        with device_allowance(calc._library, limit) as snapshot:
            with calc.prepare_batch([atoms], warm_start=True) as owner:
                for _ in range(2):
                    check_target(owner.execute(strict=True).items[0], targets[0])
                live, peak, _, rejected = snapshot()
                assert 0 < live <= peak <= limit and rejected > 0
            calc.clear_cache()
            assert snapshot()[0] == 0
    finally:
        calc.clear_cache()


@pytest.mark.parametrize("maximum_iterations", (8, 24))
def test_energy_only_iteration_allowance(
    independent_targets: Any, maximum_iterations: int
) -> None:
    """The proposal and strict tail share one allowance, even without forces."""
    atoms, _, targets = independent_targets
    calc = calculator(max_iterations=maximum_iterations)
    try:
        with calc.prepare_batch([atoms]) as owner:
            item = owner.execute(properties=("energy",)).items[0]
            assert item.iterations <= maximum_iterations
            if maximum_iterations == 24:
                check_target(item, targets[0], forces=False)
                assert item.precision["refinement_iterations"] > 0
            else:
                assert not item.precision["strict_refinement_applied"]
    finally:
        calc.clear_cache()

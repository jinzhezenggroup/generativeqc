"""Original #2840 UKS consumers, separate from matrix-provider performance."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc.batch import PreparedBatch

from benchmarks.ks_matrix_provider_qualification import final_state_gates
from benchmarks.structured_scf_endpoint_qualification import (
    basis_and_inputs,
    model,
    reference_state,
)


@lru_cache(maxsize=4)
def _reference(case: str, moved: bool) -> dict[str, Any]:
    """Reuse only independent equations for identical pinned physical inputs."""
    spec = model(case, "uks", moved=moved)
    return reference_state(spec)


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("mask", (0, 127))
@pytest.mark.parametrize("forces", (False, True), ids=("energy", "energy-forces"))
@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_KS_STATE_CUDA_TEST") != "1",
    reason="requires explicitly scheduled original-input CUDA qualification",
)
def test_original_large_spd_uks_final_state(
    representation: str,
    mask: int,
    forces: bool,
    record_property: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither a corrective update nor a rejected export can publish a state.

    Keep the retained geometry, grid, controls and independent E/D/Fock gates.
    The canonical density bound is the original 1e-8, not the older report
    helper's looser diagnostic threshold. Every cold/warm/moved consumer must
    acquire a new authenticated native state before it can be inspected.
    """
    assert os.environ.get("SLURM_JOB_ID")
    pytest.importorskip("pyscf", reason="independent original-input SCF oracle")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("original-input native endpoint called a Python force fallback")

    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", forbidden)
    monkeypatch.setattr(PreparedBatch, "_public_dft_cuda_force", forbidden)
    spec = model("large-spd-" + representation, "uks")
    basis, atoms, _, _ = basis_and_inputs(spec)
    calculator = Calculator(
        "pbe-uks",
        basis,
        device="cuda",
        precision="fp64",
        basis_representation=representation,
        density_fitting="none",
        ks_options=KsOptions(grid=GridSpec(**spec["grid"])),
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    hook = getattr(
        calculator._library, "tensor_structured_scf_qualification_for_test_v1", None
    )
    assert hook is not None, "original mask controls require a test-hook native build"
    hook.argtypes, hook.restype = [ctypes.c_int], ctypes.c_bool
    assert hook(mask)
    observations = []
    provenance = {
        "source_identity": calculator._library.generativeqc_get_source_identity().decode(),
        "library_sha256": hashlib.sha256(
            Path(calculator._library._name).read_bytes()
        ).hexdigest(),
        "scheduler": {
            name: os.environ.get(name)
            for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "CUDA_VISIBLE_DEVICES")
        },
        "mask": mask,
        "forces_requested": forces,
    }
    force_references: dict[bool, np.ndarray] = {}
    try:
        with calculator.prepare_batch(
            [atoms], charges=[spec["charge"]], multiplicities=[spec["multiplicity"]]
        ) as batch:
            for phase in range(5):
                current = model(spec["case"], "uks", moved=phase >= 3)
                item = batch.execute(
                    [current["coordinates"]] if phase >= 3 else None,
                    properties=("energy", "forces") if forces else ("energy",),
                    strict=False,
                ).items[0]
                row = {
                    "phase": phase,
                    "status": item.status,
                    "message": item.status_message,
                    "iterations": item.iterations,
                    "energy": item.energy if np.isfinite(item.energy) else None,
                    "history": item.ks_diagnostic.to_payload()["history"]
                    if item.ks_diagnostic is not None
                    else [],
                }
                observations.append(row)
                record_property("observations", json.dumps(observations))
                assert item.succeeded and item.converged, row
                assert item.executed_backend == "cuda"
                snapshot = NativeKsSnapshot(batch, 0)
                try:
                    gates = final_state_gates(
                        snapshot.metadata, snapshot.values, current
                    )
                    row["state"] = gates
                    assert gates["accepted"]
                    assert gates["density_closure_error"] < 1e-8
                    assert gates["physical_residual_rms"] < 1e-9
                    dimension, spins, natoms = map(int, snapshot.metadata[1:4])
                    offset = 2 + natoms * 4
                    count = spins * dimension * dimension
                    density, fock = (
                        snapshot.values[begin : begin + count].reshape(
                            spins, dimension, dimension
                        )
                        for begin in (offset, offset + count)
                    )
                    reference = _reference(spec["case"], phase >= 3)
                    assert reference["converged"]
                    assert item.energy == pytest.approx(reference["energy"], abs=1e-8)
                    np.testing.assert_allclose(
                        density, reference["density"], atol=1e-6, rtol=0
                    )
                    np.testing.assert_allclose(
                        fock, reference["fock"], atol=1e-6, rtol=0
                    )
                    if forces:
                        assert item.forces is not None
                        assert np.isfinite(item.forces).all()
                        row["forces"] = np.asarray(item.forces).tolist()
                        if (phase >= 3) not in force_references:
                            force_references[phase >= 3] = _independent_force(
                                batch, basis, current
                            )
                        reference_forces = force_references[phase >= 3]
                        row["reference_forces"] = reference_forces.tolist()
                        np.testing.assert_allclose(
                            item.forces, reference_forces, atol=3e-5, rtol=0
                        )
                        np.testing.assert_allclose(
                            np.asarray(item.forces).sum(axis=0), 0, atol=2e-8, rtol=0
                        )
                finally:
                    snapshot.close()
    finally:
        assert hook(-1)
        record_property("observations", json.dumps(observations))
        directory = os.environ.get("GENERATIVEQC_KS_STATE_EVIDENCE_DIR")
        if directory:
            output = Path(directory)
            output.mkdir(parents=True, exist_ok=True)
            (output / f"uks-{representation}-{mask}-{forces}.json").write_text(
                json.dumps(
                    {
                        "spec": spec,
                        "provenance": provenance,
                        "observations": observations,
                    },
                    indent=2,
                )
            )
        calculator.clear_cache()


def _independent_force(
    batch: PreparedBatch, basis: Any, spec: dict[str, Any]
) -> np.ndarray:
    """Reconverge independent PBE and full moving-grid response at the native root.

    A native seed selects the same stationary basin; no native Fock, energy,
    derivative, or convergence flag is reused by the reference calculation.
    """
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import independent_global_hybrid_gradient

    from tools.generate_validation_references import pyscf_molecule

    atoms = list(zip(spec["atomic_numbers"], spec["coordinates"], strict=True))
    _, _, inputs, _ = basis_and_inputs(spec)
    molecule, scale, _ = pyscf_molecule(inputs)
    normalization = scale[:, None] * scale[None, :]
    with NativeAO(
        atoms,
        basis=basis,
        representation=spec["basis_representation"],
        charge=spec["charge"],
        multiplicity=spec["multiplicity"],
    ) as ao:
        state = StationaryKsState.from_native(batch, ao)
        np.testing.assert_allclose(
            molecule.intor("int1e_ovlp") * normalization,
            state.overlap,
            atol=1e-12,
            rtol=0,
        )
        initial_density = state.density * normalization
        trace: list[dict[str, Any]] = []
        energy, derivative, reference = independent_global_hybrid_gradient(
            ao,
            state,
            "pbe-uks",
            xc="PBE",
            initial_density=initial_density,
            scf_gradient_tolerance=1e-9,
            scf_trace=trace,
            return_mean_field=True,
        )
        assert energy == pytest.approx(state._source.energy(), abs=1e-8)
        independent_fock = np.asarray(
            reference.get_fock(
                dm=initial_density, vhf=reference.get_veff(dm=initial_density)
            )
        ).reshape(state.fock.shape)
        for spin, block in enumerate(independent_fock):
            physical = block * normalization
            np.testing.assert_allclose(physical, state.fock[spin], atol=1e-6, rtol=0)
            commutator = (
                physical @ state.density[spin] @ state.overlap
                - state.overlap @ state.density[spin] @ physical
            )
            assert np.max(np.abs(commutator)) < 1e-9
        return -np.asarray(derivative)

"""Private same-basis GPU seed with synchronized, non-overlapping phase timers.

Only density and source coordinates cross the owner boundary. Source policy,
Fock, DIIS and convergence state remain private. The public CUDA source budget
contract is unqualified; these measurements do not promote that API.
"""

import ctypes
import os
from time import perf_counter
from typing import Any

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions, _native
from generativeqc.checkpoint import _descriptor, _pointer


def _prepare_seed(target: Any, atoms: Any, requested: str) -> dict[str, Any]:
    """Import only same-basis density after a complete independent GPU solve."""
    choices = {"none": None, "lda16": "lda-rks", "pbe16": "pbe-rks"}
    if requested not in choices:
        raise ValueError("unknown seed provider")
    record = {
        "requested": requested,
        "selected": False,
        "scope": "private GPU preliminary density; complete lifecycle charged",
        "source_reference_density_used": False,
        "public_api_qualification": False,
        "public_budget_qualification": False,
        "target_basis_identity": target._basis_metadata[0]["orbital"],
        "target_energy_tolerance": target._calculator._energy_tolerance,
        "target_density_tolerance": target._calculator._density_tolerance,
        # Absent phases remain absent measurements, including the no-seed case.
        "source_construct_seconds": None,
        "source_prepare_seconds": None,
        "source_solve_seconds": None,
        "export_seconds": None,
        "import_seconds": None,
        "source_destroy_seconds": None,
        "source_bookkeeping_seconds": None,
        "complete_source_seconds": None,
        "source_fock_builds": None,
    }
    if requested == "none":
        return record
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()
    started = perf_counter()
    phase = started
    source_calc = Calculator(
        method=choices[requested],
        basis=target._calculator._basis,
        device="cuda",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GridSpec(16, 8, 16)),
        energy_tolerance=1e-6,
        density_tolerance=1e-4,
        screening_tolerance=1e-12,
        max_iterations=64,
        diis_history=8,
    )
    cp.cuda.runtime.deviceSynchronize()
    record["source_construct_seconds"] = perf_counter() - phase
    target_calc = target._calculator
    if source_calc.ks_options.method_ir == target_calc.ks_options.method_ir:
        raise ValueError("source must retain its own functional")
    for field in ("_basis_representation", "_density_fitting_mode", "_precision_mode"):
        if getattr(source_calc, field) != getattr(target_calc, field):
            raise ValueError("source/target basis or precision mismatch: " + field)
    record.update(
        source_method=choices[requested],
        source_backend="cuda",
        source_grid=[16, 8, 16],
        source_active_ao_policy="disabled; experimental maps admit only WB97M-V",
        source_energy_tolerance=1e-6,
        source_density_tolerance=1e-4,
        source_max_iterations=64,
        source_diis_history=8,
    )
    phase = perf_counter()
    source = source_calc.prepare_batch([atoms], warm_start=True)
    cp.cuda.runtime.deviceSynchronize()
    record["source_prepare_seconds"] = perf_counter() - phase
    try:
        if (
            source._systems != target._systems
            or source._basis_metadata[0]["orbital"]
            != target._basis_metadata[0]["orbital"]
        ):
            raise ValueError("source density is not in the target AO basis")
        before = _descriptor()
        _native.check(
            target._library,
            target._library.generativeqc_batch_get_hf_warm_state(
                target._batch, 0, ctypes.byref(before)
            ),
        )
        if before.present:
            raise ValueError("preliminary source cannot overwrite a target seed")
        cp.cuda.runtime.deviceSynchronize()
        phase = perf_counter()
        result = source.execute(properties=("energy",), strict=False)
        cp.cuda.runtime.deviceSynchronize()
        record["source_solve_seconds"] = perf_counter() - phase
        item = result.items[0]
        if item.status not in (_native.STATUS_SUCCESS, _native.STATUS_NOT_CONVERGED):
            result.raise_for_failures()
        record.update(
            source_converged=bool(item.converged),
            source_status=int(item.status),
            source_iterations=int(item.iterations),
            source_fock_builds=item.fock_builds,
            source_energy_hartree=float(item.energy)
            if np.isfinite(item.energy)
            else None,
            source_scf_ao_work=read_ao_work(source),
        )
        if result.succeeded:
            cp.cuda.runtime.deviceSynchronize()
            transfer_started = perf_counter()
            state = _descriptor()
            _native.check(
                source._library,
                source._library.generativeqc_batch_get_hf_warm_state(
                    source._batch, 0, ctypes.byref(state)
                ),
            )
            if (
                not state.present
                or state.density_count <= 0
                or 8 * (state.density_count + state.coordinate_count) > 16 << 20
            ):
                raise ValueError("invalid or unbounded density export")
            density = np.empty(state.density_count, dtype=np.float64)
            coordinates = np.empty(state.coordinate_count, dtype=np.float64)
            state.density, state.coordinates = _pointer(density), _pointer(coordinates)
            _native.check(
                source._library,
                source._library.generativeqc_batch_get_hf_warm_state(
                    source._batch, 0, ctypes.byref(state)
                ),
            )
            if not np.isfinite(density).all() or not np.array_equal(
                coordinates.reshape(-1, 3),
                np.asarray([a.position for a in source._systems[0]]),
            ):
                raise ValueError("invalid exported source density or geometry")
            cp.cuda.runtime.deviceSynchronize()
            record["export_seconds"] = perf_counter() - transfer_started
            phase = perf_counter()
            _native.check(
                target._library,
                target._library.generativeqc_batch_restore_hf_warm_states(
                    target._batch, ctypes.byref(state), 1
                ),
            )
            cp.cuda.runtime.deviceSynchronize()
            record["import_seconds"] = perf_counter() - phase
            record.update(
                selected=True,
                seed_density_bytes=int(density.nbytes),
                seed_coordinate_bytes=int(coordinates.nbytes),
                export_import_seconds=perf_counter() - transfer_started,
            )
    finally:
        phase = perf_counter()
        source.close()
        del source, source_calc
        cp.cuda.runtime.deviceSynchronize()
        record["source_destroy_seconds"] = perf_counter() - phase
    record["complete_source_seconds"] = perf_counter() - started
    measured = sum(
        record[name]
        for name in (
            "source_construct_seconds",
            "source_prepare_seconds",
            "source_solve_seconds",
            "export_seconds",
            "import_seconds",
            "source_destroy_seconds",
        )
        if record[name] is not None
    )
    # Includes source policy checks, work queries, synchronization between phases,
    # and Python metadata handling. Never label this residual as GPU computation.
    record["source_bookkeeping_seconds"] = record["complete_source_seconds"] - measured
    if record["source_bookkeeping_seconds"] < 0:
        raise ValueError("overlapping source phase timers")
    return record


def prepare_seed(target: Any, atoms: Any, requested: str = "none") -> dict[str, Any]:
    """Benchmark-only RKS seed; restore the target AO policy even on failure.

    Target basis and scientific controls remain unchanged. This private native
    checkpoint experiment does not qualify a public initialization policy or a
    shared source/target memory budget. No reference-engine data are imported.
    """
    policy = os.environ.get("GENERATIVEQC_CUDA_KS_ACTIVE_AO")
    os.environ["GENERATIVEQC_CUDA_KS_ACTIVE_AO"] = "0"
    try:
        return _prepare_seed(target, atoms, requested)
    finally:
        if policy is None:
            os.environ.pop("GENERATIVEQC_CUDA_KS_ACTIVE_AO", None)
        else:
            os.environ["GENERATIVEQC_CUDA_KS_ACTIVE_AO"] = policy


COUNTERS = [
    "requested",
    "selected",
    "tiles",
    "empty_tiles",
    "min_active",
    "max_active",
    "active_sum",
    "discovery_ao_jet_values",
    "point_ao_visits",
    "point_ao_square_sum",
    "dense_point_ao_square_sum",
    "discovery_d2h_bytes",
    "reserved_device_bytes",
    "host_peak_bytes",
    "xc_evaluations",
]


class AoWork(ctypes.Structure):
    """Versioned diagnostic ABI from generativeqc.h; counts are not FLOPs."""

    _fields_ = (
        [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32)]
        + [(name, ctypes.c_uint64) for name in COUNTERS]
        + [("cutoff", ctypes.c_double), ("discovery_seconds", ctypes.c_double)]
    )


def read_ao_work(batch: Any, index: int = 0) -> dict[str, int | float]:
    """Return solve-local XC counts and one-traversal AO domains, not FLOPs."""
    function = batch._library.generativeqc_batch_get_ks_ao_selection_diagnostic_v1
    function.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(AoWork)]
    function.restype = ctypes.c_int
    value = AoWork()
    value.struct_size = ctypes.sizeof(value)
    value.abi_version = 0
    status = function(batch._batch, index, ctypes.byref(value))
    if status != 0:
        raise RuntimeError(f"AO work status {status} for system {index}")
    return {name: getattr(value, name) for name, _ in value._fields_[2:]}

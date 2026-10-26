"""Typed transport for the independent native CPU fixed-density HF audit.

Missing protocols and unsupported metadata domains return None before execution.
An admitted native failure never retries Python scientific orchestration.
"""

from __future__ import annotations

import ctypes as ct
import time
import typing
from hashlib import sha256

import numpy as np

from . import _native
from .profiles import canonical_hash


class _Controls(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("unrestricted", ct.c_uint32),
        ("density_fitted", ct.c_uint32),
        ("diis_history", ct.c_uint32),
        ("reserved", ct.c_uint32),
        ("maximum_host_bytes", ct.c_uint64),
        ("screening_tolerance", ct.c_double),
        ("metric_relative_threshold", ct.c_double),
    ]


class _Capacity(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        *[
            (name, ct.c_uint64)
            for name in (
                "ao_count",
                "spin_count",
                "density_count",
                "overlap_count",
                "provider_bytes",
                "workspace_bytes",
                "required_host_bytes",
            )
        ],
    ]


class _Result(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("overlap", ct.POINTER(ct.c_double)),
        ("overlap_capacity", ct.c_uint64),
        *[
            (name, ct.c_double)
            for name in (
                "energy",
                "energy_difference",
                "maximum_commutator",
                "rms_commutator",
            )
        ],
        ("fock_builds", ct.c_uint64),
        ("auxiliary_rank", ct.c_uint64),
    ]


def _bind(library: typing.Any) -> bool:
    query = getattr(library, "generativeqc_hf_physical_audit_cpu_query_v1", None)
    execute = getattr(library, "generativeqc_hf_physical_audit_cpu_v1", None)
    if query is None or execute is None:
        return False
    query.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.POINTER(_Controls),
        ct.POINTER(_Capacity),
    ]
    query.restype = ct.c_int32
    execute.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.POINTER(_Controls),
        ct.POINTER(ct.c_double),
        ct.c_uint64,
        ct.c_double,
        ct.POINTER(_Result),
    ]
    execute.restype = ct.c_int32
    return True


def _check(library: typing.Any, status: int, context: typing.Any = None) -> None:
    if not status:
        return
    detail = library.generativeqc_context_get_last_detail(context) if context else None
    message = (detail or library.generativeqc_status_message(status)).decode()
    error = {1: ValueError, 3: NotImplementedError, 7: MemoryError}.get(
        status, RuntimeError
    )
    raise error(message)


def audit(
    problem: typing.Any,
    calculator: typing.Any,
    atoms: typing.Any,
    density: np.ndarray,
    target_energy: float,
    *,
    charge: int,
    multiplicity: int,
    maximum_host_bytes: int,
) -> typing.Any:
    """Return native observations, or None only for an unadmitted native domain.

    The caller has already retained the incumbent complete resource preflight.
    Native capacity independently gates its scope before provider construction.
    Array copies here provide immutable input and detached presentation only.
    """
    library = getattr(calculator, "_library", None)
    if (
        getattr(calculator, "_device_name", None) != "cpu"
        or library is None
        or not _bind(library)
    ):
        return None
    if not 0 < maximum_host_bytes < 2**64 or not 0 <= calculator._diis_history < 2**32:
        return None
    from .progressive_controller import HFPhysicalResidualAudit

    started = time.perf_counter()
    unrestricted = problem.model.method == "uhf"
    fitted = calculator._density_fitting_mode != _native.DENSITY_FITTING_NONE
    request = _Controls(
        ct.sizeof(_Controls),
        _native.ABI_VERSION,
        unrestricted,
        fitted,
        calculator._diis_history,
        0,
        maximum_host_bytes,
        calculator._screening_tolerance,
        calculator._density_fitting_relative_threshold,
    )
    context, system, auxiliary = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    _native.check(
        library,
        library.generativeqc_context_create(
            ct.byref(calculator._context_descriptor()), ct.byref(context)
        ),
    )
    try:
        system = calculator._create_native_system(context, atoms, charge, multiplicity)
        if fitted and calculator._auxiliary_basis is not None:
            auxiliary = calculator._create_native_system(
                context, atoms, charge, multiplicity, calculator._auxiliary_basis
            )
        capacity = _Capacity(ct.sizeof(_Capacity), _native.ABI_VERSION)
        status = library.generativeqc_hf_physical_audit_cpu_query_v1(
            system, auxiliary, ct.byref(request), ct.byref(capacity)
        )
        if status == _native.STATUS_NOT_IMPLEMENTED:
            return None
        _check(library, status)
        if capacity.required_host_bytes > maximum_host_bytes:
            raise MemoryError("native physical verification exceeds host budget")
        n, spins = capacity.ao_count, 2 if unrestricted else 1
        if (capacity.spin_count, capacity.density_count, capacity.overlap_count) != (
            spins,
            spins * n * n,
            n * n,
        ):
            raise RuntimeError("inconsistent native physical audit shape receipt")
        if np.iscomplexobj(density) or density.shape != (spins, n, n):
            raise ValueError("physical audit requires real spin-major square density")
        owned_density = np.array(density, dtype=np.float64, order="C", copy=True)
        overlap = np.empty((n, n), dtype=np.float64)
        result = _Result(
            ct.sizeof(_Result),
            _native.ABI_VERSION,
            overlap.ctypes.data_as(ct.POINTER(ct.c_double)),
            overlap.size,
        )
        status = library.generativeqc_hf_physical_audit_cpu_v1(
            context,
            system,
            auxiliary,
            ct.byref(request),
            owned_density.ctypes.data_as(ct.POINTER(ct.c_double)),
            owned_density.size,
            target_energy,
            ct.byref(result),
        )
        _check(library, status, context)
        if result.fock_builds != 1 or not np.isfinite(overlap).all():
            raise RuntimeError("inconsistent native physical audit result receipt")
        # This versioned identity binds the requested physical model and actual
        # native consumer. It deliberately does not masquerade as the old grid
        # packing identity, since this path creates no AO-grid packing owner.
        operator = canonical_hash(
            {
                "schema": "generativeqc.hf-physical-audit/v1",
                "model": problem.model.identity,
                "provider": problem.provider_identity,
                "unrestricted": unrestricted,
                "fitted": fitted,
                "screening": request.screening_tolerance,
                "metric_cutoff": request.metric_relative_threshold if fitted else None,
                "precision": "float64",
            }
        )
        execution = canonical_hash(
            {
                "operator": operator,
                "backend": "cpu",
                "owner": "native-physical-audit-v1",
                "source": library.generativeqc_get_source_identity().decode(),
                "auxiliary_rank": result.auxiliary_rank,
            }
        )
        observations = {
            "model_identity": problem.model.identity,
            "provider_identity": problem.provider_identity,
            "operator_identity": operator,
            "execution_identity": execution,
            "density_sha256": sha256(owned_density.tobytes()).hexdigest(),
            "overlap_sha256": sha256(overlap.tobytes()).hexdigest(),
            "maximum_commutator": result.maximum_commutator,
            "rms_commutator": result.rms_commutator,
            "fixed_density_energy": result.energy,
            "energy_difference": result.energy_difference,
        }
    finally:
        if auxiliary:
            library.generativeqc_system_destroy(auxiliary)
        if system:
            library.generativeqc_system_destroy(system)
        library.generativeqc_context_destroy(context)
    # Include metadata-owner destruction as well as native provider cleanup in
    # the detached audit duration; the enclosing endpoint also times this scope.
    return HFPhysicalResidualAudit(
        **observations, seconds=time.perf_counter() - started
    )

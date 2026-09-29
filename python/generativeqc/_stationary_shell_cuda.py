"""Join prepared full-range shell derivatives to the shared stationary owner.

No method name, ERI recurrence or exchange coefficient is reconstructed here.
The optional native capability is authoritative; failures never select fallback.
"""

from __future__ import annotations

import ctypes as ct
import typing

import numpy as np

from . import _native


def seed_prepared_shell_sources(
    state: typing.Any,
    sources: typing.Any,
    source_names: tuple[str, ...],
    atoms: int,
    *,
    ecp: bool,
    maximum_host_bytes: int,
) -> tuple[tuple[str, ...], dict[str, typing.Any]]:
    """Seed [J,K] once, or report an explicitly unavailable optional capability."""
    fallback = {"route": "bounded-public-ao", "reason": "unavailable-native-shell-abi"}
    if ecp:
        return (), {**fallback, "reason": "ecp-uses-existing-stationary-route"}
    native_library = getattr(state._source, "_library", None)
    execute = getattr(
        native_library,
        "generativeqc_ks_snapshot_cuda_full_range_shell_gradient_v1",
        None,
    )
    seed = getattr(
        getattr(sources, "library", None), "stationary_seed_sources_v1", None
    )
    # Check BOTH ends before executing the provider. Older packaged artifacts
    # remain on the bounded oracle rather than silently losing shell gradients.
    if execute is None or seed is None:
        return (), fallback
    if type(atoms) is not int or not 1 <= atoms <= 128:
        raise ValueError("invalid shell source atom count")
    if len(set(source_names)) != len(source_names) or "coulomb" not in source_names:
        raise ValueError(
            "stationary shell handoff requires unique source names and Coulomb"
        )
    pointer = ct.POINTER(ct.c_double)
    execute.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        pointer,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_uint64),
    ]
    execute.restype = ct.c_int
    values = np.full((2, atoms, 3), np.nan, dtype=np.float64)
    retained = ct.c_uint64()
    batch = state._source._batch
    status = execute(
        batch._batch,
        state._source._handle,
        values.ctypes.data_as(pointer),
        values.size,
        maximum_host_bytes,
        ct.byref(retained),
    )
    if status == 3:  # GENERATIVEQC_STATUS_NOT_IMPLEMENTED: capability only.
        return (), {
            **fallback,
            "reason": "prepared-owner-has-no-full-range-shell-capability",
        }
    _native.check(native_library, status, context=batch._context)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("nonfinite prepared shell derivative result")
    has_exchange = "exact_exchange" in source_names
    if not has_exchange and np.any(values[1] != 0.0):
        raise RuntimeError("unexpected exchange derivative for a semilocal source plan")
    selected = ("coulomb", "exact_exchange") if has_exchange else ("coulomb",)
    seeds = np.zeros((len(source_names), atoms, 3), dtype=np.float64)
    for index, name in enumerate(selected):
        seeds[source_names.index(name)] = values[index]
    seed.argtypes = [ct.c_void_p, pointer, ct.c_size_t, ct.c_char_p, ct.c_size_t]
    seed.restype = ct.c_int
    sources._call(
        "stationary_seed_sources_v1",
        sources.handle,
        seeds.ctypes.data_as(pointer),
        seeds.size,
    )
    return selected, {
        "route": "prepared-direct-shell",
        "sources": selected,
        "retained_device_bytes": retained.value,
        "source_seed_h2d_bytes": seeds.nbytes,
        "shell_work_counts": None,
        "shell_work_scope": "native screening/compaction counters are not exported by v1",
    }

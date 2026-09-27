"""Normalize DFT CUDA force telemetry into one cross-functional component schema.

The normalizer is deliberately lossless about missing measurements: unavailable
timings are serialized as None rather than inferred as zero. Host-wall
components and profiler/device observations are kept in separate namespaces so
profiling synchronization cannot be added to clean endpoint timing.
"""

from __future__ import annotations

import math
import typing
from collections.abc import Mapping, Sequence

COMPONENTS = (
    "scf_fock_j",
    "scf_full_range_k",
    "scf_short_range_k",
    "scf_long_range_k",
    "semilocal_ao_grid_xc",
    "stationary_integral_derivatives",
    "semilocal_geometry_response",
    "vv10_rvv10",
    "host_packing",
    "h2d_d2h",
    "synchronization_fences",
    "compile_aot_cache_setup",
    "final_reduction_assembly",
)


def _finite_nonnegative(value: typing.Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a duration/count, not bool")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _mapping(value: typing.Any) -> Mapping[str, typing.Any]:
    return value if isinstance(value, Mapping) else {}


def _sum_present(
    values: Sequence[typing.Any], *, field: str
) -> float | None:
    present = [
        _finite_nonnegative(value, field=field)
        for value in values
        if value is not None
    ]
    if not present:
        return None
    return sum(typing.cast(list[float], present))


def _value(
    mapping: Mapping[str, typing.Any], key: str, *, field: str
) -> float | None:
    return _finite_nonnegative(mapping.get(key), field=field)


def select_force_work(raw: typing.Any, *, index: int = 0) -> Mapping[str, typing.Any]:
    """Select one force-work mapping from public-batch diagnostics or direct work."""

    if isinstance(raw, Mapping):
        nested = raw.get("work")
        if isinstance(nested, Mapping):
            return nested
        return raw
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        candidates = [
            entry
            for entry in raw
            if isinstance(entry, Mapping)
            and int(entry.get("index", index)) == index
            and isinstance(entry.get("work"), Mapping)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"expected one force-work record for index {index}, got {len(candidates)}"
            )
        return typing.cast(Mapping[str, typing.Any], candidates[0]["work"])
    raise TypeError("force work must be a mapping or indexed diagnostic sequence")


def _empty_components() -> dict[str, float | None]:
    return {name: None for name in COMPONENTS}


def _coverage(
    wall: Mapping[str, float | None], profiled_ms: Mapping[str, float | None]
) -> dict[str, list[str]]:
    return {
        "wall_seconds": [name for name, value in wall.items() if value is not None],
        "profiled_ms": [
            name for name, value in profiled_ms.items() if value is not None
        ],
        "missing_wall_seconds": [
            name for name, value in wall.items() if value is None
        ],
    }


def _normalize_wb97mv(
    work: Mapping[str, typing.Any], *, state_export_seconds: float
) -> dict[str, typing.Any]:
    component = _mapping(work.get("component_seconds"))
    wall = _empty_components()
    profiled_ms = _empty_components()

    wall["stationary_integral_derivatives"] = _value(
        component,
        "integral_derivatives",
        field="component_seconds.integral_derivatives",
    )
    wall["semilocal_geometry_response"] = _value(
        component,
        "semilocal_geometry_and_features",
        field="component_seconds.semilocal_geometry_and_features",
    )
    wall["vv10_rvv10"] = _sum_present(
        (
            component.get("vv10_pairs"),
            component.get("nonlocal_geometry"),
        ),
        field="component_seconds.vv10_rvv10",
    )
    wall["compile_aot_cache_setup"] = _value(
        component, "prepare", field="component_seconds.prepare"
    )
    wall["final_reduction_assembly"] = _value(
        component,
        "reduction_and_validation",
        field="component_seconds.reduction_and_validation",
    )
    if state_export_seconds:
        wall["host_packing"] = state_export_seconds

    native = _mapping(work.get("native_integral_resources"))
    snapshot = _mapping(work.get("snapshot_export_work"))
    traffic = {
        "one_electron_h2d_bytes": int(native.get("one_electron_h2d_bytes", 0)),
        "one_electron_d2h_bytes": int(native.get("one_electron_d2h_bytes", 0)),
        "final_state_export_d2h_bytes": int(
            native.get("final_state_export_d2h_bytes", 0)
        ),
        "snapshot_export_d2h_bytes": int(snapshot.get("d2h_bytes", 0)),
        "final_state_export_synchronizations": int(
            native.get("final_state_export_synchronizations", 0)
        ),
        "snapshot_export_synchronizations": int(
            snapshot.get("synchronizations", 0)
        ),
    }
    endpoint = _finite_nonnegative(
        work.get("endpoint_seconds"), field="endpoint_seconds"
    )
    if endpoint is not None:
        endpoint += state_export_seconds
    attributed = sum(value for value in wall.values() if value is not None)
    unattributed = None if endpoint is None else max(0.0, endpoint - attributed)
    return {
        "schema": "vibeqc.dft-force-components.v1",
        "source_route": "wb97mv-component-seconds",
        "wall_seconds": wall,
        "profiled_ms": profiled_ms,
        "traffic": traffic,
        "endpoint_seconds": endpoint,
        "attributed_wall_seconds": attributed,
        "unattributed_wall_seconds": unattributed,
        "coverage": _coverage(wall, profiled_ms),
        "notes": {
            "semilocal_geometry_response": (
                "source timer combines semilocal AO/grid feature construction "
                "with geometry response"
            ),
            "missing": "null means not measured by this telemetry source; never zero-filled",
        },
    }


def _normalize_stationary(
    work: Mapping[str, typing.Any], *, state_export_seconds: float
) -> dict[str, typing.Any]:
    timeline = _mapping(work.get("timeline"))
    phases = _mapping(timeline.get("exclusive_wall_seconds"))
    device = _mapping(work.get("device_phase_ms"))
    transfer = _mapping(work.get("transfer_work"))

    wall = _empty_components()
    profiled_ms = _empty_components()

    wall["stationary_integral_derivatives"] = _value(
        phases,
        "primitive_derivative_reduction_sync",
        field="timeline.primitive_derivative_reduction_sync",
    )
    wall["semilocal_geometry_response"] = _value(
        phases,
        "xc_geometry_and_sync",
        field="timeline.xc_geometry_and_sync",
    )
    wall["host_packing"] = _sum_present(
        (state_export_seconds, phases.get("python_packing")),
        field="timeline.host_packing",
    )
    wall["compile_aot_cache_setup"] = _sum_present(
        (
            phases.get("preparation"),
            phases.get("owner_construction"),
            phases.get("prepared_owner_lookup_or_construction"),
        ),
        field="timeline.compile_aot_cache_setup",
    )
    wall["final_reduction_assembly"] = _value(
        phases, "final_reduction", field="timeline.final_reduction"
    )

    profiled_ms["stationary_integral_derivatives"] = _sum_present(
        (
            device.get("primitive_derivative_kernel"),
            device.get("primitive_reduction"),
        ),
        field="device_phase_ms.stationary_integral_derivatives",
    )
    profiled_ms["semilocal_geometry_response"] = _sum_present(
        (device.get("geometry_kernel"), device.get("geometry_reduction")),
        field="device_phase_ms.semilocal_geometry_response",
    )
    profiled_ms["h2d_d2h"] = _sum_present(
        (
            device.get("setup_transfer_and_clear"),
            device.get("primitive_h2d"),
            device.get("geometry_h2d"),
            device.get("final_d2h_wall"),
        ),
        field="device_phase_ms.h2d_d2h",
    )
    profiled_ms["synchronization_fences"] = _value(
        device,
        "synchronization_wait_wall",
        field="device_phase_ms.synchronization_wait_wall",
    )

    traffic = {
        "source_h2d_bytes": int(transfer.get("source_h2d_bytes", work.get("h2d_bytes", 0))),
        "source_d2h_bytes": int(transfer.get("source_d2h_bytes", work.get("d2h_bytes", 0))),
        "source_h2d_calls": int(transfer.get("source_h2d_calls", work.get("h2d_calls", 0))),
        "source_d2h_calls": int(transfer.get("source_d2h_calls", work.get("d2h_calls", 0))),
        "source_synchronizations": int(work.get("synchronizations", 0)),
        "tensor_h2d_numeric_bytes": int(transfer.get("tensor_h2d_numeric_bytes", 0)),
        "tensor_d2h_bytes": int(transfer.get("tensor_d2h_bytes", 0)),
    }
    endpoint = _finite_nonnegative(
        work.get("endpoint_seconds", timeline.get("endpoint_seconds")),
        field="endpoint_seconds",
    )
    if endpoint is not None:
        endpoint += state_export_seconds
    attributed = sum(value for value in wall.values() if value is not None)
    unattributed = None if endpoint is None else max(0.0, endpoint - attributed)
    return {
        "schema": "vibeqc.dft-force-components.v1",
        "source_route": "stationary-exclusive-wall",
        "wall_seconds": wall,
        "profiled_ms": profiled_ms,
        "traffic": traffic,
        "endpoint_seconds": endpoint,
        "attributed_wall_seconds": attributed,
        "unattributed_wall_seconds": unattributed,
        "coverage": _coverage(wall, profiled_ms),
        "notes": {
            "profiled_ms": (
                "profiler/device observations are diagnostic and are not added "
                "to clean host-wall endpoint timing"
            ),
            "missing": "null means not measured by this telemetry source; never zero-filled",
        },
    }


def normalize_force_work(
    raw: typing.Any, *, index: int = 0, state_export_seconds: float = 0.0
) -> dict[str, typing.Any]:
    """Return one schema across semilocal/hybrid stationary and WB97M-V work."""

    state_export = _finite_nonnegative(
        state_export_seconds, field="state_export_seconds"
    )
    assert state_export is not None
    work = select_force_work(raw, index=index)
    if isinstance(work.get("component_seconds"), Mapping) or str(
        work.get("execution", "")
    ).startswith("cuda-complete-wb97mv"):
        return _normalize_wb97mv(work, state_export_seconds=state_export)
    return _normalize_stationary(work, state_export_seconds=state_export)

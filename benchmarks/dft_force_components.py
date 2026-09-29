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
        raise TypeError(f"{field} must be a duration/count, not bool")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _mapping(value: typing.Any) -> Mapping[str, typing.Any]:
    return value if isinstance(value, Mapping) else {}


def _sum_present(values: Sequence[typing.Any], *, field: str) -> float | None:
    present = [
        _finite_nonnegative(value, field=field) for value in values if value is not None
    ]
    if not present:
        return None
    return sum(typing.cast("list[float]", present))


def _value(mapping: Mapping[str, typing.Any], key: str, *, field: str) -> float | None:
    return _finite_nonnegative(mapping.get(key), field=field)


def _int_or_none(value: typing.Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("work counter must not be bool")
    result = int(value)
    if result < 0:
        raise ValueError("work counter must be nonnegative")
    return result


def _first_present(*values: typing.Any) -> typing.Any:
    return next((value for value in values if value is not None), None)


def _unattributed(endpoint: float | None, attributed: float) -> float | None:
    if endpoint is None:
        return None
    remainder = endpoint - attributed
    tolerance = max(1.0e-9, 1.0e-6 * max(endpoint, 1.0))
    if remainder < -tolerance:
        raise ValueError("component wall timings exceed measured endpoint")
    return max(0.0, remainder)


def select_force_work(raw: typing.Any, *, index: int = 0) -> Mapping[str, typing.Any]:
    """Select the latest force-work mapping for one prepared-batch item."""

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
        if not candidates:
            raise ValueError(f"no force-work record for index {index}")
        # Prepared-batch resource diagnostics are cumulative. The last matching
        # record is the latest force execution for this item.
        return typing.cast("Mapping[str, typing.Any]", candidates[-1]["work"])
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
        "missing_wall_seconds": [name for name, value in wall.items() if value is None],
    }


def _normalize_wb97mv(
    work: Mapping[str, typing.Any], *, state_export_seconds: float | None
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
    legacy_vv10 = _sum_present(
        (
            component.get("vv10_pairs"),
            component.get("nonlocal_geometry"),
        ),
        field="component_seconds.vv10_rvv10",
    )
    wall["vv10_rvv10"] = (
        legacy_vv10
        if legacy_vv10 is not None
        else _sum_present(
            (
                component.get("nonlocal_reset"),
                component.get("vv10_pair_enqueue"),
                component.get("nonlocal_geometry_and_pair_drain"),
            ),
            field="component_seconds.resident_vv10_rvv10",
        )
    )
    wall["compile_aot_cache_setup"] = _value(
        component, "prepare", field="component_seconds.prepare"
    )
    wall["final_reduction_assembly"] = _value(
        component,
        "reduction_and_validation",
        field="component_seconds.reduction_and_validation",
    )
    if state_export_seconds is not None:
        wall["host_packing"] = state_export_seconds

    native = _mapping(work.get("native_integral_resources"))
    snapshot = _mapping(work.get("snapshot_export_work"))
    traffic = {
        "one_electron_h2d_bytes": _int_or_none(native.get("one_electron_h2d_bytes")),
        "one_electron_d2h_bytes": _int_or_none(native.get("one_electron_d2h_bytes")),
        "final_state_export_d2h_bytes": _int_or_none(
            native.get("final_state_export_d2h_bytes")
        ),
        "snapshot_export_d2h_bytes": _int_or_none(snapshot.get("d2h_bytes")),
        "final_state_export_synchronizations": _int_or_none(
            native.get("final_state_export_synchronizations")
        ),
        "snapshot_export_synchronizations": _int_or_none(
            snapshot.get("synchronizations")
        ),
    }
    endpoint = _finite_nonnegative(
        work.get("endpoint_seconds"), field="endpoint_seconds"
    )
    if endpoint is not None and state_export_seconds is not None:
        endpoint += state_export_seconds
    attributed = sum(value for value in wall.values() if value is not None)
    return {
        "schema": "generativeqc.dft-force-components.v1",
        "source_route": "wb97mv-component-seconds",
        "wall_seconds": wall,
        "profiled_ms": profiled_ms,
        "traffic": traffic,
        "source_component_seconds": dict(component),
        "endpoint_seconds": endpoint,
        "attributed_wall_seconds": attributed,
        "unattributed_wall_seconds": _unattributed(endpoint, attributed),
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
    work: Mapping[str, typing.Any], *, state_export_seconds: float | None
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
    legacy_geometry = _value(
        phases,
        "xc_geometry_and_sync",
        field="timeline.xc_geometry_and_sync",
    )
    wall["semilocal_geometry_response"] = (
        legacy_geometry
        if legacy_geometry is not None
        else _sum_present(
            (
                phases.get("xc_geometry_enqueue"),
                phases.get("xc_geometry_drain"),
            ),
            field="timeline.xc_geometry_enqueue_drain",
        )
    )
    wall["host_packing"] = _sum_present(
        tuple(
            value
            for value in (state_export_seconds, phases.get("python_packing"))
            if value is not None
        ),
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
        "source_h2d_bytes": _int_or_none(
            _first_present(transfer.get("source_h2d_bytes"), work.get("h2d_bytes"))
        ),
        "source_d2h_bytes": _int_or_none(
            _first_present(transfer.get("source_d2h_bytes"), work.get("d2h_bytes"))
        ),
        "source_h2d_calls": _int_or_none(
            _first_present(transfer.get("source_h2d_calls"), work.get("h2d_calls"))
        ),
        "source_d2h_calls": _int_or_none(
            _first_present(transfer.get("source_d2h_calls"), work.get("d2h_calls"))
        ),
        "source_synchronizations": _int_or_none(work.get("synchronizations")),
        "tensor_h2d_numeric_bytes": _int_or_none(
            transfer.get("tensor_h2d_numeric_bytes")
        ),
        "tensor_d2h_bytes": _int_or_none(transfer.get("tensor_d2h_bytes")),
    }
    endpoint = _finite_nonnegative(
        _first_present(work.get("endpoint_seconds"), timeline.get("endpoint_seconds")),
        field="endpoint_seconds",
    )
    if endpoint is not None and state_export_seconds is not None:
        endpoint += state_export_seconds
    attributed = sum(value for value in wall.values() if value is not None)
    return {
        "schema": "generativeqc.dft-force-components.v1",
        "source_route": "stationary-exclusive-wall",
        "wall_seconds": wall,
        "profiled_ms": profiled_ms,
        "traffic": traffic,
        "endpoint_seconds": endpoint,
        "attributed_wall_seconds": attributed,
        "unattributed_wall_seconds": _unattributed(endpoint, attributed),
        "coverage": _coverage(wall, profiled_ms),
        "notes": {
            "profiled_ms": (
                "profiler/device observations are diagnostic and are not added "
                "to clean host-wall endpoint timing"
            ),
            "missing": "null means not measured by this telemetry source; never zero-filled",
        },
    }


_EXCHANGE_COMPONENT = {
    "full-range": "scf_full_range_k",
    "short-range": "scf_short_range_k",
    "long-range": "scf_long_range_k",
}
_SCF_TRACE_COMPONENT = {
    "ri_j": "scf_fock_j",
    "ri_k": "scf_full_range_k",
    "ri_k_occupied": "scf_full_range_k",
    "ri_k_short_range": "scf_short_range_k",
    "ri_k_long_range": "scf_long_range_k",
}


def _accumulate_component(
    target: dict[str, float | None], name: str, value: typing.Any, *, field: str
) -> None:
    measured = _finite_nonnegative(value, field=field)
    if measured is None:
        return
    previous = target.get(name)
    target[name] = measured if previous is None else previous + measured


def expected_scf_components(
    exchange_operators: Sequence[str],
    *,
    semilocal: bool = True,
    nonlocal_correlation: bool = False,
) -> tuple[str, ...]:
    """Return method-graph component names without using a named functional."""

    expected = ["scf_fock_j"]
    if semilocal:
        expected.append("semilocal_ao_grid_xc")
    if nonlocal_correlation:
        expected.append("vv10_rvv10")
    for operator in exchange_operators:
        try:
            component = _EXCHANGE_COMPONENT[operator]
        except KeyError as error:
            raise ValueError(f"unsupported exchange operator: {operator}") from error
        if component not in expected:
            expected.append(component)
    return tuple(expected)


def normalize_scf_trace(
    records: Sequence[Mapping[str, typing.Any]],
    *,
    exchange_operators: Sequence[str] = (),
    semilocal: bool = True,
    nonlocal_correlation: bool = False,
) -> dict[str, typing.Any]:
    """Normalize opt-in CUDA SCF traces without splitting fused J/K evidence.

    The native DF trace is intrusive diagnostic evidence. Its CUDA-event times
    never become clean endpoint wall time. A shared J/K root cannot be assigned
    to J and K separately, so that duration remains explicitly ambiguous.
    Direct/RSH providers that do not emit this trace remain missing rather than
    receiving inferred timings.
    """

    from benchmarks.df_component_ledger import aggregate

    summary = aggregate(list(records))
    profiled = _empty_components()
    ambiguous: dict[str, float] = {}
    unclassified: dict[str, float] = {}
    for group in summary["groups"]:
        if group["execution"] != "stream":
            continue
        operation = str(group["operation"])
        milliseconds = _finite_nonnegative(
            group.get("gpu_inclusive_ms"),
            field=f"scf_trace.{operation}.gpu_inclusive_ms",
        )
        if milliseconds is None:
            continue
        component = _SCF_TRACE_COMPONENT.get(operation)
        if component is not None:
            _accumulate_component(
                profiled,
                component,
                milliseconds,
                field=f"scf_trace.{operation}.gpu_inclusive_ms",
            )
        elif operation == "ri_jk_shared":
            ambiguous["scf_shared_jk"] = (
                ambiguous.get("scf_shared_jk", 0.0) + milliseconds
            )
        else:
            unclassified[operation] = unclassified.get(operation, 0.0) + milliseconds

    expected = expected_scf_components(
        exchange_operators,
        semilocal=semilocal,
        nonlocal_correlation=nonlocal_correlation,
    )
    missing = [name for name in expected if profiled[name] is None]
    return {
        "schema": "generativeqc.dft-scf-components.v1",
        "profiled_ms": profiled,
        "expected_components": list(expected),
        "missing_expected_components": missing,
        "ambiguous_profiled_ms": ambiguous,
        "unclassified_root_profiled_ms": unclassified,
        "source_summary": summary,
        "measurement_policy": (
            "opt-in CUDA-event diagnostic; do not add to clean endpoint wall time; "
            "shared J/K roots remain unsplit"
        ),
    }


def merge_scf_profile(
    force_components: Mapping[str, typing.Any],
    scf_profile: Mapping[str, typing.Any],
) -> dict[str, typing.Any]:
    """Attach measured SCF component events to one force-component record."""

    if force_components.get("schema") != "generativeqc.dft-force-components.v1":
        raise ValueError("force component schema mismatch")
    if scf_profile.get("schema") != "generativeqc.dft-scf-components.v1":
        raise ValueError("SCF component schema mismatch")
    result = {
        **force_components,
        "wall_seconds": dict(_mapping(force_components.get("wall_seconds"))),
        "profiled_ms": dict(_mapping(force_components.get("profiled_ms"))),
        "notes": dict(_mapping(force_components.get("notes"))),
    }
    incoming = _mapping(scf_profile.get("profiled_ms"))
    for name in COMPONENTS:
        value = incoming.get(name)
        if value is None:
            continue
        measured = _finite_nonnegative(value, field=f"scf_profile.{name}")
        previous = result["profiled_ms"].get(name)
        if previous is not None and not math.isclose(
            float(previous), typing.cast("float", measured), rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(f"duplicate component timing disagrees for {name}")
        result["profiled_ms"][name] = measured
    result["coverage"] = _coverage(result["wall_seconds"], result["profiled_ms"])
    result["scf_profile"] = {
        key: scf_profile.get(key)
        for key in (
            "expected_components",
            "missing_expected_components",
            "ambiguous_profiled_ms",
            "unclassified_root_profiled_ms",
            "measurement_policy",
        )
    }
    return result


def normalize_force_work(
    raw: typing.Any,
    *,
    index: int = 0,
    state_export_seconds: float | None = None,
) -> dict[str, typing.Any]:
    """Return one schema across semilocal/hybrid stationary and WB97M-V work."""

    state_export = (
        None
        if state_export_seconds is None
        else _finite_nonnegative(state_export_seconds, field="state_export_seconds")
    )
    work = select_force_work(raw, index=index)
    if isinstance(work.get("component_seconds"), Mapping) or str(
        work.get("execution", "")
    ).startswith("cuda-complete-wb97mv"):
        return _normalize_wb97mv(work, state_export_seconds=state_export)
    return _normalize_stationary(work, state_export_seconds=state_export)

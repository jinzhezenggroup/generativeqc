"""Native global capacity composition for already-loaded production libraries.

The compiler's portable model remains usable without importing the runtime.
This adapter marshals that immutable metadata to the same C++ engine used by
the installed CLI. No provider is loaded or prepared to choose a candidate.
"""

from __future__ import annotations

import ctypes
import math
import typing
from dataclasses import replace

from generativeqc_compiler.common.resources import (
    MAX_BYTES,
    ResourceBudget,
    ResourcePlan,
    ResourceRequest,
    checked_bytes,
)
from generativeqc_compiler.common.resources import (
    plan_resources as _portable_plan_resources,
)

_BASE_SPACES = ("host", "device", "pinned", "pageable")
_ABSENT = 2**64 - 1


def _native_composition(
    requests: tuple[ResourceRequest, ...],
    budget: ResourceBudget,
    query: typing.Any,
    maximum_combinations: int,
) -> dict[str, typing.Any]:
    """Marshal caller-labelled spaces; never infer a device or alter a policy.

    Device labels retain the portable model's validated ordinal semantics.
    Aggregate host/device slots cannot directly own buffers. Peak phase zero and
    an absent peak are distinguished in the wire representation.
    """
    spaces = _BASE_SPACES + tuple(
        sorted(
            {f"device:{device}" for device, _ in budget.per_device_bytes}
            | {
                estimate.space
                for request in requests
                for candidate in request.candidates
                for estimate in candidate.estimates
                if estimate.space.startswith("device:")
            }
        )
    )
    slots = {space: index for index, space in enumerate(spaces)}
    request_counts = []
    candidates = []
    buffers = []
    for request in requests:
        request_counts.append(len(request.candidates))
        ranks = {
            name: rank
            for rank, name in enumerate(
                sorted(candidate.name for candidate in request.candidates)
            )
        }
        for candidate in request.candidates:
            candidates.extend(
                (
                    candidate.relative_cost,
                    ranks[candidate.name],
                    len(candidate.estimates),
                )
            )
            for estimate in candidate.estimates:
                buffers.extend(
                    (
                        estimate.bytes,
                        slots[estimate.space],
                        estimate.first_phase,
                        estimate.last_phase,
                        int(estimate.kind != "workspace"),
                    )
                )
    limits = []
    limit_order = []
    for space, capacity, reserve in (
        ("host", budget.host_bytes, budget.host_reserve_bytes),
        ("device", budget.device_bytes, budget.device_reserve_bytes),
        ("pinned", budget.pinned_host_bytes, 0),
        *((f"device:{device}", cap, 0) for device, cap in budget.per_device_bytes),
    ):
        if capacity is not None:
            limit_order.append(space)
            limits.extend((slots[space], capacity, reserve))
    inputs = [
        (ctypes.c_uint64 * len(values))(*values)
        for values in (request_counts, candidates, buffers, limits)
    ]
    output = (ctypes.c_uint64 * (2 + len(requests) + 8 * len(spaces)))()
    pointer = ctypes.POINTER(ctypes.c_uint64)
    query.argtypes = [pointer, ctypes.c_size_t] * 4 + [
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_uint64,
        pointer,
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    arguments = tuple(value for array in inputs for value in (array, len(array)))
    status = query(
        *arguments,
        len(spaces),
        budget.headroom_fraction,
        maximum_combinations,
        output,
        len(output),
    )
    if status == 3:
        raise MemoryError("native resource composition metadata allocation failed")
    if status:
        raise ValueError(
            "native resource composition rejected descriptors or integer capacity"
        )
    if output[0] not in (0, 1, 2):
        raise ValueError("native resource composition returned an invalid status")
    if not output[1] or (output[0] == 2) != (output[1] > maximum_combinations):
        raise ValueError("native resource composition returned an invalid work count")
    result: dict[str, typing.Any] = {
        "status": int(output[0]),
        "combinations": int(output[1]),
        "selections": tuple(int(value) for value in output[2 : 2 + len(requests)]),
        "peak_bytes": {},
        "phase_peaks": {},
        "resident_bytes": {},
        "resident_phase_peaks": {},
        "minima": {},
        "limits": {},
    }
    if result["status"] != 2 and any(
        selection >= len(request.candidates)
        for request, selection in zip(requests, result["selections"], strict=True)
    ):
        raise ValueError("native resource composition returned an invalid selection")
    for index, space in enumerate(spaces):
        row = output[
            2 + len(requests) + 8 * index : 2 + len(requests) + 8 * (index + 1)
        ]
        if row[2] not in (0, 1) or row[5] not in (0, 1):
            raise ValueError(
                "native resource composition returned an invalid presence flag"
            )
        for offset, key, phase_key in (
            (0, "peak_bytes", "phase_peaks"),
            (3, "resident_bytes", "resident_phase_peaks"),
        ):
            checked_bytes(int(row[offset]), "native resource peak")
            if row[offset + 1] > MAX_BYTES + 1:
                raise ValueError(
                    "native resource composition returned an invalid peak phase"
                )
            if bool(row[offset]) != bool(row[offset + 1]) or (
                not row[offset + 2] and row[offset]
            ):
                raise ValueError(
                    "native resource composition returned an inconsistent peak"
                )
            if row[offset + 2]:
                result[key][space] = int(row[offset])
                if row[offset + 1]:
                    result[phase_key][space] = int(row[offset + 1]) - 1
        for offset, key in ((6, "minima"), (7, "limits")):
            if row[offset] != _ABSENT:
                result[key][space] = checked_bytes(
                    int(row[offset]), "native resource " + key
                )
    if set(result["limits"]) != set(limit_order):
        raise ValueError("native resource composition returned incomplete limits")
    result["limits"] = {space: result["limits"][space] for space in limit_order}
    excess = any(
        result["peak_bytes"].get(space, 0) > limit
        for space, limit in result["limits"].items()
    )
    if result["status"] != 2 and (result["status"] == 1) != excess:
        raise ValueError(
            "native resource composition returned an inconsistent budget status"
        )
    return result


def plan_resources(
    requests: typing.Any,
    budget: ResourceBudget,
    *,
    maximum_combinations: int = 65536,
    library: typing.Any = None,
) -> ResourcePlan:
    """Choose supported alternatives through the discovered native owner.

    The library must already be loaded by the caller. Missing libraries/symbols
    retain portable metadata planning; discovered native errors never retry it.
    The returned portable descriptor still independently validates/serializes its
    derived accounting. It contains no native pointer or execution lease.
    """
    query = getattr(library, "generativeqc_resource_compose_v1", None)
    if query is None:
        return _portable_plan_resources(
            requests, budget, maximum_combinations=maximum_combinations
        )
    if not isinstance(budget, ResourceBudget):
        raise TypeError("a ResourceBudget is required")
    requests = tuple(sorted(requests, key=lambda request: request.name))
    if not requests or not all(
        isinstance(request, ResourceRequest) for request in requests
    ):
        raise ValueError("at least one structured resource request is required")
    if len({request.name for request in requests}) != len(requests):
        raise ValueError("duplicate resource request owner")
    for attribute, status in (
        ("unsupported_reason", "unsupported"),
        ("infeasible_reason", "infeasible"),
    ):
        reasons = [
            f"{request.name}: {getattr(request, attribute)}"
            for request in requests
            if getattr(request, attribute)
        ]
        if reasons:
            return ResourcePlan(budget, requests, (), status, "; ".join(reasons))
    checked_bytes(maximum_combinations, "planning search limit")
    result = _native_composition(requests, budget, query, maximum_combinations)
    if result["status"] == 2:
        combinations = math.prod(len(request.candidates) for request in requests)
        return ResourcePlan(
            budget,
            requests,
            (),
            "unsupported",
            f"{combinations} provider combinations exceed planning search limit {maximum_combinations}",
        )
    selections = tuple(
        (request.name, request.candidates[selection].name)
        for request, selection in zip(requests, result["selections"], strict=True)
    )
    if result["status"] == 0:
        return ResourcePlan(budget, requests, selections, "feasible")
    failures = {
        space: {
            "peak": result["peak_bytes"].get(space, 0),
            "limit": limit,
            "minimum_across_candidates": result["minima"].get(space, 0),
        }
        for space, limit in result["limits"].items()
        if result["peak_bytes"].get(space, 0) > limit
    }
    selected = dict(selections)
    estimates = tuple(
        estimate
        for request in requests
        for candidate in request.candidates
        if selected[request.name] == candidate.name
        for estimate in candidate.estimates
    )
    dominant = sorted(estimates, key=lambda estimate: (-estimate.bytes, estimate.name))[
        :3
    ]
    return ResourcePlan(
        budget,
        requests,
        selections,
        "infeasible",
        f"no supported plan fits: {failures}; dominant allocations: "
        + ", ".join(
            f"{estimate.name}={estimate.bytes} {estimate.space}"
            for estimate in dominant
        ),
    )


def _fixed_selection_fits(plan: ResourcePlan, library: typing.Any) -> bool:
    """Check fixed supported alternatives without changing any live owner."""
    query = getattr(library, "generativeqc_resource_compose_v1", None)
    if query is None:
        peaks = plan.peak_bytes
        return all(
            peaks.get(space, 0) <= limit
            for space, limit in plan.budget.limits().items()
        )
    selected = dict(plan.selections)
    if set(selected) != {request.name for request in plan.requests}:
        raise ValueError("fixed resource admission must select every request")
    requests = tuple(
        replace(
            request,
            candidates=tuple(
                candidate
                for candidate in request.candidates
                if candidate.name == selected[request.name]
            ),
        )
        for request in plan.requests
    )
    result = _native_composition(requests, plan.budget, query, 1)
    if result["status"] == 2:
        raise ValueError(
            "native fixed resource admission exceeded its single combination"
        )
    return result["status"] == 0


def validate_resource_plan(plan: ResourcePlan, *, library: typing.Any) -> None:
    """Check a supplied explicit selection before any prepared provider exists.

    Validation does not replace user choices with the cheapest native candidate.
    This covers declared capacities only, not allocator receipts or lease state.
    """
    plan.require_feasible()
    if not _fixed_selection_fits(plan, library):
        raise ValueError(
            "native resource composition rejected the supplied plan budget"
        )


def revalidate_resource_plan(
    plan: ResourcePlan, *, library: typing.Any
) -> ResourcePlan:
    """Admit revised capacities while keeping existing selections and leases.

    Optional CPU force workspace can change after an energy-only preparation.
    This returns a provisional descriptor; the caller publishes it only after
    successful scientific execution. Native query failures cannot promote the
    plan or be retried with portable arithmetic.
    """
    if _fixed_selection_fits(plan, library):
        return replace(plan, status="feasible", diagnostic=None)
    return replace(
        plan,
        status="infeasible",
        diagnostic=plan.diagnostic or "selected resource capacities exceed budget",
    )

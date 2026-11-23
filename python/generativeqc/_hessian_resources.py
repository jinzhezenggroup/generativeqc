"""Compose native Hessian reservations with the prepared KS resource request."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

from generativeqc_compiler.common.resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
)

from .resources_planning import plan_resources

if TYPE_CHECKING:
    from generativeqc_compiler.common.resources import ResourceBudget, ResourcePlan

    from ._native_cpu_hessian import _Options as _CpuOptions
    from ._native_hessian import _Options
    from .calculator import Atom, Calculator

HESSIAN_RESOURCE_OWNER = "ks-hessian"
CPU_HESSIAN_PROVIDER = "cpu-stationary-hessian-v1"
CPU_HESSIAN_CAPACITY = "SCF-overlapped native Hessian numerical cap"


def _last_phase(requests: tuple[ResourceRequest, ...]) -> int:
    """Conservatively overlap the derivative with every possible primal phase."""
    return max(
        (
            estimate.last_phase
            for request in requests
            for candidate in request.candidates
            for estimate in candidate.estimates
        ),
        default=0,
    )


def plan_cpu_hessian(
    calculator: Calculator,
    atoms: tuple[Atom, ...],
    *,
    charge: int,
    multiplicity: int,
    dense: bool,
    budget: ResourceBudget,
    options: _CpuOptions,
    minimum: int,
    staging_bytes: int,
) -> tuple[ResourcePlan, ResourceRequest]:
    """Compose the native mandatory byte floor with all live primal owners."""
    library = calculator._library
    if not hasattr(library, "generativeqc_resource_compose_v1"):
        raise NotImplementedError(
            "native CPU Hessian lacks shared ResourceBudget planning"
        )
    systems, charges, multiplicities = (atoms,), (charge,), (multiplicity,)
    ks_options = calculator._effective_ks_options(
        systems, charges=charges, multiplicities=multiplicities
    )
    scf = calculator._resource_request(
        systems,
        charges=charges,
        multiplicities=multiplicities,
        ks_options=ks_options,
        properties=("energy",),
    )
    correction = calculator._dispersion_resource_request(systems, scf)
    prerequisites = (scf,) if correction is None else (scf, correction)
    last_phase = _last_phase(prerequisites)

    def request(capacity: int) -> ResourceRequest:
        return ResourceRequest(
            HESSIAN_RESOURCE_OWNER,
            ResourceIdentity(
                method=scf.identity.method,
                provider=CPU_HESSIAN_PROVIDER,
                backend="cpu",
                precision="fp64",
                topology=json.dumps(
                    {"scf": scf.identity.identity, "atoms": len(atoms)}
                ),
                observables=("hessian" if dense else "hvp",),
                schedule="scalar-columns" if dense else "action-only",
            ),
            (
                ResourceCandidate(
                    "bounded-response",
                    "tiled",
                    (
                        ResourceEstimate(
                            CPU_HESSIAN_CAPACITY,
                            capacity,
                            "pageable",
                            0,
                            last_phase,
                        ),
                        ResourceEstimate(
                            "Hessian geometry and direction marshalling",
                            staging_bytes,
                            "pageable",
                            0,
                            last_phase,
                        ),
                    ),
                ),
            ),
            scope_exclusions=(
                "final-frame export scratch and implicit BLAS workspace",
                "caller input storage, Python object and allocator overhead, and process RSS",
            ),
        )

    provisional = plan_resources(
        (*prerequisites, request(minimum)), budget, library=library
    ).require_feasible()
    capacity = options.numeric_bytes
    if (limit := budget.limits().get("host")) is not None:
        capacity = min(capacity, minimum + limit - provisional.peak_bytes["host"])
    response = request(capacity)
    plan = replace(
        provisional,
        requests=tuple(
            response if owner.name == response.name else owner
            for owner in provisional.requests
        ),
    )
    return plan, response


def plan_cuda_hessian(
    calculator: Calculator,
    atoms: tuple[Atom, ...],
    *,
    charge: int,
    multiplicity: int,
    dense: bool,
    budget: ResourceBudget,
    options: _Options,
) -> tuple[ResourcePlan, ResourceRequest, int]:
    """Reserve staged output and a hard CUDA cap before the SCF owner exists.

    Current KS inventories report a combined preparation/execution peak.
    Conservatively overlap the response with that peak; subtracting an
    assumed post-SCF release would undercharge any live native buffers.
    """
    from generativeqc_compiler.common.resources import ResourceBudget

    if not isinstance(budget, ResourceBudget):
        raise TypeError("resource_budget must be a ResourceBudget")
    systems, charges, multiplicities = (atoms,), (charge,), (multiplicity,)
    ks_options = calculator._effective_ks_options(
        systems, charges=charges, multiplicities=multiplicities
    )
    scf = calculator._resource_request(
        systems,
        charges=charges,
        multiplicities=multiplicities,
        ks_options=ks_options,
        properties=("energy",),
    )
    correction = calculator._dispersion_resource_request(
        systems, scf, _native_ks_required=True
    )
    prerequisites = (scf,) if correction is None else (scf, correction)
    device = f"device:{calculator._device_id}"
    coordinates = byte_product(3, len(atoms))
    output = byte_product(coordinates, coordinates if dense else 1, 8)
    direction = 0 if dense else byte_product(coordinates, 8)
    # The native arena always retains a separate result candidate. This is
    # only a shape-only floor; native admission sizes the full response.
    minimum = byte_product(2, output) + direction
    if minimum > options.device_bytes:
        raise MemoryError("native Hessian buffers exceed the endpoint device limit")
    last_phase = _last_phase(prerequisites)

    def request(capacity: int) -> ResourceRequest:
        return ResourceRequest(
            HESSIAN_RESOURCE_OWNER,
            ResourceIdentity(
                method=scf.identity.method,
                provider="cuda-stationary-hessian-v2",
                backend="cuda",
                precision="fp64",
                topology=json.dumps(
                    {
                        "scf": scf.identity.identity,
                        "coordinates": coordinates,
                        "controls": {
                            name: getattr(options, name)
                            for name, _ in options._fields_[2:]
                        },
                    }
                ),
                observables=("hessian" if dense else "hvp",),
                schedule="scalar-columns" if dense else "action-only",
            ),
            (
                ResourceCandidate(
                    "bounded-response",
                    "tiled",
                    (
                        ResourceEstimate(
                            "SCF-overlapped response and terminal device buffers",
                            capacity,
                            device,
                            0,
                            last_phase,
                        ),
                        ResourceEstimate(
                            "Hessian host candidate and terminal copy",
                            byte_product(2, output),
                            "pageable",
                            0,
                            last_phase,
                            kind="output",
                        ),
                        ResourceEstimate(
                            "Frozen nuclear direction",
                            direction,
                            "pageable",
                            0,
                            last_phase,
                            kind="output",
                        ),
                    ),
                    decisions=(("response_device_cap", str(capacity)),),
                ),
            ),
            scope_exclusions=(
                "CUDA driver, context, and library-private allocations",
                "unsampled native setup staging and Python object/allocator overhead",
            ),
        )

    provisional = plan_resources(
        (*prerequisites, request(minimum)), budget, library=calculator._library
    ).require_feasible()
    peaks = provisional.peak_bytes
    limits = budget.limits()
    capacity = options.device_bytes
    for space in ("device", device):
        if space in limits:
            capacity = min(capacity, minimum + limits[space] - peaks[space])
    response = request(capacity)
    plan = replace(
        provisional,
        requests=tuple(
            response if owner.name == HESSIAN_RESOURCE_OWNER else owner
            for owner in provisional.requests
        ),
    )
    return plan, response, capacity

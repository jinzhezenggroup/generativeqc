"""Compose CUDA Hessian capacity with the prepared KS resource request."""

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

    from ._native_hessian import _Options
    from .calculator import Atom, Calculator

HESSIAN_RESOURCE_OWNER = "ks-hessian"


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
    last_phase = max(
        (
            estimate.last_phase
            for prerequisite in prerequisites
            for candidate in prerequisite.candidates
            for estimate in candidate.estimates
        ),
        default=0,
    )

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

"""Explicit bounded cold-start densities, separate from the immutable target model."""

from __future__ import annotations

import ctypes
import json
import math
import typing
from dataclasses import asdict, dataclass, replace

from generativeqc_compiler.dft.grid import GridSpec

from . import _native


@dataclass(frozen=True)
class InitialGuessSpec:
    """An opt-in HF, coarse-LDA, or projected MINAO cold start.

    HF/LDA retain the CPU FP64 all-electron restricted exact energy domain.
    MINAO is a zero-Fock occupied-ANO projection for all-electron H-Ar targets
    followed by a bounded metric-occupation admission step, and additionally
    admits CUDA restricted KS energy/force endpoints.
    Existing explicit/imported/retained densities take precedence. Preparation
    never changes the target basis, grid, functional, precision or tolerances.
    A failed preliminary solve or exhausted preparation budget keeps the core
    guess; a failed seeded target gets one core retry. Allocation failures
    propagate. This is not an automatic profitability or ground-state claim.

    The numeric cap excludes allocator/object/runtime overhead and the target
    owner. ResourceBudget additionally composes both complete host inventories.
    """

    kind: str = "hf"
    max_iterations: int = 32
    diis_history: int = 8
    energy_tolerance: float = 1e-6
    density_tolerance: float = 1e-4
    maximum_numeric_bytes: int = 256 << 20
    grid: GridSpec | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("hf", "lda", "minao"):
            raise ValueError("initial guess kind must be 'hf', 'lda', or 'minao'")
        for name, low, high in (
            ("max_iterations", 1, 64),
            ("diis_history", 1, 16),
            ("maximum_numeric_bytes", 1, 2**63 - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name in ("energy_tolerance", "density_tolerance"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.kind in ("hf", "minao"):
            if self.grid is not None:
                raise ValueError("HF/MINAO initial guesses do not use a grid")
            return
        grid = GridSpec(8, 6, 12) if self.grid is None else self.grid
        if not isinstance(grid, GridSpec):
            raise TypeError("preliminary LDA grid must be a GridSpec")
        if (
            grid.version != 1
            or grid.element_radii
            or grid.partition_iterations != 3
            or grid.coincident_tolerance != 1e-12
            or not 2 <= grid.radial_points <= 32
            or not 2 <= grid.angular_polar <= 16
            or not 4 <= grid.angular_azimuth <= 32
        ):
            raise ValueError("preliminary LDA requires an admitted v1 coarse grid")
        object.__setattr__(self, "grid", grid)

    def to_payload(self) -> dict:
        return {"schema_version": 1, **asdict(self)}

    def native(self) -> _native.InitialGuessOptionsDescriptor:
        grid = self.grid
        return _native.InitialGuessOptionsDescriptor(
            ctypes.sizeof(_native.InitialGuessOptionsDescriptor),
            _native.ABI_VERSION,
            {"hf": 1, "lda": 2, "minao": 3}[self.kind],
            self.max_iterations,
            self.diis_history,
            self.energy_tolerance,
            self.density_tolerance,
            self.maximum_numeric_bytes,
            0 if grid is None else grid.radial_points,
            0 if grid is None else grid.angular_polar,
            0 if grid is None else grid.angular_azimuth,
        )


def require_initial_guess_library(library: object) -> None:
    query = getattr(library, "generativeqc_initial_guess_options_version", None)
    if query is None:
        raise NotImplementedError("native library lacks preliminary SCF support")
    query.argtypes, query.restype = [], ctypes.c_uint32
    if query() != 1:
        raise NotImplementedError("unsupported preliminary SCF schema")


def read_initial_guess_diagnostic(
    library: object, handle: object, index: int | None = None
) -> dict | None:
    name = (
        "generativeqc_calculation_get_initial_guess_diagnostic"
        if index is None
        else "generativeqc_batch_get_initial_guess_diagnostic"
    )
    query = getattr(library, name, None)
    if query is None:
        return None
    prefix = [ctypes.c_void_p] if index is None else [ctypes.c_void_p, ctypes.c_uint32]
    query.argtypes = [*prefix, ctypes.POINTER(_native.InitialGuessDiagnosticDescriptor)]
    query.restype = ctypes.c_int
    out = _native.InitialGuessDiagnosticDescriptor(
        ctypes.sizeof(_native.InitialGuessDiagnosticDescriptor), _native.ABI_VERSION
    )
    status = (
        query(handle, ctypes.byref(out))
        if index is None
        else query(handle, index, ctypes.byref(out))
    )
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status)
    outcomes = (
        "disabled",
        "existing_density",
        "used",
        "preparation_failed",
        "budget_skipped",
        "target_retried",
    )
    if out.requested_kind not in (1, 2, 3) or out.outcome >= len(outcomes):
        raise RuntimeError("invalid initial-guess diagnostic")
    if not math.isfinite(out.preparation_seconds) or out.preparation_seconds < 0:
        raise RuntimeError("invalid preliminary SCF preparation time")
    return {
        "kind": {1: "hf", 2: "lda", 3: "minao"}[out.requested_kind],
        "outcome": outcomes[out.outcome],
        "work_counters_complete": bool(out.work_counters_complete),
        **{
            key: getattr(out, key)
            for key in (
                "preliminary_iterations",
                "preliminary_fock_builds",
                "target_attempts",
                "discarded_target_iterations",
                "discarded_target_fock_builds",
                "preparation_numeric_capacity",
                "preparation_seconds",
            )
        },
    }


# Exact occupied-ANO source primitive inventory for H-Ar (pinned MINAO table).
_MINAO_PRIMITIVES = (
    8,
    9,
    28,
    28,
    37,
    37,
    37,
    37,
    37,
    37,
    63,
    63,
    75,
    75,
    75,
    75,
    75,
    75,
)


def _minao_numeric_capacity(n: int, numbers: typing.Sequence[int]) -> int:
    """Mirror native preliminary_numeric_capacity, including strict admission.

    16 square matrices cover caller X/raw/output and validator/eigensolver
    copies. The remaining terms conservatively sum projection buffers, linear
    eigensolver arrays, AO/primitive/center inventories and bounded through-g
    overlap scratch. This is an upper bound, not measured whole-process memory.
    """
    from generativeqc_compiler.common.resources import byte_product, checked_bytes

    if type(n) is not int or n <= 0:
        raise ValueError("invalid MINAO target AO topology")
    if any(type(z) is not int or not 1 <= z <= 18 for z in numbers):
        raise ValueError("MINAO initial guess is currently qualified for H-Ar")
    source_n = checked_bytes(
        sum(
            1 if z <= 2 else 2 if z <= 4 else 5 if z <= 10 else 6 if z <= 12 else 9
            for z in numbers
        )
    )
    primitives = checked_bytes(sum(_MINAO_PRIMITIVES[z - 1] for z in numbers))
    doubles = checked_bytes(
        byte_product(16, n, n)
        + byte_product(2, n, source_n)
        + byte_product(8, n)
        + source_n
    )
    return checked_bytes(
        byte_product(8, doubles)
        + byte_product(512, checked_bytes(n + source_n))
        + byte_product(256, len(numbers))
        + byte_product(16, primitives)
        + 8192
    )


def with_initial_guess_resources(
    request: typing.Any,
    calculator: typing.Any,
    systems: typing.Any,
    charges: typing.Any,
    multiplicities: typing.Any,
) -> typing.Any:
    """Conservatively overlap the existing target and preparation inventories.

    There is no second resource planner or uncharged memory allowance. Reserving
    both complete inventories may overestimate serialized/transient overlap,
    but cannot hide a preliminary owner behind the target's declared budget.
    """
    policy = calculator._initial_guess
    if policy is None:
        return request
    schedule = {
        **json.loads(request.identity.schedule),
        "initial_guess": policy.to_payload(),
    }
    identity = replace(request.identity, schedule=json.dumps(schedule, sort_keys=True))
    if policy.kind == "minao":
        from generativeqc_compiler.common.resources import ResourceEstimate

        items = json.loads(request.identity.topology)["items"]
        if len(items) != len(systems):
            raise ValueError("MINAO topology does not match the batch")
        retained = 0
        largest_workspace = 0
        for item in items:
            numbers = item["electrons"]["atomic_numbers"]
            if any(type(z) is not int or z < 1 or z > 18 for z in numbers):
                return replace(
                    request,
                    identity=identity,
                    candidates=(),
                    unsupported_reason="MINAO initial guess is currently qualified for H-Ar",
                )
            n = item["orbital"]["nbf"]
            if type(n) is not int or n <= 0:
                raise ValueError("invalid MINAO target AO topology")
            from generativeqc_compiler.common.resources import (
                byte_product,
                checked_bytes,
            )

            seed = byte_product(8, n, n)
            # The same complete preparation bound is used by native admission.
            # One output matrix is retained separately for every batch member;
            # preparation is serialized, so only the largest remainder is live.
            workspace = _minao_numeric_capacity(n, numbers) - seed
            retained = checked_bytes(retained + seed)
            largest_workspace = max(largest_workspace, workspace)
        extra = (
            ResourceEstimate(
                "all retained MINAO cold seeds", retained, "pageable", 0, 0
            ),
            ResourceEstimate(
                "largest serialized MINAO projection workspace",
                largest_workspace,
                "pageable",
                0,
                0,
            ),
        )
        return replace(
            request,
            identity=identity,
            candidates=tuple(
                replace(c, estimates=(*c.estimates, *extra)) for c in request.candidates
            ),
        )
    if policy.kind == "lda":
        # The native provider declines >f before constructing a source owner.
        # Use the target's already validated, ordered topology rather than
        # constructing an inadmissible KS inventory or swallowing its errors.
        items = json.loads(request.identity.topology)["items"]
        if len(items) != len(systems):
            raise ValueError("preliminary SCF topology does not match the batch")
        angular = [item["orbital"]["maximum_angular"] for item in items]
        if any(type(value) is not int or value < 0 for value in angular):
            raise ValueError("invalid preliminary SCF angular topology")
        eligible = [i for i, value in enumerate(angular) if value <= 3]
        if not eligible:
            return replace(request, identity=identity)
        systems = tuple(systems[i] for i in eligible)
        charges = tuple(items[i]["electrons"]["ionic_charge"] for i in eligible)
        multiplicities = tuple(items[i]["electrons"]["multiplicity"] for i in eligible)
    common = {
        "basis": calculator._basis,
        "basis_representation": calculator._representation_name,
        "backend": "cpu",
        "precision": "fp64",
        "charges": charges,
        "multiplicities": multiplicities,
        "diis_history": policy.diis_history,
        "max_iterations": policy.max_iterations,
        "energy_tolerance": policy.energy_tolerance,
        "density_tolerance": policy.density_tolerance,
        "screening_tolerance": 1e-12,
        "library": calculator._library,
        "name": "preliminary-scf",
    }
    if policy.kind == "hf":
        from .resources_hf import hf_resource_request

        preliminary = hf_resource_request(systems, method="rhf", **common)
    else:
        from .ks import KsOptions
        from .resources_ks import ks_resource_request

        preliminary = ks_resource_request(
            systems, method="lda-rks", ks_options=KsOptions(grid=policy.grid), **common
        )
    if not preliminary.candidates:
        return replace(
            request,
            identity=identity,
            candidates=(),
            unsupported_reason=preliminary.unsupported_reason,
            infeasible_reason=preliminary.infeasible_reason,
        )
    if len(preliminary.candidates) != 1:
        raise NotImplementedError(
            "preliminary SCF requires one explicit CPU resource provider"
        )
    extra = tuple(
        replace(e, name=f"preliminary {e.name}")
        for e in preliminary.candidates[0].estimates
    )
    return replace(
        request,
        identity=identity,
        candidates=tuple(
            replace(c, estimates=(*c.estimates, *extra)) for c in request.candidates
        ),
        scope_exclusions=tuple(
            dict.fromkeys((*request.scope_exclusions, *preliminary.scope_exclusions))
        ),
    )

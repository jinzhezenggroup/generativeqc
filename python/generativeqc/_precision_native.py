"""Transport completed precision evidence to the shared native admission policy.

Only absence of the additive protocol retains compatibility. A selected native
refusal cannot retry Python admission arithmetic. Model/gradient qualification,
observable audits and solver/seed lifetimes keep their separate owners.
"""

from __future__ import annotations

import ctypes as ct
from typing import Any

from . import _native
from ._precision_work import _ARITHMETIC_MODES, _DTYPES, _EVENT_KINDS, _OPERATOR_NAMES


class _Context(ct.Structure):
    """Copied model traits, evidence presence and unchanged solver controls."""

    _fields_ = (
        [
            (name, ct.c_uint32)
            for name in (
                "struct_size",
                "abi_version",
                "ks_model",
                "full_range_exchange",
                "range_exchange",
                "nonlocal_correlation",
                "hartree_fock",
                "density_fitted",
                "warm_start_used",
                "warm_start_fallback",
                "gradient_present",
                "gradient_fp64",
                "gradient_qualified",
                "census_presence",
                "converged",
                "cuda_execution",
                "fp64_candidate",
                "reserved",
            )
        ]
        + [(name, ct.c_double) for name in ("energy_abs", "force_max_abs", "force_rms")]
        + [
            ("requested_controls", ct.c_double * 3),
            ("standard_controls", ct.c_double * 3),
        ]
    )


class _Decision(ct.Structure):
    """Versioned booleans; never an observable error certificate."""

    _fields_ = [
        (name, ct.c_uint32)
        for name in (
            "struct_size",
            "abi_version",
            "policy_version",
            "budget_fock",
            "strict_warm",
            "standard_fallback",
        )
    ]


_OPERATORS = {name: value for value, name in _OPERATOR_NAMES.items()}
_TYPES = {name: value for value, name in _DTYPES.items()}
_MODES = {name: value for value, name in _ARITHMETIC_MODES.items()}
_EVENTS = {name: value for value, name in _EVENT_KINDS.items()}


def _uint(value: Any, bits: int = 64) -> int:
    """Reject malformed detached diagnostics instead of wrapping C integer widths."""
    if type(value) is not int or not 0 <= value < 2**bits:
        raise ValueError(f"precision evidence must be a uint{bits} integer")
    return value


def _state(value: Any) -> tuple[int, int, int]:
    """Decode only the native formatter's owner/epoch/generation identity."""
    if value is None:
        return 0, 0, 0
    fields = value.split(":") if isinstance(value, str) else ()
    if len(fields) != 4 or fields[0] != "cuda-ks":
        raise ValueError("invalid native precision state identity")
    return tuple(_uint(int(field)) for field in fields[1:])


def admission(
    calculator: Any,
    work: dict,
    *,
    result: Any = None,
    qualified_gradient: bool = False,
    precision: str | None = None,
    budget: Any = None,
) -> tuple[bool, bool, bool] | None:
    """Query the selected library using its existing public precision records.

    Missing counters stay unknown. The adapter translates names and shapes; it
    does not infer work, compare budgets or identify a qualifying method itself.
    Explicit TargetAccuracy requests still use their independent audit path.
    """
    library = getattr(calculator, "_library", None)
    query = getattr(library, "generativeqc_precision_admission_query_v1", None)
    if query is None:
        return None
    context = _Context(ct.sizeof(_Context), _native.ABI_VERSION)
    options = getattr(calculator, "_ks_options", None)
    context.ks_model = (
        options is not None and getattr(options, "_method_ir", None) is not None
    )
    if context.ks_model:
        context.full_range_exchange = options.coefficients[2] != 0
        context.range_exchange = bool(options.has_range_exchange)
        context.nonlocal_correlation = bool(options.execution_plan.nonlocal_correlation)
    context.hartree_fock = (
        getattr(getattr(calculator, "capabilities", None), "family", None)
        == "hartree_fock"
    )
    context.density_fitted = bool(getattr(calculator, "_density_fitting_mode", None))
    context.warm_start_used = getattr(result, "warm_start_used", False) is True
    context.warm_start_fallback = (
        getattr(result, "warm_start_fallback", True) is not False
    )
    gradient = work.get("gradient") or {}
    context.gradient_present = bool(gradient)
    context.gradient_fp64 = gradient.get("returned_mode") == "fp64"
    context.gradient_qualified = qualified_gradient
    valid = work.get("operator_work_counters_valid")
    context.census_presence = 1 if valid is True else 2 if valid is False else 0
    context.converged = bool(getattr(result, "converged", False))
    context.cuda_execution = getattr(result, "executed_backend", None) == "cuda"
    context.fp64_candidate = precision == "fp64"
    if budget is not None:
        from ._scf_convergence import default_scf_controls

        context.energy_abs, context.force_max_abs = (
            budget.energy_abs,
            budget.force_max_abs,
        )
        context.force_rms = 0 if budget.force_rms is None else budget.force_rms
        for index, (name, value) in enumerate(default_scf_controls(calculator).items()):
            context.standard_controls[index] = value
            context.requested_controls[index] = getattr(calculator, "_" + name, 0)

    provenance = _native.PrecisionProvenance(
        ct.sizeof(_native.PrecisionProvenance), _native.ABI_VERSION
    )
    provenance.requested_mode = (
        _native.PRECISION_FP64 if work.get("requested_mode") == "fp64" else -1
    )
    provenance.effective_bits = _uint(work.get("effective_bits") or 0, 32)
    provenance.strict_refinement_applied = work.get("strict_refinement_applied") is True
    provenance.operator_work_counters_valid = valid is True
    for name in (
        "strict_stage_fock_builds",
        "final_residual_audits",
        "skipped_final_fock_builds",
    ):
        setattr(provenance, name, _uint(work.get(name, 0)))
    # Missing must not become a measured zero for the strict warm/fallback paths.
    provenance.mixed_stage_fock_builds = _uint(
        work.get("mixed_stage_fock_builds", 2**64 - 1)
    )

    rows = work.get("operators", ())
    operators = (_native.PrecisionOperatorRecord * len(rows))()
    for out, row in zip(operators, rows, strict=True):
        out.struct_size, out.abi_version = ct.sizeof(out), _native.ABI_VERSION
        out.kind = _OPERATORS.get(row.get("name"), 0)
        for name in ("storage", "compute", "accumulation", "reduction"):
            setattr(out, name + "_dtype", _TYPES.get(row.get(name), -1))
        out.arithmetic_mode = _MODES.get(row.get("arithmetic_mode"), -1)
        out.count = _uint(row.get("count", 0))
    rows = [row for row in work.get("scf_fock_timeline", ()) if row.get("count", 0) > 0]
    events = (_native.PrecisionWorkEvent * len(rows))()
    for out, row in zip(events, rows, strict=True):
        out.struct_size, out.abi_version = ct.sizeof(out), _native.ABI_VERSION
        out.kind = _EVENTS.get(row.get("kind"), 0)
        out.owner_id, out.solve_epoch, out.state_generation = _state(row.get("state"))
    detail = _native.PrecisionWorkDetail(
        ct.sizeof(_native.PrecisionWorkDetail), _native.ABI_VERSION
    )
    detail.detail_version = _uint(work.get("detail_version") or 1, 32)
    detail.complete = work.get("complete") is True
    detail.operator_inventory_complete = work.get("operator_inventory_complete") is True
    detail.owner_id, detail.returned_solve_epoch, detail.returned_state_generation = (
        _state(work.get("returned_state_identity"))
    )
    detail.event_count, detail.operator_count = len(events), len(operators)
    output = _Decision(ct.sizeof(_Decision), _native.ABI_VERSION)
    query.argtypes = [
        ct.POINTER(_Context),
        ct.POINTER(_native.PrecisionProvenance),
        ct.POINTER(_native.PrecisionWorkDetail),
        ct.POINTER(_native.PrecisionWorkEvent),
        ct.POINTER(_native.PrecisionOperatorRecord),
        ct.POINTER(_Decision),
    ]
    query.restype = ct.c_int
    _native.check(
        library,
        query(
            ct.byref(context),
            ct.byref(provenance),
            ct.byref(detail),
            events,
            operators,
            ct.byref(output),
        ),
    )
    decisions = output.budget_fock, output.strict_warm, output.standard_fallback
    if (
        output.struct_size != ct.sizeof(_Decision)
        or output.abi_version != _native.ABI_VERSION
        or output.policy_version != 1
        or any(value not in (0, 1) for value in decisions)
    ):
        raise RuntimeError("native precision admission returned an invalid receipt")
    return tuple(bool(value) for value in decisions)

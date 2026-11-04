"""Exercise completed-candidate routing through the actual public native owner."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import _native
from generativeqc import _precision_candidate as transport
from generativeqc._precision_candidate import Disposition
from generativeqc._precision_native import _Context as AdmissionContext

LEGACY = (
    "pbe0-rks-budget-fp32-v1",
    "pbe-rks-budget-fp32-v1",
    "b3lyp-rks-budget-fp32-v1",
    "r2scan-rks-budget-fp32-v1",
    "rhf-budget-fp32-v1",
    "rhf-df-shell-budget-fp32-v1",
    "m06-2x-rks-budget-fp32-v1",
)
FITTED = tuple(
    f"{name}-df-rks{domain}-budget-fp32-v1"
    for domain in ("", "-treutler")
    for name in ("pbe", "pbe0", "b3lyp", "r2scan", "m06-2x")
) + (
    "pbe-df-rks-treutler-xc-budget-fp32-v1",
    "r2scan-df-rks-treutler-xc-budget-fp32-v1",
)
COMPOSITE = ("wb97m-v-rks-budget-fp32-v1", "wb97m-v-df-rks-budget-fp32-v1")


@pytest.fixture(scope="module")
def query(policy_library: Any) -> Any:
    """Use the full selected library, including its native canonical method catalog."""
    value = policy_library.generativeqc_precision_candidate_query_v1
    value.argtypes = [
        ct.POINTER(transport._Context),
        ct.POINTER(AdmissionContext),
        ct.POINTER(_native.PrecisionProvenance),
        ct.POINTER(_native.PrecisionWorkDetail),
        ct.POINTER(_native.PrecisionWorkEvent),
        ct.POINTER(_native.PrecisionOperatorRecord),
        ct.POINTER(transport._Decision),
    ]
    value.restype = ct.c_int
    return value


def facts(name: str | None = None, schema: int = 2, owner: int = 0) -> Any:
    value = transport._Context(ct.sizeof(transport._Context), _native.ABI_VERSION)
    value.fast_profile = value.fast_native = value.forces_requested = 1
    value.gradient_present = value.gradient_complete = 1
    value.gradient_schema_version, value.gradient_owner = schema, owner
    value.gradient_returned_mode = 1
    value.geometry_points = value.grid_points = 1
    if name is not None:
        value.qualification_present = 1
        value.qualification = name.encode()
    return value


def records() -> tuple[Any, Any, Any]:
    model = AdmissionContext(ct.sizeof(AdmissionContext), _native.ABI_VERSION)
    work = _native.PrecisionProvenance(
        ct.sizeof(_native.PrecisionProvenance), _native.ABI_VERSION
    )
    detail = _native.PrecisionWorkDetail(
        ct.sizeof(_native.PrecisionWorkDetail), _native.ABI_VERSION
    )
    detail.detail_version = 1
    return model, work, detail


def invoke(
    query: Any, candidate: Any, *, supplied: Any = None, operators: Any = None
) -> Any:
    model, work, detail = supplied or records()
    if operators is not None:
        detail.operator_count = len(operators)
    output = transport._Decision(ct.sizeof(transport._Decision), _native.ABI_VERSION)
    status = query(
        ct.byref(candidate),
        ct.byref(model),
        ct.byref(work),
        ct.byref(detail),
        None,
        operators,
        ct.byref(output),
    )
    assert status == 0
    return output


@pytest.mark.parametrize(
    "name,schema,owner",
    (
        [(name, schema, 0) for name in LEGACY for schema in (1, 2, 3)]
        + [(name, 4, 1) for name in FITTED]
        + [(name, 4, 2) for name in COMPOSITE]
        + [(None, 1, 0)]
    ),
)
def test_incumbent_gradient_domains(
    query: Any, name: str, schema: int, owner: int
) -> None:
    candidate = facts(name, schema, owner)
    model, work, detail = records()
    model.census_presence = work.operator_work_counters_valid = 1
    detail.complete = detail.operator_inventory_complete = 1
    work.strict_stage_fock_builds = work.final_residual_audits = 1
    output = invoke(query, candidate, supplied=(model, work, detail))
    assert output.qualified_gradient == 1
    assert output.disposition == Disposition.MIXED_GRADIENT
    assert output.reasons == 1
    assert output.gradient_admission.decode() == (name or LEGACY[0])
    candidate.explicit_target = 1
    output = invoke(query, candidate)
    assert output.disposition == Disposition.AUDIT and output.reasons == 1


@pytest.mark.parametrize(
    "name,schema,owner",
    (
        [(name, 4, 1) for name in LEGACY]
        + [(name, schema, 1) for name in FITTED for schema in (0, 1, 2, 3, 5)]
        + [(name, 4, owner) for name in FITTED for owner in (0, 2)]
        + [(name, 4, owner) for name in COMPOSITE for owner in (0, 1)]
        + [
            (None, 2, 0),
            ("", 1, 0),
            ("unknown", 1, 0),
            ("pbe0-rks-budget-fp32-v2", 2, 0),
        ]
    ),
)
def test_unknown_or_wrong_owner_never_gains_qualification(
    query: Any,
    name: str,
    schema: int,
    owner: int,
) -> None:
    output = invoke(query, facts(name, schema, owner))
    assert output.qualified_gradient == 0
    assert output.disposition == Disposition.AUDIT and output.reasons == 128


@pytest.mark.parametrize(
    "field,value",
    (
        ("gradient_present", 0),
        ("gradient_complete", 0),
        ("geometry_points", 0),
        ("gradient_returned_mode", 0),
        ("forces_requested", 0),
        ("fast_native", 0),
    ),
)
def test_gradient_shortcut_requires_complete_actual_work(
    query: Any, field: str, value: int
) -> None:
    candidate = facts(LEGACY[0])
    setattr(candidate, field, value)
    assert invoke(query, candidate).disposition == Disposition.AUDIT


def test_weighted_work_and_reason_precedence(query: Any) -> None:
    candidate = facts(LEGACY[0], 3)
    candidate.geometry_points = candidate.grid_points = 0
    candidate.weighted_task_counters_collected = 1
    candidate.weighted_tasks = candidate.primitive_sources = 1
    candidate.execution_retries = 1
    output = invoke(query, candidate)
    assert output.disposition == Disposition.MIXED_GRADIENT
    assert output.reasons == 1 | 16 | 32
    candidate.gradient_retries = 1
    candidate.gradient_returned_mode = 2
    output = invoke(query, candidate)
    assert output.disposition == Disposition.STRICT_GRADIENT
    assert output.reasons == 1 | 8 | 32
    candidate.primitive_sources = 0
    assert invoke(query, candidate).disposition == Disposition.AUDIT


def test_schema_one_qualification_does_not_replace_complete_strict_census(
    query: Any,
) -> None:
    candidate = facts(LEGACY[0], 1)
    output = invoke(query, candidate)
    assert output.qualified_gradient == 1 and output.disposition == Disposition.AUDIT
    model, work, detail = records()
    model.census_presence = work.operator_work_counters_valid = 1
    detail.complete = detail.operator_inventory_complete = 1
    work.strict_stage_fock_builds = 1
    assert (
        invoke(query, candidate, supplied=(model, work, detail)).disposition
        == Disposition.AUDIT
    )
    work.final_residual_audits = 1
    assert (
        invoke(query, candidate, supplied=(model, work, detail)).disposition
        == Disposition.MIXED_GRADIENT
    )


@pytest.mark.parametrize(
    "name",
    (
        "r2scan-rks-budget-fp32-v1",
        "r2scan-df-rks-budget-fp32-v1",
        "r2scan-df-rks-treutler-budget-fp32-v1",
        "r2scan-df-rks-treutler-xc-budget-fp32-v1",
    ),
)
def test_complete_correction_is_bound_to_native_canonical_identity(
    query: Any, name: str
) -> None:
    from generativeqc_compiler.method import resolve_method

    candidate = facts(name, 4 if "-df-" in name else 2, 1 if "-df-" in name else 0)
    candidate.correction_present, candidate.correction_result_flags = 1, 63
    candidate.correction_model_identity = resolve_method(
        "R2SCAN-3c", spin="unpolarized"
    ).identity.encode()
    output = invoke(query, candidate)
    assert output.qualified_gradient == 1
    assert output.gradient_admission.decode() == name.replace(
        "r2scan-", "r2scan-3c-", 1
    )
    for bit in range(6):
        candidate.correction_result_flags = 63 ^ (1 << bit)
        output = invoke(query, candidate)
        assert output.qualified_gradient == 0 and output.reasons == 4 | 128
    candidate.correction_result_flags = 63
    candidate.correction_model_identity = b"0" * 64
    assert invoke(query, candidate).qualified_gradient == 0


def test_strict_xc_requires_present_counts_and_not_a_corrected_graph(
    query: Any,
) -> None:
    candidate = facts("unknown")
    candidate.mixed_fock_count_present = 1
    model, work, detail = records()
    model.census_presence = work.operator_work_counters_valid = 1
    detail.complete = detail.operator_inventory_complete = 1
    work.mixed_stage_fock_builds = 1
    work.strict_stage_fock_builds = 2
    work.strict_refinement_applied = work.final_residual_audits = 1
    operators = (_native.PrecisionOperatorRecord * 1)()
    row = operators[0]
    row.struct_size, row.abi_version = ct.sizeof(row), _native.ABI_VERSION
    # Use the public formatter's enum mapping, not an assumed operator number.
    from generativeqc._precision_native import _MODES, _OPERATORS, _TYPES

    row.kind = _OPERATORS["xc"]
    row.compute_dtype = row.accumulation_dtype = _TYPES["fp32"]
    row.arithmetic_mode, row.count = _MODES["mixed"], 1
    supplied = model, work, detail
    assert (
        invoke(query, candidate, supplied=supplied, operators=operators).disposition
        == Disposition.STRICT_XC
    )
    candidate.mixed_fock_count_present = 0
    assert (
        invoke(query, candidate, supplied=supplied, operators=operators).disposition
        == Disposition.AUDIT
    )
    candidate.mixed_fock_count_present = candidate.correction_present = 1
    assert (
        invoke(query, candidate, supplied=supplied, operators=operators).disposition
        == Disposition.AUDIT
    )


@pytest.mark.parametrize(
    "field,value,status",
    (
        ("reserved", 1, 1),
        ("fast_profile", 2, 1),
        ("gradient_owner", 3, 1),
        ("gradient_returned_mode", 3, 1),
        ("correction_result_flags", 64, 1),
        ("struct_size", 0, 2),
        ("abi_version", _native.ABI_VERSION + 1, 2),
    ),
)
def test_invalid_request_does_not_publish(
    query: Any, field: str, value: int, status: int
) -> None:
    candidate = facts(LEGACY[0])
    model, work, detail = records()
    output = transport._Decision(ct.sizeof(transport._Decision), _native.ABI_VERSION)
    output.policy_version = 123
    before = bytes(output)
    setattr(candidate, field, value)
    assert (
        query(
            ct.byref(candidate),
            ct.byref(model),
            ct.byref(work),
            ct.byref(detail),
            None,
            None,
            ct.byref(output),
        )
        == status
    )
    assert bytes(output) == before


def test_overlap_and_reentrant_queries(query: Any) -> None:
    storage = ct.create_string_buffer(
        ct.sizeof(transport._Context) + ct.sizeof(transport._Decision)
    )
    candidate = transport._Context.from_buffer(storage)
    candidate.struct_size, candidate.abi_version = (
        ct.sizeof(candidate),
        _native.ABI_VERSION,
    )
    output = transport._Decision.from_buffer(storage, 128)
    output.struct_size, output.abi_version = ct.sizeof(output), _native.ABI_VERSION
    model, work, detail = records()
    assert (
        query(
            ct.byref(candidate),
            ct.byref(model),
            ct.byref(work),
            ct.byref(detail),
            None,
            None,
            ct.byref(output),
        )
        == 0
    )
    assert output.disposition == Disposition.STRICT
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(
            pool.map(lambda name: invoke(query, facts(name)).disposition, LEGACY * 8)
        )
    assert outcomes == [Disposition.MIXED_GRADIENT] * len(outcomes)


@pytest.mark.parametrize(
    "failure,status",
    (
        ("qualification", 1),
        ("correction_identity", 1),
        ("work_abi", 2),
        ("detail_version", 3),
        ("missing_events", 1),
        ("missing_operators", 1),
    ),
)
def test_strings_and_late_evidence_errors_remain_atomic(
    query: Any, failure: str, status: int
) -> None:
    candidate = facts(LEGACY[0])
    model, work, detail = records()
    if failure == "qualification":
        candidate.qualification = b"x" * 96
    elif failure == "correction_identity":
        candidate.correction_model_identity = b"x" * 65
    elif failure == "work_abi":
        work.abi_version += 1
    elif failure == "detail_version":
        detail.detail_version += 1
    elif failure == "missing_events":
        detail.event_count = 1
    else:
        detail.operator_count = 1
    output = transport._Decision(ct.sizeof(transport._Decision), _native.ABI_VERSION)
    output.policy_version = 123
    before = bytes(output)
    assert (
        query(
            ct.byref(candidate),
            ct.byref(model),
            ct.byref(work),
            ct.byref(detail),
            None,
            None,
            ct.byref(output),
        )
        == status
    )
    assert bytes(output) == before


def test_missing_refused_and_forged_selected_owner_never_falls_back(
    policy_library: Any,
) -> None:
    with pytest.raises(NotImplementedError, match="required precision-candidate query"):
        transport.require_candidate_classification(SimpleNamespace())
    calculator = SimpleNamespace(_library=policy_library)
    result = SimpleNamespace(precision={})
    options = {
        "profile": "fp64",
        "target": object(),
        "fast_native": False,
        "fast_fock": False,
        "forces": False,
        "precision": "fp64",
        "budget": None,
    }

    class Refused:
        def __call__(self, *args: Any) -> int:
            return 3

    calculator._library = SimpleNamespace(
        generativeqc_precision_candidate_query_v1=Refused(),
        generativeqc_status_message=policy_library.generativeqc_status_message,
    )
    with pytest.raises(NotImplementedError):
        transport.classify_candidate(calculator, result, **options)

    class Forged:
        def __call__(self, *args: Any) -> int:
            request = ct.cast(args[0], ct.POINTER(transport._Context)).contents
            request.explicit_target = 0
            output = ct.cast(args[-1], ct.POINTER(transport._Decision)).contents
            output.policy_version = 1
            output.disposition = Disposition.STRICT
            output.gradient_admission = b"unqualified"
            return 0

    calculator._library = SimpleNamespace(
        generativeqc_precision_candidate_query_v1=Forged()
    )
    with pytest.raises(RuntimeError, match="invalid receipt"):
        transport.classify_candidate(calculator, result, **options)
    options.update(profile="fast", target=None)
    with pytest.raises(RuntimeError, match="invalid receipt"):
        transport.classify_candidate(calculator, result, **options)

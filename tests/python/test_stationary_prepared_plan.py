"""Prepared stationary-plan identity and lifecycle binding tests."""

from fractions import Fraction

from vibeqc_compiler.common.prepared_execution import PreparedExecutionRequest
from vibeqc_compiler.method import (
    MethodSpec,
    original_nonlocal_correlation,
    resolve_method,
)
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.method.stationary_prepared import (
    StationaryPreparedPlan,
    compile_stationary_prepared_plan,
)

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _prepared(method: str | MethodSpec) -> StationaryPreparedPlan:
    plan = StationaryGradientPlan(
        resolve_method(method), StationaryMeanField(SCF_POINT_MODEL)
    )
    return compile_stationary_prepared_plan(plan)


def _request(
    prepared: StationaryPreparedPlan, *, schedule: str = HEX_B
) -> PreparedExecutionRequest:
    return prepared.prepared_request(
        target_identity=HEX_A,
        schedule_identity=schedule,
        workspace_identity=HEX_C,
        device=0,
    )


def test_prepared_request_binds_graph_and_lifetime_identity() -> None:
    prepared = _prepared("PBE")
    request = _request(prepared)
    assert request.kind == "stationary-gradient"
    assert request.scientific_identity == prepared.graph.identity
    assert request.schedule_identity != HEX_B
    assert len(request.schedule_identity) == 64


def test_named_aliases_with_identical_science_share_prepared_identity() -> None:
    a = MethodSpec(
        "alias-a",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
    )
    b = MethodSpec(
        "alias-b",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
    )
    pa, pb = _prepared(a), _prepared(b)
    assert pa.identity == pb.identity
    assert _request(pa).identity == _request(pb).identity


def test_lifetime_change_is_part_of_prepared_schedule_compatibility() -> None:
    pbe = _prepared("PBE")
    vv10 = MethodSpec(
        "pbe-plus-vv10",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
        nonlocal_correlation=original_nonlocal_correlation("vv10"),
    )
    nonlocal_plan = _prepared(vv10)
    assert pbe.lifetimes.identity != nonlocal_plan.lifetimes.identity
    assert _request(pbe).schedule_identity != _request(nonlocal_plan).schedule_identity


def test_external_schedule_identity_still_invalidates_replay() -> None:
    prepared = _prepared("PBE")
    assert (
        _request(prepared, schedule="b" * 64).identity
        != _request(prepared, schedule="d" * 64).identity
    )

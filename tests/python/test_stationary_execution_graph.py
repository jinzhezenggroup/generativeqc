"""Structure tests for the method-name-free stationary execution graph."""

from fractions import Fraction

from vibeqc_compiler.method import (
    MethodSpec,
    original_nonlocal_correlation,
    resolve_method,
)
from vibeqc_compiler.method.stationary_execution import (
    compile_stationary_execution_graph,
)
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)


def _plan(method: str | MethodSpec = "PBE") -> StationaryGradientPlan:
    return StationaryGradientPlan(
        resolve_method(method), StationaryMeanField(SCF_POINT_MODEL)
    )


def test_grid_feature_inventory_is_ingredient_driven() -> None:
    pbe = compile_stationary_execution_graph(_plan("PBE"))
    r2scan = compile_stationary_execution_graph(_plan("R2SCAN"))
    assert pbe.grid_features == ("rho", "gradient")
    assert r2scan.grid_features == ("rho", "gradient", "tau")


def test_nonlocal_composition_reuses_density_features_without_named_method_node() -> (
    None
):
    method = MethodSpec(
        "arbitrary-pbe-vv10-alias",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
        nonlocal_correlation=original_nonlocal_correlation("vv10"),
    )
    graph = compile_stationary_execution_graph(_plan(method))
    assert graph.node("nonlocal_pairs").inputs == (
        "grid_feature:rho",
        "grid_feature:gradient",
    )
    assert graph.value("grid_feature:rho").consumers == (
        "semilocal_geometry",
        "nonlocal_pairs",
        "nonlocal_geometry",
    )
    assert "wb97" not in repr(graph).lower()


def test_range_exchange_sources_are_graph_nodes_not_method_dispatch() -> None:
    graph = compile_stationary_execution_graph(_plan("CAM-B3LYP"))
    assert graph.node("integral:exchange_short_range").inputs == ("final_density",)
    assert graph.node("integral:exchange_long_range").inputs == ("final_density",)


def test_graph_covers_stationary_source_inventory_exactly() -> None:
    plan = _plan("CAM-B3LYP")
    graph = compile_stationary_execution_graph(plan)
    produced = {
        value.name.removeprefix("source:")
        for value in graph.values
        if value.name.startswith("source:")
    }
    assert produced == set(plan.source_names)
    assert graph.node("gradient_reduction").inputs == tuple(
        f"source:{name}" for name in plan.source_names
    )


def test_ecp_sources_are_density_consuming_integral_nodes_and_covered() -> None:
    plan = StationaryGradientPlan(
        resolve_method("PBE"),
        StationaryMeanField(SCF_POINT_MODEL, hamiltonian="scalar-semilocal-ecp"),
    )
    graph = compile_stationary_execution_graph(plan)
    assert graph.node("integral:ecp_local").inputs == ("final_density",)
    assert graph.node("integral:ecp_nonlocal").inputs == ("final_density",)
    produced = {
        value.name.removeprefix("source:")
        for value in graph.values
        if value.name.startswith("source:")
    }
    assert produced == set(plan.source_names)


def test_aliases_with_identical_science_share_graph_identity() -> None:
    first = MethodSpec(
        "first-name",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
    )
    second = MethodSpec(
        "second-name",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
    )
    a = compile_stationary_execution_graph(_plan(first))
    b = compile_stationary_execution_graph(_plan(second))
    assert a.plan_identity == b.plan_identity
    assert a.identity == b.identity

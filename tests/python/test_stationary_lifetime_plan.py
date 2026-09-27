"""Lifetime policy tests over the method-neutral stationary graph."""

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
from vibeqc_compiler.method.stationary_lifetime import (
    DEVICE_BORROWED,
    DEVICE_RETAINED,
    HOST_PUBLISHED,
    plan_stationary_lifetimes,
)


def _lifetime(method):
    gradient = StationaryGradientPlan(
        resolve_method(method), StationaryMeanField(SCF_POINT_MODEL)
    )
    graph = compile_stationary_execution_graph(gradient)
    return graph, plan_stationary_lifetimes(graph)


def test_nonlocal_shared_features_are_retained_without_method_name_policy():
    method = MethodSpec(
        "unrelated-alias",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
        nonlocal_correlation=original_nonlocal_correlation("vv10"),
    )
    graph, lifetime = _lifetime(method)
    assert lifetime.value("grid_feature:rho").placement == DEVICE_RETAINED
    assert lifetime.value("grid_feature:gradient").placement == DEVICE_RETAINED
    assert lifetime.value("nonlocal_seeds").placement == DEVICE_BORROWED
    assert "wb97" not in repr(lifetime).lower()
    assert lifetime.graph_identity == graph.identity


def test_semilocal_only_feature_can_be_borrowed_by_one_consumer():
    _, lifetime = _lifetime("PBE")
    assert lifetime.value("grid_feature:rho").placement == DEVICE_BORROWED
    assert lifetime.value("grid_feature:gradient").placement == DEVICE_BORROWED


def test_final_density_is_retained_when_integral_and_grid_consumers_share_it():
    _, lifetime = _lifetime("CAM-B3LYP")
    final_density = lifetime.value("final_density")
    assert final_density.placement == DEVICE_RETAINED
    assert "grid_features" in final_density.consumers
    assert "integral:exchange_short_range" in final_density.consumers
    assert "integral:exchange_long_range" in final_density.consumers


def test_public_gradient_is_the_only_forced_host_publication():
    graph, lifetime = _lifetime("PBE")
    assert lifetime.value("gradient").placement == HOST_PUBLISHED
    for value in graph.values:
        if value.name != "gradient":
            assert lifetime.value(value.name).placement != HOST_PUBLISHED


def test_aliases_with_identical_science_share_lifetime_identity():
    first = MethodSpec(
        "name-a",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
    )
    second = MethodSpec(
        "name-b",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
    )
    _, a = _lifetime(first)
    _, b = _lifetime(second)
    assert a.identity == b.identity

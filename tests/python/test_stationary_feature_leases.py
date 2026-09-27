"""Feature-lease projection tests for stationary CUDA adoption."""

from fractions import Fraction

from vibeqc_compiler.method import MethodSpec, original_nonlocal_correlation, resolve_method
from vibeqc_compiler.method.stationary_feature_lease import plan_stationary_feature_leases
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.method.stationary_prepared import compile_stationary_prepared_plan


def _features(method):
    gradient = StationaryGradientPlan(
        resolve_method(method), StationaryMeanField(SCF_POINT_MODEL)
    )
    return plan_stationary_feature_leases(
        compile_stationary_prepared_plan(gradient)
    )


def test_semilocal_feature_inventory_comes_from_method_requirements():
    assert _features("PBE").features == ("rho", "gradient")
    assert _features("R2SCAN").features == ("rho", "gradient", "tau")


def test_nonlocal_consumer_turns_shared_features_into_retained_leases():
    method = MethodSpec(
        "generic-meta-independent-name",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
        nonlocal_correlation=original_nonlocal_correlation("vv10"),
    )
    leases = _features(method)
    assert leases.retained_features == ("rho", "gradient")
    by_name = {lease.feature: lease for lease in leases.leases}
    assert "semilocal_geometry" in by_name["rho"].consumers
    assert "nonlocal_pairs" in by_name["rho"].consumers
    assert "nonlocal_geometry" in by_name["rho"].consumers
    assert "wb97" not in repr(leases).lower()


def test_semilocal_only_features_do_not_request_cross_consumer_retention():
    leases = _features("R2SCAN")
    assert leases.retained_features == ()


def test_feature_lease_identity_is_method_alias_independent():
    a = MethodSpec(
        "alias-feature-a",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
    )
    b = MethodSpec(
        "alias-feature-b",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
    )
    assert _features(a).identity == _features(b).identity

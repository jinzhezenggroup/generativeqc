"""Production-adoption guards for compiler-planned WB97M-V grid features."""

from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "vibeqc"
    / "_stationary_wb97mv_cuda.py"
).read_text()


def test_wb97mv_stationary_grid_features_come_from_compiler_plan() -> None:
    assert "compile_stationary_prepared_plan(plan)" in SOURCE
    assert "plan_stationary_feature_leases(prepared_plan)" in SOURCE
    assert "ingredients=grid_features" in SOURCE
    assert "points, ids, grid_features" in SOURCE
    assert 'ingredients=("rho", "gradient", "tau")' not in SOURCE
    assert 'points, ids, ("rho", "gradient", "tau")' not in SOURCE


def test_wb97mv_replay_identity_binds_compiler_execution_semantics() -> None:
    assert "prepared_plan.identity" in SOURCE
    assert "feature_plan.identity" in SOURCE
    assert '"execution_graph_identity": prepared_plan.graph.identity' in SOURCE
    assert '"lifetime_plan_identity": prepared_plan.lifetimes.identity' in SOURCE
    assert '"feature_lease_identity": feature_plan.identity' in SOURCE

"""Production-adoption guards for compiler-planned WB97M-V grid features."""

import ast
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "vibeqc"
    / "_stationary_wb97mv_cuda.py"
).read_text()
TREE = ast.parse(SOURCE)


def _calls(name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_wb97mv_stationary_grid_features_come_from_compiler_plan() -> None:
    assert "compile_stationary_prepared_plan(plan)" in SOURCE
    assert "plan_stationary_feature_leases(prepared_plan)" in SOURCE
    for name in ("CudaGrid", "resident_nonlocal_geometry"):
        calls = _calls(name)
        assert len(calls) == 1
        ingredients = next(
            keyword.value
            for keyword in calls[0].keywords
            if keyword.arg == "ingredients"
        )
        assert isinstance(ingredients, ast.Name)
        assert ingredients.id == "grid_features"
    assert 'ingredients=("rho", "gradient", "tau")' not in SOURCE


def test_wb97mv_replay_identity_binds_compiler_execution_semantics() -> None:
    assignments = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "identity" for t in node.targets)
    ]
    assert len(assignments) == 1
    identity = assignments[0].value
    assert isinstance(identity, ast.Tuple)
    values = {ast.unparse(value) for value in identity.elts}
    assert {"prepared_plan.identity", "feature_plan.identity"} <= values
    assert '"execution_graph_identity": prepared_plan.graph.identity' in SOURCE
    assert '"lifetime_plan_identity": prepared_plan.lifetimes.identity' in SOURCE
    assert '"feature_lease_identity": feature_plan.identity' in SOURCE


def test_planner_adoption_preserves_resident_owner_and_per_call_work_budget() -> None:
    assert len(_calls("_ResidentNonlocalForceOwner")) == 1
    assert len(_calls("resident_nonlocal_geometry")) == 1
    calls = _calls("_CudaSources")
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    budget = keywords["page_work_budget"]
    assert isinstance(budget, ast.Constant)
    assert budget.value == 1
    assert "work_budget" not in keywords
    assert "feature_task_with_features" not in SOURCE
    assert SOURCE.index("self._nonlocal.close()") < SOURCE.index("self._stack.close()")

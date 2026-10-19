"""Execute compiler feature/planner identity and owner-lifetime adoption."""

from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest
from generativeqc import _stationary_composite_cuda as r
from test_geometry_response_precision import harness as harness  # noqa: PLC0414


@pytest.mark.parametrize("split", [False, True])
def test_planner_semantics_and_owner_cleanup_execute(
    harness: NS, monkeypatch: pytest.MonkeyPatch, split: bool
) -> None:
    h = harness
    captures = []
    geometry_resets = []
    old_grid, old_sources = r.CudaGrid, r._CudaSources
    old_join = r.resident_nonlocal_geometry
    features = r.plan_stationary_feature_leases(h.prepared)
    features.features = ("rho", "gradient", "compiler-owned-sentinel")

    class Grid(old_grid):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captures.append(("grid", kwargs))
            super().__init__(*args, **kwargs)

        def feature_task_with_features(self, *args: Any, **kwargs: Any) -> None:
            pytest.fail("geometry execution must retain the resident feature owner")

    class Sources(old_sources):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captures.append(("sources", kwargs))
            super().__init__(*args, **kwargs)

        def reset_geometry(self, *args: Any) -> None:
            geometry_resets.append((self, args))
            super().reset_geometry(*args)

        def reset(self, *args: Any, **kwargs: Any) -> None:
            pytest.fail("geometry execution must not upload detached host density")

    def join(
        **kwargs: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
        assert kwargs["nonlocal_owner"] is h.owner._nonlocal
        assert kwargs["ingredients"] == features.features
        assert kwargs["sources"] is h.owner.sources
        assert kwargs["nonlocal_sources"] is h.owner.nonlocal_sources
        assert kwargs["sources"] is not kwargs["nonlocal_sources"]
        captures.append(("join", kwargs))
        return old_join(**kwargs)

    monkeypatch.setattr(r, "CudaGrid", Grid)
    monkeypatch.setattr(r, "_CudaSources", Sources)
    monkeypatch.setattr(r, "resident_nonlocal_geometry", join)
    monkeypatch.setattr(r, "resident_split_nonlocal_geometry", join)
    if split:
        h.source.nonlocal_grid = NS(points=np.ones((4, 3)), weights=np.ones(4))
        h.source.nonlocal_grid_spec = NS(
            partition_iterations=2, coincident_tolerance=2e-12
        )
        h.source.nonlocal_atomic_weights = np.ones(4)
    _, work = h.run()
    assert (
        next(value for name, value in captures if name == "grid")["ingredients"]
        == features.features
    )
    source_options = [value for name, value in captures if name == "sources"]
    assert len(source_options) == 2
    assert all("work_budget" not in options for options in source_options)
    assert geometry_resets == [
        (h.owner.sources, (h.source.grid_spec.coincident_tolerance,)),
        (h.owner.nonlocal_sources, (h.source.nonlocal_grid_spec.coincident_tolerance,)),
    ]
    assert source_options[0]["page_work_budget"] == max(
        1, h.basis.natom * (h.basis.natom - 1) // 2
    )
    assert source_options[1]["page_work_budget"] == 1
    assert work["execution_graph_identity"] == h.prepared.graph.identity
    assert work["lifetime_plan_identity"] == h.prepared.lifetimes.identity
    assert work["feature_lease_identity"] == features.identity
    assert h.run()[1]["prepared_execution_reused"]
    features.identity = "changed-feature-plan"
    assert not h.run()[1]["prepared_execution_reused"]
    h.prepared.identity = "changed-prepared-plan"
    assert not h.run()[1]["prepared_execution_reused"]
    h.events.clear()
    h.owner.close()
    assert h.events.index("drain") < h.events.index("release")


@pytest.mark.parametrize("atoms", [1, 3, 5])
def test_pair_page_budget_scales_without_spending_nonlocal_budget(
    harness: NS, monkeypatch: pytest.MonkeyPatch, atoms: int
) -> None:
    """Stop after construction; no fake molecular arithmetic uses the larger shape."""
    h = harness
    h.basis.natom = atoms
    h.basis.atoms = [NS(atomic_number=1)] * atoms
    options = []
    old_sources = r._CudaSources

    class Sources(old_sources):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            options.append(kwargs)
            super().__init__(*args, **kwargs)
            if len(options) == 2:
                raise RuntimeError("stop after separate source budgets")

    monkeypatch.setattr(r, "_CudaSources", Sources)
    with pytest.raises(RuntimeError, match="stop after separate source budgets"):
        h.run()
    assert [value["page_work_budget"] for value in options] == [
        max(1, atoms * (atoms - 1) // 2),
        1,
    ]
    assert all("work_budget" not in value for value in options)

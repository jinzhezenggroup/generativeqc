"""One dry tile search serves ordinary and composite stationary consumers."""

import ast
import inspect
import typing
from types import CodeType, FunctionType, SimpleNamespace

import numpy as np
import pytest
from generativeqc import _stationary_cuda as runtime
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.method.stationary_resources import (
    plan_stationary_cuda_grid_schedule,
    plan_stationary_cuda_grid_work,
)


def _ordinary(atoms: int = 24, **options: object) -> object:
    """Use production capacities, not a reduced standalone AO scratch model."""
    basis = SimpleNamespace(
        natom=atoms,
        nao=8 * atoms,
        nprimitive=22 * atoms // 3,
        numeric_bytes=2608 * atoms,
        packed=SimpleNamespace(size=3 * atoms + 44 * atoms // 3 + 128 * atoms),
    )
    arguments = {
        "plan": SimpleNamespace(spin_blocks=1),
        "target": cuda_target_info("sm_120"),
        "needs_first": True,
        "tile_points": 256,
        "primitive_tile": 4096,
        "integral_terms": 32,
        "source_names": tuple(range(8)),
        "ecp": False,
        "max_device_bytes": 512 << 20,
        "max_host_bytes": 256 << 20,
        "max_ecp_pair_samples": 100_000_000,
    }
    arguments.update(options)
    return runtime._plan_stationary_cuda_tile(
        SimpleNamespace(
            _source=SimpleNamespace(cuda_integral_derivatives=lambda *_: None)
        ),
        basis,
        **arguments,
    )


@pytest.mark.parametrize(
    "atoms,selected", [(3, 1024), (24, 1024), (48, 1024), (96, 256)]
)
def test_ordinary_automatic_choice_obeys_unchanged_host_and_device_budgets(
    atoms: int,
    selected: int,
) -> None:
    layout = plan_stationary_cuda_grid_schedule(
        grid_points=atoms * 24576,
        tile_points=None,
        admit=lambda points: _ordinary(atoms, tile_points=points),
    )
    assert layout.grid_plan.tile_points == selected
    assert layout.host_bound <= 256 << 20
    assert (
        layout.grid_plan.peak_bytes
        + layout.source_resources.allocation_bytes
        + layout.native_geometry_reserve
        <= 512 << 20
    )


@pytest.mark.parametrize("device_budget", [512 << 20, 1 << 30])
def test_96_atom_explicit_1024_does_not_silently_raise_either_budget(
    device_budget: int,
) -> None:
    with pytest.raises(ValueError, match="grid tile needs|additional-host"):
        plan_stationary_cuda_grid_schedule(
            grid_points=96 * 24576,
            tile_points=1024,
            admit=lambda points: _ordinary(
                96, tile_points=points, max_device_bytes=device_budget
            ),
        )


@pytest.mark.parametrize("points", [1, 4, 128, 256])
def test_tight_host_cap_retains_exactly_the_admitted_smaller_tile(points: int) -> None:
    baseline = _ordinary(tile_points=points)
    selected = plan_stationary_cuda_grid_schedule(
        grid_points=589824,
        tile_points=None,
        admit=lambda candidate: _ordinary(
            tile_points=candidate, max_host_bytes=baseline.host_bound
        ),
    )
    assert selected.grid_plan.tile_points == points
    assert selected.host_bound == baseline.host_bound
    with pytest.raises(ValueError, match="additional-host"):
        plan_stationary_cuda_grid_schedule(
            grid_points=589824,
            tile_points=points,
            admit=lambda candidate: _ordinary(
                tile_points=candidate, max_host_bytes=baseline.host_bound - 1
            ),
        )


def test_work_guard_participates_in_search_without_dropping_points() -> None:
    visited = []

    def admit(points: int) -> object:
        visited.append(points)
        return plan_stationary_cuda_grid_work(
            atoms=96,
            grid_points=2003,
            tile_points=points,
            max_grid_points=None,
            max_grid_pair_visits=None,
            max_pending_pair_visits=256 * 96 * 95,
        )

    work = plan_stationary_cuda_grid_schedule(
        grid_points=2003, tile_points=None, admit=admit
    )
    assert visited == [1024, 256]
    assert work.tile_points == 256
    assert sum(end - begin for begin, end in work.chunks()) == 2003
    assert work.grid_pair_visits == (1 + 2 * 2003) * 96 * 95 // 2


@pytest.mark.parametrize("explicit", [None, 256])
def test_search_is_finite_and_explicit_capacity_is_never_resized(
    explicit: int | None,
) -> None:
    visited = []

    def reject(points: int) -> None:
        visited.append(points)
        raise ValueError("full-grid owner cannot fit")

    with pytest.raises(ValueError, match="no admitted.*full-grid owner"):
        plan_stationary_cuda_grid_schedule(
            grid_points=3, tile_points=explicit, admit=reject
        )
    assert visited == ([3, 2, 1] if explicit is None else [256])


@pytest.mark.parametrize(
    "points,tile", [(0, None), (True, 256), (2**64, 256), (3, True), (3, 0), (3, 4097)]
)
def test_invalid_contract_fails_before_calling_any_owner(
    points: int, tile: int | None
) -> None:
    with pytest.raises(ValueError):
        plan_stationary_cuda_grid_schedule(
            grid_points=points,
            tile_points=tile,
            admit=lambda *_: pytest.fail("invalid shape reached admission callback"),
        )


@pytest.mark.parametrize("preference", [1, 32, 128, 512, 1024, 4096])
def test_consumer_preference_keeps_budget_fallbacks_bounded(preference: int) -> None:
    visited = []

    def admit(points: int) -> int:
        visited.append(points)
        if points > 32:
            raise ValueError("concurrent owners exceed budget")
        return points

    selected = plan_stationary_cuda_grid_schedule(
        grid_points=10_000,
        tile_points=None,
        preferred_tile_points=preference,
        admit=admit,
    )
    assert visited[0] == preference
    assert selected == min(preference, 32)
    assert all(points <= preference for points in visited)


@pytest.mark.parametrize("preference", [0, 4097, True, None, 512.0])
def test_invalid_consumer_preference_is_not_silently_ignored(
    preference: int,
) -> None:
    with pytest.raises(ValueError, match="preferred_tile_points"):
        plan_stationary_cuda_grid_schedule(
            grid_points=1000,
            tile_points=None,
            preferred_tile_points=preference,
            admit=lambda *_: pytest.fail("invalid preference reached admission"),
        )


def test_explicit_tiles_override_consumer_preference() -> None:
    assert (
        plan_stationary_cuda_grid_schedule(
            grid_points=10,
            tile_points=1024,
            preferred_tile_points=512,
            admit=lambda points: points,
        )
        == 1024
    )


def test_unexpected_owner_error_is_not_hidden_as_another_tile_rejection() -> None:
    def failed_owner(points: int) -> None:
        raise RuntimeError("not a capacity rejection")

    with pytest.raises(RuntimeError, match="not a capacity"):
        plan_stationary_cuda_grid_schedule(
            grid_points=4096, tile_points=None, admit=failed_owner
        )


def _ecp_admission(prepared: bool, host_budget: int) -> tuple[typing.Any, dict]:
    """Execute the real nested candidate callback with full ECP tensor plans."""
    from generativeqc.ks import resolve_ks_method
    from generativeqc_compiler.method.stationary_gradient import (
        SCF_POINT_MODEL,
        StationaryGradientPlan,
        StationaryMeanField,
    )

    plan = StationaryGradientPlan(
        resolve_ks_method("pbe-rks")[0],
        StationaryMeanField(SCF_POINT_MODEL, hamiltonian="scalar-semilocal-ecp"),
    )
    basis = SimpleNamespace(
        natom=2,
        nao=12,
        nprimitive=12,
        numeric_bytes=1776,
        packed=SimpleNamespace(size=222),
    )
    state = SimpleNamespace(
        grid=SimpleNamespace(points=np.empty((4096, 3))),
        _source=SimpleNamespace(
            ecp_cores=(2, 0), ecp_terms=(object(),), values=np.empty(8)
        ),
    )
    scope = vars(runtime) | {
        "state": state,
        "basis": basis,
        "na": 2,
        "n": 12,
        "plan": plan,
        "target": cuda_target_info("sm_120"),
        "needs_first": True,
        "primitive_tile": 4096,
        "integral_terms": 32,
        "source_names": runtime.stationary_runtime_sources(plan),
        "ecp": True,
        "max_device_bytes": 512 << 20,
        "max_host_bytes": host_budget,
        "max_ecp_pair_samples": 100_000_000,
        "max_grid_points": None,
        "max_grid_pair_visits": None,
        "max_pending_grid_tiles": 64,
        "max_pending_grid_pair_visits": 100_000_000,
        "prepared": object() if prepared else None,
    }
    function = next(
        node
        for node in ast.walk(
            ast.parse(inspect.getsource(runtime._complete_rks_cuda_gradient_diagnostic))
        )
        if isinstance(node, ast.FunctionDef) and node.name == "admit_tile"
    )
    compiled = compile(
        ast.Module(body=[function], type_ignores=[]),
        "<actual dry tile admission>",
        "exec",
    )
    function_code = next(
        value for value in compiled.co_consts if isinstance(value, CodeType)
    )
    return FunctionType(function_code, scope), scope


@pytest.mark.parametrize("prepared", [False, True])
def test_tight_ecp_candidate_includes_prepared_tensor_host_storage(
    monkeypatch: pytest.MonkeyPatch, prepared: bool
) -> None:
    admit, _ = _ecp_admission(False, 256 << 20)
    large, _ = admit(1024)
    assert large.ecp_pair_samples == 93_210_624
    tensor_host = sum(value.host_bytes for value in large.tensor_plans.values())
    assert tensor_host > 0
    admit, scope = _ecp_admission(prepared, large.host_bound)
    selected, _ = plan_stationary_cuda_grid_schedule(
        grid_points=4096, tile_points=None, admit=admit
    )
    assert selected.grid_plan.tile_points == (256 if prepared else 1024)
    if not prepared:
        assert admit(1024)[0].host_bound == large.host_bound
        return
    with pytest.raises(ValueError, match="prepared.*host budget"):
        plan_stationary_cuda_grid_schedule(
            grid_points=4096, tile_points=1024, admit=admit
        )

    class AdmittedBeforeCompilation(Exception):
        pass

    def stop_after_admission(*args: typing.Any) -> typing.NoReturn:
        raise AdmittedBeforeCompilation

    owner = runtime.PreparedStationaryCudaExecution()
    monkeypatch.setattr(owner, "_request", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "_layout", stop_after_admission)
    with pytest.raises(AdmittedBeforeCompilation):
        owner.ensure(
            state=scope["state"],
            basis=scope["basis"],
            contract=None,
            plan=scope["plan"],
            tensor_plans=selected.tensor_plans,
            compiler=SimpleNamespace(target=scope["target"]),
            cache="unused",
            requests=(),
            functional=1,
            ecp=True,
            device=0,
            spec=None,
            grid_plan=selected.grid_plan,
            source_bytes=selected.source_resources.allocation_bytes,
            tile_points=selected.grid_plan.tile_points,
            primitive_tile=4096,
            integral_terms=32,
            page_work_budget=16_000_000,
            max_device_bytes=512 << 20,
            max_host_bytes=large.host_bound,
            host_bound=selected.host_bound,
        )

    complete_small_host = selected.host_bound + sum(
        value.host_bytes for value in selected.tensor_plans.values()
    )
    tight, _ = _ecp_admission(True, complete_small_host - 1)
    smaller, _ = plan_stationary_cuda_grid_schedule(
        grid_points=4096, tile_points=None, admit=tight
    )
    assert smaller.grid_plan.tile_points < 256
    with pytest.raises(ValueError, match="host budget"):
        plan_stationary_cuda_grid_schedule(
            grid_points=4096, tile_points=256, admit=tight
        )

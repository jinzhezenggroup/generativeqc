"""Dead primitive roots must not remove real fallback or nuclear sources."""

from types import SimpleNamespace

import pytest
from generativeqc import _stationary_cuda as runtime
from generativeqc_compiler.integral.first_derivative_schedule import (
    COMPONENT_LABELS,
    derivative_requests,
)
from generativeqc_compiler.method.stationary_cuda import (
    plan_stationary_cuda_primitive_demand,
    qualified_sp_requests,
)


@pytest.mark.parametrize("component_mode", (False, True))
@pytest.mark.parametrize("native_required", (False, True))
@pytest.mark.parametrize("packaged", (False, True))
def test_primitive_root_projection_preserves_fallback_and_packaged_numbering(
    component_mode: bool, native_required: bool, packaged: bool
) -> None:
    requests = (
        derivative_requests(COMPONENT_LABELS)
        if component_mode
        else qualified_sp_requests()
    )
    demand = plan_stationary_cuda_primitive_demand(
        requests,
        component_mode=component_mode,
        native_integrals_required=native_required,
        packaged=packaged,
    )
    if native_required and not packaged:
        assert demand.requests == (("nuclear", ()),)
        assert not demand.component_mode
        assert not demand.integral_derivatives
    else:
        assert demand.requests is requests
        assert demand.component_mode is component_mode
        assert demand.integral_derivatives


def test_projection_cannot_omit_nuclear_repulsion() -> None:
    with pytest.raises(ValueError, match="nuclear repulsion"):
        plan_stationary_cuda_primitive_demand(
            (("kinetic", ("", "")),),
            component_mode=False,
            native_integrals_required=True,
            packaged=False,
        )


@pytest.mark.parametrize(
    "flag", ("component_mode", "native_integrals_required", "packaged")
)
def test_demand_requires_explicit_booleans(flag: str) -> None:
    options = dict.fromkeys(
        ("component_mode", "native_integrals_required", "packaged"), False
    )
    options[flag] = 1
    with pytest.raises(TypeError, match="boolean"):
        plan_stationary_cuda_primitive_demand((("nuclear", ()),), **options)


def test_ordinary_layout_still_rejects_f_shells_before_projection() -> None:
    basis = SimpleNamespace(shells=(SimpleNamespace(angular_momentum=3),))
    with pytest.raises(NotImplementedError, match="s/p/d"):
        runtime._layout(basis)


@pytest.mark.parametrize("pruned", (False, True))
def test_prepared_jit_emits_only_admitted_roots(
    monkeypatch: pytest.MonkeyPatch, pruned: bool
) -> None:
    """Exercise the actual prepared selector, stopping at the compiler boundary."""

    class CompilationReached(Exception):
        pass

    owner = runtime.PreparedStationaryCudaExecution()
    monkeypatch.setattr(owner, "_request", lambda **_: None)
    monkeypatch.setattr(
        runtime, "_layout", lambda _: (None, None, ((("xx", 1.0),),), None)
    )
    requests = (("nuclear", ()),)

    def emit(selected: object) -> str:
        assert pruned
        assert selected == requests
        return "nuclear-only"

    def emit_shards(domain: object) -> tuple:
        assert not pruned
        assert domain == ("xx",)
        return ((requests, "component-shard"),)

    def compile_source(source: object, **options: object) -> None:
        assert source == ("nuclear-only" if pruned else ("component-shard",))
        assert options["primitive_shard_width"] == (
            None if pruned else runtime.CUDA_REQUESTS_PER_UNIT
        )
        raise CompilationReached

    monkeypatch.setattr(runtime, "emit_first_derivative_cuda", emit)
    monkeypatch.setattr(runtime, "derivative_cuda_sources", emit_shards)
    monkeypatch.setattr(runtime, "compile_stationary_cuda", compile_source)
    with pytest.raises(CompilationReached):
        owner.ensure(
            state=None,
            basis=SimpleNamespace(natom=48, nao=384, nprimitive=352),
            contract=None,
            plan=None,
            tensor_plans={},
            compiler=SimpleNamespace(target=None),
            cache="unused",
            requests=requests,
            functional=1,
            ecp=False,
            device=0,
            spec=SimpleNamespace(partition_iterations=3),
            grid_plan=SimpleNamespace(peak_bytes=10),
            source_bytes=10,
            tile_points=4,
            primitive_tile=16,
            integral_terms=16,
            page_work_budget=100,
            max_device_bytes=100,
            max_host_bytes=100,
            host_bound=10,
            integral_derivatives=not pruned,
        )

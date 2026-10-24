"""The composite candidate borrows a live proof, never an XC-name assertion."""

from __future__ import annotations

import ctypes as ct
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from generativeqc import _stationary_cuda as runtime
from generativeqc_compiler.dft.cuda import GridTaskView
from test_stationary_restricted_point import _NATIVE_PREFIX
from test_stationary_task_work_budget import _block

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("attempted", (0, 13))
@pytest.mark.parametrize(
    "failure",
    (
        ArithmeticError,
        RuntimeError,
        MemoryError,
        OSError,
        runtime._Fp32GeometryUnavailable,
    ),
)
@pytest.mark.parametrize("strict_fails", (False, True))
def test_ordinary_candidate_retry_is_typed_bounded_and_drains_before_rebuild(
    attempted: int,
    failure: type[Exception],
    strict_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected whole-source attempt cannot leak into a strict retry or replay."""

    @dataclass(frozen=True)
    class Observed:
        work: dict

    baseline = dict.fromkeys(
        (
            "h2d_bytes",
            "d2h_bytes",
            "launches",
            "primitive_records",
            "xc_points",
            "grid_pair_visits",
            "task_descriptors",
            "task_batches",
            "fp32_geometry_points",
        ),
        0,
    )
    owner = runtime.PreparedStationaryCudaExecution()
    drained = []
    attempts = []
    published = Mock(wraps=owner._lease.mark_success)
    monkeypatch.setattr(owner._lease, "mark_success", published)

    def execute(state: Any, basis: Any, **kwargs: Any) -> Any:
        mixed = kwargs["fp32_semilocal_geometry"]
        attempts.append(mixed)
        request = runtime.PreparedExecutionRequest(
            "candidate", "a" * 64, "b" * 64, "c" * 64, "d" * 64
        )
        owner._lease.install(
            request,
            (runtime.PreparedArtifactBinding("e" * 64, "f" * 64),),
            host_bytes=0,
            device_bytes=0,
        )
        owner._lease.begin(request, max_host_bytes=0, max_device_bytes=0)
        owner._stack = SimpleNamespace(close=lambda: drained.append(mixed))
        owner._attempt_source_metrics = baseline
        owner.sources = SimpleNamespace(
            metrics=lambda: {**baseline, "fp32_geometry_points": attempted}
        )
        if mixed:
            raise failure("candidate failure")
        assert drained == [True]
        if strict_fails:
            raise ArithmeticError("strict failure")
        return Observed({"complete_sources": True, "fp32_geometry_points": 0})

    monkeypatch.setattr(runtime, "_complete_rks_cuda_gradient_diagnostic", execute)
    arguments = {
        "compiler": None,
        "cache": Path("unused"),
        "prepared": owner,
        "fp32_semilocal_geometry": True,
    }
    optional = failure is runtime._Fp32GeometryUnavailable
    retry = (failure is ArithmeticError and attempted > 0) or optional
    try:
        if retry and not strict_fails:
            result = runtime.complete_rks_cuda_gradient_diagnostic(
                None, None, **arguments
            )
            assert result.work["complete_sources"]
            assert result.work["fp32_geometry_points"] == 0
            assert (
                result.work["rejected_semilocal_work"]["fp32_geometry_points"]
                == attempted
            )
            if optional:
                assert result.work["fp32_geometry_unavailable"] == "candidate failure"
                assert not result.work.get("fp64_geometry_retries", 0)
            else:
                assert result.work["fp64_geometry_retries"] == 1
            assert result.work["endpoint_seconds"] > 0
            published.assert_called_once()
        else:
            with pytest.raises(
                ArithmeticError if retry else failure,
                match="strict failure" if retry else "candidate failure",
            ):
                runtime.complete_rks_cuda_gradient_diagnostic(None, None, **arguments)
            published.assert_not_called()
            assert owner._stack is None
        assert attempts == ([True, False] if retry else [True])
    finally:
        owner.close()
    assert drained == attempts


def test_optional_capability_absent_before_owner_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older artifact can deny mixed transport before a source lease exists."""

    @dataclass(frozen=True)
    class Observed:
        work: dict

    owner = runtime.PreparedStationaryCudaExecution()
    attempts = []

    def execute(state: Any, basis: Any, **kwargs: Any) -> Any:
        mixed = kwargs["fp32_semilocal_geometry"]
        attempts.append(mixed)
        if mixed:
            assert owner._attempt_source_metrics is None
            raise runtime._Fp32GeometryUnavailable("missing transport counters")
        request = runtime.PreparedExecutionRequest(
            "strict", "a" * 64, "b" * 64, "c" * 64, "d" * 64
        )
        owner._lease.install(
            request,
            (runtime.PreparedArtifactBinding("e" * 64, "f" * 64),),
            host_bytes=0,
            device_bytes=0,
        )
        owner._lease.begin(request, max_host_bytes=0, max_device_bytes=0)
        return Observed({"fp32_geometry_points": 0, "fp64_geometry_retries": 0})

    monkeypatch.setattr(runtime, "_complete_rks_cuda_gradient_diagnostic", execute)
    with owner:
        result = runtime.complete_rks_cuda_gradient_diagnostic(
            None,
            None,
            compiler=None,
            cache=Path("unused"),
            prepared=owner,
            fp32_semilocal_geometry=True,
        )
        assert result.work["rejected_semilocal_work"] == {}
        assert result.work["fp32_geometry_points"] == 0
        assert result.work["fp64_geometry_retries"] == 0
    assert attempts == [True, False]


@pytest.mark.parametrize(
    "status,error_type",
    [
        (1, RuntimeError),
        (2, MemoryError),
        (3, MemoryError),
        (4, OSError),
        (5, ArithmeticError),
    ],
)
def test_runtime_preserves_native_failure_category(
    status: int, error_type: type[Exception]
) -> None:
    """Only the explicit numerical status is eligible for an arithmetic retry."""

    def native(*args: Any) -> int:
        args[-2].value = b"native diagnostic"
        return status

    owner = object.__new__(runtime._CudaSources)
    owner.library = SimpleNamespace(probe=native)
    with pytest.raises(error_type, match="native diagnostic"):
        owner._call("probe")


@pytest.mark.parametrize("attempted", (0, 11))
@pytest.mark.parametrize(
    "failure",
    (
        ArithmeticError,
        RuntimeError,
        MemoryError,
        OSError,
        runtime._Fp32GeometryUnavailable,
    ),
)
def test_composite_retries_only_executed_numerical_candidate(
    attempted: int,
    failure: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-source retry is bounded and retains attempts instead of hiding them."""
    from generativeqc._stationary_composite_cuda import (
        PreparedCompositeStationaryCudaGradient,
    )

    baseline = dict.fromkeys(
        (
            "h2d_bytes",
            "d2h_bytes",
            "launches",
            "primitive_records",
            "xc_points",
            "grid_pair_visits",
            "task_descriptors",
            "task_batches",
            "fp32_geometry_points",
        ),
        0,
    )
    owner = PreparedCompositeStationaryCudaGradient()
    closed = Mock()
    monkeypatch.setattr(owner, "close", closed)
    owner.sources = SimpleNamespace(
        metrics=lambda: {**baseline, "fp32_geometry_points": attempted}
    )
    calls = []

    def execute(state: Any, basis: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["fp32_semilocal_geometry"])
        owner._attempt_source_metrics = baseline
        if kwargs["fp32_semilocal_geometry"]:
            raise failure("candidate failed")
        return "complete strict forces", {
            "complete_sources": True,
            "endpoint_seconds": 0.0,
        }

    monkeypatch.setattr(owner, "_execute", execute)
    arguments = {
        "compiler": None,
        "cache": Path("unused"),
        "library": Path("unused"),
        "fp32_semilocal_geometry": True,
    }
    optional = failure is runtime._Fp32GeometryUnavailable
    if (failure is ArithmeticError and attempted) or optional:
        forces, work = owner.execute(None, None, **arguments)
        assert forces == "complete strict forces" and work["complete_sources"]
        if optional:
            assert work["fp32_geometry_unavailable"] == "candidate failed"
            assert not work.get("fp64_geometry_retries", 0)
        else:
            assert work["fp64_geometry_retries"] == 1
        assert work["rejected_semilocal_work"]["fp32_geometry_points"] == attempted
        assert work["endpoint_seconds"] > 0
        assert calls == [True, False]
    else:
        with pytest.raises(failure, match="candidate failed"):
            owner.execute(None, None, **arguments)
        assert calls == [True]
    closed.assert_called_once()


def test_composite_strict_retry_failure_does_not_recurse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two failed complete attempts release both owners and never publish forces."""
    from generativeqc._stationary_composite_cuda import (
        PreparedCompositeStationaryCudaGradient,
    )

    baseline = dict.fromkeys(
        (
            "h2d_bytes",
            "d2h_bytes",
            "launches",
            "primitive_records",
            "xc_points",
            "grid_pair_visits",
            "task_descriptors",
            "task_batches",
            "fp32_geometry_points",
        ),
        0,
    )
    owner = PreparedCompositeStationaryCudaGradient()
    owner._attempt_source_metrics = baseline
    owner.sources = SimpleNamespace(
        metrics=lambda: {**baseline, "fp32_geometry_points": 7}
    )
    execute = Mock(side_effect=ArithmeticError("numerical failure"))
    closed = Mock()
    monkeypatch.setattr(owner, "_execute", execute)
    monkeypatch.setattr(owner, "close", closed)
    with pytest.raises(ArithmeticError, match="numerical failure"):
        owner.execute(
            None,
            None,
            compiler=None,
            cache=Path("unused"),
            library=Path("unused"),
            fp32_semilocal_geometry=True,
        )
    assert execute.call_count == closed.call_count == 2
    assert owner.last_work is None


@pytest.mark.parametrize("proof", (False, True))
@pytest.mark.parametrize("flags", (0, 1))
def test_candidate_requires_live_producer_without_widening(
    proof: bool, flags: int
) -> None:
    """Missing producer capability fails before submission; flags remain opaque."""
    owner = object.__new__(runtime._CudaSources)
    owner.handle = ct.c_void_p()
    owner.device = 0
    owner.profile_device = False
    owner.borrowed_streams = set()
    owner.fp32_geometry_requested = True
    owner._call = Mock()
    work = ct.pointer(ct.c_double())
    view = GridTaskView(generation=17, npoint=2, stream=123)
    task = SimpleNamespace(
        view=view,
        _owner=SimpleNamespace(device_id=0),
        density_jets=Mock(),
    )
    if proof:
        task.density_jets_binding = Mock(return_value=(work, flags))
        owner.geometry_molecular_resident_weights(
            task, 0, 4, 256, None, 512, None, functional=4
        )
        arguments = owner._call.call_args.args
        assert (
            arguments[0]
            == "stationary_geometry_molecular_resident_weights_fp32_enqueue_v1"
        )
        assert arguments[2]._obj is view
        assert arguments[-2:] == (17, flags)
        task.density_jets_binding.assert_called_once_with(4)
    else:
        with pytest.raises(NotImplementedError, match="current density binding"):
            owner.geometry_molecular_resident_weights(
                task, 0, 4, 256, None, 512, None, functional=4
            )
        owner._call.assert_not_called()
    task.density_jets.assert_not_called()


def test_candidate_native_abi_gates_proof_and_counts_only_submitted_tiles(
    tmp_path: Path,
    native_cxx: NativeCxx,
) -> None:
    """Execute the exported transport, including null/stale/empty rejection paths."""
    header = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    native = "\n".join(
        _block(header, marker)
        for marker in (
            "static int stationary_geometry_molecular_resident_weights_enqueue_impl(",
            "int stationary_geometry_molecular_resident_weights_fp32_enqueue_v1(",
            "int stationary_geometry_precision_metrics_v1(",
        )
    )
    native = (
        "namespace generativeqc_stationary_cuda {\n"
        + _block(header, "bool stationary_grid_span(")
        + "\n}\n"
        + native
    )
    native, count = re.subn(
        r"geometry_reduce\s*<<<.*?>>>\(.*?\);", "++reductions;", native, flags=re.DOTALL
    )
    assert count == 1
    prefix = (
        _NATIVE_PREFIX.replace(
            "int launches,reductions; bool proof_selected;",
            "int launches,reductions; bool proof_selected,mixed_selected;\n"
            "constexpr bool stationary_fp32_point_capable=true;",
        )
        .replace(
            "struct Owner {",
            "struct Owner { uint64_t fp32_geometry_batches{},fp32_geometry_points{};",
        )
        .replace("bool bound,bool=false", "bool bound,bool mixed=false")
        .replace(
            "++launches; proof_selected=bound;",
            "++launches; proof_selected=bound; mixed_selected=mixed;",
        )
    )
    source = tmp_path / "transport.cpp"
    source.write_text(prefix + native + _DRIVER)
    executable = native_cxx.build_executable(
        [source], tmp_path / "transport", compile_args=["-std=c++17", "-O2"]
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_candidate_work_delta_preserves_capability() -> None:
    """Attempts are cumulative launch work, not a fabricated per-point census."""
    baseline = dict.fromkeys(
        (
            "h2d_bytes",
            "d2h_bytes",
            "launches",
            "primitive_records",
            "xc_points",
            "grid_pair_visits",
            "task_descriptors",
            "task_batches",
        ),
        0,
    )
    before = {**baseline, "fp32_geometry_points": 7, "fp32_geometry_batches": 2}
    after = {
        **baseline,
        "fp32_geometry_points": 18,
        "fp32_geometry_batches": 5,
        "fp32_geometry_capable": 1,
    }
    delta = runtime._metric_delta(after, before)
    assert delta["fp32_geometry_points"] == 11
    assert delta["fp32_geometry_batches"] == 3
    assert delta["fp32_geometry_capable"] == 1
    assert "fp32_geometry_points" not in runtime._metric_delta(baseline, baseline)


_DRIVER = r"""
int main() {
  using namespace generativeqc_stationary_cuda;
  double value{}; int error{};
  for(int mode=0;mode<12;++mode) {
    Owner owner; owner.context.error=&error;
    generativeqc::dft::GridTaskView view;
    view.features=view.ao=view.points=&value;
    launches=reductions=0;
    const uint64_t generation=mode==2?16:17;
    const uint64_t flags=mode==3?2:mode==1?0:1;
    if(mode==4) view.features=nullptr;
    if(mode==5) view.npoint=0;
    size_t offset=0,per_atom=4;
    if(mode==8) { owner.molecular_points=2; per_atom=0; }
    if(mode==9) per_atom=0;
    if(mode==10) offset=48;
    if(mode==11) per_atom=SIZE_MAX;
    const int status=stationary_geometry_molecular_resident_weights_fp32_enqueue_v1(
        mode==6?nullptr:&owner,mode==7?nullptr:&view,&value,offset,per_atom,&value,&value,
        generation,flags,nullptr,0);
    const bool submitted=mode==0 || mode==1 || mode==8;
    assert(status==int(mode!=0 && mode!=1 && mode!=5 && mode!=8));
    assert(launches==int(submitted) && reductions==int(submitted));
    if(submitted) assert(mixed_selected && proof_selected==(mode==0 || mode==8));
    uint64_t metrics[3]{};
    assert(!stationary_geometry_precision_metrics_v1(&owner,metrics,3));
    assert(metrics[0]==1 && metrics[1]==uint64_t(submitted));
    assert(metrics[2]==(submitted?2:0));
    assert(stationary_geometry_precision_metrics_v1(nullptr,metrics,3));
    assert(stationary_geometry_precision_metrics_v1(&owner,nullptr,3));
    assert(stationary_geometry_precision_metrics_v1(&owner,metrics,2));
  }
}
"""

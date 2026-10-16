"""A packaged all-electron force request must not discover an NVCC compiler."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from _stationary_artifact_test_support import stationary_route
from generativeqc import _dft_gradient, _native, _stationary_cuda
from generativeqc._snapshot_grid_cache import SnapshotGridCache
from generativeqc.batch import PreparedBatch
from generativeqc_compiler import dft
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.method import resolve_method


def test_gpu_qualification_guards_bind_current_source_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check hardware-gate instrumentation even when the GPU cases are skipped."""
    from generativeqc_compiler.integral import first_derivative_schedule
    from generativeqc_compiler.method import stationary_cuda
    from generativeqc_compiler.method.stationary_gradient import StationaryGradientPlan
    from test_stationary_aot_cuda import _forbid_aot_generation

    batch = SimpleNamespace(_stationary_cuda_compiler=lambda: None)
    _forbid_aot_generation(monkeypatch, batch)
    for guarded in (
        batch._stationary_cuda_compiler,
        _stationary_cuda.compile_stationary_cuda,
        _stationary_cuda.cached_derivative_cuda_source,
        first_derivative_schedule.emit_first_derivative_cuda,
        first_derivative_schedule.derivative_cuda_sources,
        stationary_cuda.emit_stationary_wrapper_cuda,
        StationaryGradientPlan.integral_block,
        StationaryGradientPlan.reduction_program,
    ):
        with pytest.raises(pytest.fail.Exception, match="IR/AD/source generation"):
            guarded()


@pytest.mark.parametrize("entrypoint", ("cold", "prepared"))
@pytest.mark.parametrize("materialize", (False, True))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize(
    ("aot", "ecp", "component_mode", "expected_jit"),
    [
        (True, False, False, False),
        (False, False, False, True),
        (True, True, False, True),
        (True, False, True, False),
        (False, True, True, True),
        (False, False, True, True),
    ],
    ids=(
        "sp-aot",
        "missing-aot",
        "ecp-jit",
        "component-aot",
        "combined-jit",
        "component-jit",
    ),
)
def test_stationary_artifact_route_keeps_source_emission_lazy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    materialize: bool,
    spin: str,
    aot: bool,
    ecp: bool,
    component_mode: bool,
    expected_jit: bool,
) -> None:
    """Execute both real routes; a cache hit never needs the source provider."""
    from generativeqc_compiler.method.stationary_cuda import QUALIFIED_SPD_COMPONENTS

    class Selected(Exception):
        pass

    probe = stationary_route(
        monkeypatch, tmp_path, entrypoint, component_mode=component_mode, spin=spin
    )
    emitted = []
    source = object()

    def emit(requests: object, **options: object) -> object:
        assert expected_jit
        assert requests == probe.requests
        assert options["target"] is probe.target
        assert options["component_domain"] == (("xx",) if component_mode else None)
        emitted.append(True)
        return source

    def compile_source(provider: object, **options: object) -> None:
        assert expected_jit, "packaged route attempted JIT compilation"
        assert callable(provider), "source emission must be deferred until a cache miss"
        assert not emitted, "source was emitted before the compiler requested it"
        if materialize:
            assert provider() is source
        raise Selected

    def load(directory: object, **options: object) -> None:
        assert not expected_jit, "runtime JIT request used packaged scientific code"
        assert directory == tmp_path
        assert options["spin"] == spin
        assert options["functional"] == 1
        assert options["iterations"] == 3
        assert options["architecture"] == probe.target.architecture
        assert options["component_domain"] == (
            QUALIFIED_SPD_COMPONENTS if component_mode else None
        )
        raise Selected

    monkeypatch.setattr(_stationary_cuda, "cached_derivative_cuda_source", emit)
    monkeypatch.setattr(_stationary_cuda, "compile_stationary_cuda", compile_source)
    monkeypatch.setattr(_stationary_cuda, "load_stationary_aot_artifact", load)
    with pytest.raises(Selected):
        probe.run(aot=aot, ecp=ecp)
    assert bool(emitted) is (expected_jit and materialize)


@pytest.mark.parametrize("entrypoint", ("cold", "prepared"))
@pytest.mark.parametrize("component_mode", (False, True))
def test_missing_stationary_package_fails_without_jit_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    component_mode: bool,
) -> None:
    probe = stationary_route(
        monkeypatch, tmp_path, entrypoint, component_mode=component_mode
    )

    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("missing packaged stationary CUDA artifact")

    monkeypatch.setattr(_stationary_cuda, "load_stationary_aot_artifact", missing)
    with pytest.raises(FileNotFoundError, match="missing packaged"):
        probe.run(aot=True)


@pytest.mark.parametrize("method", ("PBE", "PBE0", "B3LYP"))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("missing_artifact", (False, True))
@pytest.mark.parametrize("tile_points", (None, 256))
def test_public_aot_force_does_not_probe_nvcc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_artifact: bool,
    method: str,
    spin: str,
    tile_points: int | None,
) -> None:
    class Basis:
        shells = ()
        natom, nao = 2, 2

        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(dft, "NativeAO", lambda *args, **kwargs: Basis())
    source = SimpleNamespace(
        backend="cuda",
        hamiltonian="all-electron",
        method_ir=resolve_method(method, spin=spin),
        metadata=(0, 0, 0, 0, 0, 0, 3 if method == "B3LYP" else 1),
        close=Mock(),
    )
    # Supply the force-policy workload metadata used before AOT admission;
    # compiler discovery remains forbidden by the explicit failing callback.
    state = SimpleNamespace(
        _source=source,
        identity=SimpleNamespace(ingredients=("rho", "sigma"), spin=spin),
        grid=SimpleNamespace(points=np.zeros((1, 3))),
    )

    def from_native(
        observed_batch: object, observed_basis: object, **kwargs: object
    ) -> SimpleNamespace:
        assert observed_batch is batch and isinstance(observed_basis, Basis)
        assert kwargs["index"] == 0
        assert kwargs["allow_budget_stationarity"] is True
        return state

    monkeypatch.setattr(_dft_gradient.StationaryKsState, "from_native", from_native)
    target = cuda_target_info("sm_120")

    def calculate(*args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["compiler"] is None
        assert kwargs["tile_points"] == tile_points
        assert kwargs["max_grid_points"] is None
        assert kwargs["max_grid_pair_visits"] is None
        assert kwargs["target"] is target
        assert kwargs["aot_directory"] == tmp_path
        if missing_artifact:
            raise FileNotFoundError("missing packaged stationary CUDA artifact")
        return SimpleNamespace(gradient=np.ones((2, 3)), work={"tensor_executions": 0})

    monkeypatch.setattr(
        _stationary_cuda, "complete_rks_cuda_gradient_diagnostic", calculate
    )
    batch = SimpleNamespace(
        _calculator=SimpleNamespace(
            _basis=object(),
            _representation_name="cartesian",
            _density_fitting_mode=_native.DENSITY_FITTING_NONE,
            _capabilities=SimpleNamespace(supported_properties={"energy", "forces"}),
        ),
        _stationary_cuda_execution=None,
        _snapshot_grid_cache=None,
        _charges=[0],
        _multiplicities=[1],
        _library=SimpleNamespace(_name=str(tmp_path / "libgenerativeqc.so")),
        _stationary_cuda_compiler=lambda: pytest.fail("NVCC discovery before AOT load"),
        _stationary_cuda_target=lambda: target,
    )
    if missing_artifact:
        with pytest.raises(FileNotFoundError, match="missing packaged"):
            PreparedBatch._public_dft_cuda_force(batch, 0, (), tile_points=tile_points)
    else:
        force, work = PreparedBatch._public_dft_cuda_force(
            batch, 0, (), tile_points=tile_points
        )
        np.testing.assert_array_equal(force, -np.ones((2, 3)))
        assert work["tensor_executions"] == 0
        workload = work["force_active_ao_policy"]["workload"]
        assert workload["tile_points"] == tile_points
        assert workload["tile_policy"] == (
            "budget-auto" if tile_points is None else "fixed"
        )
    source.close.assert_called_once()
    assert isinstance(batch._snapshot_grid_cache, SnapshotGridCache)


@pytest.mark.parametrize("tile_points", (True, 0, -1, 256.0, "256"))
def test_explicit_force_tile_rejects_invalid_request_before_owner_use(
    tile_points: object,
) -> None:
    with pytest.raises(ValueError, match="force tile size must be a positive integer"):
        PreparedBatch._public_dft_cuda_force(None, 0, (), tile_points=tile_points)


@pytest.mark.parametrize("method", ("PBE", "PBE0", "B3LYP"))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_public_d_shell_force_uses_component_aot_without_nvcc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str, spin: str
) -> None:
    class Basis:
        shells = (SimpleNamespace(angular_momentum=2),)
        natom, nao = 2, 5

        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(dft, "NativeAO", lambda *args, **kwargs: Basis())
    source = SimpleNamespace(
        backend="cuda",
        hamiltonian="all-electron",
        method_ir=resolve_method(method, spin=spin),
        metadata=(0, 0, 0, 0, 0, 0, 3 if method == "B3LYP" else 1),
        close=Mock(),
    )
    state = SimpleNamespace(
        _source=source,
        identity=SimpleNamespace(ingredients=("rho", "sigma"), spin=spin),
        grid=SimpleNamespace(points=np.zeros((1, 3))),
    )
    monkeypatch.setattr(
        _dft_gradient.StationaryKsState, "from_native", lambda *args, **kwargs: state
    )
    target = cuda_target_info("sm_120")

    def calculate(*args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["compiler"] is None
        assert kwargs["target"] is target
        assert kwargs["aot_directory"] == tmp_path
        assert kwargs["native_grid_library"] == tmp_path / "libgenerativeqc.so"
        return SimpleNamespace(gradient=np.ones((2, 3)), work={"tensor_executions": 0})

    monkeypatch.setattr(
        _stationary_cuda, "complete_rks_cuda_gradient_diagnostic", calculate
    )
    batch = SimpleNamespace(
        _calculator=SimpleNamespace(
            _basis=object(),
            _representation_name="spherical",
            _density_fitting_mode=_native.DENSITY_FITTING_NONE,
            _capabilities=SimpleNamespace(supported_properties={"energy", "forces"}),
        ),
        _stationary_cuda_execution=None,
        _snapshot_grid_cache=None,
        _charges=[0],
        _multiplicities=[1],
        _library=SimpleNamespace(_name=str(tmp_path / "libgenerativeqc.so")),
        _stationary_cuda_compiler=lambda: pytest.fail(
            "NVCC discovery before component AOT load"
        ),
        _stationary_cuda_target=lambda: target,
    )

    force, work = PreparedBatch._public_dft_cuda_force(batch, 0, ())

    np.testing.assert_array_equal(force, -np.ones((2, 3)))
    assert work["tensor_executions"] == 0
    source.close.assert_called_once()
    assert isinstance(batch._snapshot_grid_cache, SnapshotGridCache)


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("changed_weight", (False, True))
def test_public_hybrid_route_preserves_composition_not_method_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spin: str, changed_weight: bool
) -> None:
    """An alias reuses AOT; a different exchange fraction has bounded JIT only."""
    method = replace(resolve_method("PBE0", spin=spin), identifier="custom-pbe0")
    if changed_weight:
        method = replace(
            method,
            primitives=tuple(
                replace(node, coefficient=Fraction(3, 7))
                if node.kind == "exact_exchange"
                else node
                for node in method.primitives
            ),
        )

    class Basis:
        shells = (SimpleNamespace(angular_momentum=2),)
        natom, nao = 2, 5

        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(dft, "NativeAO", lambda *args, **kwargs: Basis())
    source = SimpleNamespace(
        backend="cuda",
        hamiltonian="all-electron",
        method_ir=method,
        metadata=(0, 0, 0, 0, 0, 0, 1),
        close=Mock(),
    )
    state = SimpleNamespace(
        _source=source,
        identity=SimpleNamespace(ingredients=("rho", "sigma"), spin=spin),
        grid=SimpleNamespace(points=np.zeros((1, 3))),
    )
    monkeypatch.setattr(
        _dft_gradient.StationaryKsState, "from_native", lambda *args, **kwargs: state
    )
    compiler = object()
    discover = Mock(return_value=compiler)

    def calculate(*args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["compiler"] is (compiler if changed_weight else None)
        assert kwargs["aot_directory"] == (None if changed_weight else tmp_path)
        assert state._source.method_ir is method
        return SimpleNamespace(gradient=np.ones((2, 3)), work={})

    monkeypatch.setattr(
        _stationary_cuda, "complete_rks_cuda_gradient_diagnostic", calculate
    )
    batch = SimpleNamespace(
        _calculator=SimpleNamespace(
            _basis=object(),
            _representation_name="spherical",
            _density_fitting_mode=_native.DENSITY_FITTING_NONE,
            _capabilities=SimpleNamespace(supported_properties={"energy", "forces"}),
        ),
        _stationary_cuda_execution=None,
        _snapshot_grid_cache=None,
        _charges=[0],
        _multiplicities=[1],
        _library=SimpleNamespace(_name=str(tmp_path / "libgenerativeqc.so")),
        _stationary_cuda_compiler=discover,
        _stationary_cuda_target=lambda: cuda_target_info("sm_120"),
    )
    force, _ = PreparedBatch._public_dft_cuda_force(batch, 0, ())
    np.testing.assert_array_equal(force, -np.ones((2, 3)))
    assert discover.call_count == int(changed_weight)
    source.close.assert_called_once()

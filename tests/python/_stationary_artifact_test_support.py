"""Run real stationary admission/selection with inert compiler and device owners."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import _stationary_cuda as runtime
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.integral import first_derivative_schedule
from generativeqc_compiler.integral.first_derivative_schedule import derivative_requests
from generativeqc_compiler.method import resolve_method

if TYPE_CHECKING:
    from pathlib import Path


def stationary_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    *,
    component_mode: bool = False,
    native_integrals: bool = False,
    spin: str = "unpolarized",
) -> SimpleNamespace:
    """Keep selection and demand real; replace external snapshot/device boundaries."""
    target = cuda_target_info("sm_120")

    class Compiler:
        def __init__(self) -> None:
            self.target = target

    monkeypatch.setattr(runtime, "CudaCompilerAdapter", Compiler)
    method = resolve_method("PBE", spin=spin)
    contract = SimpleNamespace(family="gga", spin=spin, validate=lambda state: state)
    monkeypatch.setattr(runtime, "StationaryDerivativeContract", lambda _: contract)
    monkeypatch.setattr(runtime, "native_ao_geometry_identity", lambda _: "geometry")
    basis = SimpleNamespace(
        identity="basis",
        natom=2,
        nao=1,
        nprimitive=4097 if native_integrals else 1,
        charge=0,
        multiplicity=1,
        representation="cartesian",
        packed=np.zeros(8),
        atoms=(SimpleNamespace(atomic_number=1), SimpleNamespace(atomic_number=1)),
    )
    expansions = ((("xx" if component_mode else "", 1.0),),)
    requests = (
        derivative_requests(("xx",))
        if component_mode
        else runtime.qualified_sp_requests()
    )
    monkeypatch.setattr(
        runtime,
        "_layout",
        lambda _: (np.array([[1.0, 1.0]]), np.array([[0, 0, 1]]), expansions, requests),
    )
    spec = SimpleNamespace(partition_iterations=3)
    source = SimpleNamespace(
        backend="cuda",
        hamiltonian="all-electron",
        method_ir=method,
        metadata=(3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        grid_spec=spec,
        ecp_cores=(0, 0),
        ecp_terms=(),
        functional=SimpleNamespace(ingredients=("rho", "sigma")),
        cuda_integral_derivatives=lambda **_: pytest.fail(
            "unexpected integral execution"
        ),
        ecp_derivatives=lambda: None,
    )
    state = SimpleNamespace(
        _source=source,
        identity=SimpleNamespace(
            basis_identity="basis",
            geometry_identity="geometry",
            method=method.identifier,
            functional_identity="pbe",
            regularization_identity="regularization",
        ),
        occupations=np.array([2.0]),
        density=np.zeros((2 if spin == "polarized" else 1, 1, 1)),
        grid=SimpleNamespace(points=np.zeros((1, 3))),
    )
    grid_plan = SimpleNamespace(
        peak_bytes=16,
        allocation_bytes=16,
        tile_points=1,
        active_ao_capacity=None,
        features_only=True,
    )
    layout = SimpleNamespace(
        grid_plan=grid_plan,
        tensor_plans={},
        host_bound=16,
        native_integral_host_reserve=0,
        ecp_workspace=0,
        ecp_pair_samples=0,
        source_resources=SimpleNamespace(allocation_bytes=16),
    )
    monkeypatch.setattr(runtime, "_plan_stationary_cuda_tile", lambda *a, **k: layout)
    grid_closed = []

    class Grid:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> SimpleNamespace:
            return SimpleNamespace()

        def __exit__(self, *args: object) -> None:
            # Only explicit owner cleanup may satisfy the assertion. A generator
            # context manager would also close when garbage-collected, masking
            # a missing ExitStack.close() on prepared construction failure.
            grid_closed.append(True)

    monkeypatch.setattr(runtime, "CudaGrid", Grid)
    monkeypatch.setattr(runtime, "_native_grid_artifact", lambda *a, **k: object())

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("unexpected compiler, derivative emission, or device construction")

    for name in (
        "compile_grid",
        "compile_cuda",
        "compile_stationary_cuda",
        "cached_derivative_cuda_source",
        "load_stationary_aot_artifact",
        "_CudaSources",
    ):
        monkeypatch.setattr(runtime, name, forbidden)
    # No test may accidentally generate real derivative source, even after a
    # route regression. The admitted lazy cached provider is supplied by tests.
    for name in ("emit_first_derivative_cuda", "derivative_cuda_sources"):
        monkeypatch.setattr(first_derivative_schedule, name, forbidden)
        monkeypatch.setattr(runtime, name, forbidden, raising=False)
    owner = runtime.PreparedStationaryCudaExecution()

    def run(*, aot: bool, ecp: bool = False, profile_device: bool = False) -> None:
        source.hamiltonian = "scalar-semilocal-ecp" if ecp else "all-electron"
        compiler = None if aot and not ecp else Compiler()
        options = {
            "compiler": compiler,
            "cache": tmp_path,
            "aot_directory": tmp_path if aot else None,
            "target": target,
            "native_grid_library": tmp_path / "libgrid.so",
            "profile_device": profile_device,
        }
        if entrypoint == "cold":
            runtime.complete_rks_cuda_gradient_diagnostic(
                state,
                basis,
                tile_points=1,
                **options,
            )
        else:
            demand = runtime.plan_stationary_cuda_primitive_demand(
                requests,
                component_mode=component_mode,
                native_integrals_required=native_integrals,
                packaged=aot and not ecp,
            )
            plan = runtime.StationaryGradientPlan(
                method,
                runtime.StationaryMeanField(
                    runtime.SCF_POINT_MODEL, hamiltonian=source.hamiltonian
                ),
            )
            owner.ensure(
                state=state,
                basis=basis,
                contract=contract,
                plan=plan,
                tensor_plans={},
                requests=demand.requests,
                functional=1,
                ecp=ecp,
                device=0,
                spec=spec,
                grid_plan=grid_plan,
                source_bytes=16,
                tile_points=1,
                primitive_tile=16,
                integral_terms=2,
                page_work_budget=100,
                max_device_bytes=1 << 20,
                max_host_bytes=1 << 20,
                host_bound=16,
                integral_derivatives=demand.integral_derivatives,
                **options,
            )

    return SimpleNamespace(
        run=run, target=target, requests=requests, grid_closed=grid_closed
    )

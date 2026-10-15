"""Exercise actual stationary provider admission with explicit inert device owners."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import _stationary_cpu as cpu
from generativeqc import _stationary_cuda as cuda
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.integral import first_derivative_schedule
from generativeqc_compiler.method import resolve_method

if TYPE_CHECKING:
    from pathlib import Path


class _OwnerReady(Exception):
    """Stop after real provider admission, at the first native reset boundary."""


def cuda_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: Any,
    *,
    budget: int,
    ecp: bool,
    required: bool,
    host_reserve: int,
    device_reserve: int,
) -> Any:
    """Return an admitted provider publication; this does not execute GPU forces."""
    target = cuda_target_info("sm_120")

    class Compiler:
        def __init__(self) -> None:
            self.target = target

    monkeypatch.setattr(cuda, "CudaCompilerAdapter", Compiler)
    contract = SimpleNamespace(
        family="gga", spin="unpolarized", validate=lambda state: state
    )
    monkeypatch.setattr(cuda, "StationaryDerivativeContract", lambda _: contract)
    monkeypatch.setattr(cuda, "native_ao_geometry_identity", lambda _: "geometry")
    method = resolve_method("PBE")
    bound_source = SimpleNamespace(
        **vars(source),
        backend="cuda",
        method_ir=method,
        metadata=(3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        grid_spec=SimpleNamespace(partition_iterations=3, coincident_tolerance=1e-12),
        hamiltonian="scalar-semilocal-ecp" if ecp else "all-electron",
        ecp_cores=(0, 0),
        ecp_terms=(),
        functional=SimpleNamespace(ingredients=("rho", "sigma")),
        ecp_derivatives=lambda: None,
    )
    publication = []

    def wrap(provider: Any) -> Any:
        def evaluate(*args: Any, **kwargs: Any) -> Any:
            result = provider(*args, **kwargs)
            if result is not None:
                publication.append(result[0])
            return result

        return evaluate

    for name in ("cuda_integral_derivatives", "density_fitted_integral_derivatives"):
        provider = getattr(source, name, None)
        if callable(provider):
            setattr(bound_source, name, wrap(provider))
    basis = SimpleNamespace(
        identity="basis",
        natom=2,
        nao=1,
        nprimitive=4097 if required else 1,
        charge=0,
        atoms=(SimpleNamespace(atomic_number=1), SimpleNamespace(atomic_number=1)),
    )
    state = SimpleNamespace(
        _source=bound_source,
        identity=SimpleNamespace(basis_identity="basis", geometry_identity="geometry"),
        occupations=np.array([2.0]),
        density=np.zeros((1, 1, 1)),
        weighted_density=np.zeros((1, 1, 1)),
        grid=SimpleNamespace(points=np.zeros((1, 3))),
    )
    monkeypatch.setattr(
        cuda,
        "_layout",
        lambda _: (
            np.array([[1.0, 1.0]]),
            np.array([[0, 0, 1]]),
            ((("", 1.0),),),
            cuda.qualified_sp_requests(),
        ),
    )
    layout = SimpleNamespace(
        grid_plan=SimpleNamespace(
            peak_bytes=16, tile_points=1, active_ao_capacity=None, features_only=True
        ),
        tensor_plans={},
        host_bound=16,
        native_integral_host_reserve=host_reserve,
        native_geometry_reserve=device_reserve,
        ecp_workspace=0,
        ecp_pair_samples=0,
        source_resources=SimpleNamespace(allocation_bytes=16),
    )
    monkeypatch.setattr(cuda, "_plan_stationary_cuda_tile", lambda *a, **k: layout)

    def forbid(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "provider admission must not execute compilers or derivative generation"
        )

    for name in ("cached_derivative_cuda_source", "compile_cuda", "compile_grid"):
        monkeypatch.setattr(cuda, name, forbid)
    for name in ("emit_first_derivative_cuda", "derivative_cuda_sources"):
        monkeypatch.setattr(first_derivative_schedule, name, forbid)
        monkeypatch.setattr(cuda, name, forbid, raising=False)
    monkeypatch.setattr(cuda, "compile_stationary_cuda", lambda *a, **k: object())
    monkeypatch.setattr(cuda, "_native_grid_artifact", lambda *a, **k: object())
    entered, closed = [], []
    accepted = []

    class Owner:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> Any:
            entered.append(self.name)
            return self

        def __exit__(self, *args: object) -> None:
            closed.append(self.name)

        def reset_geometry(self, *args: Any) -> None:
            accepted.append(publication[-1])
            raise _OwnerReady

        def reset(self, *args: Any) -> None:
            accepted.append(None)
            raise _OwnerReady

    monkeypatch.setattr(cuda, "CudaGrid", lambda *a, **k: Owner("grid"))
    monkeypatch.setattr(cuda, "_CudaSources", lambda *a, **k: Owner("sources"))
    try:
        try:
            cuda.complete_rks_cuda_gradient_diagnostic(
                state,
                basis,
                compiler=Compiler(),
                target=target,
                cache=tmp_path,
                native_grid_library=tmp_path / "grid.so",
                tile_points=1,
                max_device_bytes=32 + budget,
            )
        except _OwnerReady:
            assert accepted, "provider admission did not reach the owner boundary"
            return accepted[-1]
        pytest.fail("provider probe unexpectedly executed a complete force endpoint")
    finally:
        # Explicit objects have no finalizers that could conceal a missing ExitStack cleanup.
        assert closed == list(reversed(entered)), (
            "stationary owners leaked or closed out of dependency order"
        )


def cpu_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: Any,
    execution: str,
) -> bool:
    """Run the real CPU admission gate, stopping at resource planning."""

    class Admitted(Exception):
        pass

    admitted = []
    contract = SimpleNamespace(validate=lambda state: state)
    monkeypatch.setattr(cpu, "StationaryDerivativeContract", lambda _: contract)
    monkeypatch.setattr(cpu, "native_ao_geometry_identity", lambda _: "geometry")

    def plan(*args: Any, **kwargs: Any) -> None:
        admitted.append(kwargs["native_fitted_integrals"])
        raise Admitted

    monkeypatch.setattr(cpu, "_admit_work", plan)
    state = SimpleNamespace(
        _source=SimpleNamespace(**vars(source), backend="cpu"),
        identity=SimpleNamespace(basis_identity="basis", geometry_identity="geometry"),
    )
    with pytest.raises(Admitted):
        cpu.complete_rks_gradient_diagnostic(
            state,
            SimpleNamespace(identity="basis"),
            execution=execution,
            cache=tmp_path,
        )
    return admitted[-1]

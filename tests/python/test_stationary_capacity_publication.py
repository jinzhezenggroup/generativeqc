"""Execute complete orchestration with inert owners, without CUDA or compilation.

Manufactured source arrays test composition, work accounting and ownership only;
these tests do not claim numerical force or resource-allocation qualification.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self

import numpy as np
import pytest
from generativeqc import _stationary_cuda as runtime
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.method import resolve_method


def endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    provider: str,
    combined: bool,
    method: str = "PBE0",
    pair_delta: int = 0,
) -> tuple[Any, dict[str, Any]]:
    target = cuda_target_info("sm_120")
    events: list[str] = []
    pages: list[tuple[str, tuple[int, ...]]] = []
    native_calls: list[dict[str, Any]] = []
    identity = SimpleNamespace(basis_identity="basis", geometry_identity="geometry")
    model = resolve_method(method)
    source_names = runtime.stationary_runtime_sources(
        runtime.StationaryGradientPlan(
            model, runtime.StationaryMeanField(runtime.SCF_POINT_MODEL)
        )
    )
    integral_names = {"one_electron", "overlap_pulay", "coulomb", "exact_exchange"}
    values = {name: np.full((2, 3), i + 1.0) for i, name in enumerate(source_names)}
    native_values = np.stack(
        [np.full((2, 3), v) for v in (3.0, 5.0, 7.0, 11.0 if method == "PBE0" else 0.0)]
    )

    class Compiler:
        def __init__(self) -> None:
            self.target = target

    class Contract(runtime.StationaryDerivativeContract):
        # Inherit real native force-policy selection with an inert state boundary.
        spin, family = "unpolarized", "gga"

        def __init__(
            self, observed: object, *, allow_budget_stationarity: bool = False
        ) -> None:
            assert observed is identity and not allow_budget_stationarity

        def validate(self, observed: object) -> None:
            assert observed is state
            events.append("validate")

    monkeypatch.setattr(runtime, "CudaCompilerAdapter", Compiler)
    monkeypatch.setattr(runtime, "StationaryDerivativeContract", Contract)
    monkeypatch.setattr(runtime, "native_ao_geometry_identity", lambda _: "geometry")
    basis = SimpleNamespace(
        identity="basis",
        natom=2,
        nao=2,
        nprimitive=2,
        charge=0,
        atoms=(SimpleNamespace(atomic_number=1), SimpleNamespace(atomic_number=1)),
    )
    source = SimpleNamespace(
        backend="cuda",
        hamiltonian="all-electron",
        method_ir=model,
        metadata=(3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        grid_spec=SimpleNamespace(partition_iterations=3, coincident_tolerance=1e-12),
        ecp_cores=(0, 0),
        ecp_terms=(),
        functional=SimpleNamespace(ingredients=("rho", "sigma")),
        density_fitted=provider == "fitted",
        atomic_weights=np.ones(5),
        values=np.zeros(3),
        grid_cache_work={},
        export_work={},
    )

    def native(
        atoms: int, budget: int, **kwargs: Any
    ) -> tuple[np.ndarray, dict[str, int]]:
        assert atoms == 2 and budget == 160
        if provider == "fitted":
            assert not kwargs
        else:
            assert kwargs.get("range_exchange") is False
            assert kwargs.get("combined_two_electron", False) is combined
        native_calls.append(kwargs)
        return (
            native_values[:3] if combined and provider == "direct" else native_values,
            {"one_electron_host_peak_bytes": 24, "one_electron_device_peak_bytes": 32},
        )

    if provider == "fitted":
        source.density_fitted_integral_derivatives = native
        source.cuda_integral_derivatives = lambda *a, **k: pytest.fail(
            "DF used Direct derivative source"
        )
    elif provider == "direct":
        source.cuda_integral_derivatives = native
    state = SimpleNamespace(
        _source=source,
        identity=identity,
        occupations=np.array([2.0]),
        density=np.zeros((1, 2, 2)),
        weighted_density=np.zeros((1, 2, 2)),
        grid=SimpleNamespace(
            points=np.arange(15.0).reshape(5, 3),
            weights=np.ones(5),
            owners=np.zeros(5, dtype=int),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_layout",
        lambda _: (
            np.zeros((2, 2)),
            np.array([[0, 0, 1], [1, 0, 1]]),
            ((("", 1.0),), (("", 1.0),)),
            runtime.qualified_sp_requests(),
        ),
    )
    layout = SimpleNamespace(
        grid_plan=SimpleNamespace(
            peak_bytes=32,
            tile_points=2,
            active_ao_capacity=None,
            features_only=True,
            order=2,
        ),
        tensor_plans={},
        host_bound=256,
        native_integral_host_reserve=64,
        native_geometry_reserve=64,
        ecp_workspace=0,
        ecp_pair_samples=0,
        source_resources=SimpleNamespace(
            allocation_bytes=64, center_geometry_bytes=0, phased_becke_bytes=0
        ),
    )
    monkeypatch.setattr(runtime, "_plan_stationary_cuda_tile", lambda *a, **k: layout)
    artifact = SimpleNamespace(
        library=tmp_path / "inert.so",
        metadata={
            "binary_sha256": "0" * 64,
            "key": "test",
            "artifact_kind": "packaged-aot",
            "weight_programs": {},
        },
    )
    monkeypatch.setattr(runtime, "compile_stationary_cuda", lambda *a, **k: artifact)
    monkeypatch.setattr(runtime, "_native_grid_artifact", lambda *a, **k: artifact)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("orchestration control attempted compiler/device work")

    for name in ("compile_grid", "compile_cuda", "cached_derivative_cuda_source"):
        monkeypatch.setattr(runtime, name, forbidden)

    class Grid:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> Self:
            events.append("grid-open")
            return self

        def __exit__(self, *args: object) -> None:
            events.append("grid-close")

        def set_density(self, density: np.ndarray) -> None:
            np.testing.assert_array_equal(density, state.density[0])

        @contextmanager
        def feature_task(self, points: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            yield SimpleNamespace(count=len(points))

        def metrics(self) -> dict[str, int]:
            return {}

    class Sources:
        borrowed_streams = ()
        primitive_pages = primitive_page_peak_records = bulk_pack_chunks = (
            bulk_packed_descriptors
        ) = scalar_packed_descriptors = 0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.records = 0
            self.points = 0
            self.pending_tiles = 0

        def __enter__(self) -> Self:
            events.append("sources-open")
            return self

        def __exit__(self, *args: object) -> None:
            events.append("sources-close")

        def reset(self, *args: Any) -> None:
            pass

        def configure_grid(self, spec: Any, atoms: Any) -> None:
            assert spec is state._source.grid_spec
            assert tuple(atoms) == basis.atoms

        def reset_geometry(self, *args: Any) -> None:
            pass

        def integral_page(
            self, index: int, operator: str, coordinates: np.ndarray, *args: Any
        ) -> None:
            coordinates = np.asarray(coordinates)
            assert coordinates.shape[1] == (4 if operator == "four_center_eri" else 2)
            self.records += len(coordinates)
            pages.append((operator, coordinates.shape))

        def nuclear(self, atom: int, other: int, charges: Any) -> None:
            assert atom > other
            self.records += 1

        def flush(self) -> None:
            pass

        def geometry(self, task: Any, *args: Any, **kwargs: Any) -> None:
            self.pending_tiles += 1
            assert self.pending_tiles <= 1, "configured pending tile window exceeded"
            self.points += task.count

        def drain_geometry(self) -> None:
            self.pending_tiles = 0
            events.append("drain")

        def finish(self) -> dict[str, np.ndarray]:
            return {name: value.copy() for name, value in values.items()}

        def reduced(self) -> np.ndarray:
            return sum(
                (
                    value
                    for name, value in values.items()
                    if provider == "none" or name not in integral_names
                ),
                np.zeros((2, 3)),
            )

        def metrics(self) -> dict[str, int]:
            return {
                "primitive_records": self.records,
                "xc_points": self.points,
                "grid_pair_visits": 1 + 2 * self.points + pair_delta,
                "center_geometry_bytes": 0,
                "owned_device_bytes": 64,
                "h2d_bytes": 0,
                "d2h_bytes": 0,
            }

    monkeypatch.setattr(runtime, "CudaGrid", Grid)
    monkeypatch.setattr(runtime, "_CudaSources", Sources)
    monkeypatch.setenv(
        "GENERATIVEQC_DIRECT_FORCE_REDUCTION", "combined" if combined else "separate"
    )
    result = runtime._complete_rks_cuda_gradient_diagnostic(
        state,
        basis,
        compiler=Compiler(),
        cache=tmp_path,
        native_grid_library=tmp_path / "inert.so",
        target=target,
        tile_points=2,
        max_device_bytes=256,
        max_host_bytes=1024,
        max_pending_grid_tiles=1,
    )
    expected = sum(
        (
            value
            for name, value in values.items()
            if provider == "none" or name not in integral_names
        ),
        np.zeros((2, 3)),
    )
    if provider != "none":
        expected += np.sum(
            native_values[:3] if combined and provider == "direct" else native_values,
            axis=0,
        )
    return result, {
        "expected": expected,
        "events": events,
        "pages": pages,
        "native_calls": native_calls,
    }


@pytest.mark.parametrize("method", ["PBE", "PBE0"])
@pytest.mark.parametrize(
    "provider,combined",
    [
        ("none", False),
        ("direct", False),
        ("direct", True),
        ("fitted", False),
        ("fitted", True),
    ],
)
def test_complete_stationary_composition_and_work_accounting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    method: str,
    provider: str,
    combined: bool,
) -> None:
    result, observed = endpoint(
        monkeypatch, tmp_path, provider=provider, combined=combined, method=method
    )
    np.testing.assert_array_equal(result.gradient, observed["expected"])
    assert not result.gradient.flags.writeable
    assert result.work["grid_pair_visits"] == 11
    assert result.work["xc_points"] == 5
    assert result.work["additional_device_peak_bound"] == (
        96 if provider == "none" else 128
    )
    assert result.work["native_integral_host_reserve"] == 64
    assert bool(observed["native_calls"]) is (provider != "none")
    assert observed["events"][-3:] == ["sources-close", "grid-close", "validate"]
    if provider == "none":
        assert result.work["primitive_records"] == (49 if method == "PBE0" else 33)
        assert any(
            operator == "four_center_eri" and shape[1] == 4
            for operator, shape in observed["pages"]
        )
    else:
        assert result.work["primitive_records"] == 1
        assert not observed["pages"]
    if combined and provider == "direct":
        assert "two_electron" in result.components
        assert (
            "coulomb" not in result.components
            and "exact_exchange" not in result.components
        )
    else:
        assert (
            "coulomb" in result.components and "two_electron" not in result.components
        )


@pytest.mark.parametrize("delta", [-1, 1])
def test_complete_stationary_endpoint_rejects_missing_or_extra_pair_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, delta: int
) -> None:
    with pytest.raises(RuntimeError, match="source coverage"):
        endpoint(
            monkeypatch, tmp_path, provider="direct", combined=True, pair_delta=delta
        )

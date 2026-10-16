"""Execute geometry-cache and Fast precision composition through mocked CUDA ports.

The real owner, cache, canonical sum and precision receipt remain executable.
Scientific/native authentication and GPU arithmetic are qualified elsewhere.
"""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Self

import numpy as np
import pytest
from generativeqc import _stationary_composite_cuda as r
from generativeqc import _stationary_cuda as shared
from generativeqc._composite_precision import (
    COMPOSITE_GRADIENT_QUALIFICATION,
    composite_gradient_precision,
)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[NS]:
    h = NS(events=[], integral=1.0, geometry=2.0, fail=None)
    monkeypatch.setenv("GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE", "1")
    for name in tuple(r.os.environ):
        if name.startswith("GENERATIVEQC_STATIONARY_"):
            monkeypatch.delenv(name)

    class Primitive:
        spec = "nlc"
        coefficient = 1

    class SourceOwner:
        geometry_external_device = lambda *a, **k: None

        def __init__(self, *a: Any, **k: Any) -> None:
            self.fp32 = k.get("fp32_geometry", False)
            self.points = 0
            self.kinds = {}

        def __enter__(self) -> Self:
            h.events.append("acquire")
            return self

        def __exit__(self, *a: object) -> None:
            h.events.append("release")

        def reset_geometry(self, *a: Any) -> None:
            h.events.append("reset")

        def nuclear_all(self, *a: Any) -> None:
            h.events.append("nuclear")

        def metrics(self) -> dict[str, int]:
            return {"fp32_geometry_points": self.points}

    class Grid(SourceOwner):
        def set_density_device(self, **k: Any) -> None:
            h.events.append("bind-density")

    class Nonlocal:
        @staticmethod
        def required_device_bytes(*a: Any) -> int:
            return 10

        def __init__(self, *a: Any, **k: Any) -> None:
            self.fp32 = k.get("fp32_geometry", False)
            self.points = 0

        def close(self) -> None:
            h.events.append("drain")

    names = (
        "one_electron",
        "overlap_pulay",
        "coulomb",
        "exchange_short_range",
        "exchange_long_range",
        "nuclear",
        "xc",
    )
    plan = NS(
        identity="plan",
        source_names=names,
        spin_blocks=1,
        range_exchange_sources=[NS(name=x) for x in names[3:5]],
        range_exchange_primitives=[NS(operator="short"), NS(operator="long")],
        reduction_program=lambda **k: None,
    )
    prepared = NS(
        identity="prepared",
        graph=NS(identity="graph"),
        lifetimes=NS(identity="lifetime"),
    )
    features = NS(
        identity="features", features=("rho", "gradient"), retained_features=()
    )

    def layout(*a: Any, **k: Any) -> NS:
        if k["max_host_bytes"] < 1000:
            raise ValueError("capacity")
        return NS(
            grid=NS(tile_points=2, peak_bytes=10),
            sources=NS(
                allocation_bytes=10, geometry_lanes=2, becke_threads_per_point=1
            ),
            native_bytes=100,
            device_bound=100,
            host_bound=1000,
        )

    def native(_batch: Any, _handle: Any, out: Any, size: int, *rest: Any) -> int:
        h.events.append("integral")
        if h.fail == "integral":
            raise MemoryError("native OOM")
        np.ctypeslib.as_array(out, shape=(size,))[:] = h.integral
        return 0

    source = NS(
        backend="cuda",
        metadata=[8] + [0] * 12,
        hamiltonian="all-electron",
        nonlocal_density_policy=r.MOLECULAR_VV10_DENSITY_POLICY,
        functional_code=4,
        method_ir=NS(primitives=[Primitive()]),
        grid_spec=NS(partition_iterations=1, coincident_tolerance=1e-12),
        _library=NS(generativeqc_ks_snapshot_cuda_integral_gradient_v1=native),
        _handle=123,
        _batch=NS(
            _batch=1,
            _context=1,
            _calculator=NS(ks_options=NS(nonlocal_memory_budget_bytes=10000)),
        ),
        export_work={},
        atomic_weights=np.ones(2),
        density_fitted=False,
        cuda_integral_gradient_additional_device_bytes=lambda: 0,
    )

    def check() -> None:
        h.events.append("current")
        if h.fail == "stale":
            raise ValueError("stale")

    source.check_current = check
    source.cuda_resident_density = lambda: NS(
        device=0, alpha=1, beta=2, matrix_elements=4, spins=1, source_stream=1
    )
    state = NS(
        _source=source,
        identity=NS(
            basis_identity="basis",
            geometry_identity="geometry",
            to_payload=lambda: {"model": "fixed"},
        ),
        density=np.eye(2),
        grid=NS(points=np.ones((2, 3)), weights=np.ones(2)),
    )
    basis = NS(
        identity="basis",
        geometry="geometry",
        nao=2,
        natom=2,
        nprimitive=2,
        shells=[NS(angular_momentum=0)],
        atoms=[NS(atomic_number=1)] * 2,
    )

    class Contract(r.StationaryDerivativeContract):
        # Inherit real force-policy selection; replace only snapshot validation.
        def __init__(
            self, observed: object, *, allow_budget_stationarity: bool = False
        ) -> None:
            assert observed is state.identity
            assert allow_budget_stationarity is False

        def validate(self, state: Any) -> None:
            h.events.append("validate")
            if h.fail == "publication" and "geometry" in h.events:
                raise ValueError("revoked publication")
            source.check_current()

    def geometry(
        **k: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
        h.events.append("geometry")
        owner = k["sources"]
        if owner.fp32:
            if h.fail == "fp32unavailable":
                raise r._Fp32GeometryUnavailable("injected capability")
            owner.points += 2
            if h.fail == "fp32arithmetic":
                raise ArithmeticError("injected candidate arithmetic")
        value = h.geometry + (0.25 if owner.fp32 else 0.0)
        return (
            {"nuclear": np.full((2, 3), h.geometry), "xc": np.full((2, 3), value)},
            {"geometry": 0.01},
            {
                "ao_collocation_point_visits": 2,
                "geometry_point_visits": 4,
                "nonlocal_pair_sweeps": 1,
                "nonlocal_execution": "resident-full-grid-device-seeds",
            },
        )

    for name, value in {
        "NonlocalCorrelationPrimitive": Primitive,
        "_CudaSources": SourceOwner,
        "CudaGrid": Grid,
        "_ResidentNonlocalForceOwner": Nonlocal,
        "StationaryDerivativeContract": Contract,
        "native_ao_geometry_identity": lambda b: b.geometry,
        "stationary_cuda_requires_native_integrals": lambda **k: None,
        "_plan_for_state": lambda s: plan,
        "stationary_external_provider_sources": lambda p: r._COMPOSITE_EXTERNAL_SOURCES,
        "compile_stationary_prepared_plan": lambda p: prepared,
        "plan_stationary_feature_leases": lambda p: features,
        "_resolve_phased_becke_policy": lambda *a: False,
        "_resolve_becke_primitive_policy": lambda: False,
        "plan_composite_stationary_cuda_resources": layout,
        "emit_first_derivative_cuda": lambda *a: None,
        "compile_stationary_cuda": lambda *a, **k: None,
        "_native_grid_artifact": lambda *a: None,
        "resident_nonlocal_geometry": geometry,
        "_integral_source_resources": lambda *a, **k: {},
        "_metric_delta": lambda a, b: {k: a[k] - b.get(k, 0) for k in a},
    }.items():
        monkeypatch.setattr(r, name, value)
    monkeypatch.setattr(shared, "_prepare_force_memory", lambda *a: {})
    monkeypatch.setattr(r._native, "check", lambda *a, **k: None)
    h.owner = r.PreparedCompositeStationaryCudaGradient()
    h.source = source
    h.state = state
    h.basis = basis
    h.prepared = prepared

    def run(**kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return h.owner.execute(
            h.state,
            h.basis,
            compiler=NS(target=NS(architecture="sm_120")),
            cache=Path("/unused"),
            library=Path("/native"),
            max_host_bytes=kwargs.pop("max_host_bytes", 5000),
            **kwargs,
        )

    h.run = run
    yield h
    h.owner.close()


def test_identical_density_precision_mode_switches_rebuild(harness: NS) -> None:
    h = harness
    strict, _ = h.run()
    assert h.run()[1]["geometry_response_cache_hits"] == 1
    fast, fw = h.run(fp32_semilocal_geometry=True)
    assert (
        fw["geometry_response_cache_hits"] == 0
        and fw["prepared_execution_reused"] is False
    )
    assert not np.array_equal(fast, strict)
    assert h.run(fp32_semilocal_geometry=True)[1]["geometry_response_cache_hits"] == 0
    restored, rw = h.run(fp32_semilocal_geometry=False)
    assert (
        rw["geometry_response_cache_hits"] == 0
        and rw["prepared_execution_reused"] is False
    )
    np.testing.assert_array_equal(restored, strict)


@pytest.mark.parametrize("failure", ["fp32unavailable", "fp32arithmetic"])
def test_precision_retry_discards_candidate_cache(harness: NS, failure: str) -> None:
    h = harness
    strict, _ = h.run()
    h.fail = failure
    result, work = h.run(fp32_semilocal_geometry=True)
    np.testing.assert_array_equal(result, strict)
    assert work["geometry_response_cache_hits"] == 0
    assert work["fp32_semilocal_geometry_requested"] is True
    if failure == "fp32arithmetic":
        assert work["fp64_geometry_retries"] == 1
    receipt = composite_gradient_precision(
        COMPOSITE_GRADIENT_QUALIFICATION, h.state.identity, work
    )
    assert receipt["returned_mode"] == "fp64"
    strict_replay, strict_work = h.run()
    assert strict_work["geometry_response_cache_hits"] == 1
    np.testing.assert_array_equal(strict_replay, strict)
    h.fail = None
    result, work = h.run(fp32_semilocal_geometry=True)
    assert work["geometry_response_cache_hits"] == 0
    assert not np.array_equal(result, strict)


def test_precision_receipt_interaction(harness: NS) -> None:
    h = harness
    _, first = h.run(fp32_semilocal_geometry=True)
    record = composite_gradient_precision(
        COMPOSITE_GRADIENT_QUALIFICATION, h.state.identity, first
    )
    assert record["returned_mode"] == "mixed"
    _, second = h.run(fp32_semilocal_geometry=True)
    report = composite_gradient_precision(
        COMPOSITE_GRADIENT_QUALIFICATION, h.state.identity, second
    )
    assert report["returned_mode"] == "mixed"
    assert report["returned_fp32_geometry_points"] == second["grid_points"]
    assert report["attempted_fp32_geometry_points"] == second["grid_points"]
    assert h.owner._geometry_response.retained_bytes == 0

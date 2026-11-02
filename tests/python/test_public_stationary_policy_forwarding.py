"""Public force routes preserve model admission and the selected AO policy."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import (
    _composite_precision,
    _df_precision,
    _dft_gradient,
    _force_active_ao,
    _stationary_composite_cuda,
    _stationary_cuda,
)
from generativeqc.batch import PreparedBatch
from generativeqc_compiler import dft
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.method import resolve_method

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


@pytest.mark.parametrize("force_tile", [None, 1024])
@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("admitted", [False, True])
def test_public_force_preserves_admission_and_selected_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    composite: bool,
    admitted: bool,
    force_tile: int | None,
) -> None:
    class Basis:
        natom, nao = 2, 2

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    events = []
    source = SimpleNamespace(
        backend="cuda",
        hamiltonian="all-electron",
        method_ir=resolve_method("PBE0"),
        metadata=(0, 0, 0, 0, 0, 0, 1),
        close=lambda: events.append("closed"),
    )
    state = SimpleNamespace(
        _source=source,
        identity=SimpleNamespace(ingredients=("rho", "sigma"), spin="unpolarized"),
        grid=SimpleNamespace(points=np.zeros((5, 3))),
    )
    target = cuda_target_info("sm_120")
    decision = SimpleNamespace(
        cutoff=1e-16,
        cache_bytes=1024,
        producer="pre-ao-envelope-native-csr",
        max_active_fraction=0.375,
    )
    qualification = object()
    gradient = np.arange(6.0).reshape(2, 3)

    def select(workload: Any) -> Any:
        assert workload.tile_points == force_tile
        assert workload.tile_policy == (
            "budget-auto" if force_tile is None else "fixed"
        )
        assert workload.device_name == "test execution device"
        assert workload.architecture == target.architecture
        assert workload.composition == ("composite" if composite else "ordinary")
        events.append("policy")
        return decision

    def admission(calculator: Any, observed: Any, basis: Any) -> Any:
        assert (
            calculator is batch._calculator
            and observed is state
            and isinstance(basis, Basis)
        )
        events.append("admission")
        return qualification if admitted else None

    def precision(observed: Any, *args: Any) -> dict[str, str]:
        assert observed is qualification and admitted
        return {"mode": "admitted"}

    def check_options(kwargs: Any) -> None:
        assert kwargs["tile_points"] == force_tile
        prefix = "active_ao" if composite else "resident_ao"
        assert kwargs[prefix + "_producer"] == decision.producer
        assert kwargs[prefix + "_max_active_fraction"] == decision.max_active_fraction
        assert kwargs[prefix + "_cutoff"] == decision.cutoff
        assert kwargs[prefix + "_cache_bytes"] == decision.cache_bytes
        assert kwargs.get("fp32_semilocal_geometry", False) is admitted
        if not composite:
            assert (
                kwargs["max_grid_points"] is None
                and kwargs["max_grid_pair_visits"] is None
            )
        events.append("execute")

    class Composite:
        def execute(self, observed: Any, basis: Any, **kwargs: Any) -> Any:
            assert observed is state
            check_options(kwargs)
            return -gradient.copy(), {}

    def ordinary(observed: Any, basis: Any, **kwargs: Any) -> Any:
        assert observed is state
        check_options(kwargs)
        return SimpleNamespace(gradient=gradient.copy(), work={})

    def from_native(observed: Any, basis: Any, **kwargs: Any) -> Any:
        assert observed is batch and kwargs["allow_budget_stationarity"] is True
        return state

    monkeypatch.setattr(dft, "NativeAO", lambda *a, **k: Basis())
    monkeypatch.setattr(_dft_gradient.StationaryKsState, "from_native", from_native)
    monkeypatch.setattr(_force_active_ao, "QUALIFIED_FORCE_ACTIVE_AO_PROFILES", ())
    monkeypatch.setattr(_force_active_ao, "resolve_force_active_ao_policy", select)
    monkeypatch.setattr(
        _force_active_ao,
        "force_active_ao_policy_record",
        lambda observed, work: {"producer": observed.producer},
    )
    monkeypatch.setattr(
        _stationary_composite_cuda,
        "requires_composite_stationary_cuda",
        lambda _: composite,
    )
    # This forwarding fixture isolates policy admission from package selection.
    monkeypatch.setattr(
        _stationary_composite_cuda, "composite_stationary_aot_profile", lambda _: None
    )
    monkeypatch.setattr(
        _stationary_composite_cuda, "PreparedCompositeStationaryCudaGradient", Composite
    )
    monkeypatch.setattr(
        _stationary_cuda, "complete_rks_cuda_gradient_diagnostic", ordinary
    )
    monkeypatch.setattr(_df_precision, "fitted_gradient_admission", admission)
    monkeypatch.setattr(_composite_precision, "composite_gradient_admission", admission)
    monkeypatch.setattr(_df_precision, "fitted_gradient_precision", precision)
    monkeypatch.setattr(_composite_precision, "composite_gradient_precision", precision)
    batch = SimpleNamespace(
        _calculator=SimpleNamespace(
            _ks_options=SimpleNamespace(force_tile_points=force_tile),
            _basis=object(),
            _representation_name="cartesian",
            _density_fitting_mode=0,
            _capabilities=SimpleNamespace(supported_properties={"energy", "forces"}),
        ),
        _snapshot_grid_cache=None,
        _stationary_cuda_execution=None,
        _charges=[0],
        _multiplicities=[1],
        _library=SimpleNamespace(_name=str(tmp_path / "inert.so")),
        _stationary_cuda_target=lambda: target,
        _stationary_cuda_compiler=lambda: object(),
        _stationary_cuda_device_name="test execution device",
    )
    force, work = PreparedBatch._public_dft_cuda_force(batch, 0, ())
    np.testing.assert_array_equal(force, -gradient)
    assert ("gradient_precision" in work) is admitted
    assert work["force_active_ao_policy"] == {"producer": decision.producer}
    assert events.index("policy") < events.index("execute")
    assert events.index("admission") < events.index("execute")
    assert events[-1] == "closed"

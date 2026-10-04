"""Structural guards for compiler-driven composite stationary CUDA routing."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from generativeqc._stationary_composite_cuda import (
    _plan_for_state,
    requires_composite_stationary_cuda,
)
from generativeqc_compiler.dft.nonlocal_policy import MOLECULAR_VV10_DENSITY_POLICY
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_cuda import (
    stationary_external_provider_sources,
)
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)

ROOT = Path(__file__).resolve().parents[2]
BATCH = (ROOT / "python/generativeqc/batch.py").read_text()
DRIVER = (ROOT / "python/generativeqc/_stationary_composite_cuda.py").read_text()
COMPILER = (ROOT / "python/generativeqc_compiler/method/stationary_cuda.py").read_text()


def test_public_cuda_force_routing_has_no_method_name_special_case() -> None:
    begin = BATCH.index("    def _public_dft_cuda_force(")
    end = BATCH.index("\n    def ", begin + 10)
    body = BATCH[begin:end]
    assert "requires_composite_stationary_cuda(state)" in body
    assert "PreparedCompositeStationaryCudaGradient" in body
    assert "resolve_force_active_ao_policy(workload)" in body
    assert "active_ao_cutoff=decision.cutoff" in body
    assert '"resident_ao_cutoff": decision.cutoff' in body
    assert "wb97m" not in body.lower()
    assert "pbe0" not in body.lower()
    assert "b3lyp" not in body.lower()


def test_composite_route_is_selected_from_compiler_source_inventory() -> None:
    assert "stationary_external_provider_sources(plan)" in DRIVER
    assert "external-provider source inventory is not qualified" in DRIVER
    assert "_COMPOSITE_EXTERNAL_SOURCES" in DRIVER
    assert "source.method_ir" in DRIVER
    assert "_method_name" not in DRIVER
    assert "resolve_method(" not in DRIVER


@pytest.mark.parametrize("policy", [None, "unqualified-policy", "absent"])
def test_composite_route_keeps_shared_point_model_for_ordinary_dft(
    policy: str | None,
) -> None:
    state = _fake_state(
        "PBE",
        nonlocal_policy=policy,
        point_model="libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16",
    )
    if policy == "absent":
        del state._source.nonlocal_density_policy
    assert _plan_for_state(state).mean_field.point_model == SCF_POINT_MODEL


def test_composite_route_preserves_qualified_nonlocal_point_model() -> None:
    point_model = "libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16"
    state = _fake_state(
        "WB97M-V",
        nonlocal_policy=MOLECULAR_VV10_DENSITY_POLICY,
        point_model=point_model,
    )
    assert _plan_for_state(state).mean_field.point_model == point_model


def test_composite_driver_inherits_live_functional_code() -> None:
    assert "functional = int(source.functional_code)" in DRIVER
    assert "functional=functional" in DRIVER
    assert "metadata[6] != 4" not in DRIVER
    assert "functional=4" not in DRIVER


def test_compiler_owns_external_provider_inventory() -> None:
    assert "def stationary_external_provider_sources(" in COMPILER
    assert "stationary_runtime_sources(plan)" in COMPILER
    assert '"ecp_local", "ecp_nonlocal"' in COMPILER


def test_external_provider_inventory_follows_method_ir_primitives() -> None:
    pbe = StationaryGradientPlan(
        resolve_method("PBE", spin="unpolarized"),
        StationaryMeanField(SCF_POINT_MODEL),
    )
    rsh = StationaryGradientPlan(
        resolve_method("CAM-B3LYP", spin="unpolarized"),
        StationaryMeanField(SCF_POINT_MODEL),
    )
    nonlocal_rsh = StationaryGradientPlan(
        resolve_method("WB97M-V", spin="unpolarized"),
        StationaryMeanField("libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16"),
    )

    assert stationary_external_provider_sources(pbe) == ()
    assert stationary_external_provider_sources(rsh) == (
        "exchange_short_range",
        "exchange_long_range",
    )
    assert stationary_external_provider_sources(nonlocal_rsh) == (
        "exchange_short_range",
        "exchange_long_range",
        "nonlocal_ao",
        "nonlocal_grid",
        "nonlocal_weight",
    )


def _fake_state(
    identifier: str,
    *,
    nonlocal_policy: str | None = None,
    point_model: str = SCF_POINT_MODEL,
) -> SimpleNamespace:
    source = SimpleNamespace(
        method_ir=resolve_method(identifier, spin="unpolarized"),
        nonlocal_density_policy=nonlocal_policy,
        hamiltonian="all-electron",
        _batch=SimpleNamespace(
            _calculator=SimpleNamespace(
                _ks_options=SimpleNamespace(scf_domain=point_model)
            )
        ),
    )
    return SimpleNamespace(_source=source)


def test_runtime_route_fails_closed_on_unqualified_external_sources() -> None:
    assert requires_composite_stationary_cuda(_fake_state("PBE")) is False
    with pytest.raises(NotImplementedError, match="exchange_short_range"):
        requires_composite_stationary_cuda(_fake_state("CAM-B3LYP"))
    assert (
        requires_composite_stationary_cuda(
            _fake_state(
                "WB97M-V",
                nonlocal_policy=MOLECULAR_VV10_DENSITY_POLICY,
                point_model="libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16",
            )
        )
        is True
    )

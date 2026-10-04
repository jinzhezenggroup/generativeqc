"""Fail before GPU replay when qualification policies or observers disconnect."""

import os
import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks import dft_force_components
from benchmarks import readme_pbe0_integrated as runner


def force_work(
    policy: runner.QualificationPolicy, *, atoms: int = 96
) -> dict[str, Any]:
    """Use the production work schema, not an already normalized receipt."""
    selected = None
    if policy.resident_ao_cutoff is not None:
        selected = {
            "tile_count": 2,
            "dense_budget_tiles": 2 if policy.resident_ao_cache_bytes == 0 else 0,
            "discoveries": 0,
            "point_ao_square_sum": 1024 if policy.resident_ao_cache_bytes == 0 else 64,
            "dense_point_ao_square_sum": 1024,
        }
    return {
        "native_integrals_required": True,
        "stationary_integral_derivative_route": "prepared-native-complete",
        "timeline": {
            "exclusive_wall_seconds": {"prepared_stationary_integral_derivatives": 0.5}
        },
        "grid_work_plan": {"grid_points": 8, "tile_points": 4},
        "geometry_batches": 2,
        "phased_becke_batches": (
            2 if policy.phased_becke_requested and atoms > 32 else 0
        ),
        "resident_ao_selection": {
            "mode": "disabled" if selected is None else "explicit-sampled-jet-cutoff",
            "cutoff": policy.resident_ao_cutoff,
            "work": selected,
        },
    }


@pytest.mark.parametrize("name", runner.POLICIES)
@pytest.mark.parametrize("atoms", [24, 96])
def test_all_policies_check_real_normalizer_and_small_phase_fallback(
    name: str, atoms: int
) -> None:
    policy = runner.POLICIES[name]
    runner.verify_force_work(force_work(policy, atoms=atoms), policy, atoms=atoms)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("timeline", {}, "derivative wall observer"),
        ("stationary_integral_derivative_route", "bounded-ao-task", "native integral"),
        ("geometry_batches", 1, "complete grid"),
        ("phased_becke_batches", None, "Becke policy"),
        ("phased_becke_batches", 0, "Becke policy"),
        ("resident_ao_selection", None, "AO policy observer"),
    ],
)
def test_missing_observers_and_wrong_routes_fail_closed(
    key: str, value: Any, message: str
) -> None:
    policy = runner.POLICIES["local-indexed-phased"]
    work = force_work(policy)
    work[key] = value
    with pytest.raises(RuntimeError, match=message):
        runner.verify_force_work(work, policy, atoms=96)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("mode", "dense-no-resident-grid", "lost its resident grid"),
        ("work", None, "complete grid"),
        ("cutoff", 1e-8, "AO policy observer"),
    ],
)
def test_disconnected_local_ao_observer_is_not_a_work_reduction(
    key: str, value: Any, message: str
) -> None:
    policy = runner.POLICIES["local-indexed"]
    work = force_work(policy)
    work["resident_ao_selection"][key] = value
    with pytest.raises(RuntimeError, match=message):
        runner.verify_force_work(work, policy, atoms=96)


def test_zero_budget_requires_dense_fallback_and_normal_policy_requires_local_work() -> (
    None
):
    zero = runner.POLICIES["local-indexed-zero-cache"]
    work = force_work(zero)
    runner.verify_force_work(work, zero, atoms=96)
    with pytest.raises(RuntimeError, match="did not reduce"):
        runner.verify_force_work(work, runner.POLICIES["local-indexed"], atoms=96)
    work["resident_ao_selection"]["work"]["discoveries"] = 1
    with pytest.raises(RuntimeError, match="dense fallback"):
        runner.verify_force_work(work, zero, atoms=96)


def test_canonical_normalizer_is_used_not_a_shadow_module_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = dft_force_components.normalize_force_work

    def broken(*args: Any, **kwargs: Any) -> dict[str, Any]:
        record = deepcopy(original(*args, **kwargs))
        record["wall_seconds"]["stationary_integral_derivatives"] = None
        return record

    monkeypatch.setattr(dft_force_components, "normalize_force_work", broken)
    policy = runner.POLICIES["default"]
    with pytest.raises(RuntimeError, match="derivative wall observer"):
        runner.verify_force_work(force_work(policy), policy, atoms=96)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("scf_active_ao", [False, True])
def test_main_applies_policy_to_shared_caller_and_restores_all_bindings(
    monkeypatch: pytest.MonkeyPatch, fail: bool, scf_active_ao: bool
) -> None:
    policy = runner.POLICIES["local-indexed-phased"]
    calls = []

    def original_force(state: Any, basis: Any, **options: Any) -> Any:
        calls.append(options)
        return SimpleNamespace(work=force_work(policy))

    monkeypatch.setattr(
        runner._stationary_cuda,
        "_complete_rks_cuda_gradient_diagnostic",
        original_force,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "slurm-owned")
    monkeypatch.setenv("GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE", "before")
    monkeypatch.setenv("GENERATIVEQC_FORCE_DENSITY_PRODUCT_SCREENING", "1")
    inherited_ao = "0" if scf_active_ao else "1"
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", inherited_ao)
    original_resources = runner._stationary_cuda.plan_stationary_cuda_resources
    monkeypatch.setattr(runner, "plan_stationary_cuda_resources", lambda **kw: kw)

    def endpoint(spec: Any, *, qualification_policy: dict[str, Any]) -> None:
        assert spec is runner.PBE0
        assert qualification_policy["name"] == policy.name
        assert qualification_policy["scf_active_ao_requested"] is scf_active_ao
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "slurm-owned"
        assert os.environ["GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE"] == "indexed"
        assert os.environ["GENERATIVEQC_FORCE_DENSITY_PRODUCT_SCREENING"] == "0"
        assert os.environ["GENERATIVEQC_CUDA_KS_ACTIVE_AO"] == str(int(scf_active_ao))
        assert sys.argv == ["runner", "native", "--atoms", "96"]
        assert runner._stationary_cuda.plan_stationary_cuda_resources()["phased_becke"]
        runner._stationary_cuda.complete_rks_cuda_gradient_diagnostic(
            None, SimpleNamespace(natom=96), compiler=None, cache=None
        )
        if fail:
            raise RuntimeError("endpoint failure")

    monkeypatch.setattr(runner.readme_omol25, "main", endpoint)
    argv = ["runner", "--policy", policy.name, "native", "--atoms", "96"]
    if scf_active_ao:
        argv.append("--scf-active-ao")
    monkeypatch.setattr(sys, "argv", argv)
    if fail:
        with pytest.raises(RuntimeError, match="endpoint failure"):
            runner.main()
    else:
        runner.main()
    assert sys.argv is argv
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "slurm-owned"
    assert os.environ["GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE"] == "before"
    assert os.environ["GENERATIVEQC_FORCE_DENSITY_PRODUCT_SCREENING"] == "1"
    assert os.environ["GENERATIVEQC_CUDA_KS_ACTIVE_AO"] == inherited_ao
    assert (
        runner._stationary_cuda._complete_rks_cuda_gradient_diagnostic is original_force
    )
    assert runner._stationary_cuda.plan_stationary_cuda_resources is original_resources
    assert calls[0]["resident_ao_cutoff"] == 1e-16
    assert calls[0]["resident_ao_cache_bytes"] == 16 << 20


def test_reference_rejects_native_scf_selection_before_endpoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected_endpoint(*args: Any, **options: Any) -> None:
        pytest.fail("native SCF selection reached the reference endpoint")

    monkeypatch.setattr(runner.readme_omol25, "main", unexpected_endpoint)
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", "before")
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "reference", "--policy", "default", "--scf-active-ao"],
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "requires the native engine" in capsys.readouterr().err
    assert os.environ["GENERATIVEQC_CUDA_KS_ACTIVE_AO"] == "before"


def test_reference_default_does_not_inherit_native_scf_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts = []

    def endpoint(spec: Any, *, qualification_policy: dict[str, Any]) -> None:
        assert spec is runner.PBE0
        assert sys.argv == ["runner", "reference", "--atoms", "24"]
        assert os.environ["GENERATIVEQC_CUDA_KS_ACTIVE_AO"] == "0"
        receipts.append(qualification_policy)

    monkeypatch.setattr(runner.readme_omol25, "main", endpoint)
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", "1")
    monkeypatch.setattr(
        sys, "argv", ["runner", "reference", "--policy", "default", "--atoms", "24"]
    )
    runner.main()
    assert receipts[0]["scf_active_ao_requested"] is False
    assert os.environ["GENERATIVEQC_CUDA_KS_ACTIVE_AO"] == "1"


@pytest.mark.parametrize("scf_active_ao", [False, True])
def test_scf_selection_restores_an_absent_environment_key(
    monkeypatch: pytest.MonkeyPatch, scf_active_ao: bool
) -> None:
    monkeypatch.delenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", raising=False)
    with runner.qualification_policy(
        runner.POLICIES["default"], scf_active_ao=scf_active_ao
    ):
        assert os.environ["GENERATIVEQC_CUDA_KS_ACTIVE_AO"] == str(int(scf_active_ao))
    assert "GENERATIVEQC_CUDA_KS_ACTIVE_AO" not in os.environ

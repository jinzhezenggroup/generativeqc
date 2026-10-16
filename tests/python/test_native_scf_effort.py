"""Native empirical SCF policy: independent decisions, ABI and explicit compatibility."""

from __future__ import annotations

import ctypes
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pytest
from generativeqc import _scf_effort_native as native
from generativeqc.force_aware_scf import (
    ForceAwareScfPolicy,
    ScfEffortLevel,
    ScfEffortState,
    ScfForceErrorEstimator,
)


def controller(**changes: Any) -> ForceAwareScfPolicy:
    calibration = ScfForceErrorEstimator(
        methods=("rhf",),
        basis_ids=("independent-basis",),
        diagnostic_kind="density_rms",
        energy_scale=0.03,
        force_max_scale=3.0,
        force_rms_scale=2.0,
        training_families=("a", "b"),
        training_data_hash="independent-envelope-v1",
        method_basis_domains=(("rhf", "independent-basis"),),
    )
    result = ForceAwareScfPolicy(
        (
            ScfEffortLevel("loose", 0, 1e-6, 1e-4, 50),
            ScfEffortLevel("medium", 1, 1e-8, 1e-6, 80),
            ScfEffortLevel("strict", 2, 1e-10, 1e-8, 120, strict=True),
        ),
        calibration,
        optimizer_force_tolerance=1e-4,
        intermediate_energy_error=1e-5,
        intermediate_force_fraction=0.2,
        max_intermediate_force_error=5e-3,
    )
    return replace(result, **changes)


def propose(
    policy: ForceAwareScfPolicy, state: ScfEffortState, force: Any, **changes: Any
) -> Any:
    arguments = {
        "method": "rhf",
        "basis_id": "independent-basis",
        "geometry_id": "moved-v2",
        "step_index": 3,
        "current_force_max": force,
    }
    return policy.propose_next(state, **(arguments | changes))


@pytest.fixture
def library() -> Any:
    result = native.optional_library()
    assert result is not None, (
        "build the current native SCF-effort protocol for this gate"
    )
    return result


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("streak", [0, 1, 7, 2**64 - 1])
@pytest.mark.parametrize(
    "force,target", [(None, 2), (0.0, 2), (5e-4, 2), (0.001, 1), (0.1, 0)]
)
def test_native_selection_matches_independent_threshold_table(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
    level: int,
    streak: int,
    force: float | None,
    target: int,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("native policy retried Python selection/prediction")

    monkeypatch.setattr(ForceAwareScfPolicy, "_propose_next_python", forbidden)
    monkeypatch.setattr(ScfForceErrorEstimator, "predict", forbidden)
    policy = controller()
    state = ScfEffortState(level, streak)
    result = propose(policy, state, force)
    strict = force is None or force <= 5e-4
    waiting = not strict and target < level and streak == 0
    expected_level = level if waiting else target
    expected_action = (
        "strict"
        if strict
        else "hold"
        if waiting or target == level
        else "relax"
        if target < level
        else "tighten"
    )
    assert result.action == expected_action
    assert result.level is policy.levels[expected_level]
    assert result.state.level_index == expected_level
    assert result.state.safe_streak == int(waiting)
    assert result.allowed_energy_error == 1e-5
    assert result.allowed_force_error == (
        0.0 if force is None else min(5e-3, 0.2 * max(force, 1e-4))
    )
    assert len(result.state.transitions) == int(expected_level != level)
    assert state == ScfEffortState(level, streak)
    if strict:
        assert result.estimate is None
    else:
        diagnostic = (1e-4, 1e-6, 1e-8)[expected_level]
        np.testing.assert_allclose(
            (
                result.estimate.energy_abs,
                result.estimate.force_max_abs,
                result.estimate.force_rms,
            ),
            np.asarray((3e-6, 3e-4, 2e-4)) * (diagnostic / 1e-4),
            rtol=3e-16,
            atol=0,
        )
        assert result.estimate.diagnostic_value == diagnostic
        assert result.estimate.calibration_id == policy.estimator.identity
        if result.state.transitions:
            assert result.state.transitions[-1].geometry_id == "moved-v2"


def test_stationarity_and_force_allowances_are_inclusive(library: Any) -> None:
    policy = controller(relax_after=1)
    assert propose(policy, policy.initial_state(), 5e-4).action == "strict"
    assert (
        propose(policy, policy.initial_state(), np.nextafter(5e-4, np.inf)).level.name
        == "medium"
    )
    force_limit = float(np.multiply(3.0, 1e-4))
    exact = replace(
        policy,
        intermediate_energy_error=float(np.multiply(0.03, 1e-4)),
        max_intermediate_force_error=force_limit,
    )
    assert propose(exact, exact.initial_state(), 1.0).level.name == "loose"
    below = replace(exact, max_intermediate_force_error=np.nextafter(force_limit, 0))
    assert propose(below, below.initial_state(), 1.0).level.name == "medium"


@pytest.mark.parametrize(
    "method,basis", [("uhf", "independent-basis"), ("rhf", "unseen")]
)
def test_joint_domain_gate_preserves_strict_precedence(
    library: Any, method: str, basis: str
) -> None:
    policy = controller()
    fallback = propose(policy, ScfEffortState(0, 1), 0.1, method=method, basis_id=basis)
    assert fallback.action == "strict_fallback" and fallback.level.strict
    assert fallback.state.transitions[-1].reason == "calibration-domain fallback"
    assert "calibration domain" in fallback.reason
    assert (
        propose(policy, ScfEffortState(0), None, method=method, basis_id=basis).action
        == "strict"
    )
    reproducible = replace(policy, strict_reproducible=True)
    result = propose(
        reproducible, ScfEffortState(0, 7), object(), method=method, basis_id=basis
    )
    assert result.action == "strict" and result.reason == "strict reproducible mode"
    assert result.state.safe_streak == 0 and result.allowed_force_error == 0


def test_forecast_overflow_is_explicit_strict_fallback(library: Any) -> None:
    policy = controller(relax_after=1)
    policy = replace(
        policy,
        levels=(ScfEffortLevel("strict", 0, 1e-6, 2.0, 50, strict=True),),
        estimator=replace(policy.estimator, force_rms_scale=np.finfo(float).max),
    )
    result = propose(policy, policy.initial_state(), 0.1)
    assert result.action == "strict_fallback"
    assert result.reason == "force_rms must be finite and nonnegative"
    assert result.estimate is None


def test_empirical_floor_changes_forecast_not_published_diagnostic(
    library: Any,
) -> None:
    policy = controller(relax_after=1)
    policy = replace(
        policy,
        levels=tuple(
            replace(level, density_tolerance=value)
            for level, value in zip(policy.levels, (1e-16, 1e-17, 1e-18), strict=True)
        ),
    )
    result = propose(policy, policy.initial_state(), 0.1)
    assert result.action == "relax" and result.level.name == "loose"
    assert result.estimate.diagnostic_value == 1e-16
    np.testing.assert_allclose(
        (
            result.estimate.energy_abs,
            result.estimate.force_max_abs,
            result.estimate.force_rms,
        ),
        (3e-16, 3e-14, 2e-14),
        rtol=3e-16,
        atol=0,
    )


def test_largest_native_hysteresis_limit_keeps_mathematical_comparison(
    library: Any,
) -> None:
    policy = controller(relax_after=2**64 - 1)
    waiting = propose(policy, ScfEffortState(2, 2**64 - 3), 0.1)
    assert waiting.action == "hold" and waiting.state.safe_streak == 2**64 - 2
    relaxed = propose(policy, waiting.state, 0.1)
    assert relaxed.action == "relax" and relaxed.state.safe_streak == 0


def test_native_failure_never_retries_and_next_valid_call_recovers(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Failure:
        generativeqc_status_message = library.generativeqc_status_message

        def generativeqc_scf_effort_decide_v1(self, *args: Any) -> int:
            return 5

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("native failure retried compatibility")

    policy = controller(relax_after=1)
    state = policy.initial_state()
    with monkeypatch.context() as patch:
        patch.setattr(native, "optional_library", lambda: Failure())
        patch.setattr(ForceAwareScfPolicy, "_propose_next_python", forbidden)
        with pytest.raises(RuntimeError, match="GENERATIVEQC error 5"):
            propose(policy, state, 0.1)
    assert state == policy.initial_state()
    assert propose(policy, state, 0.1).action == "relax"


def test_missing_protocol_custom_predictor_and_unbounded_integers_stay_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = controller(relax_after=1)
    monkeypatch.setattr(native, "optional_library", lambda: None)
    assert propose(policy, policy.initial_state(), 0.1).action == "relax"

    def forbidden() -> Any:
        raise AssertionError("unsupported metadata queried native policy")

    monkeypatch.setattr(native, "optional_library", forbidden)

    class Conservative(ScfForceErrorEstimator):
        def predict(self, *args: Any, **kwargs: Any) -> Any:
            return replace(super().predict(*args, **kwargs), energy_abs=1.0)

    custom = Conservative(
        **{
            field.name: getattr(policy.estimator, field.name)
            for field in fields(policy.estimator)
        }
    )
    assert (
        propose(replace(policy, estimator=custom), policy.initial_state(), 0.1).action
        == "strict"
    )
    huge = replace(policy, relax_after=2**80)
    result = propose(huge, ScfEffortState(2, 2**80 - 2), 0.1)
    assert result.action == "hold" and result.state.safe_streak == 2**80 - 1


def test_selected_library_load_error_is_not_a_missing_protocol_retry(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "broken.so"
    selected.touch()
    error = OSError("selected policy library cannot load")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(native._native, "_candidate_paths", lambda: iter((selected,)))
    monkeypatch.setattr(native.ctypes, "CDLL", fail)
    with pytest.raises(OSError) as caught:
        propose(controller(), ScfEffortState(2), 0.1)
    assert caught.value is error


def test_public_abi_invalid_metadata_preserves_output(library: Any) -> None:
    descriptor = native.Descriptor(
        struct_size=ctypes.sizeof(native.Descriptor), abi_version=0
    )
    output = native.Decision(
        struct_size=ctypes.sizeof(native.Decision),
        abi_version=0,
        level_index=919,
        energy_abs=-77.0,
    )
    before = bytes(output)
    assert (
        library.generativeqc_scf_effort_decide_v1(
            ctypes.byref(descriptor), ctypes.byref(output)
        )
        == 1
    )
    assert bytes(output) == before


@pytest.mark.parametrize("mask", range(64))
def test_final_verification_preserves_all_inclusive_gates_and_status_precedence(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
    mask: int,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("native final verification retried Python gates")

    monkeypatch.setattr(ForceAwareScfPolicy, "_verify_final_python", forbidden)
    policy = controller()
    energy = policy.strict_level.energy_tolerance
    density = policy.strict_level.density_tolerance
    force = policy.optimizer_force_tolerance
    result = policy.verify_final(
        target_model_id="target",
        observed_model_id="other" if mask & 1 else "target",
        level=policy.levels[0] if mask & 2 else policy.strict_level,
        converged=not bool(mask & 4),
        energy=-1.0,
        energy_change=np.nextafter(energy, np.inf) if mask & 8 else energy,
        density_rms=np.nextafter(density, np.inf) if mask & 16 else density,
        force_max_abs=np.nextafter(force, np.inf) if mask & 32 else force,
    )
    expected = tuple(
        message
        for index, message in enumerate(
            (
                "final model identity does not match target",
                "final evaluation did not use strict SCF effort",
                "final SCF did not converge",
                "final energy-change gate is unmet",
                "final density gate is unmet",
                "optimizer force termination gate is unmet",
            )
        )
        if mask & (1 << index)
    )
    assert result.reasons == expected
    assert result.status == (
        "unverified" if mask & 1 else "observed_unmet" if mask else "observed_met"
    )


def test_native_final_failure_never_retries_python_gates(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Failure:
        generativeqc_status_message = library.generativeqc_status_message

        def generativeqc_scf_effort_verify_v1(self, *args: Any) -> int:
            return 5

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("native final failure retried Python gates")

    monkeypatch.setattr(native, "optional_library", lambda: Failure())
    monkeypatch.setattr(ForceAwareScfPolicy, "_verify_final_python", forbidden)
    policy = controller()
    with pytest.raises(RuntimeError, match="GENERATIVEQC error 5"):
        policy.verify_final(
            target_model_id="target",
            observed_model_id="target",
            level=policy.strict_level,
            converged=True,
            energy=-1.0,
            energy_change=0.0,
            density_rms=0.0,
            force_max_abs=0.0,
        )


def test_missing_protocol_preserves_final_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native, "optional_library", lambda: None)
    policy = controller()
    result = policy.verify_final(
        target_model_id="target",
        observed_model_id="target",
        level=policy.strict_level,
        converged=True,
        energy=-1.0,
        energy_change=0.0,
        density_rms=0.0,
        force_max_abs=0.0,
    )
    assert result.status == "observed_met" and result.reasons == ()


def test_custom_identity_comparison_keeps_declared_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DifferentIdentity(str):
        def __ne__(self, other: object) -> bool:
            return True

    def forbidden() -> Any:
        raise AssertionError("custom identity comparison was silently replaced")

    monkeypatch.setattr(native, "optional_library", forbidden)
    policy = controller()
    result = policy.verify_final(
        target_model_id="target",
        observed_model_id=DifferentIdentity("target"),
        level=policy.strict_level,
        converged=True,
        energy=-1.0,
        energy_change=0.0,
        density_rms=0.0,
        force_max_abs=0.0,
    )
    assert result.status == "unverified"
    assert result.reasons == ("final model identity does not match target",)

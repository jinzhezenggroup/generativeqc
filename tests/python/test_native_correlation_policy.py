"""Shared correlated policy transport, compatibility and production consumers."""

from __future__ import annotations

import ctypes
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import CBSBasis, CCSDTCBSCalculator, TargetErrorBudget, _native
from generativeqc import _correlation_convergence as policy
from generativeqc._model_resolution import snapshot_basis

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def library(required_native_cxx: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Compile the actual adapter and shared policy, with no test arithmetic."""
    directory = tmp_path_factory.mktemp("correlation-policy")
    path = required_native_cxx.build_shared(
        [ROOT / "src/api/c_api_correlation_convergence.cpp"],
        directory / "policy.so",
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    result = ctypes.CDLL(str(path))
    result.generativeqc_status_message = lambda status: b"policy rejected"
    return result


def forbid_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admitted production consumers must not recalculate policy in Python."""

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native policy retried Python arithmetic")

    monkeypatch.setattr(policy, "_compatibility_scale", forbidden)
    monkeypatch.setattr(policy, "_compatibility_limits", forbidden)


def test_public_sdk_contract(required_native_cxx: Any, tmp_path: Path) -> None:
    """Exercise public layouts, atomic failures, overlap and stopping boundaries."""
    executable = required_native_cxx.build_executable(
        [
            ROOT / "tests/native/test_correlation_policy.cpp",
            ROOT / "src/api/c_api_correlation_convergence.cpp",
        ],
        tmp_path / "policy",
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def test_policy_preserves_legacy_limits(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare randomized budgets and exact bracket neighbors with the old policy."""
    rng = random.Random(2227)
    budgets = [None, TargetErrorBudget(1e-5, 1e-4, None)]
    for value in (math.nextafter(1e-8, 0), 1e-8, math.nextafter(1e-8, math.inf)):
        budgets.append(TargetErrorBudget(value, 1e-4, None))
    budgets.extend(
        TargetErrorBudget(
            10 ** rng.uniform(-14, 2),
            10 ** rng.uniform(-14, 2),
            None if index % 3 == 0 else 10 ** rng.uniform(-14, 2),
        )
        for index in range(600)
    )
    expected = [
        policy.convergence_policy(b, library=SimpleNamespace()) for b in budgets
    ]
    forbid_compatibility(monkeypatch)
    with ThreadPoolExecutor(max_workers=8) as workers:
        actual = list(
            workers.map(
                lambda b: policy.convergence_policy(b, library=library), budgets
            )
        )
    assert actual == expected


def test_cbs_production_planning_uses_native(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual facade planning retains explicit tighter settings and CBS ceilings."""
    basis = snapshot_basis("sto-3g")
    arguments = ((CBSBasis(basis, 2), CBSBasis(snapshot_basis("def2-svp"), 3)),)
    options = {"energy_tolerance": 1e-13, "ccsd_residual_tolerance": 1e-10}
    monkeypatch.setattr(policy, "_optional_library", lambda: SimpleNamespace())
    expected = CCSDTCBSCalculator(*arguments, calculator_options=options)._convergence
    forbid_compatibility(monkeypatch)
    monkeypatch.setattr(policy, "_optional_library", lambda: library)
    actual = CCSDTCBSCalculator(*arguments, calculator_options=options)._convergence
    assert actual == expected
    assert actual["limits"]["energy_tolerance"] == 1e-13
    assert actual["limits"]["ccsd_residual_tolerance"] == 1e-10
    assert actual["limits"]["density_tolerance"] <= 1e-11
    assert CCSDTCBSCalculator(*arguments, policy="tight")._convergence["scale"] == 0


def test_selected_library_never_discovers_another(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calculator-owned transport must use its own library even if discovery differs."""

    def forbidden() -> None:
        raise AssertionError("selected library escaped to default discovery")

    monkeypatch.setattr(policy, "_optional_library", forbidden)
    budget = TargetErrorBudget(1e-5, 1e-4, 3e-5)
    expected = policy.convergence_policy(budget, library=SimpleNamespace())
    forbid_compatibility(monkeypatch)
    assert policy.convergence_policy(budget, library=library) == expected


@pytest.mark.parametrize(
    "mode", ("rejection", "bad_header", "nan_scale", "nan_limit", "zero_limit")
)
def test_admitted_errors_never_retry_python(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native refusal and invalid publication remain visible at the call boundary."""
    forbid_compatibility(monkeypatch)

    def operation(descriptor: Any, output: Any) -> int:
        if mode == "rejection":
            return _native.STATUS_INVALID_ARGUMENT
        result = ctypes.cast(output, ctypes.POINTER(policy._Limits)).contents
        result.scale = 0
        for name in policy._LIMIT_NAMES:
            setattr(result, name, 1e-9)
        if mode == "bad_header":
            result.abi_version += 1
        if mode == "nan_scale":
            result.scale = math.nan
        if mode == "nan_limit":
            result.energy_tolerance = math.nan
        if mode == "zero_limit":
            result.energy_tolerance = 0
        return _native.STATUS_SUCCESS

    selected = SimpleNamespace(
        generativeqc_correlation_convergence_query_v1=operation,
        generativeqc_status_message=lambda status: b"policy rejected",
    )
    with pytest.raises(RuntimeError, match="rejected|invalid receipt"):
        policy.convergence_policy(None, library=selected)


def test_discovery_stops_at_first_existing_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An old configured library cannot borrow a newer library's metadata policy."""
    old, newer = tmp_path / "old.so", tmp_path / "new.so"
    old.touch()
    newer.touch()
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: (tmp_path / "missing", old, newer)
    )

    def load(path: str) -> None:
        assert Path(path) == old

    monkeypatch.setattr(policy, "_load", load)
    assert policy.convergence_policy(None)[0] == 0


def test_loading_errors_are_not_missing_protocol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A broken configured library must not silently select compatibility."""
    path = tmp_path / "broken.so"
    path.write_text("not a shared library")
    monkeypatch.setattr(_native, "_candidate_paths", lambda: (path,))
    with pytest.raises(OSError):
        policy.convergence_policy(None)

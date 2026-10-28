"""Public budget-control policy boundaries and authoritative native selection."""

from __future__ import annotations

import ctypes as ct
import gc
import math
import subprocess
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import TargetErrorBudget, _native
from generativeqc import _precision_native as transport
from generativeqc import precision_policy as policy

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("energy_tolerance", "density_tolerance", "screening_tolerance")


@pytest.fixture(scope="module")
def library(required_native_cxx: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Compile the real public adapter and runtime policy with the verified cache."""
    directory = tmp_path_factory.mktemp("precision-controls")
    path = required_native_cxx.build_shared(
        [ROOT / "src/api/c_api_precision_admission.cpp"],
        directory / "controls.so",
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    value = ct.CDLL(str(path))
    value.generativeqc_status_message = lambda status: b"precision controls refused"
    value.generativeqc_application_fock_precision_version_v2 = lambda: 2
    return value


def model(library: Any) -> Any:
    """The model owner supplies facts; no equation or numerical oracle is copied."""
    return SimpleNamespace(
        _library=library,
        _density_fitting_mode=None,
        _ks_options=SimpleNamespace(
            _method_ir=SimpleNamespace(spin="unpolarized"),
            execution_plan=SimpleNamespace(self_consistent=True),
            xc_schedule="device_fused",
        ),
    )


def test_release_public_sdk(required_native_cxx: Any, tmp_path: Path) -> None:
    path = required_native_cxx.build_executable(
        [
            ROOT / "tests/native/test_precision_controls.cpp",
            ROOT / "src/api/c_api_precision_admission.cpp",
        ],
        tmp_path / "controls",
        compile_args=(
            "-std=c++20",
            "-O3",
            "-DNDEBUG",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    subprocess.run([str(path)], check=True, timeout=30, env={"PATH": "/nonexistent"})


@pytest.mark.parametrize(
    "field,limit", (("energy_abs", 1e-5), ("force_max_abs", 1e-4), ("force_rms", 3e-5))
)
def test_exact_budget_and_one_ulp_tighter(
    library: Any, field: str, limit: float
) -> None:
    values = {"energy_abs": 1e-5, "force_max_abs": 1e-4, "force_rms": 3e-5}
    calculator = model(library)
    expected = dict(zip(NAMES, (1e-6, 1e-4, 1e-10), strict=True))
    assert (
        policy._budget_fock_controls(calculator, TargetErrorBudget(**values), {})
        == expected
    )
    values[field] = math.nextafter(limit, 0.0)
    assert (
        policy._budget_fock_controls(calculator, TargetErrorBudget(**values), {}) == {}
    )


@pytest.mark.parametrize(
    "requested",
    (
        {},
        {"energy_tolerance": 1e-12},
        {"density_tolerance": 1e-2},
        dict(zip(NAMES, (1e-11, 2e-5, 1e-14), strict=True)),
    ),
)
def test_tighter_caller_ceilings_and_optional_rms(
    library: Any, requested: dict
) -> None:
    calculator = model(library)
    budget = TargetErrorBudget(1e-5, 1e-4)
    expected = {
        name: min(ceiling, requested.get(name, ceiling))
        for name, ceiling in zip(NAMES, (1e-6, 1e-4, 1e-10), strict=True)
    }
    assert policy._budget_fock_controls(calculator, budget, requested) == expected


@pytest.mark.parametrize(
    "trait", ("model", "spin", "self_consistent", "schedule", "capability")
)
def test_unsupported_traits_preserve_strict_selection(library: Any, trait: str) -> None:
    calculator = model(library)
    options = calculator._ks_options
    if trait == "model":
        options._method_ir = None
    elif trait == "spin":
        options._method_ir.spin = "polarized"
    elif trait == "self_consistent":
        options.execution_plan.self_consistent = False
    elif trait == "schedule":
        options.xc_schedule = "host_unfused"
    else:
        calculator._library = SimpleNamespace(
            generativeqc_precision_controls_query_v1=library.generativeqc_precision_controls_query_v1,
            generativeqc_status_message=library.generativeqc_status_message,
        )
    assert (
        policy._budget_fock_controls(calculator, TargetErrorBudget(1e-5, 1e-4), {})
        == {}
    )


def test_selected_policy_never_calls_compatibility(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("selected native policy retried Python")

    monkeypatch.setattr(policy, "_compatibility_budget_fock_controls", forbidden)
    calculator = model(library)
    assert policy._budget_fock_controls(calculator, TargetErrorBudget(1e-5, 1e-4), {})
    assert (
        policy._budget_fock_controls(calculator, TargetErrorBudget(1e-8, 1e-7), {})
        == {}
    )
    calculator._library = SimpleNamespace(
        generativeqc_precision_controls_query_v1=lambda *args: (
            _native.STATUS_NOT_IMPLEMENTED
        ),
        generativeqc_status_message=lambda status: b"unsupported controls",
    )
    with pytest.raises(NotImplementedError):
        policy._budget_fock_controls(calculator, TargetErrorBudget(1e-5, 1e-4), {})


def test_missing_protocol_retains_compatibility() -> None:
    calculator = model(
        SimpleNamespace(generativeqc_application_fock_precision_version_v2=lambda: 2)
    )
    assert (
        transport.budget_controls(
            calculator, TargetErrorBudget(1e-5, 1e-4), {}, available=True
        )
        is None
    )
    assert policy._budget_fock_controls(
        calculator, TargetErrorBudget(1e-5, 1e-4), {}
    ) == dict(zip(NAMES, (1e-6, 1e-4, 1e-10), strict=True))


def test_transient_binding_does_not_retain_retired_library(
    library: Any,
) -> None:
    """Transient ctypes call cycles must not retain a retired selected library."""
    selected = ct.CDLL(library._name)
    selected.generativeqc_status_message = lambda status: b"policy refused"
    selected.generativeqc_application_fock_precision_version_v2 = lambda: 2
    retained = weakref.ref(selected)
    calculator = model(selected)
    assert policy._budget_fock_controls(calculator, TargetErrorBudget(1e-5, 1e-4), {})
    del calculator, selected
    gc.collect()
    assert retained() is None

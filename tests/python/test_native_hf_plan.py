"""Executable numerical planning, public transport and compatibility boundaries."""

from __future__ import annotations

import ctypes
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import _hf_plan_native as native
from generativeqc import _native
from generativeqc.progressive_controller import DeterministicHFPlan, ProgressiveBudget
from test_progressive_controller import PROBLEM, stages

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def library(required_native_cxx: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Compile the actual public C adapter without a scientific runtime fixture."""
    directory = tmp_path_factory.mktemp("hf-plan")
    path = required_native_cxx.build_shared(
        [ROOT / "src/api/c_api_hf_plan.cpp"],
        directory / "plan.so",
        compile_args=(
            "-std=c++17",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    result = ctypes.CDLL(str(path))
    # Only status text is a test seam; every decision runs compiled production code.
    result.generativeqc_status_message = lambda status: b"invalid argument"
    return result


def forbid_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if an admitted native decision re-enters the Python policy."""

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native plan retried compatibility")

    monkeypatch.setattr(native, "_compatibility_failures", forbidden)


def test_public_sdk_contract(required_native_cxx: Any, tmp_path: Path) -> None:
    """Run the same installed-header contract used by CTest."""
    executable = required_native_cxx.build_executable(
        [ROOT / "tests/native/test_hf_plan.cpp", ROOT / "src/api/c_api_hf_plan.cpp"],
        tmp_path / "plan",
        compile_args=(
            "-std=c++17",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=30)


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"maximum_source_iterations": 19}, "source iteration limit"),
        ({"maximum_total_iterations": 119}, "planned iterations"),
        (
            {"maximum_estimated_cost_units": math.nextafter(5.0, 0.0)},
            "estimated stage cost",
        ),
    ],
)
def test_public_plan_uses_native_rejection(
    library: Any, monkeypatch: pytest.MonkeyPatch, options: dict, reason: str
) -> None:
    forbid_compatibility(monkeypatch)
    with pytest.raises(ValueError, match=reason):
        DeterministicHFPlan(
            PROBLEM, stages(), ProgressiveBudget(**options), _library=library
        )


def test_exact_limits_and_identity_are_preserved(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = stages()
    budget = ProgressiveBudget(
        maximum_source_iterations=20,
        maximum_total_iterations=120,
        maximum_estimated_cost_units=5.0,
    )
    portable = DeterministicHFPlan(PROBLEM, (source, target), budget, _library=library)
    forbid_compatibility(monkeypatch)
    monkeypatch.setattr(native, "require_library", lambda: library)
    plan = DeterministicHFPlan(PROBLEM, (source, target), budget)
    assert plan == portable and plan.identity == portable.identity
    assert asdict(plan) == asdict(portable) and "_library" not in asdict(plan)
    assert replace(plan) == plan
    with pytest.raises(ValueError, match="every stage cost estimate"):
        DeterministicHFPlan(
            PROBLEM, (replace(source, estimated_cost_units=None), target), budget
        )
    with pytest.raises(ValueError, match="cannot be stricter"):
        DeterministicHFPlan(
            PROBLEM,
            (
                replace(
                    source, convergence=replace(source.convergence, energy_change=1e-11)
                ),
                target,
            ),
            budget,
        )


def test_decisions_match_independent_arithmetic_and_are_detached(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integer oracle does not wrap; random powers span the binary64 cost range."""
    forbid_compatibility(monkeypatch)
    rng = random.Random(2227)
    source, target = stages()
    samples = []
    for _ in range(300):
        counts = [rng.choice((1, 20, 100, 120, 2**63, 2**64 - 1)) for _ in range(4)]
        costs = [
            rng.choice((None, 1.0, 4.0, 5.0, math.ldexp(1.0, rng.randint(-1074, 1023))))
            for _ in range(3)
        ]
        tolerances = [rng.choice((1e-5, 1e-8, 1e-10)) for _ in range(4)]
        left = replace(
            source,
            convergence=replace(
                source.convergence,
                maximum_iterations=counts[0],
                energy_change=tolerances[0],
                density_rms=tolerances[1],
            ),
            estimated_cost_units=costs[0],
        )
        right = replace(
            target,
            convergence=replace(
                target.convergence,
                maximum_iterations=counts[1],
                energy_change=tolerances[2],
                density_rms=tolerances[3],
            ),
            estimated_cost_units=costs[1],
        )
        budget = ProgressiveBudget(
            maximum_source_iterations=counts[2],
            maximum_total_iterations=counts[3],
            maximum_estimated_cost_units=costs[2],
        )
        failed = [
            tolerances[0] < tolerances[2] or tolerances[1] < tolerances[3],
            counts[0] > counts[2],
            counts[0] + counts[1] > counts[3],
            False,
            False,
        ]
        if costs[2] is not None:
            failed[3] = costs[0] is None or costs[1] is None
            if not failed[3]:
                failed[4] = costs[0] + costs[1] > costs[2]
        expected = sum(1 << bit for bit, flag in enumerate(failed) if flag)
        samples.append((left, right, budget, expected))
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(
            pool.map(
                lambda sample: native.failures(*sample[:3], library=library), samples
            )
        )
    assert actual == [sample[3] for sample in samples]


def test_selected_old_library_and_wide_integer_compatibility(
    monkeypatch: pytest.MonkeyPatch, library: Any
) -> None:
    def forbidden() -> None:
        raise AssertionError("selected owner searched for another library")

    monkeypatch.setattr(native, "require_library", forbidden)
    source, target = stages()
    with pytest.raises(RuntimeError, match="requires the current native plan protocol"):
        native.failures(source, target, ProgressiveBudget(), library=SimpleNamespace())
    wide = replace(
        source, convergence=replace(source.convergence, maximum_iterations=2**80)
    )
    assert native.failures(wide, target, ProgressiveBudget(), library=library) == 6
    assert (
        native.failures(
            wide,
            target,
            ProgressiveBudget(
                maximum_source_iterations=2**80, maximum_total_iterations=2**80 + 100
            ),
            library=library,
        )
        == 0
    )


@pytest.mark.parametrize("missing", ["library", "protocol"])
def test_standalone_plan_requires_current_native_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """Typed standalone plans cannot silently select Python numerical admission."""
    selected = tmp_path / "selected.so"
    if missing == "protocol":
        selected.touch()
        monkeypatch.setattr(native.ctypes, "CDLL", lambda _: SimpleNamespace())
    monkeypatch.setattr(native._native, "_candidate_paths", lambda: iter((selected,)))
    forbid_compatibility(monkeypatch)
    with pytest.raises(RuntimeError, match="HF planning requires the current native"):
        DeterministicHFPlan(PROBLEM, stages(), ProgressiveBudget())


@pytest.mark.parametrize("corrupt", [False, True])
def test_native_error_or_invalid_receipt_never_retries(
    monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    forbid_compatibility(monkeypatch)

    def operation(descriptor: Any, output: Any) -> int:
        if corrupt:
            ctypes.cast(output, ctypes.POINTER(native.Decision)).contents.failures = 32
            return _native.STATUS_SUCCESS
        return _native.STATUS_INVALID_ARGUMENT

    library = SimpleNamespace(
        generativeqc_hf_plan_query_v1=operation,
        generativeqc_status_message=lambda status: b"invalid argument",
    )
    with pytest.raises(
        RuntimeError, match="inconsistent receipt" if corrupt else "invalid argument"
    ):
        native.failures(*stages(), ProgressiveBudget(), library=library)

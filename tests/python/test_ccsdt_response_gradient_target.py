"""Exercise the real CUDA qualification setup without loading native owners."""

from __future__ import annotations

from typing import Any

import _cc_cuda_test_support as cuda_support
import pytest


@pytest.fixture
def compiler_setup(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[tuple]]:
    """Call the shared qualification setup with intercepted compiler boundaries."""
    calls: list[tuple] = []

    def find_nvcc() -> str:
        calls.append(("nvcc",))
        return "allocated-nvcc"

    def target_info(architecture: str) -> str:
        calls.append(("target", architecture))
        if architecture not in {"sm_80", "sm_89", "sm_90", "sm_120"}:
            raise ValueError("unsupported target")
        return architecture

    def compiler(nvcc: str, target: str) -> tuple[str, str]:
        calls.append(("compiler", nvcc, target))
        return nvcc, target

    monkeypatch.setattr(cuda_support, "find_nvcc", find_nvcc)
    monkeypatch.setattr(cuda_support, "cuda_target_info", target_info)
    monkeypatch.setattr(cuda_support, "CudaCompilerAdapter", compiler)
    monkeypatch.setenv("SLURM_JOB_ID", "host-control-flow-test")
    return cuda_support.response_gradient_compiler, calls


@pytest.mark.parametrize("architecture", [None, "", "   "])
def test_target_is_required_before_compiler_lookup(
    compiler_setup: tuple[Any, list[tuple]],
    monkeypatch: pytest.MonkeyPatch,
    architecture: str | None,
) -> None:
    setup, calls = compiler_setup
    if architecture is None:
        monkeypatch.delenv("GENERATIVEQC_TENSOR_ARCH", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_TENSOR_ARCH", architecture)
    with pytest.raises(pytest.fail.Exception, match="GENERATIVEQC_TENSOR_ARCH"):
        setup()
    assert calls == []


@pytest.mark.parametrize(
    "architecture", ["sm_80", "sm_89", "sm_90", "sm_120", " sm_90 "]
)
def test_explicit_target_is_preserved(
    compiler_setup: tuple[Any, list[tuple]],
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
) -> None:
    setup, calls = compiler_setup
    monkeypatch.setenv("GENERATIVEQC_TENSOR_ARCH", architecture)
    assert setup() == ("allocated-nvcc", architecture.strip())
    assert calls[-1] == ("compiler", "allocated-nvcc", architecture.strip())


def test_invalid_target_has_no_default_fallback(
    compiler_setup: tuple[Any, list[tuple]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup, calls = compiler_setup
    monkeypatch.setenv("GENERATIVEQC_TENSOR_ARCH", "not-a-target")
    with pytest.raises(ValueError, match="unsupported target"):
        setup()
    assert not any(call[0] == "compiler" for call in calls)


def test_slurm_requirement_is_preserved(
    compiler_setup: tuple[Any, list[tuple]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup, calls = compiler_setup
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("GENERATIVEQC_TENSOR_ARCH", "sm_90")
    with pytest.raises(AssertionError, match="requires Slurm"):
        setup()
    assert calls == []


def test_missing_nvcc_has_no_compiler_fallback(
    compiler_setup: tuple[Any, list[tuple]], monkeypatch: pytest.MonkeyPatch
) -> None:
    setup, calls = compiler_setup
    monkeypatch.setenv("GENERATIVEQC_TENSOR_ARCH", "sm_90")
    monkeypatch.setattr(cuda_support, "find_nvcc", lambda: None)
    with pytest.raises(AssertionError):
        setup()
    assert calls == []


def test_compiler_construction_failure_propagates(
    compiler_setup: tuple[Any, list[tuple]], monkeypatch: pytest.MonkeyPatch
) -> None:
    setup, calls = compiler_setup
    monkeypatch.setenv("GENERATIVEQC_TENSOR_ARCH", "sm_90")

    def fail(nvcc: str, target: str) -> None:
        assert (nvcc, target) == ("allocated-nvcc", "sm_90")
        raise RuntimeError("compiler initialization failed")

    monkeypatch.setattr(cuda_support, "CudaCompilerAdapter", fail)
    with pytest.raises(RuntimeError, match="compiler initialization failed"):
        setup()
    assert calls == [("nvcc",), ("target", "sm_90")]

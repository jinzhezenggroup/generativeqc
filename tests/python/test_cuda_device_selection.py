"""Runtime target discovery respects visible ordinals without a real GPU."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from generativeqc import _cuda_device, profiles
from generativeqc.experimental import array_api as xp


@pytest.mark.parametrize(
    "capability", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]
)
def test_driver_uses_selected_visible_ordinal(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
) -> None:
    """Device names/topology and physical ordinal zero never select the target."""

    def get_device(output: Any, ordinal: int) -> int:
        assert ordinal == 2
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = 7
        return 0

    def compute_capability(major: Any, minor: Any, handle: ctypes.c_int) -> int:
        assert handle.value == 7
        for output, value in zip((major, minor), capability, strict=True):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = value
        return 0

    driver = SimpleNamespace(
        cuInit=Mock(return_value=0),
        cuDeviceGet=Mock(side_effect=get_device),
        cuDeviceComputeCapability=Mock(side_effect=compute_capability),
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda name: driver)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")
    assert _cuda_device.device_architecture(2) == f"sm_{capability[0]}{capability[1]}"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b,GPU-c"


@pytest.mark.parametrize(
    "stage", ["cuInit", "cuDeviceGet", "cuDeviceComputeCapability"]
)
def test_driver_failure_does_not_guess_a_target(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    driver = SimpleNamespace(
        **{
            name: Mock(return_value=100 if name == stage else 0)
            for name in ("cuInit", "cuDeviceGet", "cuDeviceComputeCapability")
        }
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda name: driver)
    with pytest.raises(RuntimeError, match=stage):
        _cuda_device.device_architecture()


def test_missing_driver_does_not_guess_a_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ctypes, "CDLL", Mock(side_effect=OSError("driver missing")))
    with pytest.raises(RuntimeError, match="NVIDIA driver"):
        _cuda_device.device_architecture()


@pytest.mark.parametrize("device", [-1, True, 2**31, 0.5])
def test_invalid_device_is_rejected_before_driver_access(
    monkeypatch: pytest.MonkeyPatch, device: Any
) -> None:
    monkeypatch.setattr(
        ctypes, "CDLL", Mock(side_effect=AssertionError("driver access"))
    )
    with pytest.raises(ValueError, match="ordinal"):
        _cuda_device.device_architecture(device)


@pytest.mark.parametrize("explicit", [None, "sm_90"])
def test_array_api_compiles_for_detected_or_explicit_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, explicit: str | None
) -> None:
    from generativeqc_compiler.common import cuda_adapter
    from generativeqc_compiler.tensor import cuda_execute

    probe = Mock(return_value="sm_86")
    monkeypatch.setattr(_cuda_device, "device_architecture", probe)
    monkeypatch.setattr(profiles, "find_nvcc", lambda: Path("nvcc"))
    monkeypatch.setattr(
        cuda_adapter, "CudaCompilerAdapter", lambda *args, **kwargs: None
    )
    targets = []

    def compile_plan(plan: Any, compiler: Any, cache: Path) -> None:
        targets.append(plan.target.architecture)

    def prepare(plan: Any, artifact: Any, *, device: int) -> SimpleNamespace:
        assert device == 2
        return SimpleNamespace(
            execute=lambda feeds: SimpleNamespace(outputs={"output": feeds["values"]}),
            close=lambda: None,
        )

    monkeypatch.setattr(cuda_execute, "compile_cuda", compile_plan)
    monkeypatch.setattr(cuda_execute, "PreparedCuda", prepare)
    with_target = {} if explicit is None else {"cuda_arch": explicit}
    compiled = xp.compile(
        lambda values: values, backend="cuda", device=2, cache=tmp_path, **with_target
    )
    # Tracing/lowering stays offline even with automatic device selection.
    values = np.asarray([1.0, 2.0])
    compiled.lower(values)
    assert not probe.called
    compiled(values)
    assert targets == [explicit or "sm_86"]
    if explicit is None:
        probe.assert_called_with(2)
    else:
        assert not probe.called
    compiled.close()

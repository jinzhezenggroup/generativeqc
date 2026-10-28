"""Exercise benchmark deployment choices without a physical GPU or toolkit."""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc_compiler.integral import batch_benchmark, benchmark
from generativeqc_compiler.integral.tuning import emission, process

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from conftest import NativeCxx


@pytest.mark.parametrize(
    "environment_builder",
    [
        benchmark._runtime_environment,
        batch_benchmark._runtime_environment,
        process._runtime_environment,
    ],
)
@pytest.mark.parametrize("visible", [None, "", "2,0", "GPU-allocated-uuid"])
def test_benchmark_environment_preserves_visibility(
    environment_builder: Callable[[Path], dict[str, str]],
    visible: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "existing"))
    before = dict(os.environ)
    result = environment_builder(tmp_path / "toolkit/bin/nvcc")
    assert result.get("CUDA_VISIBLE_DEVICES") == visible
    assert result["LD_LIBRARY_PATH"] == f"{tmp_path}/toolkit/lib64:{tmp_path}/existing"
    assert dict(os.environ) == before


def test_batch_cli_discovers_toolkit_and_forwards_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    compiler = tmp_path / "custom toolkit/bin/nvcc"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("#!/bin/sh\nexit 0\n")
    compiler.chmod(0o755)
    for name in ("GENERATIVEQC_NVCC", "CUDACXX", "CUDAToolkit_ROOT", "CUDA_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_HOME", str(compiler.parent.parent))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        batch_benchmark.sys,
        "argv",
        [
            "batch",
            "--architecture",
            "sm_86",
            "--shell-class",
            "psss",
            "--device-id",
            "2",
        ],
    )
    monkeypatch.setattr(
        batch_benchmark,
        "_run_batch",
        lambda args: {
            "nvcc": str(args.nvcc),
            "device_id": args.device_id,
            "accepted_shell_classes": ["psss"],
        },
    )
    batch_benchmark.main()
    assert json.loads(capsys.readouterr().out) == {
        "nvcc": str(compiler),
        "device_id": 2,
        "accepted_shell_classes": ["psss"],
    }


# The driver calls the real CUDA ABI names, implemented here by a tiny host
# fixture. Different visible ordinals have different architectures; a reset to
# zero therefore fails the generated target gate even without GPU hardware.
_CUDA_STUB = r"""
#pragma once
#include <cstddef>
#include <cstdio>
using cudaError_t = int;
constexpr int cudaSuccess = 0;
constexpr int cudaDevAttrMaxSharedMemoryPerBlockOptin = 1;
inline int current_device = 0;
struct cudaDeviceProp {
  char name[32] = "synthetic";
  int major = 8, minor = 0, warpSize = 32, maxThreadsPerBlock = 1024;
  int maxThreadsPerMultiProcessor = 1536, maxBlocksPerMultiProcessor = 16;
  int regsPerMultiprocessor = 65536, multiProcessorCount = 12;
  std::size_t sharedMemPerBlock = 49152, sharedMemPerMultiprocessor = 65536;
};
inline int cudaSetDevice(int value) {
  if (value < 0 || value > 2) return 7;
  current_device = value; return 0;
}
inline int cudaFree(void*) { std::printf("initialized device=%d\n", current_device); return 0; }
inline const char* cudaGetErrorString(int) { return "invalid visible ordinal"; }
inline int cudaGetDevice(int* value) { *value = current_device; return 0; }
inline int cudaGetDeviceProperties(cudaDeviceProp* value, int device) {
  if (device != current_device) return 7;
  value->minor = device == 2 ? 6 : 0; return 0;
}
inline int cudaDriverGetVersion(int* value) { *value = 12090; return 0; }
inline int cudaRuntimeGetVersion(int* value) { *value = 12090; return 0; }
inline int cudaDeviceGetAttribute(int* value, int, int device) {
  if (device != current_device) return 7;
  *value = 65536; return 0;
}
"""


@pytest.mark.parametrize("driver", ["schedule", "batch"])
@pytest.mark.parametrize("device_id", [0, 2, 3])
def test_generated_drivers_select_requested_visible_gpu(
    tmp_path: Path, native_cxx: NativeCxx, driver: str, device_id: int
) -> None:
    (tmp_path / "cuda_runtime.h").write_text(_CUDA_STUB)
    source = tmp_path / "driver.cpp"
    source.write_text(
        emission.emit_schedule_driver((), "sm_86", device_id=device_id)
        if driver == "schedule"
        else batch_benchmark.emit_batch_driver((), device_id=device_id)
    )
    executable = tmp_path / "driver"
    native_cxx.build_executable(
        [source], executable, compile_args=("-std=c++17", f"-I{tmp_path}")
    )
    result = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=10
    )
    if device_id == 3:
        assert result.returncode == 2
        assert "initialized device=" not in result.stdout
    else:
        assert f"initialized device={device_id}" in result.stdout
        assert result.returncode == (
            5 if driver == "schedule" and device_id == 0 else 0
        )
        if driver == "schedule":
            record = json.loads(result.stdout.splitlines()[1])
            assert record["device_id"] == device_id


@pytest.mark.parametrize("invalid", [-1, 2**31, True, "2"])
def test_driver_rejects_invalid_device_before_source_emission(invalid: Any) -> None:
    with pytest.raises(ValueError, match="ordinal"):
        emission.emit_schedule_driver((), "sm_86", device_id=invalid)
    with pytest.raises(ValueError, match="ordinal"):
        batch_benchmark.emit_batch_driver((), device_id=invalid)

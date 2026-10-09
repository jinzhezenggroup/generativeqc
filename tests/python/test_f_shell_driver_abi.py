"""Compile and execute the real fixture driver's CUDA argument marshalling."""

import json
import re
import struct
import subprocess
from pathlib import Path
from typing import Any

from tools.generativeqc_validation.f_shell import generated_symbols, source_audit
from tools.generativeqc_validation.f_shell_cuda import emit_numerical_driver

ROOT = Path(__file__).resolve().parents[2]

# The host stub provides allocation and argument interception only. No integral,
# Fock, force, atomic, or CUDA execution is simulated by this ABI regression.
CUDA_STUB = r"""
#pragma once
#include <cstdlib>
#include <cstring>
#define __global__
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
struct dim3 { unsigned x; explicit dim3(unsigned value) : x(value) {} };
struct cudaDeviceProp { int major = 12, minor = 0; char name[16] = "ABI host stub"; };
struct cudaFuncAttributes {
  int maxThreadsPerBlock = 1024, numRegs = 32;
  std::size_t localSizeBytes = 0, sharedSizeBytes = 0;
};
inline const char* cudaGetErrorString(int) { return "host stub failure"; }
template<class T> int cudaMalloc(T** pointer, std::size_t bytes) {
  *pointer = static_cast<T*>(std::malloc(bytes)); return *pointer ? 0 : 1;
}
inline int cudaFree(void* pointer) { std::free(pointer); return 0; }
inline int cudaMemcpy(void* dst, const void* src, std::size_t bytes, int) {
  std::memcpy(dst, src, bytes); return 0;
}
inline int cudaMemset(void* dst, int value, std::size_t bytes) {
  std::memset(dst, value, bytes); return 0;
}
inline int cudaGetDevice(int* value) { *value = 0; return 0; }
inline int cudaGetDeviceProperties(cudaDeviceProp*, int) { return 0; }
inline int cudaDriverGetVersion(int* value) { *value = 12090; return 0; }
inline int cudaRuntimeGetVersion(int* value) { *value = 12090; return 0; }
inline int cudaFuncGetAttributes(cudaFuncAttributes*, const void*) { return 0; }
inline int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* blocks, const void*, unsigned, int) {
  *blocks = 1; return 0;
}
inline int cudaGetLastError() { return 0; }
inline int cudaDeviceSynchronize() { return 0; }
int cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t, void*);
"""

LAUNCH_INTERCEPT = r"""
int cudaLaunchKernel(const void* function, dim3 grid, dim3 block, void** args,
                     std::size_t shared, void* stream) {
  for (const auto& kernel : kernels) if (kernel.function == function) {
    if (grid.x != 1 || block.x != kernel.threads || shared || stream)
      throw std::runtime_error("unexpected launch geometry");
    if (kernel.persistent) {
      if (**static_cast<std::uint32_t**>(args[9]) != 0 ||
          **static_cast<std::uint32_t**>(args[10]) != 1 ||
          **static_cast<std::uint32_t**>(args[11]) != 0)
        throw std::runtime_error("persistent queue ABI mismatch");
    } else if (*static_cast<std::size_t*>(args[9]) != 1) {
      throw std::runtime_error("direct task count ABI mismatch");
    }
    double* output = nullptr;
    if (kernel.force) {
      output = *static_cast<double**>(args[8]);
    } else {
      const auto value = *static_cast<generativeqc::runtime::CompensatedOutput*>(args[8]);
      if (value.correction != nullptr)
        throw std::runtime_error("Fock correction argument must be null");
      output = value.sum;
    }
    if (!output) throw std::runtime_error("missing output plane");
    output[0] = kernel.force ? 101.0 : 202.0;
    return 0;
  }
  throw std::runtime_error("unknown kernel");
}
"""


def test_numerical_driver_matches_generated_signatures_and_launch_values(
    tmp_path: Path, native_cxx: Any
) -> None:
    """Check all eight real wrapper signatures and direct/persistent argv slots."""
    _, source = source_audit("fsss")
    definitions = []
    for symbol in generated_symbols("fsss"):
        signature = re.search(r"void " + symbol + r"\([^{}]*\)\s*\{", source)
        assert signature is not None
        definitions.append('extern "C" ' + signature[0] + "}\n")
    (tmp_path / "cuda_runtime.h").write_text(CUDA_STUB)
    driver = tmp_path / "driver.cpp"
    driver.write_text(
        emit_numerical_driver("fsss") + "\n" + "\n".join(definitions) + LAUNCH_INTERCEPT
    )
    executable = tmp_path / "driver"
    native_cxx.build_executable(
        [driver],
        executable,
        compile_args=["-std=c++17", "-I", str(tmp_path), "-I", str(ROOT / "src")],
    )
    fixture = tmp_path / "fixture.bin"
    integers = (13, 2, 1, 0, 0, 10, 11, 12, 0, 1, 2, 3, 1, 1, 1, 1)
    values = (0.0,) * 12 + (1.0,) * (13 + 16 + 3 * 13 * 13)
    fixture.write_bytes(
        b"VQF13501"
        + struct.pack("<4I4Q8I", *integers)
        + struct.pack(f"<{len(values)}d", *values)
    )
    result = subprocess.run(
        [str(executable), str(fixture)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    outputs = json.loads(result.stdout.splitlines()[1])["outputs"]
    assert len(outputs) == 8
    for name, values in outputs.items():
        assert values[0] == (101 if "force" in name else 202)
        assert all(value == 0 for value in values[1:])

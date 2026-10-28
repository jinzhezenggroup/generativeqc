"""Execute parameter-family force producers with independent physical XC oracles."""

from __future__ import annotations

import ctypes
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code
from generativeqc_compiler.xc.parameter_cuda_package import parameterized_cuda_package
from test_libxc_automatic_point_native import _inputs, _oracle

if TYPE_CHECKING:
    from conftest import NativeCxx

NAMES = ("MGGA_X_KTBM_0", "MGGA_X_KTBM_1")


@pytest.fixture(scope="module")
def stationary_library(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> ctypes.CDLL:
    """Compile real generated bodies with CUDA launch syntax simulated on host.

    This exercises parameter rows, physical layout and failure propagation;
    separate finite Slurm qualification covers actual device execution.
    """
    directory = tmp_path_factory.mktemp("parameter-stationary")
    (directory / "tensor").mkdir()
    (directory / "cuda_runtime_api.h").write_text("using cudaStream_t = void*;\n")
    (directory / "tensor/cuda_runtime.cuh").write_text("""#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#define __device__
#define __global__
#define __grid_constant__
#define __forceinline__ inline
using std::isfinite;
using std::min;
struct HostDim { unsigned x; };
HostDim blockIdx{0}, threadIdx{0}, blockDim{1}, gridDim{1};
int atomicCAS(int* p, int expected, int value) {
  int old = *p; if (old == expected) *p = value; return old;
}
int atomicExch(int* p, int value) { int old = *p; *p = value; return old; }
namespace generativeqc_tensor {
using I = std::int64_t;
double finite(double x, int* error, int code) {
  if (!std::isfinite(x)) atomicCAS(error, 0, code); return x;
}
}
""")
    profiles = tuple(
        f"libxc_{automatic_functional_code(name)}_{spin}"
        for name, spin in zip(NAMES, ("rks", "uks"), strict=True)
    )
    workers, providers = parameterized_cuda_package(NAMES, force_profiles=profiles)
    assert all(provider.stationary_launcher for provider in providers.values())
    _, energy_only = parameterized_cuda_package(NAMES)
    assert not any(provider.stationary_launcher for provider in energy_only.values())
    source = "\n".join((*workers.values(), *(p.source for p in providers.values())))
    source = re.sub(r"<<<.*?>>>", "", source, flags=re.DOTALL)
    cases = "\n".join(
        f"case {index}: {providers[name].namespace}::launch_stationary(nullptr, "
        "features, count, spins, values.data(), &error, &producer_error); break;"
        for index, name in enumerate(NAMES)
    )
    source += (
        """
#include <vector>
extern "C" int evaluate(unsigned entry, std::size_t spins, std::size_t count,
                        const double* features, double* output,
                        int error, int producer_error) {
  std::vector<generativeqc::dft::DevicePointValue> values(count);
  try {
    for (blockIdx.x = 0; blockIdx.x < std::max(std::size_t{1}, count); ++blockIdx.x) {
      switch (entry) {
"""
        + cases
        + """
        default: return -2;
      }
    }
  } catch (const std::invalid_argument&) { return -1; }
  if (error) return error;
  for (std::size_t i = 0; i < count; ++i) {
    const auto& value = values[i]; double* out = output + 11*i;
    out[0] = value.energy;
    for (unsigned spin = 0; spin < 2; ++spin) {
      out[1+spin] = value.rho[spin]; out[9+spin] = value.kinetic[spin];
      for (unsigned axis = 0; axis < 3; ++axis)
        out[3+3*spin+axis] = value.gradient[spin][axis];
    }
  }
  return 0;
}
"""
    )
    path = directory / "launch.cpp"
    path.write_text(source)
    root = Path(__file__).resolve().parents[2]
    binary = native_cxx.build_shared(
        [path],
        directory / "launch.so",
        compile_args=(
            "-std=c++20",
            "-O3",
            "-ffp-contract=off",
            f"-I{directory}",
            f"-I{root / 'src'}",
        ),
    )
    library = ctypes.CDLL(str(binary))
    vector = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    library.evaluate.argtypes = (
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_size_t,
        vector,
        vector,
        ctypes.c_int,
        ctypes.c_int,
    )
    library.evaluate.restype = ctypes.c_int
    return library


@pytest.mark.parametrize("index,spins", ((0, 1), (1, 2)))
@pytest.mark.parametrize("selection", ("0", "1"))
def test_stationary_parameter_rows_match_independent_libxc(
    stationary_library: ctypes.CDLL,
    index: int,
    spins: int,
    selection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Energy shortcuts cannot discard physical derivatives in force tiles."""
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", selection)
    inputs = _inputs(NAMES[index], spins - 1)
    packed = np.ascontiguousarray(inputs[:, [0, 2, 3, 4, 8, 1, 5, 6, 7, 9]].T)
    actual = np.empty((len(inputs), 11))
    assert (
        stationary_library.evaluate(index, spins, len(inputs), packed, actual, 0, 0)
        == 0
    )
    observed = (
        actual
        if spins == 2
        else np.column_stack(
            (
                actual[:, 0],
                actual[:, 1:3].mean(axis=1),
                0.5 * (actual[:, 3:6] + actual[:, 6:9]),
                actual[:, 9:11].mean(axis=1),
            )
        )
    )
    np.testing.assert_allclose(
        observed, _oracle(NAMES[index], inputs, spins - 1), atol=2e-11, rtol=2e-9
    )
    packed[0, 10] = -1.0
    assert (
        stationary_library.evaluate(index, spins, len(inputs), packed, actual, 0, 0)
        != 0
    )


@pytest.mark.parametrize(
    "index,spins,count,error,producer,expected",
    (
        (0, 2, 1, 0, 0, -1),
        (1, 1, 1, 0, 0, -1),
        (0, 0, 1, 0, 0, -1),
        (0, 1, 0, 0, 0, -1),
        (0, 1, 1, 7, 0, 1),
        (1, 2, 1, 0, 5, 1),
    ),
)
def test_stationary_parameter_admission_and_producer_errors(
    stationary_library: ctypes.CDLL,
    index: int,
    spins: int,
    count: int,
    error: int,
    producer: int,
    expected: int,
) -> None:
    """Reject unpackaged spins and never publish an upstream failed tile."""
    packed = np.zeros((10, max(count, 1)))
    output = np.full((max(count, 1), 11), 991.0)
    assert (
        stationary_library.evaluate(
            index, spins, count, packed, output, error, producer
        )
        == expected
    )
    assert np.all(output == 991.0)

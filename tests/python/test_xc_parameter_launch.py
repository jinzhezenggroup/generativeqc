"""Execute shared launchers, adapters and compact layouts against independent XC."""

from __future__ import annotations

import ctypes
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code
from generativeqc_compiler.xc.parameter_cuda_package import parameterized_cuda_package
from test_libxc_automatic_cuda_xc import _pack_tiles, _unpack_tiles
from test_libxc_automatic_point_native import _inputs, _oracle

if TYPE_CHECKING:
    from conftest import NativeCxx

NAMES = (
    "GGA_XC_TH_FCFO",
    "GGA_XC_TH_FCO",
    "GGA_X_BKL1",
    "GGA_X_BKL2",
    "LDA_C_BR78",
    "LDA_C_WIGNER",
    "MGGA_X_KTBM_0",
    "MGGA_X_KTBM_GAP",
)


@pytest.fixture(scope="module")
def launch_library(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> ctypes.CDLL:
    """Run exact generated launch bodies with one host-simulated grid thread.

    Only CUDA launch syntax and execution qualifiers are stubbed. The grid-stride
    loop still visits every point, so partial tiles, physical spin embedding,
    coefficient rows and the actual host selector execute together. Real CUDA
    compilation and device performance remain separately qualified.
    """
    directory = tmp_path_factory.mktemp("xc-parameter-launch")
    (directory / "tensor").mkdir()
    (directory / "cuda_runtime_api.h").write_text("using cudaStream_t = void*;\n")
    (directory / "tensor/cuda_runtime.cuh").write_text(
        """#pragma once
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
constexpr HostDim blockIdx{0}, threadIdx{0}, blockDim{1}, gridDim{1};
inline int atomicCAS(int* address, int expected, int desired) {
  const int old = *address;
  if (old == expected) *address = desired;
  return old;
}
namespace generativeqc_tensor {
using I = std::int64_t;
inline double finite(double value, int* error, int code) {
  if (!std::isfinite(value)) atomicCAS(error, 0, code);
  return value;
}
}
"""
    )
    workers, providers = parameterized_cuda_package(NAMES, policy="discover")
    source = "\n".join((*workers.values(), *(p.source for p in providers.values())))
    source = re.sub(r"<<<.*?>>>", "", source, flags=re.DOTALL)
    cases = []
    for index, name in enumerate(NAMES):
        namespace = providers[name].namespace
        code = automatic_functional_code(name)
        cases.append(f"""
    case {index}:
      if (tile)
        {namespace}::launch_batch(nullptr, features, weights, count, spins,
            coefficients, totals, &error, {code}, 1.0, 1.0, tile);
      else
        {namespace}::launch(nullptr, features, weights, count, spins,
            coefficients, totals, &error, {code}, 1.0, 1.0, nullptr);
      return error;
""")
    source += (
        """
extern "C" int evaluate(unsigned entry, std::size_t spins, std::size_t tile,
                        const double* features, const double* weights,
                        double* coefficients, double* totals, std::size_t count) {
  int error = 0;
  switch (entry) {
"""
        + "".join(cases)
        + "\n    default: return -1;\n  }\n}\n"
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
        vector,
        vector,
        ctypes.c_size_t,
    )
    library.evaluate.restype = ctypes.c_int
    return library


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("tile", (0, 16, 127))
@pytest.mark.parametrize("selection", ("0", "1"))
def test_shared_launch_matches_independent_physical_spin_oracle(
    launch_library: ctypes.CDLL,
    name: str,
    spins: int,
    tile: int,
    selection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paired adaptation must preserve U1, P2, disabled selection and tail strides."""
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_EQUAL_SPIN", selection)
    inputs = np.resize(_inputs(name, spins - 1), (263, 10))
    terms = 1 if name.startswith("LDA") else 5 if name.startswith("MGGA") else 4
    physical = np.stack(
        [inputs[:, [0, 2, 3, 4, 8]], inputs[:, [1, 5, 6, 7, 9]]], axis=1
    )
    if spins == 1:
        physical = physical.sum(axis=1, keepdims=True)
    count = len(inputs)
    features = physical[:, :, :terms].reshape(count, -1)
    packed = _pack_tiles(features, tile or count)
    weights = np.linspace(0.1, 1.1, count)
    coefficients = np.full(packed.shape, np.nan)
    totals = np.full(3 * count, np.nan)
    assert (
        launch_library.evaluate(
            NAMES.index(name), spins, tile, packed, weights, coefficients, totals, count
        )
        == 0
    )
    oracle = _oracle(name, inputs, spins - 1)
    expected = (
        (
            oracle[:, 1:6]
            if spins == 1
            else np.column_stack(
                [oracle[:, [1, 3, 4, 5, 9]], oracle[:, [2, 6, 7, 8, 10]]]
            )
        )
        .reshape(count, spins, 5)[:, :, :terms]
        .reshape(count, -1)
    )
    np.testing.assert_allclose(
        _unpack_tiles(coefficients, count, spins * terms, tile or count),
        expected,
        rtol=4e-11,
        atol=4e-11,
    )
    actual_totals = _unpack_tiles(totals, count, 3, tile or count)
    np.testing.assert_allclose(
        actual_totals[:, 0], weights * oracle[:, 0], rtol=4e-11, atol=4e-11
    )
    np.testing.assert_allclose(
        actual_totals[:, 1:], weights[:, None] * inputs[:, :2], rtol=1e-15, atol=1e-30
    )
    # Invalid physical input must still reach the device-visible error channel.
    packed[0] = -1.0
    assert (
        launch_library.evaluate(
            NAMES.index(name), spins, tile, packed, weights, coefficients, totals, count
        )
        == 3
    )

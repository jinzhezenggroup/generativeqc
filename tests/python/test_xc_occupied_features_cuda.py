"""Opt-in scheduled CUDA tests of emitted occupied projection and feature code."""

import ctypes as ct
import os
from pathlib import Path

import numpy as np
import pytest
from _cpp_source_support import cpp_function_definition, cpp_record_definition
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.dft.feature_policy import emit_feature_policy
from generativeqc_compiler.dft.xc_contraction_cuda import (
    DEFAULT_XC_MATRIX_SCHEDULE,
    _emit_occupied_features,
    _emit_tiled,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_OCCUPIED_XC") != "1",
    reason="requires explicit finite Slurm GPU allocation",
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def occupied_probe(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    directory = tmp_path_factory.mktemp("occupied-xc-cuda")
    source, library = directory / "probe.cu", directory / "probe.so"
    declaration = cpp_record_definition(
        (ROOT / "src/dft/cuda_xc.hpp").read_text(), "CudaXcOccupiedDensity"
    )
    source.write_text(
        PREAMBLE
        + declaration
        + ";\nnamespace generativeqc_grid_policy {\n"
        + emit_feature_policy(device=True)
        + "}\n"
        + cpp_function_definition(
            _emit_tiled(DEFAULT_XC_MATRIX_SCHEDULE),
            "tiled_density_product_body",
            include_template=True,
        )
        + _emit_occupied_features(DEFAULT_XC_MATRIX_SCHEDULE)
        + DRIVER
    )
    result = CudaCompilerAdapter(
        Path(
            os.environ.get("GENERATIVEQC_NVCC", "/group/software/cuda-12.9.1/bin/nvcc")
        ),
        cuda_target_info("sm_120"),
    ).compile_shared(source, library, options=("--fmad=false", "-lineinfo"))
    assert result.returncode == 0, result.stderr
    native = ct.CDLL(str(library))
    function = native.occupied_probe
    function.argtypes = [ct.c_void_p] * 4 + [ct.c_size_t] * 7 + [ct.c_void_p]
    function.restype = ct.c_int
    return native


@pytest.mark.parametrize("occupations", [(1,), (5,), (33,), (5, 3), (0, 7)])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e-308, 1e-155, 1e90])
def test_projection_and_features_match_independent_density(
    occupied_probe: ct.CDLL, occupations: tuple[int, ...], mapped: bool, scale: float
) -> None:
    rng = np.random.default_rng(20261017)
    active, points, spins = 41, 19, len(occupations)
    nao = 55 if mapped else active
    ids = (
        np.sort(rng.choice(nao, active, replace=False)).astype(np.uintp)
        if mapped
        else None
    )
    selected = ids if mapped else np.arange(active)
    frames = rng.normal(size=(spins, nao, nao)) * 0.1 * scale
    ao = np.ascontiguousarray(rng.normal(size=(4, points, active)) * 0.1)
    if scale == 1e-308:
        ao[1:] *= 1e307
    elif scale != 1.0:
        ao /= scale
    reference = np.zeros((spins, 5, points))
    matrices = np.zeros((spins, nao, nao))
    for spin, count in enumerate(occupations):
        matrices[spin] = (
            frames[spin, :, :count]
            @ frames[spin, :, :count].T
            * (2.0 if spins == 1 else 1.0)
        )
        density = matrices[spin][np.ix_(selected, selected)]
        # The matrix consumer symmetrizes each triangle after FP64 storage.
        # Halving subnormals before addition is observably different from
        # either an unrounded C*C.T oracle or halving their sum afterwards.
        density = 0.5 * density + 0.5 * density.T
        work = ao @ density
        reference[spin, 0] = np.sum(ao[0] * work[0], axis=-1)
        reference[spin, 1:4] = 2.0 * np.sum(ao[1:4] * work[0], axis=-1)
        reference[spin, 4] = 0.5 * np.sum(ao[1:4] * work[1:4], axis=(0, 2))
        frames[spin, :, count:] = np.nan
    coefficients = np.ascontiguousarray(frames.transpose(0, 2, 1))
    actual = np.zeros_like(reference)
    status = occupied_probe.occupied_probe(
        matrices.ctypes.data,
        coefficients.ctypes.data,
        ao.ctypes.data,
        None if ids is None else ids.ctypes.data,
        nao,
        active,
        points,
        spins,
        occupations[0],
        occupations[1] if spins == 2 else 0,
        max(occupations),
        actual.ctypes.data,
    )
    assert status == 0
    np.testing.assert_allclose(actual, reference, rtol=5e-13, atol=5e-14)


PREAMBLE = r"""
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
using I = std::int64_t;
namespace generativeqc_tensor {
inline unsigned blocks(I count, unsigned width) { return (count + width - 1) / width; }
inline void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}
__device__ double finite(double value, int* error, int code) {
  if (!isfinite(value)) atomicExch(error, code);
  return value;
}
"""

DRIVER = r"""
template<class Scalar> struct DeviceBuffer {
  Scalar* value{};
  explicit DeviceBuffer(size_t count) {
    generativeqc_tensor::cuda_check(cudaMalloc(&value, count * sizeof(Scalar)));
  }
  ~DeviceBuffer() { if (value) cudaFree(value); }
};
struct Stream {
  cudaStream_t value{};
  Stream() { generativeqc_tensor::cuda_check(cudaStreamCreate(&value)); }
  ~Stream() { if (value) cudaStreamDestroy(value); }
};
extern "C" int occupied_probe(const double* density, const double* coefficients, const double* ao,
    const size_t* ids, size_t nao, size_t active, size_t points, size_t spins,
    size_t alpha, size_t beta, size_t rank, double* output) {
  try {
    using generativeqc_tensor::cuda_check;
    Stream stream;
    DeviceBuffer<double> frame(spins * nao * nao), orbitals(4 * points * active);
    DeviceBuffer<double> matrix(spins * nao * nao);
    DeviceBuffer<double> work(spins * 4 * points * rank), features(spins * 5 * points);
    DeviceBuffer<size_t> map(ids ? active : 1);
    DeviceBuffer<int> error(1);
    cuda_check(cudaMemsetAsync(error.value, 0, sizeof(int), stream.value));
    cuda_check(cudaMemcpyAsync(frame.value, coefficients, spins * nao * nao * sizeof(double),
                               cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(matrix.value, density, spins * nao * nao * sizeof(double),
                               cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(orbitals.value, ao, 4 * points * active * sizeof(double),
                               cudaMemcpyHostToDevice, stream.value));
    if (ids)
      cuda_check(cudaMemcpyAsync(map.value, ids, active * sizeof(size_t),
                                 cudaMemcpyHostToDevice, stream.value));
    CudaXcOccupiedDensity witness{matrix.value, frame.value, nao, spins, {alpha, beta}, 1};
    launch_occupied_density(stream.value, witness, orbitals.value, active, points,
                            work.value, features.value, 5, error.value, ids ? map.value : nullptr);
    int failure = 0;
    cuda_check(cudaMemcpyAsync(output, features.value, spins * 5 * points * sizeof(double),
                               cudaMemcpyDeviceToHost, stream.value));
    cuda_check(cudaMemcpyAsync(&failure, error.value, sizeof(int),
                               cudaMemcpyDeviceToHost, stream.value));
    cuda_check(cudaStreamSynchronize(stream.value));
    return failure;
  } catch (...) {
    return 99;
  }
}
"""

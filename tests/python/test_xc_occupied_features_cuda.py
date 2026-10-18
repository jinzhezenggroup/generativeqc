"""Opt-in scheduled CUDA tests of emitted occupied projection and feature code."""

import ctypes as ct
import os
from pathlib import Path

import numpy as np
import pytest
from _cpp_source_support import cpp_function_definition, cpp_record_definition
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.dft.ao_cuda import _NATIVE_XC_CONTRACTION_KERNELS
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
        + cpp_function_definition(
            _NATIVE_XC_CONTRACTION_KERNELS, "density_product", include_template=True
        )
        + cpp_function_definition(
            _NATIVE_XC_CONTRACTION_KERNELS,
            "density_features_body",
            include_template=True,
        )
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
    native.matrix_probe.argtypes = [ct.c_void_p] * 3 + [ct.c_size_t] * 4 + [ct.c_void_p]
    native.matrix_probe.restype = ct.c_int
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


@pytest.mark.parametrize("active", [17, 41])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("occupations", [(1,), (1, 1), (0, 1), (1, 0)])
@pytest.mark.parametrize("scale", [1e-6, 100.0, 1e6, 1e165])
def test_finite_cancellation_and_overflow_match_original_cuda_consumer(
    occupied_probe: ct.CDLL,
    active: int,
    mapped: bool,
    occupations: tuple[int, ...],
    scale: float,
) -> None:
    """Use the retained emitted matrix dispatch, not an unrounded C*C.T oracle."""
    points, spins = 19, len(occupations)
    nao = active + 7 if mapped else active
    ids = np.arange(active, dtype=np.uintp) + (2 if mapped else 0)
    frame = np.zeros((nao, nao))
    frame[ids[:2], 0] = [0.6, 0.8]
    frame[ids[:2], 1] = [0.8, -0.6]
    for column, row in enumerate(sorted(set(range(nao)) - set(ids[:2])), start=2):
        frame[row, column] = 1.0
    matrices = np.zeros((spins, nao, nao))
    for spin, occupied in enumerate(occupations):
        if occupied:
            matrices[spin] = ((2.0 if spins == 1 else 1.0) * frame[:, :1]) @ frame[
                :, :1
            ].T
    coefficients = np.ascontiguousarray(np.broadcast_to(frame.T, (spins, nao, nao)))
    ao = np.zeros((4, points, active))
    for jet in range(4):
        ao[jet, :, 0] = (jet + 1) * 4.0 * scale
        ao[jet, :, 1] = -(jet + 1) * 3.0 * scale
    actual = np.zeros((spins, 5, points))
    expected = np.zeros_like(actual)
    mapping = ids.ctypes.data if mapped else None
    reference_status = occupied_probe.matrix_probe(
        matrices.ctypes.data,
        ao.ctypes.data,
        mapping,
        nao,
        active,
        points,
        spins,
        expected.ctypes.data,
    )
    actual_status = occupied_probe.occupied_probe(
        matrices.ctypes.data,
        coefficients.ctypes.data,
        ao.ctypes.data,
        mapping,
        nao,
        active,
        points,
        spins,
        occupations[0],
        occupations[1] if spins == 2 else 0,
        1,
        actual.ctypes.data,
    )
    assert actual_status == reference_status
    if scale == 1e165:
        assert reference_status != 0
        assert np.any(~np.isfinite(actual))
    else:
        assert reference_status == 0
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-14)
        if scale == 1e6:
            assert np.max(np.abs(actual)) > 0.001


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
    DeviceBuffer<double> work(spins * 4 * points * active), features(spins * 5 * points);
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

template<bool Cooperative>
__global__ void reference_features(const double* ao, const double* work, I active, I count,
    I spins, double* features, int* error) {
  density_features_body<Cooperative>(ao, work, active, count, spins, 4, 4, 5, 0, features, error);
}
extern "C" int matrix_probe(const double* density, const double* ao, const size_t* ids,
    size_t nao, size_t active, size_t points, size_t spins, double* output) {
  try {
    using generativeqc_tensor::cuda_check;
    Stream stream;
    DeviceBuffer<double> matrix(spins * nao * nao), orbitals(4 * points * active);
    DeviceBuffer<double> work(spins * 4 * points * active), features(spins * 5 * points);
    DeviceBuffer<size_t> map(ids ? active : 1);
    DeviceBuffer<int> error(1);
    cuda_check(cudaMemsetAsync(error.value, 0, sizeof(int), stream.value));
    cuda_check(cudaMemcpyAsync(matrix.value, density, spins * nao * nao * sizeof(double),
                               cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(orbitals.value, ao, 4 * points * active * sizeof(double),
                               cudaMemcpyHostToDevice, stream.value));
    if (ids)
      cuda_check(cudaMemcpyAsync(map.value, ids, active * sizeof(size_t),
                                 cudaMemcpyHostToDevice, stream.value));
    density_product<false><<<generativeqc_tensor::blocks(spins * 4 * points * active, 128),
        128, 0, stream.value>>>(matrix.value, orbitals.value, active, points, spins, 4,
                                work.value, error.value, ids ? map.value : nullptr, nao);
    cuda_check(cudaGetLastError());
    if (active >= 32)
      reference_features<true><<<generativeqc_tensor::blocks(spins * points * 32, 128),
          128, 0, stream.value>>>(orbitals.value, work.value, active, points, spins, features.value, error.value);
    else
      reference_features<false><<<generativeqc_tensor::blocks(spins * points, 128),
          128, 0, stream.value>>>(orbitals.value, work.value, active, points, spins, features.value, error.value);
    cuda_check(cudaGetLastError());
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

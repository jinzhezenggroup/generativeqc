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
from generativeqc_compiler.dft.xc_tile_batch_cuda import emit_native_xc_tile_batches
from generativeqc_compiler.tensor.scf_cuda import emit_density_cuda

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
    batches = emit_native_xc_tile_batches()
    source.write_text(
        PREAMBLE
        + emit_density_cuda()
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
            _emit_tiled(DEFAULT_XC_MATRIX_SCHEDULE),
            "tiled_density_product",
            include_template=True,
        )
        + cpp_function_definition(
            _NATIVE_XC_CONTRACTION_KERNELS,
            "density_features_body",
            include_template=True,
        )
        + cpp_record_definition(
            (ROOT / "src/dft/cuda_xc_tile.hpp").read_text(), "CudaXcCompactTile"
        )
        + ";\n"
        + cpp_function_definition(batches, "compact_tile_ids")
        + cpp_function_definition(
            batches, "batch_density_products", include_template=True
        )
        + cpp_function_definition(
            batches, "batch_density_features", include_template=True
        )
        + DRIVER
        + cpp_function_definition(DRIVER, "occupied_probe")
        .replace("occupied_probe(", "proposal_probe(")
        .replace("{alpha, beta}, 1};", "{alpha, beta}, 1, true};")
        + cpp_function_definition(DRIVER, "compact_probe")
        .replace("compact_probe(", "compact_proposal_probe(")
        .replace("{alpha,beta},1};", "{alpha,beta},1,true};")
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
    native.proposal_probe.argtypes = function.argtypes
    native.proposal_probe.restype = ct.c_int
    native.matrix_probe.argtypes = [ct.c_void_p] * 3 + [ct.c_size_t] * 4 + [ct.c_void_p]
    native.matrix_probe.restype = ct.c_int
    native.form_density.argtypes = [ct.c_void_p] * 2 + [ct.c_size_t] * 2 + [ct.c_void_p]
    native.form_density.restype = ct.c_int
    native.compact_probe.argtypes = (
        [ct.c_void_p, ct.c_size_t]
        + [ct.c_void_p] * 4
        + [ct.c_size_t] * 4
        + [ct.c_void_p] * 2
    )
    native.compact_probe.restype = ct.c_int
    native.compact_proposal_probe.argtypes = native.compact_probe.argtypes
    native.compact_proposal_probe.restype = ct.c_int
    return native


@pytest.mark.parametrize("occupations", [(1,), (5,), (33,), (5, 3), (0, 7)])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e-308, 1e-155, 1e90])
@pytest.mark.parametrize("provisional", [False, True])
def test_projection_and_features_match_independent_density(
    occupied_probe: ct.CDLL,
    occupations: tuple[int, ...],
    mapped: bool,
    scale: float,
    provisional: bool,
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
    function = (
        occupied_probe.proposal_probe if provisional else occupied_probe.occupied_probe
    )
    status = function(
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
    dense_features = np.zeros_like(actual)
    assert (
        occupied_probe.matrix_probe(
            matrices.ctypes.data,
            ao.ctypes.data,
            None if ids is None else ids.ctypes.data,
            nao,
            active,
            points,
            spins,
            dense_features.ctypes.data,
        )
        == 0
    )
    if provisional:
        np.testing.assert_allclose(actual, dense_features, rtol=5e-13, atol=5e-14)
    else:
        np.testing.assert_array_equal(
            actual.view(np.uint64), dense_features.view(np.uint64)
        )


@pytest.mark.parametrize("active", [17, 31, 32, 41])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("occupations", [(3,), (3, 2), (0, 3)])
def test_gpu_formed_gram_has_exact_original_density_features(
    occupied_probe: ct.CDLL,
    active: int,
    mapped: bool,
    occupations: tuple[int, ...],
) -> None:
    """A real GPU Gram producer and orthonormal frames feed both consumers."""
    rng = np.random.default_rng(20261104)
    nao, points, spins = active + 7 if mapped else active, 3, len(occupations)
    ids = np.arange(active, dtype=np.uintp) + (2 if mapped else 0)
    coefficients = np.ascontiguousarray(
        np.stack([np.linalg.qr(rng.normal(size=(nao, nao)))[0].T for _ in occupations])
    )
    counts = np.ascontiguousarray(occupations, dtype=np.int32)
    density = np.zeros((spins, nao, nao))
    assert (
        occupied_probe.form_density(
            coefficients.ctypes.data,
            counts.ctypes.data,
            nao,
            spins,
            density.ctypes.data,
        )
        == 0
    )
    for spin, count in enumerate(occupations):
        frame = coefficients[spin, :count].T
        np.testing.assert_allclose(
            density[spin],
            ((2.0 if spins == 1 else 1.0) * frame) @ frame.T,
            rtol=5e-15,
            atol=5e-16,
        )
    ao = np.ascontiguousarray(rng.normal(size=(4, points, active)) * 1e-4)
    actual, expected = np.zeros((spins, 5, points)), np.zeros((spins, 5, points))
    mapping = ids.ctypes.data if mapped else None
    assert (
        occupied_probe.occupied_probe(
            density.ctypes.data,
            coefficients.ctypes.data,
            ao.ctypes.data,
            mapping,
            nao,
            active,
            points,
            spins,
            occupations[0],
            occupations[1] if spins == 2 else 0,
            max(occupations),
            actual.ctypes.data,
        )
        == 0
    )
    assert (
        occupied_probe.matrix_probe(
            density.ctypes.data,
            ao.ctypes.data,
            mapping,
            nao,
            active,
            points,
            spins,
            expected.ctypes.data,
        )
        == 0
    )
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


class CompactTile(ct.Structure):
    """Native batch ABI; each tile retains its own point/AO/scratch strides."""

    _fields_ = [
        (name, ct.c_size_t)
        for name in (
            "begin",
            "count",
            "active",
            "map_offset",
            "ao_offset",
            "work_offset",
            "potential_offset",
        )
    ]


@pytest.mark.parametrize("occupations", [(3,), (3, 2), (0, 3)])
@pytest.mark.parametrize("scale", [1e-4, 1e165])
@pytest.mark.parametrize("provisional", [False, True])
def test_compact_gpu_batches_preserve_original_features_and_scratch_owners(
    occupied_probe: ct.CDLL,
    occupations: tuple[int, ...],
    scale: float,
    provisional: bool,
) -> None:
    """Run actual batch specializations, including capacity fallback in one group."""
    nao, spins = 48, len(occupations)
    rng = np.random.default_rng(20261105)
    coefficients = np.ascontiguousarray(
        np.stack([np.linalg.qr(rng.normal(size=(nao, nao)))[0].T for _ in occupations])
    )
    counts = np.ascontiguousarray(occupations, dtype=np.int32)
    density = np.zeros((spins, nao, nao))
    assert (
        occupied_probe.form_density(
            coefficients.ctypes.data,
            counts.ctypes.data,
            nao,
            spins,
            density.ctypes.data,
        )
        == 0
    )
    tiles = (CompactTile * 3)(
        CompactTile(11, 3, 17, 5, 7, 11, 0),
        CompactTile(14, 2, 33, 27, 218, spins * 204 + 22, 0),
        CompactTile(16, 4, 5, 65, 489, spins * 468 + 33, 0),
    )
    ao = np.zeros(569)
    ids = np.zeros(70, dtype=np.uintp)
    work = np.zeros(spins * 548 + 44)
    features = np.zeros(spins * 45 + 7)
    expected = np.full_like(features, -456.0)
    protected_work = np.ones(len(work), dtype=bool)
    for tile in tiles:
        mapping = np.sort(rng.choice(nao, tile.active, replace=False)).astype(np.uintp)
        ids[tile.map_offset : tile.map_offset + tile.active] = mapping
        panel = np.ascontiguousarray(
            rng.normal(size=(4, tile.count, tile.active)) * scale
        )
        ao[tile.ao_offset : tile.ao_offset + panel.size] = panel.ravel()
        dense = np.zeros((spins, 5, tile.count))
        status = occupied_probe.matrix_probe(
            density.ctypes.data,
            panel.ctypes.data,
            mapping.ctypes.data,
            nao,
            tile.active,
            tile.count,
            spins,
            dense.ctypes.data,
        )
        assert status == (1 if scale == 1e165 else 0)
        offset = (tile.begin - tiles[0].begin) * spins * 5
        expected[offset : offset + dense.size] = dense.ravel()
        protected_work[
            tile.work_offset : tile.work_offset + spins * 4 * tile.count * tile.active
        ] = False
    function = (
        occupied_probe.compact_proposal_probe
        if provisional
        else occupied_probe.compact_probe
    )
    status = function(
        tiles,
        len(tiles),
        density.ctypes.data,
        coefficients.ctypes.data,
        ao.ctypes.data,
        ids.ctypes.data,
        nao,
        spins,
        occupations[0],
        occupations[1] if spins == 2 else 0,
        features.ctypes.data,
        work.ctypes.data,
    )
    assert status == (1 if scale == 1e165 else 0)
    if scale == 1e165:
        np.testing.assert_array_equal(np.isfinite(features), np.isfinite(expected))
    elif provisional:
        np.testing.assert_allclose(features, expected, rtol=5e-13, atol=5e-14)
    else:
        np.testing.assert_array_equal(
            features.view(np.uint64), expected.view(np.uint64)
        )
    np.testing.assert_array_equal(work[protected_work], -123.0)


@pytest.mark.parametrize("active", [17, 32, 41])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("occupations", [(1,), (1, 1), (0, 1), (1, 0)])
def test_provisional_gpu_path_does_not_read_dense_matrix(
    occupied_probe: ct.CDLL, active: int, mapped: bool, occupations: tuple[int, ...]
) -> None:
    """Poison D to distinguish work avoidance from exact verification on device.

    This synthetic low-level witness deliberately violates Gram provenance;
    physical SCF publication must instead use the independent strict target.
    """
    nao, points, spins = active + 3 * mapped, 3, len(occupations)
    ids = np.arange(active, dtype=np.uintp) + int(mapped)
    coefficients = np.zeros((spins, nao, nao))
    for spin, occupied in enumerate(occupations):
        if occupied:
            coefficients[spin, 0, ids[0]] = 1.0
    ao = np.full((4, points, active), 0.125)
    density = np.full((spins, nao, nao), np.nan)
    actual = np.zeros((spins, 5, points))
    arguments = (
        density.ctypes.data,
        coefficients.ctypes.data,
        ao.ctypes.data,
        ids.ctypes.data if mapped else None,
        nao,
        active,
        points,
        spins,
        occupations[0],
        occupations[1] if spins == 2 else 0,
        1,
        actual.ctypes.data,
    )
    assert occupied_probe.proposal_probe(*arguments) == 0
    expected = np.zeros_like(actual)
    for spin, occupied in enumerate(occupations):
        expected[spin] = (
            (2.0 if spins == 1 else 1.0)
            * occupied
            * np.array([0.015625, 0.03125, 0.03125, 0.03125, 0.0234375])[:, None]
        )
    np.testing.assert_array_equal(actual, expected)
    assert occupied_probe.occupied_probe(*arguments) == 1


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("all_jets", [False, True])
def test_finite_orthonormal_underflow_uses_original_stored_density(
    occupied_probe: ct.CDLL, mapped: bool, all_jets: bool
) -> None:
    """Stored Gram underflow must not be undone by reassociating finite AO jets."""
    active, points, spins = 16, 1, 1
    nao = 23 if mapped else active
    ids = np.arange(active, dtype=np.uintp) + (3 if mapped else 0)
    for tiny, ao_scale, expected_rho in ((1e-200, 1e200, 0.0), (1.0, 0.25, 0.125)):
        frame = np.eye(nao)
        if tiny != 1.0:
            frame[np.ix_(ids[:2], ids[:2])] = [[tiny, 1.0], [1.0, -tiny]]
        occupied = frame[:, ids[:1]]
        stored = np.ascontiguousarray((2.0 * occupied) @ occupied.T)
        coefficients = np.ascontiguousarray(
            frame[:, np.r_[int(ids[0]), np.delete(np.arange(nao), ids[0])]].T
        )
        counts = np.array([1], dtype=np.int32)
        gpu_stored = np.zeros_like(stored)
        assert (
            occupied_probe.form_density(
                coefficients.ctypes.data,
                counts.ctypes.data,
                nao,
                spins,
                gpu_stored.ctypes.data,
            )
            == 0
        )
        np.testing.assert_array_equal(gpu_stored, stored)
        stored = gpu_stored
        ao = np.zeros((4, points, active))
        ao[0, 0, 0] = ao_scale
        if all_jets:
            for jet in range(1, 4):
                ao[jet, 0, 0] = (jet + 1) * ao_scale
        actual = np.zeros((spins, 5, points))
        expected = np.zeros_like(actual)
        mapping = ids.ctypes.data if mapped else None
        assert (
            occupied_probe.matrix_probe(
                stored.ctypes.data,
                ao.ctypes.data,
                mapping,
                nao,
                active,
                points,
                spins,
                expected.ctypes.data,
            )
            == 0
        )
        assert (
            occupied_probe.occupied_probe(
                stored.ctypes.data,
                coefficients.ctypes.data,
                ao.ctypes.data,
                mapping,
                nao,
                active,
                points,
                spins,
                1,
                0,
                1,
                actual.ctypes.data,
            )
            == 0
        )
        assert np.isfinite(actual).all()
        assert expected[0, 0, 0] == expected_rho
        np.testing.assert_array_equal(actual, expected)


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
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        if scale == 1e6:
            assert np.max(np.abs(actual)) > 0.001


PREAMBLE = r"""
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <vector>
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
extern "C" int form_density(const double* coefficients, const std::int32_t* occupations,
    size_t nao, size_t spins, double* output) {
  try {
    using generativeqc_tensor::cuda_check;
    Stream stream;
    DeviceBuffer<double> frame(spins * nao * nao), density(spins * nao * nao);
    DeviceBuffer<std::int32_t> counts(spins);
    cuda_check(cudaMemcpyAsync(frame.value, coefficients, spins * nao * nao * sizeof(double),
                               cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(counts.value, occupations, spins * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream.value));
    const auto blocks = generativeqc_tensor::blocks(spins * nao * nao, 128);
    if (spins == 1)
      generativeqc::scf::generated::occupied_density_kernel<2><<<blocks, 128, 0, stream.value>>>(
          1, spins, nao, counts.value, frame.value, nullptr, density.value);
    else
      generativeqc::scf::generated::occupied_density_kernel<1><<<blocks, 128, 0, stream.value>>>(
          1, spins, nao, counts.value, frame.value, nullptr, density.value);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(output, density.value, spins * nao * nao * sizeof(double),
                               cudaMemcpyDeviceToHost, stream.value));
    cuda_check(cudaStreamSynchronize(stream.value));
    return 0;
  } catch (...) {
    return 99;
  }
}
extern "C" int compact_probe(const CudaXcCompactTile* tiles, size_t tile_count,
    const double* density, const double* coefficients, const double* ao, const size_t* ids,
    size_t nao, size_t spins, size_t alpha, size_t beta, double* output, double* scratch) {
  try {
    using generativeqc_tensor::cuda_check;
    size_t ao_count=0, work_count=0, map_count=0, feature_count=0, max_active=0, max_points=0;
    const size_t begin=tiles[0].begin;
    for (size_t index=0; index<tile_count; ++index) {
      const auto& tile=tiles[index];
      ao_count=std::max(ao_count,tile.ao_offset+4*tile.count*tile.active);
      work_count=std::max(work_count,tile.work_offset+spins*4*tile.count*tile.active);
      map_count=std::max(map_count,tile.map_offset+tile.active);
      feature_count=std::max(feature_count,(tile.begin+tile.count-begin)*spins*5);
      max_active=std::max(max_active,tile.active);
      max_points=std::max(max_points,tile.count);
    }
    work_count+=11;
    feature_count+=7;
    Stream stream;
    DeviceBuffer<CudaXcCompactTile> metadata(tile_count);
    DeviceBuffer<double> matrix(spins*nao*nao), frame(spins*nao*nao), orbitals(ao_count),
        work(work_count), features(feature_count);
    DeviceBuffer<size_t> map(map_count);
    DeviceBuffer<int> error(1);
    std::vector<double> work_initial(work_count,-123.0), feature_initial(feature_count,-456.0);
    cuda_check(cudaMemsetAsync(error.value,0,sizeof(int),stream.value));
    cuda_check(cudaMemcpyAsync(metadata.value,tiles,tile_count*sizeof(CudaXcCompactTile),
                               cudaMemcpyHostToDevice,stream.value));
    cuda_check(cudaMemcpyAsync(matrix.value,density,spins*nao*nao*sizeof(double),
                               cudaMemcpyHostToDevice,stream.value));
    cuda_check(cudaMemcpyAsync(frame.value,coefficients,spins*nao*nao*sizeof(double),
                               cudaMemcpyHostToDevice,stream.value));
    cuda_check(cudaMemcpyAsync(orbitals.value,ao,ao_count*sizeof(double),
                               cudaMemcpyHostToDevice,stream.value));
    cuda_check(cudaMemcpyAsync(map.value,ids,map_count*sizeof(size_t),
                               cudaMemcpyHostToDevice,stream.value));
    cuda_check(cudaMemcpyAsync(work.value,work_initial.data(),work_count*sizeof(double),
                               cudaMemcpyHostToDevice,stream.value));
    cuda_check(cudaMemcpyAsync(features.value,feature_initial.data(),feature_count*sizeof(double),
                               cudaMemcpyHostToDevice,stream.value));
    CudaXcOccupiedDensity witness{matrix.value,frame.value,nao,spins,{alpha,beta},1};
    batch_density_products<double,true><<<dim3((max_active+15)/16,(max_points+15)/16,
        tile_count*spins*4),dim3(16,16),0,stream.value>>>(metadata.value,matrix.value,
            orbitals.value,work.value,nao,spins,4,map.value,error.value,witness);
    cuda_check(cudaGetLastError());
    batch_density_features<true><<<dim3((spins*max_points*32+127)/128,tile_count),
        128,0,stream.value>>>(metadata.value,orbitals.value,work.value,features.value,
            begin,spins,4,4,5,2,map.value,error.value,witness);
    cuda_check(cudaGetLastError());
    int failure=0;
    cuda_check(cudaMemcpyAsync(output,features.value,feature_count*sizeof(double),
                               cudaMemcpyDeviceToHost,stream.value));
    cuda_check(cudaMemcpyAsync(scratch,work.value,work_count*sizeof(double),
                               cudaMemcpyDeviceToHost,stream.value));
    cuda_check(cudaMemcpyAsync(&failure,error.value,sizeof(int),cudaMemcpyDeviceToHost,stream.value));
    cuda_check(cudaStreamSynchronize(stream.value));
    return failure;
  } catch (...) {
    return 99;
  }
}
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
    tiled_density_product<false><<<dim3((active + 15) / 16, (points + 15) / 16, spins * 4),
        dim3(16, 16), 0, stream.value>>>(matrix.value, orbitals.value, active, points, 4,
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

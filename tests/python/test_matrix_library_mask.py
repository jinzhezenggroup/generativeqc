"""Compile the complete matrix adapter with vendor/queue doubles.

NumPy independently checks routing, strides, scale and inactive output bytes.
The selected-copy kernel's actual body executes with host launch indices. Queue
replay changes the pointed-to mask and inputs without resubmitting the adapter.
These are host contract tests, not CUDA/Graph/cuBLAS device qualification.
"""

from __future__ import annotations

import ctypes
import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
PTR = ctypes.c_void_p


def _kernel(source: str) -> str:
    start = source.index("__global__ void copy_selected_matrices_kernel(")
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def adapter(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> ctypes.CDLL:
    folder = tmp_path_factory.mktemp("matrix-mask")
    fixture = ROOT / "tests/native/fixtures/matrix_mask"
    copy = folder / "copy.cpp"
    copy.write_text(
        '#include "cuda_runtime.h"\n#include <cstdint>\n'
        "#define __global__\ndim3 blockIdx, blockDim, threadIdx;\n"
        + _kernel((ROOT / "src/scf/cuda/scf_matrix_kernels.cu").read_text())
        + """
void host_selected_copy(dim3 grid, dim3 block, int batch, int spins, int n,
                        const std::uint8_t* mask, const double* source, double* output) {
  blockDim = block;
  for (blockIdx.x = 0; blockIdx.x < grid.x; ++blockIdx.x)
    for (threadIdx.x = 0; threadIdx.x < block.x; ++threadIdx.x)
      copy_selected_matrices_kernel(batch, spins, n, mask, source, output);
}
"""
    )
    library = required_native_cxx.build_shared(
        [ROOT / "src/scf/cuda/matrix_library.cpp", fixture / "mock.cpp", copy],
        folder / "probe.so",
        compile_args=(
            "-std=c++17",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{fixture}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    probe = ctypes.CDLL(str(library))
    probe.probe_submit.argtypes = (
        [ctypes.c_int] * 7
        + [PTR] * 5
        + [ctypes.c_size_t, ctypes.c_int, ctypes.c_double]
    )
    probe.probe_submit.restype = ctypes.c_int
    probe.probe_reset.argtypes = [ctypes.c_int, ctypes.c_int]
    probe.probe_replay.argtypes = []
    probe.probe_stat.argtypes = [ctypes.c_int]
    probe.probe_stat.restype = ctypes.c_int
    return probe


def _pointer(array: np.ndarray | None) -> int | None:
    return None if array is None else array.ctypes.data


def _submit(
    adapter: ctypes.CDLL,
    spec: tuple[int, ...],
    left: np.ndarray | None,
    right: np.ndarray | None,
    mask: np.ndarray | None,
    output: np.ndarray | None,
    scratch: np.ndarray | None,
    *,
    capacity: int | None = None,
    library: int = 1,
    scale: float = 1.0,
) -> int:
    return adapter.probe_submit(
        *spec,
        *map(_pointer, (left, right, mask, output, scratch)),
        len(scratch) if capacity is None and scratch is not None else (capacity or 0),
        library,
        scale,
    )


def _matrices(seed: int, batch: int, states: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logical = rng.normal(size=(batch, states, n, n))
    return logical.transpose(0, 1, 3, 2).copy().ravel()


def _expected(
    spec: tuple[int, ...],
    left: np.ndarray | None,
    right: np.ndarray | None,
    scale: float,
) -> np.ndarray:
    spin_api, batch, spins, n, left_spin, transpose, right_spin = spec
    l = left.reshape(batch, spins if left_spin else 1, n, n).transpose(0, 1, 3, 2)
    r = right.reshape(batch, spins if right_spin else 1, n, n).transpose(0, 1, 3, 2)
    if transpose:
        l = l.swapaxes(-1, -2)
    result = (l @ r) * (1.0 if spin_api else scale)
    result = np.broadcast_to(result, (batch, spins, n, n))
    return result.transpose(0, 1, 3, 2).copy().reshape(batch, -1)


SPECS = [
    (api, batch, spins, n, ls, trans, rs)
    for n, batch, trans in itertools.product((1, 3, 17), (1, 3), (0, 1))
    for api, spins, ls, rs in [(0, 1, 0, 0)]
    + [
        (1, spins, ls, rs)
        for spins, ls, rs in itertools.product((1, 2), (0, 1), (0, 1))
    ]
]


@pytest.mark.parametrize("spec", SPECS)
def test_masked_vendor_publication_and_dynamic_replay(
    adapter: ctypes.CDLL, spec: tuple[int, ...]
) -> None:
    api, batch, spins, n, ls, _, rs = spec
    left = _matrices(101, batch, spins if ls else 1, n)
    right = _matrices(203, batch, spins if rs else 1, n)
    output = np.full(batch * spins * n * n, 9876.125)
    scratch = np.full_like(output, np.nan)
    mask = np.ones(batch, dtype=np.uint8)
    scale = -1.25
    adapter.probe_reset(0, 0)
    assert _submit(adapter, spec, left, right, mask, output, scratch, scale=scale) == 0
    assert np.all(output == 9876.125), "submission must not synchronously publish"
    assert adapter.probe_stat(0) == (spins if api else 1)
    assert adapter.probe_stat(1) == ((spins if api else 1) if batch > 1 else 0)
    assert adapter.probe_stat(2) == 1 and adapter.probe_stat(3) == 0
    assert adapter.probe_stat(4) == 0, "no stream synchronization"
    # The same queued operations observe changed device mask values each replay.
    for mode in ("active", "mixed", "inactive", "failed", "reactivated"):
        mask[:] = 1
        left[:] = _matrices(101, batch, spins if ls else 1, n)
        if mode in ("inactive", "failed"):
            mask[:] = 0
        elif mode == "mixed":
            mask[::2] = 0
        elif mode == "reactivated":
            mask[:] = 255  # Any nonzero mask byte is active.
            left *= 0.5
        if mode == "failed":
            left[:] = np.nan
        # Noncanonical NaN payload, negative zero and finite sentinels must be
        # retained bit-for-bit for inactive systems, not recomputed as beta*C.
        bits = output.view(np.uint64)
        bits[:] = np.resize(
            np.array(
                [0x8000000000000000, 0x7FF8000000000123, 0x40C34A1000000000],
                dtype=np.uint64,
            ),
            len(bits),
        )
        before = bits.copy().reshape(batch, -1)
        adapter.probe_replay()
        actual = output.reshape(batch, -1)
        active = mask != 0
        expected = _expected(spec, left, right, scale)
        np.testing.assert_allclose(
            actual[active], expected[active], rtol=2e-13, atol=2e-13
        )
        np.testing.assert_array_equal(bits.reshape(batch, -1)[~active], before[~active])


@pytest.mark.parametrize("spec", [SPECS[0], (1, 3, 2, 3, 1, 1, 0)])
def test_unmasked_library_and_native_do_not_require_scratch(
    adapter: ctypes.CDLL, spec: tuple[int, ...]
) -> None:
    _, batch, spins, n, ls, _, rs = spec
    left = _matrices(9, batch, spins if ls else 1, n)
    right = _matrices(7, batch, spins if rs else 1, n)
    output = np.full(batch * spins * n * n, 7654.0)
    adapter.probe_reset(0, 0)
    assert _submit(adapter, spec, left, right, None, output, None, scale=1.5) == 0
    assert adapter.probe_stat(2) == 0 and adapter.probe_stat(3) == 0
    adapter.probe_replay()
    np.testing.assert_allclose(
        output.reshape(batch, -1),
        _expected(spec, left, right, 1.5),
        rtol=2e-13,
        atol=2e-13,
    )
    adapter.probe_reset(0, 0)
    mask = np.zeros(batch, dtype=np.uint8)
    assert _submit(adapter, spec, left, right, mask, output, None, library=0) == 0
    assert (
        adapter.probe_stat(0) == 0
        and adapter.probe_stat(2) == 0
        and adapter.probe_stat(3) == 1
    )


@pytest.mark.parametrize("api", [0, 1])
@pytest.mark.parametrize(
    "bad", ["missing", "short", "left", "right", "output", "partial", "mask"]
)
def test_invalid_scratch_rejects_before_enqueue(
    adapter: ctypes.CDLL, api: int, bad: str
) -> None:
    spec = (api, 3, 2 if api else 1, 3, api, 0, api)
    count = 3 * (2 if api else 1) * 9
    left = np.ones(count)
    right = np.ones(count)
    output = np.full(count + 1, 123.0)
    scratch = np.zeros(count)
    mask_storage = np.zeros(count)
    mask = mask_storage.view(np.uint8)[:3]
    selected = {
        "missing": None,
        "short": scratch,
        "left": left,
        "right": right,
        "output": output,
        "partial": output[1:],
        "mask": mask_storage,
    }[bad]
    adapter.probe_reset(0, 0)
    status = _submit(
        adapter,
        spec,
        left,
        right,
        mask,
        output,
        selected,
        capacity=count - 1 if bad == "short" else count,
    )
    assert status == 1
    assert [adapter.probe_stat(i) for i in range(6)] == [0] * 6
    np.testing.assert_array_equal(output, 123.0)


@pytest.mark.parametrize("failure", ["gemm1", "gemm2", "copy"])
def test_submission_failures_do_not_publish_partial_masked_results(
    adapter: ctypes.CDLL, failure: str
) -> None:
    spec = (1, 3, 2, 3, 1, 0, 1)
    left = np.ones(54)
    right = np.ones(54)
    output = np.full(54, 123.0)
    scratch = np.zeros(54)
    mask = np.ones(3, dtype=np.uint8)
    adapter.probe_reset(
        1 if failure == "gemm1" else 2 if failure == "gemm2" else 0,
        42 if failure == "copy" else 0,
    )
    assert _submit(adapter, spec, left, right, mask, output, scratch) == 6
    adapter.probe_replay()
    np.testing.assert_array_equal(output, 123.0)
    assert adapter.probe_stat(2) == (1 if failure == "copy" else 0)
    assert adapter.probe_stat(3) == 0 and adapter.probe_stat(4) == 0


@pytest.mark.parametrize(
    "batch,spins,n",
    [
        (0, 1, 1),
        (-1, 1, 1),
        (1, 0, 1),
        (1, -1, 1),
        (1, 1, 0),
        (1, 1, -1),
        (2**31 - 1, 2**31 - 1, 2**31 - 1),
        (1, 1, 262144),
    ],
)
def test_invalid_shape_rejects_before_enqueue(
    adapter: ctypes.CDLL, batch: int, spins: int, n: int
) -> None:
    adapter.probe_reset(0, 0)
    spec = (1, batch, spins, n, 1, 0, 1)
    assert _submit(adapter, spec, None, None, None, None, None) == 1
    assert [adapter.probe_stat(i) for i in range(6)] == [0] * 6


def _definition(source: str, signature: str) -> str:
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def layouts(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> ctypes.CDLL:
    folder = tmp_path_factory.mktemp("matrix-mask-layout")
    fixture = ROOT / "tests/native/fixtures/matrix_mask"
    ks = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    unit = folder / "layout.cpp"
    unit.write_text(
        """
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "scf/cuda/arena.hpp"
#include "scf/cuda/matrix_library.hpp"
#include "dft/cuda_ks_kernels.hpp"
using namespace generativeqc::scf::cuda_execution;
namespace cuda_ks_detail = generativeqc::dft::cuda_ks_detail;
constexpr unsigned kCudaKsChunkCapacity = 2;
"""
        + _definition(ks, "std::size_t product(")
        + "\n"
        + _definition(ks, "std::size_t sum(")
        + "\n"
        + _definition(ks, "struct KsStateStorage")
        + ";\n"
        + """
extern "C" void ks_layout(int n, int spins, int history, int exchange, int range, int incremental,
                           std::size_t* result) {
  KsStateStorage dry, actual;
  const auto bytes = dry.partition(n, spins, history, exchange, range, incremental, true, nullptr);
  std::vector<double> storage((bytes + 7) / 8);
  const auto actual_bytes = actual.partition(n, spins, history, exchange, range, incremental,
                                              true, storage.data());
  assert(bytes == actual_bytes && dry.masked_matrix_output == nullptr);
  const auto begin = reinterpret_cast<std::uintptr_t>(actual.masked_matrix_output);
  const auto end = reinterpret_cast<std::uintptr_t>(actual.cold_seed);
  const auto base = reinterpret_cast<std::uintptr_t>(storage.data());
  assert(begin >= base && end <= base + bytes && end >= begin);
  for (auto* p : {actual.hcore, actual.overlap, actual.x, actual.j, actual.density,
                  actual.proposal, actual.warm, actual.warm_orbitals, actual.fock,
                  actual.residual, actual.tmp1, actual.tmp2, actual.effective,
                  actual.cold_seed, actual.fock_history, actual.residual_history,
                  actual.final_coefficients}) {
    const auto address = reinterpret_cast<std::uintptr_t>(p);
    assert(address < begin || address >= end);
  }
  result[0] = bytes;
  result[1] = end - begin;
  result[2] = begin - base;
}
extern "C" void hf_layout(int batch, int n, int spins, int history, std::size_t* result) {
  ArenaLayout native{}, library{};
  for (int library_route = 0; library_route < 2; ++library_route) {
    auto& layout = library_route ? library : native;
    assert(make_layout(batch, n, n, 2, 2, 3, 1, 0, 27, 0, 0, 0, 0, 0, 0, 0,
                        6, history, 0, spins, false, false, false, false, false,
                        false, false, false, layout, false, library_route));
  }
  result[0] = native.bytes;
  result[1] = library.bytes;
  result[2] = library.masked_matrix_output;
  result[3] = library.eigensystem;
  assert(native.masked_matrix_output == native.eigensystem);
  assert(library.masked_matrix_output >= library.temporary + batch * spins * n * n * sizeof(double));
}
"""
    )
    library = required_native_cxx.build_shared(
        [ROOT / "src/scf/cuda/arena.cpp", unit],
        folder / "layouts.so",
        compile_args=(
            "-std=c++20",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{fixture}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
    )
    result = ctypes.CDLL(str(library))
    result.ks_layout.argtypes = [ctypes.c_int] * 6 + [PTR]
    result.hf_layout.argtypes = [ctypes.c_int] * 4 + [PTR]
    return result


@pytest.mark.parametrize(
    "n,spins,history", list(itertools.product((1, 16, 17, 33), (1, 2), (1, 8)))
)
def test_ks_dry_run_charges_disjoint_bounded_mask_workspace(
    layouts: ctypes.CDLL, n: int, spins: int, history: int
) -> None:
    for exchange, range_correction, incremental in itertools.product((0, 1), repeat=3):
        result = (ctypes.c_size_t * 3)()
        layouts.ks_layout(
            n, spins, history, exchange, range_correction, incremental, result
        )
        assert result[1] == (n * n * spins * 8 if n >= 17 else 0)
        assert result[2] + result[1] <= result[0]


@pytest.mark.parametrize(
    "batch,n,spins,history",
    list(itertools.product((1, 3), (1, 17, 33), (1, 2), (1, 8))),
)
def test_hf_arena_charges_exact_span_only_for_library(
    layouts: ctypes.CDLL, batch: int, n: int, spins: int, history: int
) -> None:
    result = (ctypes.c_size_t * 4)()
    layouts.hf_layout(batch, n, spins, history, result)
    assert result[1] - result[0] == batch * spins * n * n * 8
    assert result[3] - result[2] == batch * spins * n * n * 8
    assert result[3] <= result[1]


def test_signed_cuda_grid_boundary_without_allocations_or_replay(
    adapter: ctypes.CDLL,
) -> None:
    # Synthetic disjoint address ranges are admitted but never dereferenced:
    # the vendor/transport doubles only retain operations until explicit replay.
    addresses = [1 << 40, 2 << 40, 5 << 40, 3 << 40, 4 << 40]
    batch = 2**31 - 1
    elements = batch * 2 * 4 * 4
    adapter.probe_reset(0, 0)
    assert (
        adapter.probe_submit(1, batch, 2, 4, 1, 0, 1, *addresses, elements, 1, 1.0) == 0
    )
    assert adapter.probe_stat(0) == 2 and adapter.probe_stat(2) == 1
    adapter.probe_reset(0, 0)
    assert adapter.probe_submit(0, 2**30, 1, 8, 0, 0, 0, *addresses, 2**36, 1, 1.0) == 1
    assert [adapter.probe_stat(i) for i in range(6)] == [0] * 6

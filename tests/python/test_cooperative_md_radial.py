"""Execute the emitted MD stages with real host-thread publication barriers.

These gates qualify shared arithmetic, tails, buffer lifetime and bounds, not
CUDA performance. The independent closed-derivative oracle uses 80-digit Boys
integrals and factorial multiplicities, not the production MD recurrence.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import mpmath as mp
import numpy as np
import pytest
from generativeqc_compiler.integral.md_j_engine_cuda import (
    build_md_radial_stage,
    cooperative_md_radial_workspace,
    emit_cooperative_md_radial_support,
)
from test_md_j_engine_math import _closed_radial

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from conftest import NativeCxx


@pytest.fixture(scope="module")
def cooperative_radial(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Any:
    """Run actual generated CUDA function bodies with a C++20 CTA emulator."""
    folder = tmp_path_factory.mktemp("cooperative-md-radial")
    source: Path = folder / "radial.cpp"
    obj, library = folder / "radial.o", folder / "radial.so"
    source.write_text(
        r"""
#include <algorithm>
#include <barrier>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>
#define __device__
#define __constant__
struct Dimension { unsigned x; };
thread_local Dimension threadIdx;
Dimension blockDim;
thread_local std::barrier<>* publication;
thread_local bool warp_only, wrong_scope;
void block_publication() {
  wrong_scope |= warp_only;
  publication->arrive_and_wait();
}
#define __syncthreads() block_publication()
#define __syncwarp(...) publication->arrive_and_wait()
template<class Value> Value __ldg(const Value* pointer) { return *pointer; }
template<class Scalar> struct Vec3 { Scalar x, y, z; };
"""
        + emit_cooperative_md_radial_support()
        + r"""
template<unsigned Order>
int run(unsigned lanes, double rho, const double* displacement,
        const double* boys, double* output) {
  constexpr unsigned Count = (Order+1U)*(Order+2U)*(Order+3U)/6U;
  constexpr unsigned Elements = 2U*Count + Order + 1U;
  std::vector<double> guarded(Elements+2U, std::nan(""));
  guarded.front() = guarded.back() = 918273.0;
  blockDim.x = lanes;
  std::barrier barrier(static_cast<std::ptrdiff_t>(lanes));
  std::vector<std::thread> workers;
  for (unsigned rank=0; rank<lanes; ++rank) workers.emplace_back([&,rank] {
    threadIdx.x=rank;
    publication=&barrier;
    const auto* roots=cooperative_md_radial<Order>(
        rho, {displacement[0],displacement[1],displacement[2]}, boys, guarded.data()+1U);
    for (unsigned root=rank; root<Count; root+=lanes) output[root]=roots[root];
  });
  for (auto& worker: workers) worker.join();
  return guarded.front()!=918273.0 || guarded.back()!=918273.0;
}
extern "C" int evaluate(unsigned order, unsigned lanes, double rho,
                        const double* displacement, const double* boys, double* output) {
  if (!lanes || lanes>256) return 2;
  if (order==7) return run<7>(lanes,rho,displacement,boys,output);
  if (order==8) return run<8>(lanes,rho,displacement,boys,output);
  return 3;
}
template<unsigned Order>
int run_warps(unsigned warps, const double* rhos, const double* displacements,
              const double* boys, double* output) {
  constexpr unsigned Count = (Order+1U)*(Order+2U)*(Order+3U)/6U;
  constexpr unsigned Elements = 2U*Count + Order + 1U;
  std::vector<double> guarded(warps*(Elements+2U), std::nan(""));
  std::vector<unsigned> scope_faults(warps*32U);
  std::vector<std::unique_ptr<std::barrier<>>> barriers;
  for (unsigned warp=0; warp<warps; ++warp) {
    guarded[warp*(Elements+2U)] = guarded[(warp+1U)*(Elements+2U)-1U] = 918273.0;
    barriers.push_back(std::make_unique<std::barrier<>>(32));
  }
  blockDim.x = warps*32U;
  std::vector<std::thread> workers;
  for (unsigned rank=0; rank<blockDim.x; ++rank) workers.emplace_back([&,rank] {
    const unsigned warp=rank/32U, lane=rank%32U;
    threadIdx.x=rank;
    warp_only=true;
    publication=barriers[warp].get();
    const auto* roots=cooperative_md_radial<Order,true>(rhos[warp],
        {displacements[warp*3U],displacements[warp*3U+1U],displacements[warp*3U+2U]},
        boys+warp*(Order+1U), guarded.data()+warp*(Elements+2U)+1U);
    for (unsigned root=lane; root<Count; root+=32U) output[warp*Count+root]=roots[root];
    scope_faults[rank]=wrong_scope;
  });
  for (auto& worker: workers) worker.join();
  if (std::any_of(scope_faults.begin(),scope_faults.end(),[](unsigned fault) { return fault!=0; }))
    return 4;
  for (unsigned warp=0; warp<warps; ++warp)
    if (guarded[warp*(Elements+2U)]!=918273.0 ||
        guarded[(warp+1U)*(Elements+2U)-1U]!=918273.0) return 1;
  return 0;
}
extern "C" int evaluate_warps(unsigned order, unsigned warps, const double* rhos,
                              const double* displacements, const double* boys, double* output) {
  if (!warps || warps>8U) return 2;
  if (order==5U) return run_warps<5>(warps,rhos,displacements,boys,output);
  if (order==6U) return run_warps<6>(warps,rhos,displacements,boys,output);
  return 3;
}
"""
    )
    required_native_cxx.compile_object(
        source, obj, args=("-std=c++20", "-O2", "-fPIC", "-pthread")
    )
    required_native_cxx.link([obj], library, args=("-shared", "-pthread"))
    owner = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    owner.evaluate.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_double,
        array,
        array,
        array,
    ]
    owner.evaluate.restype = ctypes.c_int
    owner.evaluate_warps.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        array,
        array,
        array,
        array,
    ]
    owner.evaluate_warps.restype = ctypes.c_int
    return owner


@pytest.mark.parametrize("order", (5, 6))
@pytest.mark.parametrize("warps", (1, 2, 4, 8))
def test_warp_radial_has_independent_storage_and_publications(
    cooperative_radial: Any, order: int, warps: int
) -> None:
    """Concurrent warps use distinct inputs, roots and canaries, never a CTA barrier."""
    rhos = np.asarray([0.17 * (warp + 1) for warp in range(warps)])
    displacements = np.asarray(
        [(0.2 + warp, -0.7, 0.1 * warp) for warp in range(warps)]
    )
    moments = []
    expected = []
    bounds = []
    with mp.workdps(80):
        for rho, displacement in zip(rhos, displacements, strict=True):
            argument = mp.mpf(rho) * sum(mp.mpf(value) ** 2 for value in displacement)
            moments.append(
                [
                    mp.gammainc(degree + mp.mpf("0.5"), 0, argument)
                    / (2 * argument ** (degree + mp.mpf("0.5")))
                    for degree in range(order + 1)
                ]
            )
            roots, scales = _closed_radial(order, rho, tuple(displacement))
            expected.append(roots)
            bounds.append(scales)
    boys = np.asarray(moments, dtype=np.float64)
    saved_boys = boys.copy()
    expected_array = np.asarray(expected)
    actual = np.full_like(expected_array, np.nan)
    assert (
        cooperative_radial.evaluate_warps(
            order, warps, rhos, displacements, boys, actual
        )
        == 0
    )
    np.testing.assert_array_equal(boys, saved_boys)
    assert np.all(np.isfinite(actual))
    assert np.all(
        np.abs(actual - expected_array) <= 5e-14 * np.asarray(bounds) + 1e-290
    )


@pytest.mark.parametrize("order", (7, 8))
@pytest.mark.parametrize("lanes", (1, 32, 256))
@pytest.mark.parametrize(
    ("rho", "displacement"),
    (
        (0.0, (0.3, -0.8, 0.6)),
        (0.37, (0.0, 0.0, 0.0)),
        (1e-10, (3e4, -8e4, 6e4)),
        (100.0, (0.03, -0.08, 0.06)),
    ),
)
def test_cooperative_radial_matches_independent_closed_derivatives(
    cooperative_radial: Any,
    order: int,
    lanes: int,
    rho: float,
    displacement: tuple[float, float, float],
) -> None:
    """Inactive lanes retire every stage; all roots and canaries must survive."""
    with mp.workdps(80):
        argument = mp.mpf(rho) * sum(mp.mpf(value) ** 2 for value in displacement)
        boys = np.asarray(
            [
                mp.mpf(1) / (2 * degree + 1)
                if argument == 0
                else mp.gammainc(degree + mp.mpf("0.5"), 0, argument)
                / (2 * argument ** (degree + mp.mpf("0.5")))
                for degree in range(9)
            ],
            dtype=np.float64,
        )
    saved_boys = boys.copy()
    expected, bounds = _closed_radial(order, rho, displacement)
    actual = np.full_like(expected, np.nan)
    assert (
        cooperative_radial.evaluate(
            order, lanes, rho, np.asarray(displacement), boys, actual
        )
        == 0
    )
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(boys, saved_boys)
    assert np.all(np.abs(actual - expected) <= 5e-14 * bounds + 1e-290)


@pytest.mark.parametrize("stage", range(1, 12))
def test_radial_stage_reads_only_complete_previous_domain(stage: int) -> None:
    """Every dependency fits the prior publication, including compacted slots."""
    previous_count = stage * (stage + 1) * (stage + 2) // 6
    current_count = (stage + 1) * (stage + 2) * (stage + 3) // 6
    steps = build_md_radial_stage(stage)
    assert tuple(step.destination for step in steps) == tuple(range(1, current_count))
    for step in steps:
        assert 0 <= step.first < previous_count
        assert step.second is None or 0 <= step.second < previous_count
        assert 0 <= step.axis < 3
        assert (step.second is None) == (step.multiplicity == 0)
    if stage <= 8:
        assert cooperative_md_radial_workspace(stage) == 2 * current_count + stage + 1


@pytest.mark.parametrize("order", (-1, 0, 9, True, 7.0))
def test_cooperative_workspace_rejects_unsupported_domains(order: int) -> None:
    with pytest.raises(ValueError, match="between one and eight"):
        cooperative_md_radial_workspace(order)

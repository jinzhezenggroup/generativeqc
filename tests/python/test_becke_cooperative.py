"""Host-thread execution of the actual cooperative helper, never a GPU benchmark."""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.method.stationary_resources import (
    plan_stationary_cuda_resources,
)
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials

HARNESS = r"""
#include <atomic>
#include <barrier>
#include <cstdint>
#include <thread>
#include <vector>
static std::atomic<size_t> pairs_evaluated, norms_evaluated, logs_evaluated;
static auto counted_log(double x) { ++logs_evaluated; return local_log(x); }
static auto counted_pair(double x) { ++pairs_evaluated; return local_becke(x); }
static auto counted_norm(double x,double y,double z) {
  ++norms_evaluated; return local_norm(x,y,z);
}
struct Team {
  size_t lane, lanes;
  std::barrier<>* barrier;
  size_t rank() const { return lane; }
  size_t size() const { return lanes; }
  void sync() const { barrier->arrive_and_wait(); }
};
extern "C" int run(const double* points,size_t np,const double* centers,size_t na,
    const int64_t* owners,const double* seeds,bool cached,size_t threads,double* output,
    size_t* counts) {
  using namespace generativeqc_grid_adjoint;
  pairs_evaluated=norms_evaluated=logs_evaluated=0;
  std::vector<CenterPair> geometry(na*(na-1)/2);
  auto* cached_pairs=cached && !geometry.empty()?geometry.data():nullptr;
  if(!prepare_center_geometry(centers,na,1e-12,cached_pairs,counted_norm,local_ratio_geometry))
    return -2;
  std::vector<double> gradient(3*na+2,0),logs(na),products(na),bar_product(na),bar_distance(na);
  gradient.front()=gradient.back()=987654;
  std::vector<size_t> zeros(na);
  std::vector<std::array<double,4>> distances(na);
  std::vector<PointPair> states(geometry.size()+2);
  states.front().factor={987654,987654}; states.back().factor={987654,987654};
  std::atomic<bool> valid{true};
  if(threads) {
    std::barrier barrier(static_cast<std::ptrdiff_t>(threads));
    std::vector<std::thread> workers;
    for(size_t lane=0;lane<threads;++lane) workers.emplace_back([&,lane]{
      for(size_t p=0;p<np;++p)
        if(!contract_point_cooperative(points+3*p,centers,na,owners[p],seeds[p],
            gradient.data()+1,logs.data(),products.data(),bar_product.data(),bar_distance.data(),
            zeros.data(),distances.data(),states.data()+1,Team{lane,threads,&barrier},counted_norm,
            local_ratio,counted_log,counted_pair,cached_pairs,local_ratio_prepared)) {
          valid=false; break;
        }
    });
    for(auto& worker:workers) worker.join();
  } else {
    for(size_t p=0;p<np;++p)
      if(!contract_point_prepared(points+3*p,centers,na,owners[p],seeds[p],gradient.data()+1,
          logs.data(),products.data(),bar_product.data(),bar_distance.data(),zeros.data(),
          distances.data(),counted_norm,local_ratio,counted_log,counted_pair,cached_pairs,
          local_ratio_prepared)) { valid=false; break; }
  }
  if(gradient.front()!=987654 || gradient.back()!=987654 ||
     states.front().factor[0]!=987654 || states.back().factor[0]!=987654) return -3;
  for(size_t i=1;i<=3*na;++i) if(!std::isfinite(gradient[i])) valid=false;
  if(!valid) return -2;
  std::copy(gradient.begin()+1,gradient.end()-1,output);
  counts[0]=pairs_evaluated; counts[1]=norms_evaluated; counts[2]=logs_evaluated;
  return 0;
}
"""


@pytest.fixture(scope="module", params=[1, 3, 5])
def helper(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ct.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("becke-cooperative")
    source = directory / "probe.cpp"
    source.write_text(emit_grid_adjoint() + emit_grid_partials(request.param) + HARNESS)
    library = directory / "probe.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-pthread",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = ct.CDLL(str(library))
    pointer = ct.POINTER(ct.c_double)
    result.run.argtypes = [
        pointer,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
        ct.POINTER(ct.c_int64),
        pointer,
        ct.c_bool,
        ct.c_size_t,
        pointer,
        ct.POINTER(ct.c_size_t),
    ]
    result.run.restype = ct.c_int
    result.iterations = request.param
    return result


def run(
    helper: ct.CDLL,
    points: np.ndarray,
    centers: np.ndarray,
    owners: np.ndarray,
    seeds: np.ndarray,
    cached: bool,
    threads: int,
) -> tuple[int, np.ndarray, tuple[int, int, int]]:
    pointer = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_double))
    output = np.full_like(centers, 12345.0)
    counts = (ct.c_size_t * 3)()
    status = helper.run(
        pointer(points),
        len(points),
        pointer(centers),
        len(centers),
        owners.ctypes.data_as(ct.POINTER(ct.c_int64)),
        pointer(seeds),
        cached,
        threads,
        pointer(output),
        counts,
    )
    return status, output, tuple(counts)


@pytest.mark.parametrize("atoms", [1, 2, 3, 12, 31, 32])
@pytest.mark.parametrize("cached", [False, True])
def test_cooperative_matches_generic_and_reuses_pair_state(
    helper: ct.CDLL, atoms: int, cached: bool
) -> None:
    rng = np.random.default_rng(7300 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 2
    points = rng.normal(size=(17, 3)) * 3
    owners = rng.integers(atoms, size=len(points), dtype=np.int64)
    seeds = rng.normal(size=len(points))
    reference = run(helper, points, centers, owners, seeds, cached, 0)
    for threads in (1, 7, 32):
        actual = run(helper, points, centers, owners, seeds, cached, threads)
        assert reference[0] == actual[0] == 0
        np.testing.assert_allclose(actual[1], reference[1], rtol=5e-13, atol=2e-13)
        pairs = atoms * (atoms - 1) // 2
        assert actual[2][0] == len(points) * pairs
        assert reference[2][0] == 2 * actual[2][0]
        assert actual[2][2] <= reference[2][2]
        if cached:
            assert actual[2][1] == pairs + len(points) * atoms
    # Reprepare every changed geometry; forward state never survives a point.
    moved = centers + rng.normal(size=centers.shape) * 0.1
    for geometry in (moved, centers):
        actual = run(helper, points, geometry, owners, seeds, cached, 32)
        reference = run(helper, points, geometry, owners, seeds, cached, 0)
        assert actual[0] == reference[0] == 0
        np.testing.assert_allclose(actual[1], reference[1], rtol=5e-13, atol=2e-13)


@pytest.mark.parametrize(
    "case",
    [
        "saturated",
        "rounded_zero",
        "positive_zero",
        "negative_zero",
        "coincident",
        "collision",
        "nonfinite",
        "tiny",
        "huge",
        "empty",
    ],
)
def test_cooperative_edge_semantics(helper: ct.CDLL, case: str) -> None:
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.1, 0.0]])
    points = np.array([[-1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    owners = np.array([0, 1], dtype=np.int64)
    seeds = np.array([0.3, -0.7])
    if case == "rounded_zero":
        points[0, 1], owners[0], seeds[0] = 2e-5, 1, 1e12
    elif case in {"positive_zero", "negative_zero"}:
        seeds.fill(-0.0 if case == "negative_zero" else 0.0)
    elif case == "coincident":
        centers[1] = centers[0]
    elif case == "collision":
        points[-1] = centers[0]
    elif case == "nonfinite":
        points[-1, 1] = np.inf
    elif case in {"tiny", "huge"}:
        points *= 1e-150 if case == "tiny" else 1e150
    elif case == "empty":
        points, owners, seeds = points[:0], owners[:0], seeds[:0]
    for cached in (False, True):
        reference = run(helper, points, centers, owners, seeds, cached, 0)
        actual = run(helper, points, centers, owners, seeds, cached, 32)
        assert actual[0] == reference[0]
        np.testing.assert_allclose(actual[1], reference[1], rtol=5e-13, atol=2e-13)
        if reference[0]:
            np.testing.assert_array_equal(actual[1], 12345.0)


def test_explicit_cooperative_selection_keeps_point_lanes() -> None:
    shape = {
        "atoms": 12,
        "aos": 96,
        "primitives": 240,
        "points": 4096,
        "tasks": 256,
        "spins": 2,
        "sources": 8,
    }
    target = cuda_target_info("sm_120")
    generic = plan_stationary_cuda_resources(
        **shape, target=target, budget_bytes=1 << 30, cooperative_becke=False
    )
    cooperative = plan_stationary_cuda_resources(
        **shape, target=target, budget_bytes=1 << 30, cooperative_becke=True
    )
    assert generic.becke_threads_per_point == 1 and generic.becke_shared_bytes == 0
    assert cooperative.becke_threads_per_point == 32
    assert cooperative.becke_shared_bytes == 16 + 66 * 64
    assert cooperative.geometry_lanes == generic.geometry_lanes
    assert cooperative.allocation_bytes == generic.allocation_bytes
    for atoms in (1,):
        plan = plan_stationary_cuda_resources(
            **{**shape, "atoms": atoms},
            target=target,
            budget_bytes=1 << 30,
            cooperative_becke=True,
        )
        assert plan.becke_threads_per_point == 1 and plan.becke_shared_bytes == 0
    for atoms in (33, 96, 128):
        common = dict(shape, atoms=atoms)
        generic = plan_stationary_cuda_resources(
            **common, target=target, budget_bytes=1 << 30, cooperative_becke=False
        )
        plan = plan_stationary_cuda_resources(
            **common, target=target, budget_bytes=1 << 30, cooperative_becke=True
        )
        assert plan.becke_threads_per_point == 32
        assert plan.becke_shared_bytes == 16 + 64 * (4 * (2 * atoms - 5) // 2)
        assert plan.becke_shared_bytes <= 32 << 10
        assert plan.geometry_lanes == generic.geometry_lanes
        assert plan.allocation_bytes == generic.allocation_bytes
        for field in ("shared_memory_per_block", "tuning_maximum_shared_bytes"):
            rejected = plan_stationary_cuda_resources(
                **common,
                target=replace(target, **{field: plan.becke_shared_bytes - 1}),
                budget_bytes=1 << 30,
                cooperative_becke=True,
            )
            assert rejected.becke_threads_per_point == 1
            assert rejected.becke_shared_bytes == 0
    for limited in (
        replace(target, shared_memory_per_block=1024),
        replace(target, tuning_maximum_shared_bytes=1024),
        replace(target, warp_size=16, maximum_threads_per_block=16),
    ):
        plan = plan_stationary_cuda_resources(
            **shape, target=limited, budget_bytes=1 << 30, cooperative_becke=True
        )
        assert plan.becke_threads_per_point == 1 and plan.becke_shared_bytes == 0
    with pytest.raises(ValueError, match="must be boolean"):
        plan_stationary_cuda_resources(
            **shape, target=target, budget_bytes=1 << 30, cooperative_becke=1
        )


def test_cooperative_independent_decimal_fd_translation_and_permutation(
    helper: ct.CDLL,
) -> None:
    from decimal import Decimal, localcontext

    from test_grid_response import CENTERS, DC, POINTS, decimal_partition

    iterations = helper.iterations
    owners = np.array([0, 1, 2], dtype=np.int64)
    seeds = np.array([0.3, -0.2, 0.7])
    status, gradient, _ = run(helper, POINTS, CENTERS, owners, seeds, True, 32)
    assert status == 0
    with localcontext() as context:
        context.prec = 60
        h = Decimal("1e-16")

        def moved(
            values: np.ndarray, motion: np.ndarray, sign: int
        ) -> list[list[Decimal]]:
            return [
                [
                    Decimal(str(x)) + sign * h * Decimal(str(dx))
                    for x, dx in zip(row, direction)
                ]
                for row, direction in zip(values, motion)
            ]

        plus, minus = [
            decimal_partition(
                moved(POINTS, DC[owners], sign), moved(CENTERS, DC, sign), iterations
            )
            for sign in (1, -1)
        ]
        expected = sum(
            Decimal(str(seed)) * (p[owner] - m[owner]) / (2 * h)
            for seed, p, m, owner in zip(seeds, plus, minus, owners)
        )
        assert abs(np.sum(gradient * DC) - float(expected)) < 2e-14
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=3e-15)
    shift = np.array([0.3, -0.7, 0.2])
    translated = run(helper, POINTS + shift, CENTERS + shift, owners, seeds, True, 32)
    np.testing.assert_allclose(translated[1], gradient, atol=2e-15)
    order = [2, 0, 1]
    permuted = run(
        helper, POINTS, CENTERS[order], np.argsort(order)[owners], seeds, True, 32
    )
    np.testing.assert_allclose(permuted[1], gradient[order], atol=2e-15)
    pieces = [
        run(helper, POINTS[a:b], CENTERS, owners[a:b], seeds[a:b], True, 32)[1]
        for a, b in ((0, 2), (2, 3), (3, 3))
    ]
    np.testing.assert_allclose(sum(pieces), gradient, atol=2e-15)


def test_native_cooperative_configuration_validates_actual_device_caps(
    tmp_path: Path,
) -> None:
    from test_stationary_task_work_budget import _block

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    header = (root / "src/dft/stationary_gradient_cuda.cuh").read_text()
    source = tmp_path / "configure.cpp"
    source.write_text(
        r"""
#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <cstring>
using std::size_t;
namespace generativeqc_grid_adjoint { struct PointPair { double v[8]; }; }
namespace generativeqc_stationary_cuda {
constexpr size_t stationary_becke_control_bytes=16, stationary_becke_max_atoms=128,
                 stationary_becke_retained_max_atoms=32, stationary_becke_pair_tile_rows=4,
                 stationary_becke_threads=32;
struct Context { int device=0; void check_device() {} };
struct Owner {
  Context context;
  bool failed=false,topology_ready=false;
  size_t atoms=12,geometry_lanes=2048,becke_threads_per_point=1,becke_shared_bytes=0;
};
void error_text(char* out,size_t n,const char* message) {
  if(n) { std::strncpy(out,message,n-1); out[n-1]=0; }
}
struct cudaDeviceProp {
  int maxThreadsPerBlock=1024,maxThreadsDim[3]{1024,1024,64},maxGridSize[3]{99999,1,1};
  size_t sharedMemPerBlock=49152;
};
static cudaDeviceProp actual;
void cuda_check(int code) { if(code) throw std::runtime_error("device error"); }
int cudaGetDeviceProperties(cudaDeviceProp* out,int) { *out=actual; return 0; }
"""
        + "template<class F>\n"
        + _block(header, "int guarded(")
        + "\n}\n"
        + _block(header, "int stationary_configure_becke(")
        + r"""
int main() {
  using namespace generativeqc_stationary_cuda;
  Owner p; char error[256]{};
  const size_t required=16+66*64;
  if(stationary_configure_becke(&p,32,required,error,256)) return 1;
  if(p.becke_threads_per_point!=32 || p.becke_shared_bytes!=required) return 2;
  if(stationary_configure_becke(&p,1,0,error,256) || p.becke_threads_per_point!=1) return 3;
  for(int limit=0;limit<4;++limit) {
    actual=cudaDeviceProp{};
    if(limit==0) actual.sharedMemPerBlock=required-1;
    if(limit==1) actual.maxThreadsPerBlock=16;
    if(limit==2) actual.maxThreadsDim[0]=16;
    if(limit==3) actual.maxGridSize[0]=2047;
    if(stationary_configure_becke(&p,32,required,error,256)) return 4;
    if(p.becke_threads_per_point!=1 || p.becke_shared_bytes) return 5;
  }
  actual=cudaDeviceProp{};
  actual.sharedMemPerBlock=required;
  if(stationary_configure_becke(&p,32,required,error,256) || p.becke_threads_per_point!=32) return 6;
  if(!stationary_configure_becke(&p,16,required,error,256)) return 7;
  if(!stationary_configure_becke(&p,32,required-1,error,256)) return 8;
  p.atoms=33;
  if(!stationary_configure_becke(&p,32,required,error,256)) return 9;
  for(size_t atoms:{size_t(33),size_t(96),size_t(128)}) {
    p.atoms=atoms;
    const size_t tiled=16+64*(4*(2*atoms-5)/2);
    actual.sharedMemPerBlock=tiled;
    if(stationary_configure_becke(&p,32,tiled,error,256) || p.becke_threads_per_point!=32 ||
       p.becke_shared_bytes!=tiled) return 13;
    actual.sharedMemPerBlock=tiled-1;
    if(stationary_configure_becke(&p,32,tiled,error,256) || p.becke_threads_per_point!=1 ||
       p.becke_shared_bytes) return 14;
  }
  p.atoms=1;
  if(!stationary_configure_becke(&p,32,required,error,256)) return 10;
  p.atoms=12; p.topology_ready=true;
  if(!stationary_configure_becke(&p,1,0,error,256)) return 11;
  if(!stationary_configure_becke(nullptr,1,0,error,256)) return 12;
  return 0;
}
"""
    )
    binary = tmp_path / "configure"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(binary)], check=True, timeout=10)


def test_generic_saturation_keeps_lazy_log_work(helper: ct.CDLL) -> None:
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points = np.array([[-1.0, 0.0, 0.0]])
    owners = np.array([0], dtype=np.int64)
    seeds = np.ones(1)
    for threads in (0, 32):
        result = run(helper, points, centers, owners, seeds, True, threads)
        assert result[0] == 0
        assert result[2][2] == 1
        np.testing.assert_array_equal(result[1], 0)


def test_single_atom_point_reuse_then_collision_is_collective(helper: ct.CDLL) -> None:
    centers = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    owners = np.zeros(2, dtype=np.int64)
    seeds = np.ones(2)
    for threads in (1, 7, 32):
        result = run(helper, points, centers, owners, seeds, False, threads)
        assert result[0] == -2
        np.testing.assert_array_equal(result[1], 12345.0)

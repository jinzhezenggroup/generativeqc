"""Execute the actual emitted Becke helper with retained and direct geometry.

These are CPU/source-work gates, not NVIDIA performance measurements. The
independent Decimal/finite-difference scientific gates live in test_grid_native.
"""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials

HARNESS = r"""
#include <cstdint>
#include <vector>
static size_t norms;
static std::array<double, 4> counted_norm(double x, double y, double z) {
  ++norms;
  return local_norm(x, y, z);
}
extern "C" int run(const double* points, size_t np, const double* centers, size_t na,
                   const int64_t* owners, const double* seeds, bool cached,
                   double tolerance, double* output, size_t* count) {
  using namespace generativeqc_grid_adjoint;
  norms = 0;
  std::vector<CenterPair> pairs(na * (na - 1) / 2);
  CenterPair* geometry = cached && pairs.size() ? pairs.data() : nullptr;
  if (!prepare_center_geometry(centers, na, tolerance, geometry, counted_norm,
                               local_ratio_geometry)) return -2;
  std::vector<double> gradient(3 * na), logs(na), products(na), bar_product(na), bar_distance(na);
  std::vector<size_t> zeros(na);
  std::vector<std::array<double, 4>> distances(na);
  for (size_t p = 0; p < np; ++p) {
    if (!contract_point_prepared(points + 3*p, centers, na, owners[p], seeds[p],
                                 gradient.data(), logs.data(), products.data(),
                                 bar_product.data(), bar_distance.data(), zeros.data(),
                                 distances.data(), counted_norm, local_ratio, local_log,
                                 local_becke, geometry, local_ratio_prepared)) return -2;
  }
  for (double v : gradient) if (!std::isfinite(v)) return -2;
  std::copy(gradient.begin(), gradient.end(), output);
  *count = norms;
  return 0;
}
extern "C" void ratios(double a, double b, double* output) {
  const auto geometry = local_ratio_geometry(b);
  const auto direct = local_ratio(a, b);
  const auto prepared = local_ratio_prepared(a, geometry.data());
  std::copy(direct.begin(), direct.end(), output);
  std::copy(prepared.begin(), prepared.end(), output + 3);
}
"""


@pytest.fixture(scope="module", params=[1, 2, 3, 4, 5])
def helper(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ct.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp(f"becke-{request.param}")
    source = directory / "helper.cpp"
    source.write_text(emit_grid_adjoint() + emit_grid_partials(request.param) + HARNESS)
    library = directory / "helper.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
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
    )
    lib = ct.CDLL(str(library))
    pointer = ct.POINTER(ct.c_double)
    lib.run.argtypes = [
        pointer,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
        ct.POINTER(ct.c_int64),
        pointer,
        ct.c_bool,
        ct.c_double,
        pointer,
        ct.POINTER(ct.c_size_t),
    ]
    lib.run.restype = ct.c_int
    lib.ratios.argtypes = [ct.c_double, ct.c_double, pointer]
    return lib


def run(
    helper: ct.CDLL,
    points: np.ndarray,
    centers: np.ndarray,
    owners: np.ndarray,
    seeds: np.ndarray,
    cached: bool,
    tolerance: float = 1e-12,
) -> tuple[int, np.ndarray, int]:
    ptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_double))
    result = np.full_like(centers, 12345.0)
    count = ct.c_size_t()
    status = helper.run(
        ptr(points),
        len(points),
        ptr(centers),
        len(centers),
        owners.ctypes.data_as(ct.POINTER(ct.c_int64)),
        ptr(seeds),
        cached,
        tolerance,
        ptr(result),
        ct.byref(count),
    )
    return status, result, count.value


@pytest.mark.parametrize(
    "atoms,points", [(1, 17), (2, 1), (3, 257), (7, 17), (12, 4096), (16, 33), (128, 1)]
)
def test_exact_helper_parity_and_measured_norm_work(
    helper: ct.CDLL, atoms: int, points: int
) -> None:
    rng = np.random.default_rng(3107 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 2
    xyz = rng.normal(size=(points, 3)) * 3
    owners = rng.integers(atoms, size=points, dtype=np.int64)
    seeds = rng.normal(size=points)
    direct = run(helper, xyz, centers, owners, seeds, False)
    cached = run(helper, xyz, centers, owners, seeds, True)
    assert direct[0] == cached[0] == 0
    np.testing.assert_array_equal(cached[1].view(np.uint64), direct[1].view(np.uint64))
    pairs = atoms * (atoms - 1) // 2
    assert direct[2] == pairs + points * (atoms + 2 * pairs)
    assert cached[2] == pairs + points * atoms
    # A fresh same-size geometry is always prepared before reuse, including a
    # return to the original geometry. Ragged calls do not retain point state.
    for moved in (centers + rng.normal(size=(atoms, 3)) * 0.1, centers):
        expected = run(helper, xyz, moved, owners, seeds, False)
        actual = run(helper, xyz, moved, owners, seeds, True)
        assert expected[0] == actual[0] == 0
        np.testing.assert_array_equal(
            actual[1].view(np.uint64), expected[1].view(np.uint64)
        )
    pieces = [
        run(helper, xyz[a:b], centers, owners[a:b], seeds[a:b], True)[1]
        for a, b in ((0, min(5, points)), (min(5, points), points), (points, points))
    ]
    np.testing.assert_allclose(sum(pieces), cached[1], rtol=5e-14, atol=2e-14)


@pytest.mark.parametrize(
    "case",
    [
        "saturated",
        "rounded_zero",
        "zero_seed",
        "coincident",
        "tolerance",
        "point_collision",
        "nonfinite",
        "empty",
        "empty_invalid",
    ],
)
def test_edge_semantics_and_transactional_failure(helper: ct.CDLL, case: str) -> None:
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.1, 0.0]])
    points = np.array([[-1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    owners = np.array([0, 1], dtype=np.int64)
    seeds = np.array([0.3, -0.7])
    tolerance = 1e-12
    if case == "rounded_zero":
        points[0, 1] = 2e-5
        owners[0] = 1
        seeds[0] = 1e12
    elif case == "zero_seed":
        seeds.fill(0)
    elif case == "coincident":
        centers[1] = centers[0]
    elif case == "tolerance":
        tolerance = 2.0
    elif case == "point_collision":
        points[-1] = centers[0]
    elif case == "nonfinite":
        centers[1, 1] = np.inf
    elif case.startswith("empty"):
        points, owners, seeds = points[:0], owners[:0], seeds[:0]
        if case == "empty_invalid":
            centers[1] = centers[0]
    direct = run(helper, points, centers, owners, seeds, False, tolerance)
    cached = run(helper, points, centers, owners, seeds, True, tolerance)
    assert cached[0] == direct[0]
    np.testing.assert_array_equal(cached[1].view(np.uint64), direct[1].view(np.uint64))
    if case in {
        "coincident",
        "tolerance",
        "point_collision",
        "nonfinite",
        "empty_invalid",
    }:
        assert cached[0] == -2
        np.testing.assert_array_equal(cached[1], 12345.0)


def test_ratio_cut_preserves_ieee_extremes(helper: ct.CDLL) -> None:
    result = np.zeros(6)
    pointer = result.ctypes.data_as(ct.POINTER(ct.c_double))
    for a in (0.0, -0.0, 1.0, -1.0, 1e-300, 1e300):
        for b in (np.nextafter(0.0, 1.0), 1e-300, 1e-150, 1.0, 1e150, 1e300):
            helper.ratios(a, b, pointer)
            np.testing.assert_array_equal(np.isnan(result[:3]), np.isnan(result[3:]))
            finite = ~np.isnan(result[:3])
            np.testing.assert_array_equal(
                result[:3][finite].view(np.uint64), result[3:][finite].view(np.uint64)
            )


def test_cpu_optional_cache_allocation_failure_keeps_direct_route(
    tmp_path: Path,
) -> None:

    from generativeqc_compiler.xc.grid_native import emit_grid_contraction

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "allocation.cpp"
    source.write_text(
        r"""
#include <cstdlib>
#include <cstring>
#include <new>
static bool fail_table=false;
static int failed_allocations=0;
void* operator new(std::size_t n) {
  if(fail_table && n==48*6) { ++failed_allocations; throw std::bad_alloc(); }
  if(void* p=std::malloc(n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
"""
        + emit_grid_contraction(3)
        + r"""
int main() {
  const double centers[12]{0,0,0,1,0,0,0,1,0,0,0,1};
  const double points[9]{.2,.3,.4,.3,-.2,.1,.1,.7,-.1},seeds[3]{.3,-.5,.9};
  const int64_t owners[3]{0,1,2};
  double direct[12],cached[12],fallback[12];
  const auto call=[&](double* output,size_t budget) {
    return grid_contract(points,3,centers,4,owners,seeds,output,12,budget,10000,1e-12);
  };
  if(call(direct,8*(30+120)) || call(cached,10000)) return 1;
  fail_table=true;
  if(call(fallback,10000) || failed_allocations!=1) return 2;
  if(std::memcmp(direct,cached,sizeof(direct)) || std::memcmp(cached,fallback,sizeof(cached))) return 3;
  return 0;
}
"""
    )
    executable = tmp_path / "allocation"
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            f"-I{root / 'src/dft'}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)

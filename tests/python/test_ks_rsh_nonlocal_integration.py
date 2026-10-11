"""Execute the production fused-RSH/resident-XC submission join on the host.

Provider and CUDA calls are test doubles. This checks dispatch, ordering and
failure propagation, not CUDA numerics or performance.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "generativeqc/generativeqc.h"
constexpr unsigned kCudaKsChunkCapacity = 2;
std::vector<std::string> events;
std::string fail_phase;
generativeqc_status fused_status = GENERATIVEQC_STATUS_SUCCESS;
generativeqc_status primary_status = GENERATIVEQC_STATUS_SUCCESS;
generativeqc_status correction_status = GENERATIVEQC_STATUS_SUCCESS;
void event(const char* name) {
  events.emplace_back(name);
  if (fail_phase == name) throw std::runtime_error(name);
}
void check(int status, const std::string& = {}) {
  if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error("status");
}
int cudaGetLastError() { return 0; }
struct MatrixView { double* scratch; std::size_t capacity; };
const int* expected_provider = nullptr;
const double* expected_density = nullptr;
double *expected_j = nullptr, *expected_exchange = nullptr, *expected_scratch = nullptr;
int* expected_error = nullptr;
std::size_t expected_matrix = 0, expected_elements = 0;
struct MatrixProducts {
  MatrixView view(double* scratch, std::size_t capacity) {
    assert(scratch == expected_scratch && capacity == expected_elements);
    event("matrices");
    return {scratch, capacity};
  }
};
namespace scf {
template<class... T> generativeqc_status enqueue_prepared_cuda_rsh_values(T&&...) {
  event("fused"); return fused_status;
}
generativeqc_status enqueue_prepared_cuda_fock(
    const int& provider, const double* alpha, const double* beta, std::size_t matrix,
    double* coulomb, double* alpha_exchange, double* beta_exchange, int* error,
    bool mixed, std::string&, std::uint64_t* mixed_work, const MatrixView* matrices) {
  assert(&provider == expected_provider && alpha == expected_density && !beta);
  assert(matrix == expected_matrix && coulomb == expected_j);
  assert(alpha_exchange == expected_exchange && !beta_exchange && error == expected_error);
  assert(!mixed && !mixed_work);
  assert(matrices && matrices->scratch == expected_scratch);
  assert(matrices->capacity == expected_elements);
  event("primary"); return primary_status;
}
template<class... T> generativeqc_status enqueue_prepared_cuda_exchange_correction(T&&...) {
  event("correction"); return correction_status;
}
}
struct CudaXcView { double* potential; };
struct Xc {
  double result = 23.0;
  template<class... T> CudaXcView enqueue_replay_body(T&&...) {
    event("semilocal"); return {&result};
  }
  template<class... T> CudaXcView enqueue_replay_density_features(T&&...) {
    event("features"); return {&result};
  }
  struct Grid { double *weights, *points; };
  Grid grid_view() { event("grid"); return {&result, &result}; }
  template<class... T> void enqueue_replay_nonlocal_potential(T&&...) {
    event("nonlocal"); result += 7.0;
  }
};
namespace generated { constexpr double kMolecularVv10DensityThreshold = 1e-12; }
namespace nlc {
template<class... T> void enqueue_vv10_molecular_domain_cuda(T&&...) { event("domain"); }
template<class... T> void enqueue_vv10_cuda_device(T&&...) { event("pairs"); }
}
namespace cuda_ks_detail {
void assemble_fock(void*, std::size_t, unsigned, double*, double*, double*, double,
                   double*, double, double* potential, int*, double* fock) {
  event("assemble"); *fock = *potential;
}
}
struct Owner {
  int provider = 0, correction = 0, *range_correction = &correction;
  bool has_range_correction = true, device_nonlocal = true;
  Xc xc_owner;
  Xc* xc = &xc_owner;
  double buffer = 0.0, result = 0.0;
  double density_storage[4]{}, j_storage[4]{}, exchange_storage[4]{}, scratch[4]{};
  double *density = density_storage, *j = j_storage, *exchange = exchange_storage;
  double *range_exchange = &buffer, *masked_matrix_output = scratch;
  MatrixProducts matrix_products;
  double *hcore = &buffer, *fock = &result;
  double exchange_coefficient = -0.5, range_exchange_coefficient = -0.2;
  double *nonlocal_raw_density = &buffer, *nonlocal_raw_gradient = &buffer;
  double *nonlocal_effective_weights = &buffer, *nonlocal_effective_density = &buffer;
  double *nonlocal_effective_gradient = &buffer, *nonlocal_workspace = &buffer;
  double *nonlocal_vrho = &buffer, *nonlocal_vsigma = &buffer;
  int error = 0, *jk_error = &error, *range_jk_error = &error, *enabled = &error;
  int *nonlocal_domain_error = &error, *nonlocal_pair_error = &error;
  int device = 0;
  void* stream = nullptr;
  std::size_t matrix = 4, elements = 4, n = 2;
  unsigned spins = 1;
  struct { std::size_t npoint = 1; } xc_layout;
  struct { std::size_t workspace_bytes = 8; } nonlocal_layout;
  struct Nonlocal { int parameters() { return 0; } } nonlocal_owner;
  Nonlocal* nonlocal_correlation = &nonlocal_owner;
  template<class F> void run_resident_nonlocal_cuda(F operation) { operation(); }
  template<class... Arguments> generativeqc_status enqueue_primary_jk(Arguments&&...) {
    event("primary"); return primary_status;
  }
"""

DRIVER = r"""
};
int main(int argc, char** argv) {
  if (argc != 4) return 99;
  const std::string route = argv[1];
  fail_phase = argv[3];
  Owner owner;
  expected_provider = &owner.provider;
  expected_density = owner.density;
  expected_j = owner.j;
  expected_exchange = owner.exchange;
  expected_error = owner.jk_error;
  expected_scratch = owner.masked_matrix_output;
  expected_matrix = owner.matrix;
  expected_elements = owner.elements;
  owner.device_nonlocal = std::string(argv[2]) == "1";
  owner.has_range_correction = route != "no-range";
  if (route == "fallback" || route == "primary-failure" || route == "correction-failure")
    fused_status = GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  if (route == "fused-runtime") fused_status = GENERATIVEQC_STATUS_CUDA_ERROR;
  if (route == "fused-numerical") fused_status = GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  if (route == "fused-invalid") fused_status = GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (route == "fused-allocation") fused_status = GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  if (route == "primary-failure") primary_status = GENERATIVEQC_STATUS_CUDA_ERROR;
  if (route == "correction-failure") correction_status = GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  bool failed = false;
  try { owner.enqueue_one(route == "overflow" ? 2 : 0); }
  catch (const std::exception&) { failed = true; }
  if (!failed && owner.result != (owner.device_nonlocal ? 30.0 : 23.0)) return 98;
  std::cout << (failed ? "FAIL" : "OK");
  for (const auto& name : events) std::cout << ':' << name;
  std::cout << '\n';
}
"""


@pytest.fixture(scope="module")
def submission_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    helper_start = source.index("  generativeqc_status enqueue_fock(")
    helper_end = source.index("  void multiply_matrix(", helper_start)
    helper = source[helper_start:helper_end]
    start = source.index("  void enqueue_one(unsigned slot)")
    end = source.index("    const auto blocks =", start)
    body = source[start:end] + "  }\n"
    folder = tmp_path_factory.mktemp("ks-rsh-nonlocal-join")
    unit, binary = folder / "probe.cpp", folder / "probe"
    unit.write_text(PREFIX + helper + body + DRIVER)
    return native_cxx.build_executable(
        [unit],
        binary,
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "include"),
        ),
        compile_timeout=45,
    )


def _run(probe: Path, route: str, nonlocal_body: bool, failure: str = "") -> str:
    result = subprocess.run(
        [str(probe), route, str(int(nonlocal_body)), failure],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("nonlocal_body", (False, True))
@pytest.mark.parametrize(
    ("route", "expected"),
    (
        ("fused", "OK:fused"),
        ("fallback", "OK:fused:matrices:primary:correction"),
        ("no-range", "OK:primary"),
        ("fused-runtime", "FAIL:fused"),
        ("fused-numerical", "FAIL:fused"),
        ("fused-invalid", "FAIL:fused"),
        ("fused-allocation", "FAIL:fused"),
        ("primary-failure", "FAIL:fused:matrices:primary"),
        ("correction-failure", "FAIL:fused:matrices:primary:correction"),
        ("overflow", "FAIL"),
    ),
)
def test_fock_dispatch_finishes_before_xc(
    submission_probe: Path, route: str, expected: str, nonlocal_body: bool
) -> None:
    if expected.startswith("OK"):
        expected += (
            ":features:grid:domain:pairs:nonlocal:assemble"
            if nonlocal_body
            else ":semilocal:assemble"
        )
    assert _run(submission_probe, route, nonlocal_body) == expected


@pytest.mark.parametrize("route", ("fused", "fallback"))
@pytest.mark.parametrize(
    "phase", ("features", "grid", "domain", "pairs", "nonlocal", "assemble")
)
def test_nonlocal_failure_stops_submission(
    submission_probe: Path, route: str, phase: str
) -> None:
    phases = ["features", "grid", "domain", "pairs", "nonlocal", "assemble"]
    fock = (
        ["fused"]
        if route == "fused"
        else ["fused", "matrices", "primary", "correction"]
    )
    expected = ":".join(["FAIL", *fock, *phases[: phases.index(phase) + 1]])
    assert _run(submission_probe, route, True, phase) == expected


@pytest.mark.parametrize("nonlocal_body", (False, True))
@pytest.mark.parametrize("phase", ("matrices", "primary", "correction"))
def test_fallback_failure_stops_before_xc(
    submission_probe: Path, phase: str, nonlocal_body: bool
) -> None:
    phases = ["fused", "matrices", "primary", "correction"]
    expected = ":".join(["FAIL", *phases[: phases.index(phase) + 1]])
    assert _run(submission_probe, "fallback", nonlocal_body, phase) == expected

"""Execute the production fused-RSH/resident-XC submission join on the host.

Provider and CUDA calls are test doubles. This checks dispatch, ordering and
failure propagation, not CUDA numerics or performance.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <cstddef>
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
namespace scf {
template<class... T> generativeqc_status enqueue_prepared_cuda_rsh_values(T&&...) {
  event("fused"); return fused_status;
}
template<class... T> generativeqc_status enqueue_prepared_cuda_fock(T&&...) {
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
  double *density = &buffer, *j = &buffer, *exchange = &buffer, *range_exchange = &buffer;
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
  std::size_t matrix = 1, elements = 1, n = 1;
  unsigned spins = 1;
  struct { std::size_t npoint = 1; } xc_layout;
  struct { std::size_t workspace_bytes = 8; } nonlocal_layout;
  struct Nonlocal { int parameters() { return 0; } } nonlocal_owner;
  Nonlocal* nonlocal_correlation = &nonlocal_owner;
  template<class F> void run_resident_nonlocal_cuda(F operation) { operation(); }
"""

DRIVER = r"""
};
int main(int argc, char** argv) {
  if (argc != 4) return 99;
  const std::string route = argv[1];
  fail_phase = argv[3];
  Owner owner;
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
def submission_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    start = source.index("  void enqueue_one(unsigned slot)")
    end = source.index("    const auto blocks =", start)
    body = source[start:end] + "  }\n"
    folder = tmp_path_factory.mktemp("ks-rsh-nonlocal-join")
    unit, binary = folder / "probe.cpp", folder / "probe"
    unit.write_text(PREFIX + body + DRIVER)
    result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "include"),
            str(unit),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    return binary


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
        ("fallback", "OK:fused:primary:correction"),
        ("no-range", "OK:primary"),
        ("fused-runtime", "FAIL:fused"),
        ("fused-numerical", "FAIL:fused"),
        ("fused-invalid", "FAIL:fused"),
        ("fused-allocation", "FAIL:fused"),
        ("primary-failure", "FAIL:fused:primary"),
        ("correction-failure", "FAIL:fused:primary:correction"),
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
    fock = ["fused"] if route == "fused" else ["fused", "primary", "correction"]
    expected = ":".join(["FAIL", *fock, *phases[: phases.index(phase) + 1]])
    assert _run(submission_probe, route, True, phase) == expected

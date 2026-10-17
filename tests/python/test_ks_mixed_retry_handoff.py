"""Execute mixed-error recovery and the next device-XC dispatch without a GPU.

Only CUDA/producer work is doubled. The production retry transaction and complete
DeviceFused XC branch select whether the next action uses the fresh strict target.
This is lifecycle coverage, not a substitute for independent GPU numerical gates.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition, cpp_record_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _block(source: str, condition: str) -> str:
    start = source.index(condition)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def retry_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    retry = _block(cpp_function_definition(source, "finish_legacy"), "if (is_failed)")
    device_xc = _block(
        cpp_function_definition(source, "stage_xc"),
        "if (options.xc_execution_schedule",
    )
    folder = tmp_path_factory.mktemp("ks-mixed-retry")
    cpp, binary = folder / "retry.cpp", folder / "retry"
    cpp.write_text(
        PREFIX.replace(
            "@OCCUPIED_DENSITY@",
            cpp_record_definition(
                (ROOT / "src/dft/cuda_xc.hpp").read_text(), "CudaXcOccupiedDensity"
            )
            + ";",
        )
        + "bool retry() {\n"
        + retry
        + "\nreturn false;\n}\n"
        + "CudaXcView stage_xc(std::uint64_t next_generation, Phase phase) {\n"
        + device_xc
        + '\nthrow std::runtime_error("unexpected host route");\n}\n};\n'
        + MAIN
    )
    native_cxx.build_executable([cpp], binary, compile_args=["-std=c++20", "-O0"])
    return binary


@pytest.mark.parametrize(
    "component", range(7), ids=["j", "density", "xc", "range", "nlc", "all", "matrix"]
)
@pytest.mark.parametrize("stage", range(3), ids=["seed", "frozen", "target"])
@pytest.mark.parametrize("source", range(3), ids=["absent", "mixed", "strict"])
def test_failed_mixed_operator_retries_fresh_strict_target(
    retry_probe: Path, component: int, stage: int, source: int
) -> None:
    result = subprocess.run(
        [str(retry_probe), str(component), str(stage), str(source)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
@OCCUPIED_DENSITY@
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
int cudaMemsetAsync(void* p, int value, std::size_t n, void*) {
  std::memset(p, value, n); return 0;
}
void check(int status) { require(!status, "unexpected device failure"); }
namespace generativeqc::runtime { enum class PrecisionPhase { Admitted, StrictAudit }; }
using Phase = generativeqc::runtime::PrecisionPhase;
namespace runtime::cuda_trace {
struct TraceRegion { TraceRegion(const char*, void*) {} void finish() {} };
void trace_counter(const char*, std::uint64_t) {}
template<class F> void trace_call(const char*, void*, F work) { work(); }
}
namespace scf { struct ScfOptions { enum class XcExecutionSchedule { DeviceFused }; }; }
namespace generated { constexpr double kMolecularVv10DensityThreshold = 1e-10; }
int domains = 0;
namespace nlc {
template<class... T> void enqueue_vv10_molecular_domain_cuda(T...) { ++domains; }
}
struct CudaXcView {};
struct Xc {
  int semilocal = 0, features = 0, potentials = 0;
  template<class... T> void enqueue(T...) { ++semilocal; }
  template<class... T> void enqueue_density_features(T...) { ++features; }
  template<class... T> void enqueue_nonlocal_potential(T...) { ++potentials; }
  CudaXcView view(std::uint64_t) { return {}; }
  struct Grid { const double *weights{}, *points{}; };
  Grid grid_view() { return {}; }
};
struct Source { bool mixed; bool mixed_recurrence() const { return mixed; } };
struct Owner {
  bool is_failed = true, pending_incremental_delta = true;
  bool pending_mixed_coulomb{}, pending_mixed_density{}, pending_mixed_point{};
  bool pending_mixed_contractions{};
  bool pending_mixed_range_exchange{}, pending_mixed_nonlocal{};
  bool strict_refinement{}, phase_source_cold = true, stabilize_occupations = true;
  bool final_closure = true, is_active{}, occupied_fitted_factor_ready = true;
  bool pending_fitted_occupied = true, final_fitted_projection_ready = true;
  bool nonlocal_seed_stage{}, nonlocal_frozen_stage{}, nonlocal_outer_active = true;
  bool device_nonlocal = true;
  bool occupied_xc_requested{}, occupied_xc_frame_ready{}, device_chunk_mode{};
  struct { bool any_lower_precision() const { return false; } } precision_schedule;
  double* tmp1{};
  std::size_t n{1}, spins{1}, occupations[2]{1, 0};
  unsigned final_corrections = 2, nonlocal_exact_epoch_iterations = 7;
  int history_count_data = 4, history_head_data = 2;
  int *history_count = &history_count_data, *history_head = &history_head_data;
  double previous_energy = 3, previous_energy_correction = 4;
  struct { struct { unsigned execution_retries{}; } precision;
    unsigned iterations = 5, fock_builds = 5;
  } output;
  struct { scf::ScfOptions::XcExecutionSchedule xc_execution_schedule =
    scf::ScfOptions::XcExecutionSchedule::DeviceFused; } options;
  unsigned retries{}, preparations{}, retirements{}, pairs{};
  bool last_pair_mixed = true;
  std::unique_ptr<Source> phase_source;
  void retire_phase_source(bool audit) {
    require(!audit, "failed source must not be certified");
    ++retirements; phase_source.reset();
  }
  void prepare_phase_source(bool) {
    require(strict_refinement && !nonlocal_seed_stage && !nonlocal_frozen_stage &&
                !nonlocal_outer_active && nonlocal_exact_epoch_iterations == 0,
            "provisional state survived until strict source preparation");
    require(!phase_source || !phase_source->mixed_recurrence(), "mixed lease survived retirement");
    ++preparations;
  }
  void record_precision_retry() { ++retries; }
  template<class F> void run_resident_nonlocal_cuda(F work) { work(); }
  void enqueue_nonlocal_pairs(const double*, bool mixed) { ++pairs; last_pair_mixed = mixed; }
  void* stream{};
  double density_value = 8.5;
  double *density = &density_value, *nonlocal_raw_density{}, *nonlocal_raw_gradient{};
  double *nonlocal_effective_weights{}, *nonlocal_effective_density{}, *nonlocal_effective_gradient{};
  double *nonlocal_vrho{}, *nonlocal_vsigma{}, *nonlocal_energy{};
  int *nonlocal_domain_error{};
  std::size_t elements = 1;
  struct { std::size_t npoint = 24; } xc_layout;
  std::unique_ptr<Xc> xc = std::make_unique<Xc>();
"""

MAIN = r"""
int main(int argc, char** argv) {
 try {
  require(argc == 4, "expected component, provisional stage and lease selection");
  const int component = std::stoi(argv[1]), stage = std::stoi(argv[2]), source = std::stoi(argv[3]);
  Owner p;
  p.pending_mixed_coulomb = component == 0 || component == 5;
  p.pending_mixed_density = component == 1 || component == 5;
  p.pending_mixed_point = component == 2 || component == 5;
  p.pending_mixed_range_exchange = component == 3 || component == 5;
  p.pending_mixed_nonlocal = component == 4 || component == 5;
  p.pending_mixed_contractions = component == 5 || component == 6;
  p.nonlocal_seed_stage = stage == 0; p.nonlocal_frozen_stage = stage == 1;
  if (source) p.phase_source = std::make_unique<Source>(Source{source == 1});
  require(p.retry() && p.is_active && !p.is_failed && p.strict_refinement, "strict retry was not selected");
  require(p.retirements == unsigned(source == 1) && p.preparations == 1, "source lifecycle changed");
  require(p.retries == 1 && p.output.precision.execution_retries == 1, "retry census changed");
  require(!p.history_count_data && !p.history_head_data, "nonlinear history survived");
  require(std::isinf(p.previous_energy) && p.previous_energy > 0 && p.previous_energy_correction == 0,
          "strict energy epoch retained an approximate baseline");
  require(!p.pending_incremental_delta && !p.stabilize_occupations && !p.final_closure && !p.final_corrections,
          "failed proposal state survived");
  require(p.density_value == 8.5 && p.output.iterations == 5 && p.output.fock_builds == 5,
          "retry changed density or invented completed work");
  p.stage_xc(2, Phase::StrictAudit);
  require(domains == 1 && p.pairs == 1 && !p.last_pair_mixed, "retry did not evaluate fresh strict NLC");
  require(!p.xc->semilocal && p.xc->features == 1 && p.xc->potentials == 1,
          "retry selected a seed-only or frozen-potential target");
  // A failure with no admitted mixed component remains terminal.
  Owner strict;
  require(!strict.retry() && strict.is_failed && !strict.is_active && !strict.retries,
          "strict failure was silently retried");
  require(!strict.occupied_fitted_factor_ready && !strict.pending_fitted_occupied &&
              !strict.final_fitted_projection_ready, "failed strict state remained reusable");
 } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
"""

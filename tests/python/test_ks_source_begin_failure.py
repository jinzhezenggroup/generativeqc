"""Exercise the source-admission transaction with injected host/CUDA failures."""

import subprocess
from pathlib import Path

import pytest
from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("scenario", [1, 2, 3, 4])
def test_phase_source_begin_failure_is_terminal_and_releases_the_lease(
    tmp_path: Path, native_cxx: NativeCxx, scenario: int
) -> None:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    begin = source.index("    is_active = false;\n    started = true;")
    end = source.index("\n  unsigned configured_chunk_width()", begin)
    # Keep the real begin transaction and catch. Only preparation/device work
    # are doubles; neither scientific integrals nor a CUDA context is needed.
    transaction = source[begin:end]
    cpp = tmp_path / "begin.cpp"
    cpp.write_text(
        r"""
#include <cassert>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include "scf/coulomb_exchange_work.hpp"
constexpr int GENERATIVEQC_PRECISION_FP64 = 64;
constexpr int cudaMemcpyDeviceToDevice = 0, cudaMemcpyHostToDevice = 1;
int cudaMemsetAsync(void*, int, std::size_t, void*) { return 0; }
int cudaMemcpyAsync(void*, const void*, std::size_t, int, void*) { return 0; }
int cudaStreamSynchronize(void*) { return 0; }
int cudaGetLastError() { return 0; }
void check(int status) { if (status) throw std::runtime_error("device"); }
enum class CudaKsSeedOrigin { External, NativeColdGuess };
namespace scf {
enum class FockBackend { Cuda };
enum class FockApproximation { Exact };
enum class FockOperator { ShortRange, LongRange };
struct ScfOptions { enum class XcExecutionSchedule { DeviceFused }; };
}
namespace cuda_ks_detail { template<class... T> void reset_control(T...) {} }
namespace runtime::cuda_trace { void trace_counter(const char*, std::uint64_t) {} }
int active_scenario = 0;
struct Region {
  bool bound() const { return false; }
  bool matches(int) const { return false; }
  bool failed() const { return false; }
  void bind(int) {
    if (active_scenario == 3) throw std::bad_alloc();
  }
  void recover() {} void invalidate() {} void mark_failure(const char*) {}
};
struct Family {
  double cuda_global_hybrid_exact_exchange = 0;
  bool component_coefficients_are_native_scales = false;
};
const Family* semilocal_family_metadata_from_code(int) { return nullptr; }
struct Owner {
  bool is_active = true, started = false, is_failed = true;
  bool stabilize_occupations{}, final_closure{}, strict_refinement{}, phase_source_cold{};
  bool pending_mixed_coulomb{}, pending_mixed_density{}, mixed_precision_executed{};
  bool pending_mixed_exchange = true, pending_mixed_resident = true;
  bool pending_mixed_range_exchange = true, pending_mixed_range_resident = true;
  bool pending_mixed_nonlocal = true;
  bool pending_mixed_point = true;
  bool pending_mixed_contractions = true;
  bool experimental_mixed_point{}, budget_fock{};
  unsigned final_corrections{}, refinement_iterations{}, pending_iterations{};
  struct { std::optional<int> precision_mode;
    std::array<double, 3> application_precision_budget{};
    scf::ScfOptions::XcExecutionSchedule xc_execution_schedule{};
    double semilocal_exchange_scale = 1, semilocal_correlation_scale = 1;
  } options;
  struct { struct { int requested_mode{}; } precision;
    struct { std::uint64_t owner_id{}; std::vector<int> operators; } precision_work;
    struct {
      generativeqc::scf::CoulombExchangeWork coulomb_exchange_work =
          generativeqc::scf::CoulombExchangeWork::empty();
    } dft_diagnostic;
  } output;
  std::uint64_t owner = 1, solve_epoch = 1;
  std::uint64_t warm_density_generation{}, warm_factor_generation{};
  bool has_exchange = true, has_range_correction{}, fitted_coulomb{}, incremental_direct_jk{};
  const void* automatic_program{};
  bool device_nonlocal{}, device_chunk_mode{};
  bool nonlocal_seed_stage{}, nonlocal_cache_enabled{};
  int *nonlocal_domain_error{}, *nonlocal_pair_error{};
  std::uint64_t* nonlocal_cache_counters{};
  void* nonlocal_correlation{};
  double exchange_coefficient{};
  unsigned spins = 1;
  int functional{};
  struct Correction {
    scf::FockBackend backend{};
    struct { unsigned derivative_order{};
      struct { bool present{}; scf::FockApproximation approximation{};
        scf::FockOperator op{}; double omega{};
      } coulomb, exchange;
    } spec;
  };
  std::optional<Correction> range_correction;
  struct { bool any_lower_precision() const { return false; } } precision_schedule;
  struct Provider { struct System { std::vector<int> ecp_terms; } value;
    const System& system() const { return value; }
  } provider;
  static constexpr unsigned kCudaKsChunkCapacity = 2;
  unsigned configured_chunk_width() const { return scenario == 3 ? 2 : 1; }
  int device_chunk_binding() const { return 0; }
  Region device_chunk_region;
  void* stream{};
  int history_count_slot{}, history_head_slot{};
  int* history_count = &history_count_slot; int* history_head = &history_head_slot;
  double* incremental_max_abs_delta_density{};
  int occupations[2]{};
  void* control{}; void* enabled{}; void* spin_enabled{};
  double density[1]{}, warm[1]{}, cold_seed[1]{};
  std::size_t elements = 1;
  struct { std::size_t density_h2d_bytes{}, synchronizations{}; } movement;
  bool warm_energy_baseline{}, occupied_fitted_factor_ready{};
  struct { std::size_t state_device_bytes{}, xc_device_bytes{}, provider_device_bytes{}; } resource;
  double range_warm_energy{}, range_warm_energy_correction{};
  double warm_energy{}, warm_energy_correction{}, previous_energy{}, previous_energy_correction{};
  std::unique_ptr<int> phase_source;
  int scenario{}, retired{};
  void prepare_phase_source(bool, bool) {
    if (scenario == 1) throw std::runtime_error("source CUDA failure");
    phase_source = std::make_unique<int>(1);
    if (scenario == 2) throw std::bad_alloc();
  }
  void prepare_cold_fitted_factor() {
    if (scenario == 4) throw std::runtime_error("cold factor preparation failure");
  }
  void retire_phase_source(bool) { ++retired; phase_source.reset(); }
  void restore_warm_seed() { throw std::logic_error("unexpected warm seed in cold begin probe"); }
  void begin() {
    const std::vector<double>* input = nullptr;
    const auto seed_origin = CudaKsSeedOrigin::External;
    const bool use_warm = false, retain_range_source = false;
    // This probe exercises optional source-admission failures; its ordinary
    // begin transaction has no provisional nonlocal initializer.
    const bool nonlocal_initialization = false;
    const auto initial_synchronizations = movement.synchronizations;
    std::vector<double> prepared;
"""
        + transaction
        + r"""
};
int main(int argc, char** argv) {
    assert(argc == 2);
    const auto scenario = std::atoi(argv[1]);
    active_scenario = scenario;
    Owner owner;
    owner.scenario = scenario;
    if (scenario == 3) owner.has_exchange = false;
    bool thrown = false;
    try { owner.begin(); } catch (...) { thrown = true; }
    assert(thrown);
    assert(owner.is_failed && !owner.is_active);
    assert(!owner.output.dft_diagnostic.coulomb_exchange_work.complete);
    assert(!owner.pending_mixed_point && !owner.pending_mixed_exchange &&
           !owner.pending_mixed_resident);
    assert(!owner.pending_mixed_contractions);
    assert(!owner.pending_mixed_range_exchange && !owner.pending_mixed_range_resident &&
           !owner.pending_mixed_nonlocal);
    assert(!owner.phase_source && owner.retired == 1);
    owner.pending_mixed_point = owner.pending_mixed_exchange = owner.pending_mixed_resident = true;
    owner.pending_mixed_contractions = true;
    owner.pending_mixed_range_exchange = owner.pending_mixed_range_resident = true;
    owner.pending_mixed_nonlocal = true;
    owner.scenario = 0;
    active_scenario = 0;
    owner.begin();
    assert(!owner.is_failed && owner.is_active && owner.phase_source);
    assert(!owner.pending_mixed_point && !owner.pending_mixed_exchange &&
           !owner.pending_mixed_resident);
    assert(!owner.pending_mixed_contractions);
    assert(!owner.pending_mixed_range_exchange && !owner.pending_mixed_range_resident &&
           !owner.pending_mixed_nonlocal);
}
"""
    )
    executable = tmp_path / "begin"
    native_cxx.build_executable(
        [cpp], executable, compile_args=("-std=c++17", f"-I{ROOT / 'src'}")
    )
    subprocess.run([str(executable), str(scenario)], check=True, timeout=10)

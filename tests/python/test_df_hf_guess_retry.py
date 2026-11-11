"""Execute the actual DF guess and CUDA SCF retry control with numerical boundaries injected."""

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner
from _df_scf_test_support import df_convergence_owner

ROOT = Path(__file__).resolve().parents[2]

_SHIM = r"""
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "methods/df_hf_guess.hpp"
#include "generativeqc/generativeqc.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/mean_field.hpp"
#include "scf/density_fitting.hpp"
#include "molecule/basis.hpp"
using namespace generativeqc;
using scf::ScfOptions;
using scf::ScfResult;
using scf::cuda_df_iteration_diis_history;
using scf::cuda_df_orbital_convergence;
using Matrix=std::vector<double>;
struct EigenResult { Matrix vectors; };
unsigned device_cycles{}, host_cycles{}, preparations{};
int device_behavior{};
bool expected_orbital_convergence{};
namespace host_trace {
struct Region { Region(const char*) {} void finish(){} };
enum class EigenReason {fallback};
template<class F> auto call(const char*,F f) {return f();}
template<class F> auto with_reason(EigenReason,F f) {return f();}
}
namespace generativeqc::runtime::df_progress {
void number(const char*,auto){}
void label(const char*,const char*){}
}
namespace initial_guess {
using PreliminaryDiagnostic=scf::initial_guess::PreliminaryDiagnostic;
struct OverlapOrthogonalizer {};
Matrix prepare_overlap_orthogonalizer(const core::System&,const Matrix&,std::size_t,OverlapOrthogonalizer*,int){return {1.0};}
}
struct DensityFittingScfData {
 struct {std::size_t nbf=1; Matrix hcore{1}, overlap{1}; double nuclear_repulsion{};} one_electron;
 struct {std::size_t value_bytes{};} resolved_budget;
};
DensityFittingScfData prepare_density_fitting_data(const core::System&,const core::System&,double,int,std::size_t,bool,unsigned){++preparations;return {};}
using CudaDensityFittingPlanPtr=std::shared_ptr<int>;
CudaDensityFittingPlanPtr make_cuda_density_fitting_plan(const DensityFittingScfData&,const ScfOptions&,int,std::size_t,bool,void*,const core::System*,const core::System*){return std::make_shared<int>(0);}
int df_setup_eigen(int*){return 0;}
int df_initial_orbital_request(){return 0;}
std::size_t cuda_density_fitting_seed_host_numeric_bytes(int*) {return 0;}
std::size_t df_seed_retained_capacity(const DensityFittingScfData&) {return 0;}
std::optional<Matrix> prepare_df_huckel_density(const core::System&,const DensityFittingScfData&,
    const ScfOptions&,const Matrix*,initial_guess::PreliminaryDiagnostic&,int,std::size_t) {
  return std::nullopt;
}
Matrix prepare_initial_density(const core::System&,const auto&,const Matrix&,std::size_t,const Matrix*,std::optional<EigenResult>&,int,int){return {1.0};}
void discard_density_fitting_tensor_storage(DensityFittingScfData&){}
struct CudaDensityFittingDeviceScfItem {bool converged{};unsigned iterations{};double energy{},energy_change{},density_rms{};std::uint64_t fock_builds{};};
generativeqc_status run_cuda_density_fitting_rhf_device_scf(int*,const Matrix&,const Matrix&,const Matrix&,const std::vector<std::int32_t>&,const Matrix&,unsigned max_iterations,double,double,Matrix& final_density,std::vector<CudaDensityFittingDeviceScfItem>& records,std::string&,const Matrix&,unsigned,bool qualified_owner,bool orbital_convergence){
  assert(qualified_owner && orbital_convergence == expected_orbital_convergence);
  if (device_behavior == 2) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  if (device_behavior == 3) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  if (device_behavior == 4) return GENERATIVEQC_STATUS_SUCCESS;
  const unsigned cycles = device_behavior == 1 ? 13U : max_iterations;
  device_cycles += cycles;
  final_density = {2.0};
  // Execution census deliberately differs from the returned SCF history.
  records={{device_behavior == 1,cycles,0,1,1,cycles+2U}};
  return GENERATIVEQC_STATUS_SUCCESS;
}
void finalize_density_fitting_rhf(const DensityFittingScfData&,const Matrix&,std::size_t,Matrix& d,const ScfOptions&,ScfResult& r,int*,std::size_t=0,bool=false,bool=false){r.density=d;}
struct Diis {Diis(unsigned){} Matrix update(const Matrix& f,const Matrix&){return f;}};
generativeqc_status execute_cuda_density_fitting_rhf_jk(int*,const Matrix&,Matrix& j,Matrix& k,std::string&){++host_cycles;j={0};k={0};return GENERATIVEQC_STATUS_SUCCESS;}
double electronic_energy(const Matrix&,const Matrix&,const Matrix&){return host_cycles;}
Matrix commutator_residual(const Matrix&,const Matrix&,const Matrix&,std::size_t){return {1};}
EigenResult iteration_df_eigen(const Matrix&,const Matrix&,const Matrix&,std::size_t,int*){return {{1}};}
Matrix density_from_orbitals(const Matrix&,std::size_t,std::size_t){return {1};}
double density_rms(const Matrix&,const Matrix&){return 1.0;}
"""

_TAIL = r"""namespace generativeqc::molecule {
std::size_t ao_count(const core::System& s) noexcept {return s.shells.size();}
std::size_t cartesian_ao_count(const core::System& system) noexcept {return system.shells.size();}
generativeqc_status validate_and_normalize(core::System&,std::string&,unsigned maximum_angular_momentum){
  assert(maximum_angular_momentum == 4);
  return GENERATIVEQC_STATUS_SUCCESS;
}
}
namespace generativeqc::scf {
std::size_t density_fitting_source_metadata_bytes(std::size_t, std::size_t, std::size_t,
    std::size_t, std::size_t, std::size_t) {return 0;}
std::size_t density_fitting_scf_diis_device_bytes(std::size_t, std::size_t, unsigned) noexcept {return 0;}
DensityFittingTilePlan plan_requested_density_fitting_tiles(DfPairStorageRequest, std::size_t,
    std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
    std::size_t, bool, std::size_t, bool method_owned_packing) {
  assert(!method_owned_packing);
  DensityFittingTilePlan plan;
  plan.stores_full_three_center = true;
  return plan;
}
ScfResult run_rhf_density_fitting_cuda(const core::System& s,const core::System& a,const ScfOptions& o,int d,const std::vector<double>* density,initial_guess::OverlapOrthogonalizer*) {return ::run_rhf_density_fitting_cuda_impl(s,a,o,d,density,nullptr);}
void validate_hf_warm_density(const core::System&,generativeqc_method,const std::vector<double>&,const initial_guess::EigenOperation&, const initial_guess::MatrixProductOperation&){}
}

int main(int argc, char** argv) {
  assert(argc == 2);
  setenv("GENERATIVEQC_CUDA_RHF_CONVERGENCE", argv[1], 1);
  unsetenv("GENERATIVEQC_DF_DISABLE_DEVICE_DIIS");
  expected_orbital_convergence = std::string(argv[1]) != "strict";
  unsetenv("GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS");
  core::System system, auxiliary;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  system.electron_count = 2;
  system.atoms.resize(1);
  system.atoms[0].atomic_number = 1;
  system.shells.resize(230);
  generativeqc_method_descriptor descriptor{};
  descriptor.correlation_memory_budget_bytes = 1ULL << 30;
  using namespace methods::detail;
  auto prepare = [&] {
    device_cycles = host_cycles = 0;
    return prepare_df_hf_guess(system, auxiliary, descriptor, 0, 0);
  };
  // The failed compact solve used the full allowance. It must not restart
  // another 32-cycle SCF loop inside the supposedly bounded preparation.
  auto guess = prepare();
  assert(device_cycles == 32 && host_cycles == 0);
  assert(guess.iterations == 32 && guess.work_counters_complete);
  assert(guess.outcome == DFHFGuessOutcome::Failed && guess.density.empty());
  device_behavior = 1;
  guess = prepare();
  assert(device_cycles == 13 && host_cycles == 0);
  assert(guess.iterations == 13 && guess.work_counters_complete);
  assert(guess.outcome == DFHFGuessOutcome::Used && !guess.density.empty());
  for (int refusal : {2, 3, 4}) {
    device_behavior = refusal;
    guess = prepare();
    assert(device_cycles == 0 && host_cycles == 0);
    assert(!guess.work_counters_complete && guess.density.empty());
    assert(guess.outcome == DFHFGuessOutcome::Failed);
  }
  // The new policy is guess-specific: ordinary DF-RHF keeps its established
  // host-orchestrated retry when its compact device attempt does not converge.
  device_behavior = 0;
  device_cycles = host_cycles = 0;
  ScfOptions options;
  options.max_iterations = 32;
  assert(options.density_fitting_host_retry);
  const auto ordinary = run_rhf_density_fitting_cuda_impl(
      system, auxiliary, options, 0, nullptr, nullptr);
  assert(device_cycles == 32 && host_cycles == 32);
  assert(!ordinary.converged && ordinary.iterations == 32);
  assert(ordinary.fock_builds == 66 && ordinary.preliminary_guess.work_counters_complete);
  device_behavior = 1;
  const auto completed = run_rhf_density_fitting_cuda_impl(
      system, auxiliary, options, 0, nullptr, nullptr);
  assert(completed.converged && completed.iterations == 13 && completed.fock_builds == 15);

  // The real admission helper must run before any DF preparation or SCF work.
  device_cycles = host_cycles = preparations = 0;
  options.preliminary_guess.emplace();
  try {
    run_rhf_density_fitting_cuda_impl(system, auxiliary, options, 0, nullptr, nullptr);
    assert(false);
  } catch (const std::invalid_argument&) {}
  assert(preparations == 0 && device_cycles == 0 && host_cycles == 0);
}
"""


@pytest.mark.parametrize("convergence", ["", "strict", "orbital-gradient"])
def test_guess_has_one_scf_attempt_and_ordinary_df_keeps_retry(
    tmp_path: Path, convergence: str
) -> None:
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("requires host C++")
    source = (ROOT / "src/scf/rhf.cpp").read_text()
    start = source.index("ScfResult run_rhf_density_fitting_cuda_impl(")
    stop = source.index("\nScfResult run_uhf_density_fitting_cuda_impl(", start)
    execution = (ROOT / "src/scf/fock_execution.cpp").read_text()
    guard_start = execution.index("void reject_cuda_df_preliminary_guess(")
    guard_stop = execution.index(
        "\nResolvedFockBuild fock_strategy_for_execution(", guard_start
    )
    admission = (
        "namespace generativeqc::scf {\n" + execution[guard_start:guard_stop] + "}\n"
    )
    probe = tmp_path / "df_guess_retry.cpp"
    probe.write_text(
        df_convergence_owner() + _SHIM + admission + source[start:stop] + _TAIL
    )
    binary = tmp_path / "df_guess_retry"
    compile_owner(
        compiler, tmp_path, [probe, ROOT / "src/methods/df_hf_guess.cpp"], binary
    )
    subprocess.run(
        [str(binary), convergence], check=True, capture_output=True, timeout=10
    )


def test_df_convergence_owner_preserves_eligibility(tmp_path: Path) -> None:
    """Unsupported owners stay strict; explicit unsupported requests fail."""
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("requires host C++")
    probe = tmp_path / "df_convergence.cpp"
    probe.write_text(
        df_convergence_owner()
        + r"""
#include <cassert>
int main() {
  using namespace generativeqc::scf;
  for (unsigned condition = 0; condition < 7; ++condition) {
    ScfOptions options;
    bool qualified = true;
    unsetenv("GENERATIVEQC_DF_DISABLE_DEVICE_DIIS");
    switch (condition) {
      case 1: qualified = false; break;
      case 2: options.export_physical_reference = true; break;
      case 3: options.diis_history = 1; break;
      case 4: options.precision_mode = GENERATIVEQC_PRECISION_AUTO; break;
      case 5: options.application_precision_budget[0] = 1e-8; break;
      case 6: setenv("GENERATIVEQC_DF_DISABLE_DEVICE_DIIS", "1", 1); break;
    }
    unsetenv("GENERATIVEQC_CUDA_RHF_CONVERGENCE");
    assert(cuda_df_orbital_convergence(options, qualified) == (condition == 0));
    setenv("GENERATIVEQC_CUDA_RHF_CONVERGENCE", "strict", 1);
    assert(!cuda_df_orbital_convergence(options, qualified));
    setenv("GENERATIVEQC_CUDA_RHF_CONVERGENCE", "orbital-gradient", 1);
    bool rejected = false;
    try { assert(cuda_df_orbital_convergence(options, qualified)); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected == (condition != 0));
  }
  unsetenv("GENERATIVEQC_DF_DISABLE_DEVICE_DIIS");
  for (bool energy : {false, true}) {
    ScfOptions tight;
    (energy ? tight.energy_tolerance : tight.density_tolerance) = 1e-12;
    for (const char* request : {"", "strict", "orbital-gradient"}) {
      setenv("GENERATIVEQC_CUDA_RHF_CONVERGENCE", request, 1);
      assert(!cuda_df_orbital_convergence(tight, true));
    }
  }
  setenv("GENERATIVEQC_CUDA_RHF_CONVERGENCE", "invalid", 1);
  try { cuda_df_orbital_convergence(ScfOptions{}, true); assert(false); }
  catch (const std::invalid_argument&) {}
}
"""
    )
    binary = tmp_path / "df_convergence"
    compile_owner(compiler, tmp_path, [probe], binary)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)


def test_incomplete_preliminary_work_is_not_published(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("requires host C++")
    source = (ROOT / "benchmarks/df_hf_preconvergence.cpp").read_text()
    start = source.index("    const auto field =")
    stop = source.index("    const auto array =", start)
    fields = "\n".join(
        line
        for line in source.splitlines()
        if 'field("pre_iterations",' in line
        or 'field("pre_work_counters_complete",' in line
    )
    probe = tmp_path / "preliminary_reporting.cpp"
    probe.write_text(
        r"""
#include <cassert>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
int main() {
  for (bool pre_work_counters_complete : {true, false}) {
    unsigned pre_cycles = 13;
    std::ostringstream output;
"""
        + source[start:stop]
        + fields
        + r"""
    assert(output.str() == (pre_work_counters_complete
        ? "\"pre_iterations\": 13,\n\"pre_work_counters_complete\": true,\n"
        : "\"pre_iterations\": null,\n\"pre_work_counters_complete\": false,\n"));
  }
}
"""
    )
    binary = tmp_path / "preliminary_reporting"
    compile_owner(compiler, tmp_path, [probe], binary)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)

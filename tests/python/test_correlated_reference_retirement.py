"""Host-only execution of the production correlated-plan retirement contract.

These probes exercise real admission code and method callbacks. CUDA operations
are replaced by shape-only planners or failure-injection stand-ins; no device,
integrals, SCF solve, or chemical energy is evaluated.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _compile_run(tmp_path: Path, code: str, sources: tuple[Path, ...] = ()) -> str:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = tmp_path / "retirement.cpp"
    source.write_text(code, encoding="utf-8")
    objects = []
    for index, path in enumerate((source, *sources)):
        obj = tmp_path / f"retirement_{index}.o"
        command = [
            cache,
            compiler,
            "-std=c++20",
            "-O0",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(tmp_path),
            "-c",
            str(path),
            "-o",
            str(obj),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=90
        )
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(obj))
    executable = tmp_path / "retirement"
    result = subprocess.run(
        [compiler, *objects, "-o", str(executable)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


PLAN_STANDINS = r"""
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "methods/correlated_cuda_reference.hpp"
int live_phase_owners = 0;
int destroyed_plans = 0;
struct PhaseOwner {
  PhaseOwner() { ++live_phase_owners; }
  ~PhaseOwner() { --live_phase_owners; }
};
namespace generativeqc::scf {
struct CudaRhfBucketPlan {
  std::size_t numeric_bytes, device_bytes;
};
std::size_t hf_cuda_retained_numeric_bytes(const CudaRhfBucketPlan* plan) noexcept {
  return plan ? plan->numeric_bytes : 0;
}
std::size_t hf_cuda_owned_device_bytes(const CudaRhfBucketPlan* plan) noexcept {
  return plan ? plan->device_bytes : 0;
}
void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan* plan) noexcept {
  if (!plan) return;
  // A failed phase must destroy all of its partial output/workspace first.
  assert(live_phase_owners == 0);
  ++destroyed_plans;
  delete plan;
}
}
using generativeqc::methods::detail::CorrelatedCudaReferencePlan;
using generativeqc::methods::detail::run_with_cuda_reference_budget;
using generativeqc::methods::detail::run_with_rccsd_reference_source;
"""


def test_retry_is_memory_only_single_and_unwinds_before_retirement(
    tmp_path: Path,
) -> None:
    output = _compile_run(
        tmp_path,
        PLAN_STANDINS
        + r"""
template<class Failure> void memory_retry() {
  CorrelatedCudaReferencePlan owner;
  *owner.slot() = new generativeqc::scf::CudaRhfBucketPlan{30, 10};
  const auto detached = std::make_shared<const std::vector<double>>(
      std::initializer_list<double>{1.0, 0.0, 0.0, 1.0});
  const auto* reference_identity = detached.get();
  const bool reference_plan_reused = true;
  const auto before = destroyed_plans;
  std::vector<std::size_t> budgets;
  auto result = run_with_cuda_reference_budget(owner.slot(), 100, [&](std::size_t budget) {
    PhaseOwner partial_output;
    budgets.push_back(budget);
    if (budgets.size() == 1) throw Failure();
    assert(detached.get() == reference_identity && detached->at(0) == 1.0);
    return std::make_unique<int>(17);
  });
  assert((budgets == std::vector<std::size_t>{70, 100}));
  assert(*result == 17 && destroyed_plans == before + 1);
  assert(reference_plan_reused && !owner.get() && owner.owned_device_bytes() == 0);
  assert(owner.retained_numeric_bytes() == 0 && live_phase_owners == 0);
}
struct AdmissionFailure : std::length_error {
  AdmissionFailure() : std::length_error("admission") {}
};
template<class Failure> void no_numerical_retry() {
  CorrelatedCudaReferencePlan owner;
  *owner.slot() = new generativeqc::scf::CudaRhfBucketPlan{30, 10};
  int calls = 0;
  const auto before = destroyed_plans;
  bool caught = false;
  try {
    run_with_cuda_reference_budget(owner.slot(), 100, [&](std::size_t budget) {
      PhaseOwner partial_output;
      assert(budget == 70);
      ++calls;
      throw Failure("numerical or invalid input");
    });
  } catch (const Failure& error) {
    caught = std::string(error.what()) == "numerical or invalid input";
  }
  assert(caught && calls == 1 && destroyed_plans == before && owner.get());
}
int main() {
  memory_retry<AdmissionFailure>();
  memory_retry<std::bad_alloc>();
  no_numerical_retry<std::invalid_argument>();
  no_numerical_retry<std::runtime_error>();
  no_numerical_retry<std::domain_error>();
  {
    CorrelatedCudaReferencePlan owner;
    *owner.slot() = new generativeqc::scf::CudaRhfBucketPlan{30, 10};
    int value = 7, calls = 0;
    const auto before = destroyed_plans;
    int& borrowed = run_with_cuda_reference_budget(owner.slot(), 100,
        [&](std::size_t budget) -> int& { ++calls; assert(budget == 70); return value; });
    assert(&borrowed == &value && calls == 1 && destroyed_plans == before);
    assert(owner.retained_numeric_bytes() == 30 && owner.owned_device_bytes() == 10);
  }
  // Exhaustion must retire before the first downstream allocation, including
  // the exact equality boundary and a reservation greater than the budget.
  for (const auto retained : {100U, 101U}) {
    CorrelatedCudaReferencePlan owner;
    *owner.slot() = new generativeqc::scf::CudaRhfBucketPlan{retained, 10};
    int calls = 0;
    run_with_cuda_reference_budget(owner.slot(), 100, [&](std::size_t budget) {
      ++calls; assert(budget == 100 && !owner.get());
    });
    assert(calls == 1);
  }
  // Neither a null slot nor an empty owner creates an extra attempt.
  for (bool absent_slot : {false, true}) {
    CorrelatedCudaReferencePlan owner;
    int calls = 0;
    bool caught = false;
    const auto before = destroyed_plans;
    try {
      run_with_cuda_reference_budget(absent_slot ? nullptr : owner.slot(), 100,
          [&](std::size_t budget) { ++calls; assert(budget == 100); throw std::bad_alloc(); });
    } catch (const std::bad_alloc&) { caught = true; }
    assert(caught && calls == 1 && destroyed_plans == before);
  }
  for (bool allocation_failure : {false, true}) {
    CorrelatedCudaReferencePlan owner;
    *owner.slot() = new generativeqc::scf::CudaRhfBucketPlan{30, 10};
    int calls = 0;
    const auto before = destroyed_plans;
    bool caught = false;
    try {
      run_with_cuda_reference_budget(owner.slot(), 100, [&](std::size_t budget) {
        PhaseOwner partial_output;
        ++calls;
        assert(budget == (calls == 1 ? 70 : 100));
        if (allocation_failure) throw std::bad_alloc();
        throw std::length_error("still too large after retirement");
      });
    } catch (const std::bad_alloc&) { caught = allocation_failure; }
      catch (const std::length_error& error) {
        caught = !allocation_failure &&
                 std::string(error.what()) == "still too large after retirement";
      }
    assert(caught && calls == 2 && destroyed_plans == before + 1 && !owner.get());
  }
  std::cout << "memory-only, one retry, full reservation and unwind gates passed\n";
}
""",
    )
    assert "unwind gates passed" in output


def test_exact_h2_mp2_capacity_is_restored_after_plan_retirement(
    tmp_path: Path,
) -> None:
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.integral.direct_resident_schedule import (
        emit_direct_resident_psss_schedule_header,
    )
    from generativeqc_compiler.mp2.equations import energy_program
    from generativeqc_compiler.tensor.cuda_plan import plan_cuda
    from generativeqc_compiler.tensor.optimize import prepare_for_backend

    # Compute the real compiler's tile-1 allocation instead of substituting a
    # guessed kernel allowance. This needs no CUDA compiler or runtime.
    program = prepare_for_backend(energy_program((1, 1, 1, 1)), "cuda")
    kernel = plan_cuda(
        program,
        cuda_target_info("sm_90"),
        max_bytes=1 << 20,
        reassociate_contractions=False,
    )
    (tmp_path / "generated_direct_resident_psss_schedule.cuh").write_text(
        emit_direct_resident_psss_schedule_header(), encoding="utf-8"
    )
    provider = (ROOT / "src/posthf/native_provider.cpp").read_text(encoding="utf-8")
    provider = provider[
        provider.index("namespace generativeqc::posthf {") : provider.index(
            "std::vector<std::vector<double>> NativeBlockProvider::get_many("
        )
    ]
    provider += r"""
std::vector<double> NativeBlockProvider::get(const MOSlots&, bool, int,
    generativeqc_tensor::Metrics*, ProviderWork*) const {
  throw std::logic_error("shape probe must not evaluate integrals");
}
}
"""
    energy = (ROOT / "src/posthf/mp2_energy.cpp").read_text(encoding="utf-8")
    energy = energy[
        energy.index("namespace generativeqc::mp2 {") : energy.index(
            "  struct KernelState {"
        )
    ]
    energy += "return result;\n}\n}\n"
    reference = (ROOT / "src/scf/cuda/reference_export.cuh").read_text(encoding="utf-8")
    reference = reference[
        reference.index("inline std::size_t check_capacity(") : reference.index(
            "/** Stage the final physical"
        )
    ]
    headers = r"""
#include <cmath>
#include <limits>
#include <type_traits>
#include "posthf/mp2_energy.hpp"
#include "posthf/mp2_cpu_generated.hpp"
#include "posthf/mp2_cuda_plan.hpp"
#include "posthf/mp2_schedule_generated.hpp"
#include "scf/cuda/arena.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda/reference_eri_policy.hpp"
"""
    cuda_plan = (
        "namespace generativeqc::mp2::generated { CudaPlan cuda_plan(unsigned tile, int) {"
        ' if (tile != 1) throw std::logic_error("shape probe only generated tile 1");'
        f" return {{nullptr, nullptr, nullptr, {kernel.peak_bytes},"
        f' {kernel.allocation_bytes}, "host-plan-only"}}; }} }}\n'
    )
    code = (
        PLAN_STANDINS
        + headers
        + cuda_plan
        + provider
        + energy
        + "namespace generativeqc::scf::reference_detail {\n"
        + reference
        + "}\n"
        + H2_ADMISSION_MAIN
    )
    output = _compile_run(
        tmp_path,
        code,
        (
            ROOT / "src/scf/cuda/arena.cpp",
            ROOT / "src/scf/cuda/topology.cpp",
            ROOT / "src/molecule/basis.cpp",
        ),
    )
    assert (
        "exact_budget=113386440 retained_device_lower_bound=2584 attempts=2" in output
    )


H2_ADMISSION_MAIN = r"""
using namespace generativeqc;
struct ShapeSource final : integrals::ElectronInteractionSource {
  core::System system;
  ShapeSource() {
    system.atoms = {{1,{0,0,0},0},{1,{0,0,1.4},0}};
    system.electron_count = 2;
    for (unsigned atom = 0; atom < 2; ++atom)
      system.shells.push_back({atom,0,{{3.42525091,0.15432897},
          {0.62391373,0.53532814},{0.16885540,0.44463454}}});
  }
  const core::System& orbital() const override { return system; }
  std::size_t nbf() const override { return 2; }
  std::size_t naux() const override { return 0; }
  std::size_t retained_numeric_bytes() const override { return posthf::source_capacity(system); }
  bool supports(Operator op) const noexcept override { return op == Operator::eri; }
  void read(Operator, const std::array<std::size_t,4>&, const std::array<std::size_t,4>&,
            double*, std::size_t) const override {
    throw std::logic_error("shape probe must not evaluate integrals");
  }
};
int main() {
  ShapeSource source;
  hf::PhysicalReference ref;
  ref.nbf = 2; ref.nocc = 1;
  ref.coefficients = {1,0,0,1}; ref.orbital_energies = {-1,1};
  posthf::NativeBlockProvider provider(source, ref, 1ULL << 40, 1);
  const auto kernel_reserve = mp2::generated::cuda_plan(1,0).numeric_bytes + 32 + 16 + 64;
  const auto budget = provider.batch_bytes({1,1,1,1},1,true) + kernel_reserve;
  using namespace scf::cuda_execution;
  HostBatch host;
  assert(pack_host_batch({source.system},{nullptr},host,false,true,false));
  ArenaLayout layout{};
  assert(make_layout(1,2,2,2,2,host.shell_pair_first.size(),
      host.system_shell_pair_block_offsets.back(),0,host.shell_pair_primitive_offsets.back(),
      0,0,0,0,0,0,0,host.primitive_exponents.size(),8,100,1,
      false,false,false,false,false,false,false,false,layout));
  const auto reference_peak = scf::reference_detail::base_capacity(layout.bytes,host,4,0,budget);
  const auto cache = reference_eri_cache_bytes(16,0,false,reference_peak,budget);
  const auto retained = layout.bytes + cache;
  assert(budget == 113386440 && layout.bytes == 2456 && cache == 128);
  assert(posthf::rhf_reference_capacity(source.system,8,false) < budget);
  assert(reference_peak + cache < budget);
  const auto original = mp2::conventional_energy(ref,source,budget,1e-12,8,true,0);
  assert(original.numeric_capacity_bytes == budget);
  // Even the device-only lower bound reproduces the regression. Production
  // now also reserves retained host/topology/provider allocations separately.
  for (const auto reservation : {retained, layout.bytes}) {
    bool rejected = false;
    try { (void)mp2::conventional_energy(ref,source,budget-reservation,1e-12,8,true,0); }
    catch (const std::length_error&) { rejected = true; }
    assert(rejected);
  }
  CorrelatedCudaReferencePlan owner;
  *owner.slot() = new scf::CudaRhfBucketPlan{retained, retained};
  const auto* coefficients = ref.coefficients.data();
  std::vector<std::size_t> attempts;
  const auto rescued = run_with_cuda_reference_budget(owner.slot(),budget,
      [&](std::size_t available) {
        attempts.push_back(available);
        return mp2::conventional_energy(ref,source,available,1e-12,8,true,0);
      });
  assert((attempts == std::vector<std::size_t>{budget-retained,budget}));
  assert(rescued.numeric_capacity_bytes == budget && !owner.get());
  assert(ref.coefficients.data() == coefficients && ref.orbital_energies[0] == -1);
  std::cout << "exact_budget=" << budget << " retained_device_lower_bound=" << retained
            << " attempts=" << attempts.size() << '\n';
}
"""


def _between(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def test_actual_cc_provider_and_solver_callbacks_preserve_retry_state(
    tmp_path: Path,
) -> None:
    method = (ROOT / "src/methods/rccsd_method.cpp").read_text(encoding="utf-8")
    provider = _between(
        method,
        "    posthf::ProviderWork provider_work;",
        "    const double problem_seconds =",
    )
    solver = _between(
        method[method.index('    allocation_stage = "CC resident solve";') :],
        "    run_with_cuda_reference_budget(",
        "    const double solver_seconds =",
    )
    code = (
        PLAN_STANDINS
        + CC_CALLBACK_STANDINS
        + r"""
void provider_probe(int failure, bool resident) {
  reset_injection(failure);
  CorrelatedCudaReferencePlan owner;
  *owner.slot() = new scf::CudaRhfBucketPlan{30,10};
  auto* cuda_reference_plan = owner.slot();
  const auto retained_plan_bytes = [&] { return owner.retained_numeric_bytes(); };
  scf::PreparedFockPlan* prepared_exact = nullptr;
  core::System system;
  auto reference = std::make_shared<hf::PhysicalReference>();
  cc::SolverOptions solver_options, correlation_options;
  solver_options.max_bytes = 100;
  const bool cuda = true, retain_df_response = true;
  const core::System* correlation_auxiliary = nullptr;
  const scf::cuda_execution::CudaDfSourcePolicy* correlation_policy = nullptr;
  ProbeExecution execution;
  ProbeState state;
  if (resident) {
    state.reference_interaction_source=std::make_shared<ResidentSource>(
        std::exchange(*owner.slot(),nullptr));
    // A resident source owns the executable itself; only the compact route
    // can retry a later raw-allocation failure by retiring a separate cache.
    raw_constructions=1;
  } else {
    state.reference_interaction_source=std::make_shared<CompactSource>();
  }
  bool rejected = false;
  const auto before = destroyed_plans;
  try {
"""
        + provider
        + r"""
    assert(failure != 2);
    assert(state.problem.provider_peak_bytes == 40 && provider_phase_peak == 40);
    assert((build_budgets == (resident ? std::vector<std::size_t>{100,100}
                                      : std::vector<std::size_t>{70,100})));
    assert(raw_constructions == 2 && destroyed_sources == 1);
    assert(!state.reference_interaction_source);
    assert(provider_work.source_scans == 2 && provider_work.source_values == 6);
    assert(provider_metrics.input_ms == 2 && state.df_source.source_identity == 2);
    assert(destroyed_plans == before + 1 && !owner.get());
  } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected == (failure == 2));
  if (rejected) {
    assert(build_budgets.size() == 1 && state.problem.provider_peak_bytes == 99);
    assert(raw_constructions == (resident ? 1 : 0) && destroyed_sources == 0);
    assert(bool(owner.get()) == !resident && state.reference_interaction_source);
    assert(destroyed_plans == before && !state.df_source.response_state);
  }
  // The accepted detached DF output is allowed to survive the retry; the
  // failed attempt must already have released its output before retirement.
  state.df_source = {};
}
void solver_probe(int failure, bool resident) {
  reset_injection(failure);
  CorrelatedCudaReferencePlan owner;
  *owner.slot() = new scf::CudaRhfBucketPlan{30,10};
  auto* cuda_reference_plan = owner.slot();
  cc::SolverOptions solver_options, correlation_options;
  solver_options.max_bytes = 100;
  const bool cuda = true;
  const std::size_t exact_source_retained = resident ? 30 : 17;
  ProbeState state;
  if (resident) {
    state.reference_interaction_source=std::make_shared<ResidentSource>(
        std::exchange(*owner.slot(),nullptr));
    solve_failures=1;
  } else {
    state.reference_interaction_source=std::make_shared<CompactSource>();
  }
  state.problem.reference_retained_bytes = 23 + exact_source_retained;
  const auto retire_optional_source = [&] {
    assert(live_phase_owners == 0);
    if (!state.reference_interaction_source) return false;
    state.reference_interaction_source.reset();
    return true;
  };
  ProbeExecution execution;
  bool rejected = false;
  const auto before = destroyed_plans;
  try {
"""
        + solver
        + r"""
    assert(failure != 2);
    if (failure == 3) {
      assert(state.solved.status == cc::SolveStatus::NumericalFailure);
      assert(solve_budgets.size() == 1 && bool(owner.get()) == !resident &&
             state.reference_interaction_source);
      assert(destroyed_plans == before);
    } else {
      assert((solve_budgets == (resident ? std::vector<std::size_t>{100,100}
                                        : std::vector<std::size_t>{70,70,100})));
      assert((solve_retained == (resident ? std::vector<std::size_t>{53,23}
                                         : std::vector<std::size_t>{40,23,23})));
      assert(state.solved.converged() && !owner.get() && !state.reference_interaction_source);
      assert(destroyed_sources == 1);
      assert(destroyed_plans == before + 1 && correlation_options.max_bytes == 100);
    }
  } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected == (failure == 2));
  if (rejected) {
    assert(solve_budgets.size() == 1 && state.solved.reason == "unpublished");
    assert(bool(owner.get()) == !resident && state.reference_interaction_source &&
           destroyed_plans == before && destroyed_sources == 0);
  }
}
int main() {
  for (bool resident : {false,true}) {
    for (int failure : {0,1,2}) provider_probe(failure,resident);
    for (int failure : {0,1,2,3}) solver_probe(failure,resident);
  }
  std::cout << "actual provider/solver callback rollback and cumulative work gates passed\n";
}
"""
    )
    assert "cumulative work gates passed" in _compile_run(tmp_path, code)


CC_CALLBACK_STANDINS = r"""
#include "cc/df_source.hpp"
#include "cc/solver.hpp"
#include "posthf/native_provider.hpp"
using namespace generativeqc;
int injected_failure = 0, raw_constructions = 0, destroyed_sources = 0;
int live_source_views = 0, solve_failures = 2;
std::vector<std::size_t> build_budgets, solve_budgets, solve_retained;
void reset_injection(int failure) {
  assert(live_phase_owners == 0 && live_source_views == 0);
  injected_failure = failure;
  raw_constructions = destroyed_sources = 0;
  solve_failures = 2;
  build_budgets.clear(); solve_budgets.clear(); solve_retained.clear();
}
void injected_exception() {
  if (injected_failure == 2) throw std::invalid_argument("invalid scientific state");
  if (injected_failure == 1) throw std::bad_alloc();
  throw std::length_error("phase admission");
}
struct ProbeExecution {
  int device_id() const { return 0; }
  bool cuda_requested() const { return true; }
};
namespace generativeqc::cc {
class DFSourceState : public PhaseOwner {};
SolverResult solve_cuda(const Problem& problem, const SolverOptions& options, int) {
  PhaseOwner partial_result;
  solve_budgets.push_back(options.max_bytes);
  solve_retained.push_back(problem.reference_retained_bytes);
  if (injected_failure == 3) {
    SolverResult result;
    result.status = SolveStatus::NumericalFailure;
    return result;
  }
  if (solve_budgets.size() <= static_cast<std::size_t>(solve_failures)) injected_exception();
  SolverResult result;
  result.status = SolveStatus::Converged;
  return result;
}
SolverResult solve_cpu(const Problem&, const SolverOptions&) {
  throw std::logic_error("CUDA callback dispatched a CPU solve");
}
}
namespace generativeqc::posthf {
struct RawSource::Impl { core::System system; };
RawSource::RawSource(core::System system, const core::System*) {
  ++raw_constructions;
  if (raw_constructions == 1) throw std::bad_alloc();
  impl_ = std::make_unique<Impl>();
  impl_->system = std::move(system);
}
RawSource::~RawSource() = default;
const core::System& RawSource::orbital() const { return impl_->system; }
const core::System& RawSource::auxiliary() const { return impl_->system; }
std::size_t RawSource::nbf() const { return 2; }
std::size_t RawSource::naux() const { return 0; }
std::size_t RawSource::retained_numeric_bytes() const { return 1; }
void RawSource::read(Operator, const std::array<std::size_t,4>&,
                     const std::array<std::size_t,4>&, double*, std::size_t) const {
  throw std::logic_error("failure injection must not evaluate integrals");
}
}
namespace generativeqc::scf {
class PreparedFockPlan {
 public:
  core::System system;
  ~PreparedFockPlan() {
    assert(live_phase_owners == 0 && live_source_views == 0);
    ++destroyed_sources;
  }
};
class PreparedFockInteractionSourceView : public integrals::ElectronInteractionSource {
 public:
  explicit PreparedFockInteractionSourceView(const PreparedFockPlan& p) : plan(p) {
    ++live_source_views;
  }
  ~PreparedFockInteractionSourceView() { --live_source_views; }
  const core::System& orbital() const override { return plan.system; }
  std::size_t nbf() const override { return 2; }
  std::size_t naux() const override { return 0; }
  std::size_t retained_numeric_bytes() const override { return 17; }
  bool supports(Operator) const noexcept override { return true; }
  void read(Operator, const std::array<std::size_t,4>&, const std::array<std::size_t,4>&,
            double*, std::size_t) const override {
    throw std::logic_error("failure injection must not evaluate integrals");
  }
 private:
  const PreparedFockPlan& plan;
};
}
struct ExactSource : integrals::ElectronInteractionSource {
  core::System system;
  ~ExactSource() {
    assert(live_phase_owners == 0 && live_source_views == 0);
    ++destroyed_sources;
  }
  const core::System& orbital() const override { return system; }
  std::size_t nbf() const override { return 2; }
  std::size_t naux() const override { return 0; }
  std::size_t retained_numeric_bytes() const override { return 17; }
  bool supports(Operator) const noexcept override { return true; }
  void read(Operator,const std::array<std::size_t,4>&,const std::array<std::size_t,4>&,
            double*,std::size_t) const override { throw std::logic_error("no numerical work"); }
};
struct CompactSource final : ExactSource {};
struct ResidentSource final : ExactSource {
  std::unique_ptr<scf::CudaRhfBucketPlan,void(*)(scf::CudaRhfBucketPlan*)> plan;
  explicit ResidentSource(scf::CudaRhfBucketPlan* p):plan(p,scf::destroy_rhf_cuda_bucket_plan) {}
  std::size_t retained_numeric_bytes() const override { return plan->numeric_bytes; }
};
struct ProbeState {
  std::shared_ptr<const integrals::ElectronInteractionSource> reference_interaction_source;
  cc::Problem problem;
  cc::SolverResult solved;
  cc::DFSourceResult df_source;
  std::shared_ptr<const hf::PhysicalReference> reference = std::make_shared<hf::PhysicalReference>();
  std::vector<double> eps_o{-1}, eps_v{1};
  std::size_t budget = 70, reference_execution_plan_bytes = 30;
  std::size_t reference_execution_plan_device_bytes = 10, external_reservation_bytes = 11;
  generativeqc_correlation_diagnostic diagnostic{};
  ProbeState() {
    problem.provider_peak_bytes = 99; solved.reason = "unpublished";
    diagnostic.numeric_capacity_bytes = 97;
    diagnostic.reference_execution_plan_owned_device_bytes = 10;
    diagnostic.reference_execution_plan_reused = 1;
  }
};
cc::Problem build_problem(const core::System&, const integrals::ElectronInteractionSource* source,
    const hf::PhysicalReference&, const cc::SolverOptions& options, bool, int,
    posthf::ProviderWork& work, generativeqc_tensor::Metrics& metrics, const core::System*,
    cc::DFSourceResult* retained, const scf::cuda_execution::CudaDfSourcePolicy*) {
  PhaseOwner partial_problem;
  assert(source && source->nbf() == 2 && retained && !retained->response_state);
  build_budgets.push_back(options.max_bytes);
  ++work.source_scans; work.source_values += 3; ++metrics.input_ms;
  retained->response_state = std::make_shared<cc::DFSourceState>();
  retained->source_identity = work.source_scans;
  if (build_budgets.size() == 1) injected_exception();
  cc::Problem result;
  result.provider_peak_bytes = 40;
  return result;
}
"""


def _statement(source: str, start: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(";", begin) + 1] + "\n"


def test_actual_mp2_and_cc_force_triples_callbacks_restore_budgets(
    tmp_path: Path,
) -> None:
    mp2 = (ROOT / "src/methods/mp2_method.cpp").read_text(encoding="utf-8")
    cc = (ROOT / "src/methods/rccsd_method.cpp").read_text(encoding="utf-8")
    triples = (ROOT / "src/methods/rccsdt_method.cpp").read_text(encoding="utf-8")
    mp2_energy = _between(
        mp2,
        "      std::unique_ptr<posthf::RawSource> raw_source;",
        "      Result result;",
    )
    mp2_force = _between(
        mp2,
        "        response::GmresOptions response_options;",
        "        result.forces =",
    )
    mp2_force_entry = _statement(mp2, "const bool force_started_with_reference_plan =")
    mp2_measurement = _statement(mp2, "last_->measured_endpoint_peak_bytes =")
    mp2_diagnostic = "".join(
        _statement(mp2, field)
        for field in (
            "diagnostic.numeric_capacity_bytes = std::max(reference_capacity, energy_capacity)",
            "diagnostic.reference_execution_plan_reused =",
            "diagnostic.reference_execution_plan_owned_device_bytes =",
        )
    )
    cc_force = _between(
        cc,
        "    std::unique_ptr<posthf::RawSource> force_raw_source;",
        "    auto diagnostic = state.diagnostic;",
    )
    cc_force_accounting = _between(
        cc,
        "    const auto force_capacity =",
        "    execution_.observe_numeric_peak(runtime::ExecutionMemorySpace::Host",
    )
    triples_run = _between(
        triples,
        "      const auto phase_budget =",
        "      const double triples_seconds =",
    )
    triples_accounting = _statement(
        triples, "diagnostic.numeric_capacity_bytes = std::max<std::uint64_t>("
    )
    triples_force = _between(
        triples,
        "        std::unique_ptr<posthf::RawSource> force_raw_source;",
        "        if (execution_.cuda_requested()) {",
    )
    triples_force_accounting = _between(
        triples,
        "        const auto force_capacity =",
        "        execution_.observe_numeric_peak(runtime::ExecutionMemorySpace::Host",
    )
    # Include the real post-callback scientific-provenance acceptance gates.
    cc_acceptance = _between(
        cc[cc.index("    auto diagnostic = state.diagnostic;") :],
        "      if (!force.lambda.cuda_actions",
        "      execution_.observe_numeric_peak(",
    )
    triples_acceptance = _between(
        triples[
            triples.index("        auto force = run_with_rccsd_reference_source(") :
        ],
        "          if (!force.lambda.cuda_actions",
        "          execution_.observe_numeric_peak(",
    )
    code = (
        PLAN_STANDINS
        + CC_CALLBACK_STANDINS
        + ENDPOINT_CALLBACK_STANDINS
        + r"""
void mp2_probe(int failure, std::size_t warm_capacity) {
  reset_endpoint(failure);
  CorrelatedCudaReferencePlan cuda_reference_plan_;
  *cuda_reference_plan_.slot() = new scf::CudaRhfBucketPlan{30,10};
  auto* reference_plan = cuda_reference_plan_.slot();
  const std::size_t phase_budget = 100, reference_capacity = 80;
  const bool density_fitted_ = false, fitted_cuda_ = false, cuda = true;
  const bool reference_plan_reused = true;
  const double threshold_ = 1e-12;
  struct { double density_fitting_relative_threshold = 1e-10; } options_;
  struct { int device_id = 0; } context_;
  core::System system_;
  std::optional<core::System> auxiliary_;
  scf::PreparedFockPlan* prepared_exact = nullptr;
  hf::PhysicalReference ref;
  std::optional<mp2::ConventionalForceResult> force_diagnostic;
  bool rejected = false;
  try {
"""
        + mp2_energy
        + mp2_force_entry
        + mp2_force
        + r"""
    generativeqc_correlation_diagnostic diagnostic{};
"""
        + mp2_diagnostic
        + "    auto* last_ = &diagnostic;\n"
        + mp2_measurement
        + r"""
    if (failure == 3) {
      assert((force_budgets == std::vector<std::size_t>{100}) && raw_constructions == 2);
      assert(diagnostic.numeric_capacity_bytes == std::max(reference_capacity,70+warm_capacity));
    } else {
      assert((force_budgets == std::vector<std::size_t>{70,100}));
      assert(diagnostic.numeric_capacity_bytes == 100+warm_capacity);
    }
    assert(diagnostic.reference_execution_plan_reused == 1);
    assert(diagnostic.reference_execution_plan_owned_device_bytes == 0);
    assert(force_diagnostic && force_diagnostic->forces.at(0) == 5);
    assert(force_diagnostic->measured_endpoint_peak_bytes == 45);
    assert(force_started_with_reference_plan == (failure != 3));
    // A successful retry's allocator telemetry omits the reference owner
    // present during the failed force attempt. Only retirement before force
    // begins, with no warm reservation, permits this nonzero measurement.
    assert(diagnostic.measured_endpoint_peak_bytes ==
           (!warm_capacity && failure == 3 ? 45 : 0));
  } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected == (failure == 2));
  if (rejected) assert(force_budgets.size() == 1 && !force_diagnostic && reference_plan && *reference_plan);
}
void cc_force_probe(int failure, bool borrowed=false) {
  reset_endpoint(failure);
  CorrelatedCudaReferencePlan cuda_reference_plan_;
  *cuda_reference_plan_.slot() = new scf::CudaRhfBucketPlan{30,10};
  std::unique_ptr<scf::PreparedFockPlan> cpu_exact_plan_;
  core::System system_;
  ProbeExecution execution_;
  ProbeState state;
  if(borrowed) {
    state.reference_interaction_source=std::make_shared<ResidentSource>(
        std::exchange(*cuda_reference_plan_.slot(),nullptr));
    state.problem.reference_retained_bytes=30;
    state.budget=100;state.reference_execution_plan_bytes=0;
    state.reference_execution_plan_device_bytes=0;
  }
  const auto* reference_identity = state.reference.get();
  bool rejected = false;
  try {
"""
        + cc_force
        + r"""
    auto diagnostic = state.diagnostic;
"""
        + cc_acceptance
        + cc_force_accounting
        + r"""
    assert(failure != 2 && failure != 4);
    assert(state.reference.get() == reference_identity && state.budget == 100);
    assert(diagnostic.numeric_capacity_bytes == 97 && diagnostic.planned_endpoint_peak_bytes == 97);
    assert(diagnostic.reference_execution_plan_reused == 1);
    assert(diagnostic.reference_execution_plan_owned_device_bytes == 0);
    if (failure == 3) assert((force_budgets == std::vector<std::size_t>{100}) && raw_constructions == 2);
    else assert((force_budgets == (borrowed ? std::vector<std::size_t>{100,100}
                                           : std::vector<std::size_t>{70,100})));
    if(borrowed) assert(!state.reference_interaction_source &&
                       state.problem.reference_retained_bytes==0 && !cuda_reference_plan_.get());
  } catch (const std::runtime_error&) { rejected = true; }
    catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected == (failure == 2 || failure == 4));
  if (rejected) {
    assert(force_budgets.size()==1 && bool(cuda_reference_plan_.get())==!borrowed);
    assert(bool(state.reference_interaction_source)==borrowed);
  }
}
void triples_probe(int failure, bool fail_triples, bool borrowed=false) {
  reset_endpoint(failure);
  CorrelatedCudaReferencePlan cuda_reference_plan_;
  *cuda_reference_plan_.slot() = new scf::CudaRhfBucketPlan{30,10};
  std::unique_ptr<scf::PreparedFockPlan> cpu_exact_plan_;
  core::System system_;
  ProbeExecution execution_;
  ProbeState state;
  if(borrowed) {
    state.reference_interaction_source=std::make_shared<ResidentSource>(
        std::exchange(*cuda_reference_plan_.slot(),nullptr));
    state.problem.reference_retained_bytes=30;
    state.budget=100;state.reference_execution_plan_bytes=0;
    state.reference_execution_plan_device_bytes=0;
  }
  const auto* reference_identity = state.reference.get();
  const auto retained_capacity = [&] { return 20+state.problem.reference_retained_bytes; };
  auto retained = retained_capacity();
  const double denominator_threshold = 1e-10, force_denominator_threshold = 1e-10;
  double triples_energy = 0, triples_minimum_denominator = 0;
  std::size_t triples_virtual_count = 0, triples_workspace_bytes = 0;
  triples_should_fail = fail_triples;
  bool rejected = false;
  try {
"""
        + triples_run
        + r"""
    auto diagnostic = state.diagnostic;
"""
        + triples_accounting
        + r"""
    const auto completed_triples_peak = diagnostic.numeric_capacity_bytes;
    if (fail_triples) {
      assert((triples_budgets == std::vector<std::size_t>{50,80}));
      assert(triples_energy == -0.25 && triples_virtual_count == 2 && state.budget == 100);
      assert(!cuda_reference_plan_.get() && diagnostic.reference_execution_plan_reused == 1);
      return;
    }
"""
        + triples_force
        + triples_acceptance
        + triples_force_accounting
        + r"""
    assert(failure != 2 && failure != 4);
    assert(completed_triples_peak == 111 && diagnostic.numeric_capacity_bytes == 111);
    assert(diagnostic.planned_endpoint_peak_bytes == 111 && state.budget == 100);
    assert(diagnostic.reference_execution_plan_reused == 1);
    assert(diagnostic.reference_execution_plan_owned_device_bytes == 0);
    assert(state.reference.get() == reference_identity && state.reference_execution_plan_bytes == 0);
    if (failure == 3) assert((force_budgets == std::vector<std::size_t>{100}) && raw_constructions == 2);
    else assert((force_budgets == (borrowed ? std::vector<std::size_t>{100,100}
                                           : std::vector<std::size_t>{70,100})));
    if(borrowed) assert(!state.reference_interaction_source &&
                       state.problem.reference_retained_bytes==0 && !cuda_reference_plan_.get());
  } catch (const std::runtime_error&) { rejected = true; }
    catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected == (failure == 2 || failure == 4));
  if (rejected) {
    assert(bool(cuda_reference_plan_.get())==!borrowed);
    assert(bool(state.reference_interaction_source)==borrowed);
  }
}
int main() {
  for (std::size_t warm_capacity : {0U,11U})
    for (int failure : {0,1,2,3}) mp2_probe(failure,warm_capacity);
  for (int failure : {0,1,2,3,4}) {
    cc_force_probe(failure);
    triples_probe(failure, false);
  }
  for (int failure : {0,1,2}) triples_probe(failure, true);
  for(int failure : {0,1,2,4}) {
    cc_force_probe(failure,true);
    triples_probe(failure,false,true);
  }
  for(int failure : {0,1,2}) triples_probe(failure,true,true);
  std::cout << "actual MP2/CC force/triples budget and completed-peak gates passed\n";
}
"""
    )
    assert "completed-peak gates passed" in _compile_run(tmp_path, code)


ENDPOINT_CALLBACK_STANDINS = r"""
#include <limits>
#include <tuple>
#include "response/native_gmres.hpp"
using posthf::checked_add;
bool triples_should_fail = false;
std::vector<std::size_t> force_budgets, triples_budgets;
void reset_endpoint(int failure) {
  reset_injection(failure);
  // Mode 3 injects an allocation failure while lazily constructing RawSource.
  raw_constructions = failure == 3 ? 0 : 1;
  force_budgets.clear(); triples_budgets.clear(); triples_should_fail = false;
}
struct MethodError : std::runtime_error {
  MethodError(generativeqc_status, const char* message) : std::runtime_error(message) {}
};
void admit_force(std::size_t budget, std::size_t stage_budget) {
  force_budgets.push_back(budget);
  assert(stage_budget == budget);
  if (injected_failure != 3 && injected_failure != 4 && force_budgets.size() == 1)
    injected_exception();
}
struct ProbeForce {
  struct { bool cuda_actions = true; std::size_t owned_device_bytes = 1; } lambda;
  bool cuda_response_actions = true;
  std::size_t response_owned_device_bytes = 1, numeric_capacity_bytes = 50;
};
namespace generativeqc::cc {
template<class... Args> ProbeForce rccsd_force_cuda(const Args&... args) {
  PhaseOwner partial_force;
  const auto values = std::tie(args...);
  admit_force(std::get<7>(values),std::get<9>(values));
  ProbeForce result;
  if (injected_failure == 4) result.lambda.cuda_actions = false;
  return result;
}
template<class... Args> ProbeForce rccsdt_force_cuda(const Args&... args) {
  return rccsd_force_cuda(args...);
}
template<class... Args> ProbeForce rccsd_force_cpu(const Args&...) {
  throw std::logic_error("CUDA force callback dispatched a CPU owner");
}
template<class... Args> ProbeForce rccsdt_force_cpu(const Args&...) {
  throw std::logic_error("CUDA force callback dispatched a CPU owner");
}
namespace triples {
struct ProbeTriples {
  double energy = -0.25, minimum_absolute_denominator = 2;
  std::size_t virtual_triples = 2, workspace_bytes = 50;
};
template<class... Args> ProbeTriples evaluate_cuda(const Args&... args) {
  PhaseOwner partial_triples;
  const auto values = std::tie(args...);
  triples_budgets.push_back(std::get<11>(values));
  if (triples_should_fail && triples_budgets.size() == 1) injected_exception();
  return {};
}
namespace generated {
template<class... Args> ProbeTriples evaluate(const Args&...) {
  throw std::logic_error("CUDA triples callback dispatched a CPU owner");
}
}
}
}
namespace generativeqc::mp2 {
struct ProbeEnergy { std::size_t numeric_capacity_bytes = 70; };
struct ConventionalForceResult {
  std::vector<double> forces{5};
  std::size_t measured_endpoint_peak_bytes = 45;
};
template<class... Args> ProbeEnergy conventional_energy(const Args&... args) {
  assert(std::get<2>(std::tie(args...)) == (injected_failure == 3 ? 100 : 70));
  return {};
}
template<class... Args> ProbeEnergy density_fitted_energy(const Args&...) {
  throw std::logic_error("conventional callback dispatched fitted energy");
}
template<class... Args> ConventionalForceResult conventional_force_cuda(const Args&... args) {
  PhaseOwner partial_force;
  const auto values = std::tie(args...);
  admit_force(std::get<2>(values), std::get<5>(values).max_workspace_bytes);
  return {};
}
template<class... Args> ConventionalForceResult conventional_force_cpu(const Args&...) {
  throw std::logic_error("CUDA callback dispatched CPU MP2 force");
}
template<class... Args> ConventionalForceResult density_fitted_force_cpu(const Args&...) {
  throw std::logic_error("conventional callback dispatched fitted force");
}
}
"""

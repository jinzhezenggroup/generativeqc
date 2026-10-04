"""Host-execute native-reference dispatch and optional CUDA source admission."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, program: str) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source, executable = tmp_path / "probe.cpp", tmp_path / "probe"
    source.write_text(program)
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cuda_source_checked_phase_admission(tmp_path: Path) -> None:
    _run(tmp_path, CAPACITY)


def test_cuda_reference_finishes_before_optional_source(tmp_path: Path) -> None:
    text = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    assert text.index("const auto problem_started =") < text.index(
        "const auto source_plan = plan_correlated_cuda_source("
    )
    retained = text[
        text.index("std::size_t retained_reference_bytes(") : text.index(
            "cc::Problem build_problem("
        )
    ]
    start = text.index("RccsdNativeState execute_rccsd_prepared(")
    body = text[
        start : text.index('    allocation_stage = "MO provider/problem";', start)
    ]
    # Execute the production reference and source-preparation prefix. Correlated
    # kernels are outside this probe; their existing source-lifetime test remains.
    body += "return {bool(prepared_exact), source_preparation_peak, bool(borrowed_reference_source)}; } catch (...) { throw; } }\n"
    helper = (ROOT / "src/methods/correlated_cuda_reference.hpp").read_text()
    helper = helper[
        helper.index("template <class Operation>") : helper.index("/** Method-layer")
    ]
    _run(tmp_path, PREFIX + helper + retained + body + MAIN)


def test_cuda_reference_resident_interaction_handoff_is_preferred() -> None:
    entry = (ROOT / "src/scf/cuda_hf_entry.cpp").read_text()
    direct = (ROOT / "src/scf/cuda/direct_jk_kernels.cu").read_text()
    rccsd = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    rccsdt = (ROOT / "src/methods/rccsdt_method.cpp").read_text()
    assert "exact_reference_source_identity" in entry
    assert "run_rhf_cuda_cached(&owner.plan" in entry
    assert "std::exchange(*plan, nullptr)" in entry
    assert "launch_copy_resident_eri_tile" in entry
    assert "copy_resident_eri_tile_kernel" in direct
    assert (
        "if (cuda && cuda_source_cache && !correlation_auxiliary && "
        "!borrowed_reference_source)" in rccsd
    )
    assert rccsd.index("if (state.reference_interaction_source)") < rccsd.index(
        "else if (prepared_exact)"
    )
    assert "state.reference_interaction_source.get()" in rccsdt


def test_provider_schedule_checks_attempt_deltas(tmp_path: Path) -> None:
    text = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    assert "const auto initial_work = provider_work;" in text
    validation = text[
        text.index(
            "  const auto source_scans = provider_work.source_scans"
        ) : text.index("  p.provider_peak_bytes =")
    ]
    _run(
        tmp_path,
        r"""
#include <vector>
#include "posthf/native_provider.hpp"
using namespace generativeqc;
void validate(const posthf::ProviderWork& initial_work, const posthf::ProviderWork& provider_work) {
  constexpr std::size_t n=2;
  struct {std::vector<int> batches{0};} reuse;
  struct {std::size_t source_reads=4;} source_tile_plan;
"""
        + validation
        + r"""
}
int main() {
  posthf::ProviderWork before{}, after{};
  before.source_scans=2; before.source_reads=7; before.source_values=27;
  after=before; after.source_scans+=1; after.source_reads+=4; after.source_values+=16;
  validate(before,after);
  if (after.source_scans!=3 || after.source_reads!=11 || after.source_values!=43) return 1;
  for (int field=0;field<3;++field) {
    auto invalid=after;
    if (field==0) ++invalid.source_scans;
    if (field==1) ++invalid.source_reads;
    if (field==2) ++invalid.source_values;
    bool refused=false;
    try { validate(before,invalid); } catch (const std::logic_error&) { refused=true; }
    if (!refused) return 2;
  }
}
""",
    )


CAPACITY = r"""
#include <limits>
#include "methods/correlated_cuda_source.hpp"
int main() {
  using generativeqc::methods::detail::plan_correlated_cuda_source;
  for (std::size_t n : {2U,14U,100U,1000U}) {
    for (std::size_t width : {32U,64U,128U}) {
      const auto plan=plan_correlated_cuda_source(n,2*n,n,n,3*n,n/2,n/2,
                                                 width,128*n,4096,8192,1ULL<<50);
      if (!plan.admitted || plan.device_bytes!=8192 ||
          plan.peak_bytes!=std::max(plan.one_electron_phase_bytes,plan.direct_phase_bytes)) return 1;
      const auto exact=plan_correlated_cuda_source(n,2*n,n,n,3*n,n/2,n/2,
                                                  width,128*n,4096,8192,plan.peak_bytes);
      const auto short_plan=plan_correlated_cuda_source(n,2*n,n,n,3*n,n/2,n/2,
                                                       width,128*n,4096,8192,plan.peak_bytes-1);
      if (!exact.admitted || short_plan.admitted) return 2;
      const auto larger_reference=plan_correlated_cuda_source(n,2*n,n,n,3*n,n/2,n/2,
                                                             width,128*n,4196,8192,1ULL<<50);
      if (larger_reference.peak_bytes!=plan.peak_bytes+100) return 3;
    }
  }
  bool overflow=false, invalid=false;
  try { (void)plan_correlated_cuda_source(2,2,1,std::numeric_limits<std::size_t>::max(),
                                         2,1,1,32,0,0,1,1); }
  catch (const std::overflow_error&) { overflow=true; }
  try { (void)plan_correlated_cuda_source(2,2,1,2,2,1,1,0,0,0,1,1); }
  catch (const std::invalid_argument&) { invalid=true; }
  return !(overflow && invalid);
}
"""

PREFIX = r"""
#include <chrono>
#include <memory>
#include <utility>
#include <vector>
#include "hf/reference.hpp"
#include "methods/correlated_cuda_source.hpp"
#include "scf/cuda/df_source_domain.hpp"
#define GENERATIVEQC_HAS_CUDA 1
int live=0,native_calls=0,host_calls=0,allocations=0;
int source_attempts=0;
int warm_failure=0;
bool converged=true,fail_source=false,borrow_resident=false;
std::size_t given_budget=0;
namespace generativeqc::integrals { struct ElectronInteractionSource { std::shared_ptr<void> owner; }; }
namespace generativeqc::molecule {
std::size_t ao_count(const core::System& s) noexcept { return s.shells.size(); }
std::size_t cartesian_ao_count(const core::System& s) noexcept { return s.shells.size(); }
}
namespace generativeqc::runtime {
struct ExecutionContext { bool cuda=true; bool cuda_requested() const {return cuda;}
                          int device_id() const {return 0;} };
}
namespace generativeqc::cc { struct SolverOptions {std::size_t max_bytes=1<<20;}; }
namespace generativeqc::scf {
namespace cuda_execution { constexpr std::size_t kResidentPsssThreads=128; }
enum class FockSpin { Restricted }; enum class FockBackend { Cuda };
// Match the real FockBuildSpec default: value-only consumers must opt out.
struct FockSpec {unsigned derivative_order=1;};
FockSpec make_hf_fock_spec(FockSpin) {return {};}
int resolve_fock_build(FockSpec spec,FockBackend,double) {
  if (spec.derivative_order) throw std::runtime_error("unadmitted source derivatives");
  return 0;
}
struct ScfOptions {int resolved_fock_build=0;};
struct CudaRhfBucketPlan {};
std::size_t hf_cuda_owned_device_bytes(const CudaRhfBucketPlan*) noexcept { return 0; }
std::size_t hf_cuda_retained_numeric_bytes(const CudaRhfBucketPlan* plan) noexcept {
  return plan ? 64 : 0;
}
void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan* plan) noexcept { delete plan; }
struct PreparedFockPlan {
  PreparedFockPlan() {++live;}
  PreparedFockPlan(const core::System&,std::nullptr_t,int,int,std::size_t budget) {
    if (!native_calls || live) throw std::runtime_error("source preceded native reference");
    given_budget=budget;
    std::vector<double> partial_storage(16);
    ++source_attempts;
    if (fail_source) throw std::bad_alloc();
    ++live; ++allocations;
  }
  ~PreparedFockPlan() {--live;}
  int strategy() const {return 0;}
};
struct ScfResult {
  bool converged=false;
  std::vector<double> density;
  std::shared_ptr<const hf::PhysicalReference> reference;
};
ScfResult physical() {
  auto ref=std::make_shared<hf::PhysicalReference>(); ref->nbf=2; ref->nocc=1;
  ref->density.resize(4);
  return {converged,{},converged ? ref : nullptr};
}
ScfResult run_prepared_fock_strategy(const PreparedFockPlan&,const ScfOptions&,
                                    const std::vector<double>*) {
  ++host_calls; return physical();
}
ScfResult run_rhf_cuda(
    const core::System&,const ScfOptions&,int,const std::vector<double>* seed,
    std::shared_ptr<const integrals::ElectronInteractionSource>* source) {
  if (live) throw std::runtime_error("prior source survived into native reference");
  ++native_calls;
  if (seed && warm_failure==1) throw std::runtime_error("injected warm reference failure");
  if (seed && warm_failure==2) return {};
  if (borrow_resident && source) *source=std::make_shared<integrals::ElectronInteractionSource>();
  return physical();
}
ScfResult run_rhf_cuda_cached(CudaRhfBucketPlan** plan,const core::System& system,
                              const ScfOptions& options,int device,
                              const std::vector<double>* seed,bool* reused,
                              std::shared_ptr<const integrals::ElectronInteractionSource>* source) {
  const bool had_plan=*plan!=nullptr;
  if (!*plan) *plan=new CudaRhfBucketPlan;
  if (reused) *reused=had_plan;
  auto result=run_rhf_cuda(system,options,device,seed,source);
  if(borrow_resident && source && *source) {
    auto resident=std::make_shared<integrals::ElectronInteractionSource>();
    resident->owner=std::shared_ptr<CudaRhfBucketPlan>(std::exchange(*plan,nullptr));
    *source=std::move(resident);
  }
  return result;
}
ScfResult run_rhf(const core::System&,const ScfOptions&,const std::vector<double>*) {
  ++host_calls; return physical();
}
std::size_t cuda_direct_jk_device_bytes(std::size_t,std::size_t,std::size_t,
                                      std::size_t,std::size_t,unsigned) {return 8192;}
}
namespace generativeqc::methods::detail {
struct MethodError : std::runtime_error {MethodError(int,const char* s):std::runtime_error(s){}};
struct RccsdNativeState {bool prepared; std::size_t peak; bool borrowed;};
"""

MAIN = r"""
}
int main() {
  using namespace generativeqc;
  core::System system; system.atoms.resize(1); system.shells.resize(2);
  for (auto& shell : system.shells) shell.primitives.resize(1);
  runtime::ExecutionContext execution; scf::ScfOptions reference; cc::SolverOptions solver;
  std::unique_ptr<scf::PreparedFockPlan> cache;
  const std::vector<double> seed(4,0.5);
  for (int mode=0;mode<7;++mode) {
    cache=std::make_unique<scf::PreparedFockPlan>();
    native_calls=host_calls=allocations=0; given_budget=0;
    converged=mode!=2; fail_source=mode==3; solver.max_bytes=mode==1 ? 1 : 1<<20;
    warm_failure=mode>=5 ? mode-4 : 0;
    bool fallback=false;
    try {
      auto result=methods::detail::execute_rccsd_prepared(
          execution,system,reference,solver,0,cache.get(),mode>=5 ? &seed : nullptr,
          &fallback,&cache);
      if (mode==2 || native_calls!=(mode>=5 ? 2 : 1) || host_calls) return 1;
      if (result.prepared!=(mode==0 || mode>=4)) return 2;
      if ((mode==0 || mode>=3) && given_budget!=8192) return 3;
      if (mode==1 && allocations) return 4;
      if (fallback!=(mode>=5)) return 8;
    } catch (const methods::detail::MethodError&) {
      if (mode!=2 || allocations || cache) return 5;
    }
    cache.reset();
    if (live) return 6;
  }
  // An admitted constructor attempt may allocate and unwind before failing.
  // Its complete conservative bound still includes the simultaneous RHF plan.
  // A rejected estimate never starts construction and contributes no peak.
  const auto source_plan=methods::detail::plan_correlated_cuda_source(
      2,2,1,2,2,2,0,scf::cuda_execution::kResidentPsssThreads,
      posthf::source_capacity(system),
      methods::detail::retained_reference_bytes(*scf::physical().reference),8192,1<<20);
  if (!source_plan.admitted) return 9;
  for (int mode=0;mode<3;++mode) {
    auto* plan=new scf::CudaRhfBucketPlan;
    native_calls=host_calls=allocations=source_attempts=0;
    given_budget=0; converged=true; warm_failure=0; fail_source=mode==1;
    solver.max_bytes=source_plan.peak_bytes+64-(mode==2);
    auto result=methods::detail::execute_rccsd_prepared(
        execution,system,reference,solver,0,nullptr,nullptr,nullptr,&cache,
        nullptr,false,nullptr,&plan);
    if (result.peak!=(mode==2 ? 0 : source_plan.peak_bytes+64)) return 10;
    if (source_attempts!=(mode==2 ? 0 : 1) || allocations!=(mode==0)) return 11;
    if (result.prepared!=(mode==0) || !plan || native_calls!=1) return 12;
    cache.reset();
    scf::destroy_rhf_cuda_bucket_plan(plan);
    if (live) return 13;
  }
  // The real dispatch must use the returned resident source and skip a second
  // PreparedFock owner in both cached and uncached reference adapters.
  borrow_resident=true; converged=true; fail_source=false; warm_failure=0;
  for(bool cached : {false,true}) {
    auto* plan=cached?new scf::CudaRhfBucketPlan:nullptr;
    native_calls=host_calls=allocations=source_attempts=0;
    solver.max_bytes=1<<20;
    const auto borrowed=methods::detail::execute_rccsd_prepared(
        execution,system,reference,solver,0,nullptr,nullptr,nullptr,&cache,
        nullptr,false,nullptr,cached?&plan:nullptr);
    if(!borrowed.borrowed || borrowed.prepared || borrowed.peak || plan || cache ||
       source_attempts || allocations || native_calls!=1) return 14;
  }
  borrow_resident=false;
  execution.cuda=false; converged=true; fail_source=false;
  cache=std::make_unique<scf::PreparedFockPlan>(); native_calls=host_calls=0;
  auto* original=cache.get();
  auto result=methods::detail::execute_rccsd_prepared(
      execution,system,reference,solver,0,cache.get(),nullptr,nullptr,&cache);
  if (!result.prepared || cache.get()!=original || native_calls || host_calls!=1) return 7;
}
"""

"""Host-execute native-reference dispatch and optional CUDA source admission."""

import shutil
import subprocess
from pathlib import Path

import pytest
from _cpp_source_support import cpp_function_definition

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, program: str) -> None:
    compiler = shutil.which("c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source, executable = tmp_path / "probe.cpp", tmp_path / "probe"
    source.write_text(program)
    compiled = subprocess.run(
        [
            cache,
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


def test_cuda_reference_handoffs_preserve_single_source_owner(tmp_path: Path) -> None:
    text = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    start = text.index("RccsdNativeState execute_rccsd_prepared(")
    body = text[
        start : text.index('    allocation_stage = "MO provider/problem";', start)
    ]
    # Execute the real reference/adapter prefix. Both resident and detached
    # adapters can populate a source; the consumer must retain just one alias.
    body += (
        "return {bool(prepared_exact), source_preparation_peak, "
        "bool(borrowed_reference_source), borrowed_reference_source.use_count(), "
        "bool(reference_source.source), correlation_options.max_bytes, "
        "reference_plan_reused}; } catch (...) { throw; } }\n"
    )
    helper = (ROOT / "src/methods/correlated_cuda_reference.hpp").read_text()
    helper = helper[
        helper.index("template <class Operation>") : helper.index("/** Method-layer")
    ]
    warm = cpp_function_definition(
        (ROOT / "src/methods/correlated_warm_reference.hpp").read_text(),
        "solve_reference",
        include_template=True,
    )
    # Execute the actual shared policy with real method errors and SCF result
    # types. Only scientific providers and their counted storage are doubles.
    warm = "namespace warm_reference {\n" + warm + "\n}\n"
    _run(tmp_path, PREFIX + warm + helper + body + MAIN)


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
    assert "plan_correlated_cuda_source" not in rccsd
    assert rccsd.index("if (state.reference_interaction_source)") < rccsd.index(
        "else if (prepared_exact)"
    )
    assert "state.reference_interaction_source.get()" in rccsdt


PREFIX = r"""
#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>
#include <typeinfo>
#include <vector>
#include "generativeqc/generativeqc.h"
#include "hf/reference.hpp"
#include "methods/execution_failure.hpp"
#include "scf/cuda/df_source_domain.hpp"
#include "scf/rhf_source_handoff.hpp"
#define GENERATIVEQC_HAS_CUDA 1
int live=0,source_live=0,plan_live=0,native_calls=0,host_calls=0;
int source_mode=0,warm_failure=0,df_calls=0,df_live=0;
bool converged=true,expect_source=true,df_retryable=true;
namespace generativeqc::integrals {
struct ElectronInteractionSource {
  ElectronInteractionSource() { ++source_live; }
  ~ElectronInteractionSource() { --source_live; }
  std::shared_ptr<void> owner;
};
}
namespace generativeqc::cc { struct SolverOptions { std::size_t max_bytes=1024; }; }
namespace generativeqc::scf {
struct CudaRhfBucketPlan {
  CudaRhfBucketPlan() { ++plan_live; }
  ~CudaRhfBucketPlan() { --plan_live; }
};
std::size_t hf_cuda_retained_numeric_bytes(const CudaRhfBucketPlan* plan) noexcept {
  return plan ? 64 : 0;
}
void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan* plan) noexcept { delete plan; }
struct PreparedFockPlan {
  PreparedFockPlan() { ++live; }
  ~PreparedFockPlan() { --live; }
  ResolvedFockBuild strategy() const { return {}; }
};
ScfResult physical() {
  auto ref=std::make_shared<hf::PhysicalReference>(); ref->nbf=2; ref->nocc=1;
  ref->density.resize(4);
  ScfResult result; result.converged=converged;
  result.reference=converged ? ref : nullptr; return result;
}
ScfResult run_prepared_fock_strategy(const PreparedFockPlan&,const ScfOptions&,
                                    const std::vector<double>*) {
  ++host_calls; return physical();
}
ScfResult run_rhf_cuda(
    const core::System&,const ScfOptions&,int,const std::vector<double>* seed,
    std::shared_ptr<const integrals::ElectronInteractionSource>* source=nullptr,
    CudaRhfSourceHandoff* handoff=nullptr) {
  if (live || source_live) throw std::logic_error("prior source survived into native reference");
  if (bool(source)!=expect_source || bool(handoff)!=expect_source)
    throw std::logic_error("wrong conventional/DF source dispatch");
  ++native_calls;
  if (source) source->reset();
  if (handoff) *handoff={};
  if (converged && source_mode==1 && source) {
    *source=std::make_shared<integrals::ElectronInteractionSource>();
    // Exercise an adapter that reports the same source in both output slots.
    if (handoff) handoff->source=*source;
  } else if (converged && source_mode==2 && handoff) {
    handoff->source=std::make_shared<integrals::ElectronInteractionSource>();
    handoff->retained_numeric_bytes=32;
    handoff->numeric_peak_bytes=256;
  } else if (source_mode==3 && handoff) {
    handoff->required_peak_bytes=2048;
    handoff->resource_fallback=true;
  }
  if (seed && warm_failure==1) throw std::invalid_argument("injected warm reference failure");
  if (warm_failure==3) throw std::runtime_error("injected terminal reference failure");
  if (warm_failure==4) throw std::bad_alloc();
  if (warm_failure==5) throw methods::MethodError(
      GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "injected capability refusal");
  if (warm_failure==6) throw std::length_error("injected reference capacity refusal");
  if (seed && warm_failure==2) return {};
  return physical();
}
ScfResult run_rhf_cuda_cached(
    CudaRhfBucketPlan** plan,const core::System& system,const ScfOptions& options,int device,
    const std::vector<double>* seed,bool* reused,
    std::shared_ptr<const integrals::ElectronInteractionSource>* source=nullptr,
    CudaRhfSourceHandoff* handoff=nullptr) {
  const bool had_plan=*plan!=nullptr;
  if (!*plan) *plan=new CudaRhfBucketPlan;
  if (reused) *reused=had_plan;
  auto result=run_rhf_cuda(system,options,device,seed,source,handoff);
  if (result.converged && source && *source) {
    auto resident=std::make_shared<integrals::ElectronInteractionSource>();
    resident->owner=std::shared_ptr<CudaRhfBucketPlan>(std::exchange(*plan,nullptr));
    *source=std::move(resident);
    if (handoff) handoff->source=*source;
  }
  return result;
}
ScfResult run_rhf(const core::System&,const ScfOptions&,const std::vector<double>*) {
  ++host_calls; return physical();
}
}
// This exact-source probe must never claim a successful RI-JK solve. A
// counted refusal also checks that warm retries reconstruct the same model.
namespace generativeqc::hf {
struct DFReference {
  DFReference() { if (df_live++) std::abort(); }
  ~DFReference() { --df_live; }
  scf::ScfResult solve(const core::System&, const core::System&, scf::ScfOptions,
                       int, const std::vector<double>*) {
    ++df_calls;
    if (df_retryable) throw std::invalid_argument("injected RI-JK refusal");
    throw std::runtime_error("injected RI-JK refusal");
  }
};
}
namespace generativeqc::methods::detail {
struct RccsdNativeState {
  bool prepared; std::size_t peak; bool borrowed; long owners; bool handoff_alias;
  std::size_t budget; bool reused;
};
"""

MAIN = r"""
}
int main() {
  using namespace generativeqc;
  core::System system,auxiliary;
  core::ContextState context; context.requested_backend=core::kCudaBackend;
  runtime::ExecutionContext execution(context);
  scf::ScfOptions reference; cc::SolverOptions solver;
  std::unique_ptr<scf::PreparedFockPlan> cache;
  const std::vector<double> seed(4,0.5);
  for (bool cached : {false,true}) {
    for (int mode=0;mode<4;++mode) {
      for (int failure=0;failure<3;++failure) {
        source_mode=mode; warm_failure=failure; expect_source=true; converged=true;
        cache=std::make_unique<scf::PreparedFockPlan>();
        auto* plan=cached ? new scf::CudaRhfBucketPlan : nullptr;
        native_calls=host_calls=0;
        bool fallback=false;
        const auto result=methods::detail::execute_rccsd_prepared(
            execution,system,reference,solver,0,cache.get(),failure ? &seed : nullptr,
            &fallback,&cache,nullptr,false,nullptr,cached ? &plan : nullptr);
        const bool supplied=mode==1 || mode==2;
        if (result.prepared || result.borrowed!=supplied || cache || live || source_live) return 1;
        if (result.owners!=(supplied ? 1 : 0) || result.handoff_alias) return 2;
        if (result.peak!=(mode==2 ? 256u : 0u)) return 3;
        if (native_calls!=(failure ? 2 : 1) || host_calls || fallback!=bool(failure)) return 4;
        if (bool(plan)!=(cached && mode!=1)) return 5;
        if (result.budget!=(cached && mode!=1 ? 960u : 1024u) || result.reused!=cached) return 6;
        scf::destroy_rhf_cuda_bucket_plan(plan);
        if (plan_live) return 7;
      }
    }
  }
  // A failed physical reference cannot publish or retain a scientific source.
  for (bool cached : {false,true}) {
    auto* plan=cached ? new scf::CudaRhfBucketPlan : nullptr;
    converged=false; native_calls=host_calls=0; warm_failure=0;
    try {
      (void)methods::detail::execute_rccsd_prepared(
          execution,system,reference,solver,0,nullptr,nullptr,nullptr,&cache,
          nullptr,false,nullptr,cached ? &plan : nullptr);
      return 8;
    } catch (const methods::MethodError&) {}
    if (native_calls!=1 || host_calls || source_live) return 9;
    scf::destroy_rhf_cuda_bucket_plan(plan);
  }
  // DF references do not request an exact ERI source from either adapter.
  converged=true; expect_source=false; source_mode=2;
  for (bool cached : {false,true}) {
    auto* plan=cached ? new scf::CudaRhfBucketPlan : nullptr;
    native_calls=host_calls=0;
    const auto result=methods::detail::execute_rccsd_prepared(
        execution,system,reference,solver,0,nullptr,nullptr,nullptr,&cache,
        &auxiliary,false,nullptr,cached ? &plan : nullptr);
    if (result.borrowed || result.peak || result.prepared || native_calls!=1 || host_calls) return 10;
    scf::destroy_rhf_cuda_bucket_plan(plan);
  }
  // A full-DF refusal cannot fall through to the exact adapters. A seeded
  // retry releases its previous fitted owner before constructing the next.
  for (bool seeded : {false,true}) {
    native_calls=host_calls=df_calls=0;
    try {
      (void)methods::detail::execute_rccsd_prepared(
          execution,system,reference,solver,0,nullptr,seeded ? &seed : nullptr,
          nullptr,&cache,&auxiliary,false,nullptr,nullptr,false,true);
      return 12;
    } catch (const std::invalid_argument& error) {
      if (std::string(error.what())!="injected RI-JK refusal") return 13;
    }
    if (native_calls || host_calls || df_live || df_calls!=(seeded ? 2 : 1)) return 14;
  }
  // Untyped, resource and capability errors stay terminal even with a seed.
  // The failing attempt may already own a source; unwinding must retire it.
  converged=true; expect_source=true; source_mode=1;
  for (bool cached : {false,true}) {
    for (bool seeded : {false,true}) {
      for (int failure=3;failure<=6;++failure) {
        warm_failure=failure;
        cache=std::make_unique<scf::PreparedFockPlan>();
        auto* plan=cached ? new scf::CudaRhfBucketPlan : nullptr;
        native_calls=host_calls=0;
        bool fallback=true;
        try {
          (void)methods::detail::execute_rccsd_prepared(
              execution,system,reference,solver,0,cache.get(),seeded ? &seed : nullptr,
              &fallback,&cache,nullptr,false,nullptr,cached ? &plan : nullptr);
          return 15;
        } catch (const std::exception& error) {
          const auto expected=failure==3 ? GENERATIVEQC_STATUS_NUMERICAL_FAILURE
              : failure==5 ? GENERATIVEQC_STATUS_NOT_IMPLEMENTED
              : GENERATIVEQC_STATUS_OUT_OF_MEMORY;
          if (methods::item_exception_status()!=expected) return 16;
          if (failure==3 && (typeid(error)!=typeid(std::runtime_error) ||
              std::string(error.what())!="injected terminal reference failure")) return 17;
          if (failure==4 && typeid(error)!=typeid(std::bad_alloc)) return 18;
          if (failure==5 && (typeid(error)!=typeid(methods::MethodError) ||
              std::string(error.what())!="injected capability refusal")) return 19;
          if (failure==6 && (typeid(error)!=typeid(std::length_error) ||
              std::string(error.what())!="injected reference capacity refusal")) return 20;
        }
        if (native_calls!=1 || host_calls || fallback || cache || live || source_live) return 21;
        scf::destroy_rhf_cuda_bucket_plan(plan);
        if (plan_live) return 22;
      }
    }
  }
  // An untyped fitted-provider refusal cannot authorize a cold retry either.
  warm_failure=0; df_retryable=false;
  for (bool seeded : {false,true}) {
    native_calls=host_calls=df_calls=0;
    bool fallback=true;
    try {
      (void)methods::detail::execute_rccsd_prepared(
          execution,system,reference,solver,0,nullptr,seeded ? &seed : nullptr,
          &fallback,&cache,&auxiliary,false,nullptr,nullptr,false,true);
      return 23;
    } catch (const std::runtime_error& error) {
      if (typeid(error)!=typeid(std::runtime_error) ||
          std::string(error.what())!="injected RI-JK refusal") return 24;
    }
    if (native_calls || host_calls || df_live || df_calls!=1 || fallback) return 25;
  }
  // CPU preparation remains cached and never enters the CUDA adapters.
  context.requested_backend=core::kCpuReferenceBackend;
  execution=runtime::ExecutionContext(context); cache=std::make_unique<scf::PreparedFockPlan>();
  native_calls=host_calls=0;
  auto* original=cache.get();
  const auto result=methods::detail::execute_rccsd_prepared(
      execution,system,reference,solver,0,cache.get(),nullptr,nullptr,&cache);
  if (!result.prepared || cache.get()!=original || native_calls || host_calls!=1 ||
      result.borrowed || result.peak || plan_live || source_live) return 11;
}
"""

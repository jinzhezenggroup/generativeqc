"""Host-execute production plan accounting and bucket admission without CUDA work.

The harness uses the real owner/topology types, packer, accounting functions,
cache lifecycle, and reference capacity expressions. Only CUDA execution and
runtime policy probes are replaced; no integrals or SCF iterations run.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from _eigen_handle_test_support import empty_eigen_owner_units

ROOT = Path(__file__).resolve().parents[2]


def between(source: str, start: str, end: str) -> str:
    return source[source.index(start) : source.index(end, source.index(start))]


@pytest.fixture(scope="module")
def capacity_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("correlated-reference-capacity")
    (directory / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t=void*; using cudaGraph_t=void*; "
        "using cudaGraphExec_t=void*; using cudaError_t=int; constexpr int cudaSuccess=0;\n"
    )
    (directory / "cuda_runtime.h").write_text(
        '#pragma once\n#include "cuda_runtime_api.h"\n'
        "struct dim3 { unsigned x=1,y=1,z=1; };\n"
    )
    (directory / "cublas_v2.h").write_text(
        "#pragma once\nusing cublasHandle_t=void*;\n"
    )
    eigen_units = empty_eigen_owner_units(directory)
    # The actual topology packer consumes this compiler-owned schedule.
    from generativeqc_compiler.integral.direct_resident_schedule import (
        emit_direct_resident_psss_schedule_header,
    )

    (directory / "generated_direct_resident_psss_schedule.cuh").write_text(
        emit_direct_resident_psss_schedule_header()
    )
    driver = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    reference = (ROOT / "src/scf/cuda/reference_export.cuh").read_text()
    capacity = between(reference, "inline std::size_t check_capacity(", "/** Stage")
    peak = between(
        driver,
        "    resources.reference_peak_bytes_ = reference_detail::check_capacity(\n"
        "        std::max(retained_reference_peak,",
        "    if (resources.reference_eri_ != nullptr)",
    )
    snapshot = between(
        driver,
        "      plan.reference_admitted_plan_host_bytes =",
        "      // Download has synchronized",
    )
    # Exercise every declared numeric vector, including capacity retained at size zero.
    topology = between(
        (ROOT / "src/scf/cuda/topology.hpp").read_text(), "struct HostBatch {", "\n};"
    )
    owner = between(
        (ROOT / "src/scf/cuda/rhf_bucket_internal.hpp").read_text(),
        "struct CudaRhfBucketPlan {",
        "\n};",
    )
    fields = re.findall(r"std::vector<[^>]+> (\w+);", topology)
    fields.remove("ecp_systems")
    plan_fields = re.findall(r"std::vector<[^>]+> (\w+);", owner)
    fill = (
        "\n".join(f"  charge(p.topology.{name});" for name in fields)
        + "\n"
        + "\n".join(f"  charge(p.{name});" for name in plan_fields)
    )
    source = directory / "probe.cpp"
    source.write_text(
        PREFIX
        + "namespace generativeqc::scf::reference_detail {\n"
        + capacity
        + "}\n"
        + DRIVER.replace("@PEAK@", peak).replace("@SNAPSHOT@", snapshot)
        + MAIN.replace("@FILL@", fill)
    )
    objects = []
    for unit in (
        source,
        ROOT / "src/scf/cuda/rhf_bucket.cpp",
        ROOT / "src/scf/cuda/topology.cpp",
        ROOT / "src/molecule/basis.cpp",
        *eigen_units,
    ):
        target = directory / (unit.stem + ".o")
        result = subprocess.run(
            [
                cache,
                compiler,
                "-std=c++20",
                "-O0",
                "-ffunction-sections",
                "-fdata-sections",
                "-I" + str(directory),
                "-I" + str(ROOT / "src"),
                "-I" + str(ROOT / "include"),
                "-c",
                str(unit),
                "-o",
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(target))
    executable = directory / "probe"
    result = subprocess.run(
        [compiler, *objects, "-Wl,--gc-sections", "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize(
    "mode",
    ["accounting", "warm", "short", "scientific", "growth", "freeze", "overflow"],
)
def test_reference_plan_capacity(capacity_probe: Path, mode: str) -> None:
    result = subprocess.run(
        [str(capacity_probe), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_non_cuda_query_has_matching_noexcept_stub() -> None:
    source = (ROOT / "src/scf/rhf.cpp").read_text()
    assert (
        "std::size_t hf_cuda_retained_numeric_bytes(const CudaRhfBucketPlan*) noexcept { return 0; }"
        in source
    )


PREFIX = r"""
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/reference_eri_policy.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/cuda/rhf_policy.hpp"
using namespace generativeqc;
void require(bool good,const char* message) { if(!good) throw std::runtime_error(message); }
int created=0, destroyed=0;
namespace generativeqc::scf::cuda_execution {
CudaResources::~CudaResources() { if(arena_) ++destroyed; }
RhfIterationGraphs::~RhfIterationGraphs() = default;
}
namespace generativeqc::scf::cuda_policy {
bool reuse_converged_fock_requested() noexcept { return false; }
bool graph_native_eigensolver_override_requested() noexcept { return false; }
bool bounded_direct_streaming_override_requested() noexcept { return false; }
bool bounded_fock_class_timing_requested() noexcept { return false; }
bool bounded_direct_fock_only_diagnostic_requested() noexcept { return false; }
std::optional<std::uint64_t> bounded_direct_primary_streaming_fock_mask_requested() noexcept { return {}; }
unsigned one_electron_value_mapping_requested() noexcept { return 0; }
MixedPrecisionFockPolicy resolve_mixed_precision_fock_policy(
    std::optional<generativeqc_precision_mode>,double,double,double) noexcept { return {}; }
}
namespace generativeqc::scf {
using namespace cuda_execution;
using namespace cuda_policy;
FockBuildSpec make_hf_fock_spec(FockSpin spin,FockApproximation) {
  FockBuildSpec spec; spec.spin=spin; return spec;
}
ResolvedFockBuild resolve_fock_build(FockBuildSpec spec,FockBackend backend,double tolerance,double) {
  ResolvedFockBuild result; result.spec=spec; result.backend=backend;
  result.screening_tolerance=tolerance; return result;
}
void require_exact_direct_strategy(const ResolvedFockBuild&,FockSpin,FockBackend) {}
}
"""

DRIVER = r"""
namespace generativeqc::scf {
std::vector<RhfBucketItem> execute_hf_cuda_bucket_driver(CudaRhfBucketPlan& plan,
    const HostBatch& host,const ScfOptions& options,int device_id,bool unrestricted,
    bool shell_class_profiling,bool inactive_eigensolver_profiling) {
  const bool first_setup=!plan.initialized;
  // Preserve the real driver's exact-option invariant after lifecycle admission.
  require(first_setup || same_hf_bucket_options(plan.options,options),"stale driver budget");
  require(first_setup || compatible_hf_bucket_options(plan,host,options),"unsafe driver reuse");
  const auto retained_reference_peak=first_setup?0:hf_cuda_reference_reuse_capacity(plan,host);
  if(first_setup) {
    ++created; plan.options=options; plan.topology=host;
    plan.topology.positions.clear(); plan.topology.warm_mask.clear(); plan.topology.warm_density.clear();
    plan.resources.device_id_=device_id; plan.resources.arena_=&plan; plan.layout.bytes=2456;
    plan.unrestricted=unrestricted; plan.shell_class_profiling=shell_class_profiling;
    plan.inactive_eigensolver_profiling=inactive_eigensolver_profiling;
  }
  auto& resources=plan.resources;
  const auto required=reference_detail::base_capacity(
      plan.layout.bytes,host,host.nbf*host.nbf,0,options.reference_memory_budget_bytes);
  if(first_setup) {
    resources.reference_eri_bytes_=reference_eri_cache_bytes(
        host.nbf*host.nbf*host.nbf*host.nbf,0,options.compute_forces,required,
        options.reference_memory_budget_bytes);
    if(resources.reference_eri_bytes_) resources.reference_eri_=reinterpret_cast<double*>(&plan);
  }
@PEAK@
  plan.initialized=true;
@SNAPSHOT@
  RhfBucketItem result; result.status=GENERATIVEQC_STATUS_SUCCESS; result.scf.converged=true;
  auto reference=std::make_shared<PhysicalReference>();
  reference->numeric_capacity_bytes=resources.reference_peak_bytes_;
  result.scf.reference=std::move(reference);
  return {result};
}
}
"""

MAIN = r"""
using namespace generativeqc::scf;
using namespace generativeqc::scf::cuda_execution;
struct Owner {
  CudaRhfBucketPlan* p=nullptr;
  ~Owner() { destroy_rhf_cuda_bucket_plan(p); }
};
core::System h2() {
  core::System s; s.atoms={{1,{0.,0.,0.}},{1,{0.,0.,1.4}}};
  s.shells={{0,0,{{1.,1.}}},{1,0,{{1.,1.}}}}; s.electron_count=2; return s;
}
ScfOptions options() {
  ScfOptions o; o.compute_forces=false; o.export_physical_reference=true;
  o.reference_memory_budget_bytes=256ULL<<20; return o;
}
RhfBucketItem run(Owner& owner,const ScfOptions& o,bool warm=false) {
  const auto system=h2(); std::vector<double> density(4,0.5);
  auto rows=run_rhf_cuda_bucket_cached(&owner.p,{system},o,{warm?&density:nullptr},0);
  require(rows.size()==1 && rows[0].scf.converged,"host execution failed");
  require(rows[0].scf.reference->numeric_capacity_bytes<=o.reference_memory_budget_bytes,
          "reported capacity exceeded accepted budget");
  return rows[0];
}
void accounting() {
  CudaRhfBucketPlan p;
  require(hf_cuda_retained_numeric_bytes(nullptr)==0 && hf_cuda_retained_numeric_bytes(&p)==0,
          "empty plan accounting");
  p.resources.arena_=&p; p.layout.bytes=4096;
  p.resources.solver_workspace_=&p; p.resources.solver_workspace_bytes_=256;
  p.resources.reference_eri_=reinterpret_cast<double*>(&p); p.resources.reference_eri_bytes_=128;
  p.resources.direct_tile_validation_=reinterpret_cast<DirectTileValidationRecord*>(&p);
  p.resources.solver_host_workspace_=&p; p.resources.solver_host_workspace_bytes_=1024;
  p.resources.provider_retained_bytes_=2048;
  const auto device=4096+256+128+sizeof(DirectTileValidationRecord);
  require(hf_cuda_owned_device_bytes(&p)==device,"device-only accounting changed");
  std::size_t expected=device+1024+2048;
  auto charge=[&](auto& v) { v.reserve(3); expected+=v.capacity()*sizeof(typename std::decay_t<decltype(v)>::value_type); };
@FILL@
  p.last_ppps_queue_profile.emplace(); charge(p.last_ppps_queue_profile->ket_count_histogram);
  p.last_inactive_eigensolver_profile.emplace(); charge(*p.last_inactive_eigensolver_profile);
  p.topology.ecp_systems.resize(1); auto& ecp=p.topology.ecp_systems[0];
  charge(ecp.atoms); charge(ecp.ecp_terms); ecp.shells.resize(1); charge(ecp.shells[0].primitives);
  require(hf_cuda_retained_numeric_bytes(&p)==expected,"omitted retained host/provider capacity");
  require(hf_cuda_owned_device_bytes(&p)==device,"host bytes polluted device diagnostic");
  p.resources.solver_host_workspace_=nullptr;
  require(hf_cuda_retained_numeric_bytes(&p)==expected-1024,"unallocated host workspace charged");
}
void warm() {
  Owner owner; auto cold_options=options(); auto cold=run(owner,cold_options);
  const auto count=created; const auto peak=cold.scf.reference->numeric_capacity_bytes;
  auto warm_options=cold_options; warm_options.reference_memory_budget_bytes-=80;
  require(run(owner,warm_options,true).execution_plan_reused,"first warm lost plan");
  require(created==count && owner.p->resources.reference_peak_bytes_==peak,"warm over-reservation");
  clear_rhf_cuda_bucket_warm_starts(owner.p);
  require(run(owner,cold_options).execution_plan_reused,"clear budget transition lost plan");
  set_rhf_cuda_bucket_warm_start_updates(owner.p,false);
  require(run(owner,cold_options,true).execution_plan_reused,"freeze budget transition lost plan");
  auto exact=cold_options; exact.reference_memory_budget_bytes=peak;
  require(run(owner,exact,true).execution_plan_reused,"exact complete peak not reusable");
}
void short_budget() {
  Owner owner; auto o=options(); auto cold=run(owner,o);
  const auto peak=cold.scf.reference->numeric_capacity_bytes;
  const auto eri=owner.p->resources.reference_eri_bytes_; require(eri>0,"test needs optional ERIs");
  const auto before=created; o.reference_memory_budget_bytes=peak-1;
  require(!run(owner,o,true).execution_plan_reused,"one-byte-short cache was reused");
  require(created==before+1 && owner.p->resources.reference_eri_==nullptr,"bounded fallback not rebuilt");
  require(owner.p->resources.reference_peak_bytes_==peak-eri,"fallback capacity changed");
  o.reference_memory_budget_bytes=peak-eri;
  require(run(owner,o,true).execution_plan_reused,"exact nonresident bound lost plan");
  o.reference_memory_budget_bytes-=1; bool refused=false;
  try { (void)run(owner,o); } catch(const std::length_error&) { refused=true; }
  require(refused,"one byte below mandatory reference admitted");
}
void scientific() {
  for(int field=0;field!=11;++field) {
    Owner owner; auto o=options(); run(owner,o); auto changed=o;
    switch(field) {
      case 0: changed.energy_tolerance*=2; break;
      case 1: changed.density_tolerance*=2; break;
      case 2: changed.screening_tolerance*=2; break;
      case 3: ++changed.diis_history; break;
      case 4: ++changed.max_iterations; break;
      case 5: changed.compute_forces=true; break;
      case 6: changed.precision_mode=GENERATIVEQC_PRECISION_FP64; break;
      case 7: changed.incremental_direct_jk=true; break;
      case 8: ++changed.incremental_direct_jk_rebuild_interval; break;
      case 9: changed.incremental_direct_jk_density_rms_threshold=1e-3; break;
      case 10:
        changed.resolved_fock_build=owner.p->options.resolved_fock_build;
        changed.resolved_fock_build->spec.exchange.coefficient=-0.25; break;
    }
    require(!run(owner,changed).execution_plan_reused,"scientific option mismatch reused plan");
  }
}
void growth() {
  Owner owner; auto o=options(); run(owner,o);
  HostBatch candidate; require(pack_host_batch({h2()},{nullptr},candidate,false,true,false),"packing");
  const auto old=owner.p->resources.reference_peak_bytes_;
  const auto initial=hf_cuda_reference_reuse_capacity(*owner.p,candidate);
  require(initial==old,"unchanged host grew admission");
  owner.p->resident_warm_density.reserve(20);
  const auto delta=runtime::vector_bytes(owner.p->resident_warm_density);
  require(hf_cuda_reference_reuse_capacity(*owner.p,candidate)==old+delta,"host growth not charged");
  clear_rhf_cuda_bucket_warm_starts(owner.p);
  require(hf_cuda_reference_reuse_capacity(*owner.p,candidate)==old+delta,"clear freed capacity on paper");
  auto exact=o; exact.reference_memory_budget_bytes=old+delta;
  require(run(owner,exact).execution_plan_reused,"grown host exact bound lost plan");
  require(owner.p->resources.reference_peak_bytes_==old+delta,"driver erased extra admission");
  require(hf_cuda_reference_reuse_capacity(*owner.p,candidate)==old+delta,"host growth double charged");
  candidate.warm_density.reserve(40); const auto extra=host_batch_numeric_bytes(candidate)-owner.p->reference_admitted_candidate_host_bytes;
  require(hf_cuda_reference_reuse_capacity(*owner.p,candidate)==old+delta+4*extra,"candidate growth not charged");
  const auto original_options=owner.p->options; auto lower=original_options;
  lower.reference_memory_budget_bytes=old+delta-1;
  require(!compatible_hf_bucket_options(*owner.p,owner.p->topology,lower),"retained growth ignored");
  require(!run(owner,lower).execution_plan_reused,"host growth budget did not rebuild");
}
void freeze() {
  Owner owner; auto o=options(); run(owner,o);
  owner.p->resident_warm_positions.assign(6,1.0);
  owner.p->resident_warm_density.assign(4,0.5);
  owner.p->resident_previous_energy.assign(1,-1.0);
  set_rhf_cuda_bucket_warm_start_updates(owner.p,false);
  const auto bytes=hf_cuda_retained_numeric_bytes(owner.p);
  const auto peak=hf_cuda_reference_reuse_capacity(*owner.p,owner.p->topology);
  require(owner.p->frozen_warm_density.size()==4,"freeze did not acquire host payload");
  auto exact=o; exact.reference_memory_budget_bytes=peak;
  require(run(owner,exact,true).execution_plan_reused,"frozen host exact admission lost plan");
  clear_rhf_cuda_bucket_warm_starts(owner.p);
  set_rhf_cuda_bucket_warm_start_updates(owner.p,true);
  require(hf_cuda_retained_numeric_bytes(owner.p)==bytes,"clear/thaw forgot retained capacity");
  require(run(owner,exact).execution_plan_reused,"cleared host exact admission lost plan");
}
void overflow() {
  Owner owner; auto o=options(); run(owner,o);
  owner.p->resources.provider_retained_bytes_=std::numeric_limits<std::size_t>::max();
  require(hf_cuda_retained_numeric_bytes(owner.p)==std::numeric_limits<std::size_t>::max(),"query did not saturate");
  o.reference_memory_budget_bytes=std::numeric_limits<std::size_t>::max();
  require(!compatible_hf_bucket_options(*owner.p,owner.p->topology,o),"saturated capacity admitted");
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"test mode"); const std::string mode=argv[1];
    if(mode=="accounting") accounting(); else if(mode=="warm") warm();
    else if(mode=="short") short_budget(); else if(mode=="scientific") scientific();
    else if(mode=="growth") growth(); else if(mode=="freeze") freeze();
    else if(mode=="overflow") overflow(); else return 2;
    std::cout<<mode<<" passed\n"; return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
"""

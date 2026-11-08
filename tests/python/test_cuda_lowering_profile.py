"""Execute generated admission and actual provider preparation with host doubles.

These probes establish selection, error and resource semantics, not CUDA
arithmetic or complete molecular endpoint performance.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common.native_lowering import native_lowering_portfolio
from generativeqc_compiler.dft.xc_density_lowering import density_portfolio
from generativeqc_compiler.dft.xc_density_provider import emit_density_provider
from generativeqc_compiler.dft.xc_lowering_profile import emit_indexed_fp64_profile
from generativeqc_compiler.dft.xc_potential_lowering import emit_potential_portfolio
from test_panel_product_cleanup import CUDA, PREFIX

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]

SUPPORT = r"""
#include "runtime/cuda_device_facts.hpp"
#include "tensor/cuda_symmetric_product.hpp"
constexpr int CUBLAS_OP_T=1, CUBLAS_FILL_MODE_LOWER=0;
int cublasDsyr2k(void*,int,int,int,int,const double*,const double*,int,
                const double*,int,const double*,double*,int){std::abort();}
namespace generativeqc_tensor {
std::size_t blocks(std::size_t count,std::size_t threads){return (count+threads-1)/threads;}
}
namespace generativeqc::runtime {
CudaTargetInfo probed_target{};
int probe_error=0;
bool forbid_probe=false;
cudaError_t cuda_device_facts(int ordinal,CudaTargetInfo& target,char (&name)[256]) noexcept {
  assert(!forbid_probe && ordinal==device);
  target=probed_target; name[0]='\0'; return probe_error;
}
}
namespace generativeqc::dft {
enum class CudaXcAoPrecision {Fp64,Fp32ComputeFp64Storage};
struct CudaXcLayout {
  std::size_t nao=512,npoint=1U<<19,work_jets=4,tile_points=256,spins=1,ao_map_entries=1;
  std::uint32_t functional{};
  bool local_ao=true,response=false,fp32_contractions=false;
  CudaXcAoPrecision ao_precision=CudaXcAoPrecision::Fp64;
};
}
"""

DRIVER = r"""
int main(int argc,char** argv) {
  assert(argc==2);
  using namespace generativeqc::runtime;
  using namespace generativeqc::tensor;
  using namespace generativeqc::dft;
  using namespace generativeqc::dft::cuda_xc_detail;
  const std::string mode=argv[1];
  unsetenv("GENERATIVEQC_CUDA_XC_GGA_GEMM");
  CudaXcLayout layout;
  probed_target.compute_capability_major=8;
  probed_target.fp32_fp64_ratio=2;
  std::size_t budget=128ULL<<20;
  bool library=true;
  if(mode=="consumer") {probed_target.compute_capability_major=12;probed_target.fp32_fp64_ratio=64;library=false;}
  if(mode=="unknown") {probed_target.fp32_fp64_ratio=0;library=false;}
  if(mode=="old") {probed_target.compute_capability_major=7;library=false;}
  if(mode=="small-ao") {--layout.nao;forbid_probe=true;library=false;}
  if(mode=="small-grid") {--layout.npoint;forbid_probe=true;library=false;}
  if(mode=="gga") {layout.work_jets=1;forbid_probe=true;library=false;}
  if(mode.starts_with("gga-")) {
    layout.work_jets=1;
    layout.functional=semilocal_family_code(SemilocalFamily::Pbe);
  }
  if(mode=="gga-b3lyp") layout.functional=semilocal_family_code(SemilocalFamily::B3lyp);
  if(mode=="gga-enabled") setenv("GENERATIVEQC_CUDA_XC_GGA_GEMM","1",1);
  if(mode=="gga-rollback" || mode=="meta-rollback") {
    setenv("GENERATIVEQC_CUDA_XC_GGA_GEMM","0",1);
    if(mode=="gga-rollback") {forbid_probe=true;library=false;}
  }
  if(mode=="gga-invalid") {setenv("GENERATIVEQC_CUDA_XC_GGA_GEMM","2",1);forbid_probe=true;}
  if(mode=="gga-uks") {layout.spins=2;forbid_probe=true;library=false;}
  if(mode=="gga-other") {
    layout.functional=semilocal_family_code(SemilocalFamily::Wb97mv);
    forbid_probe=true;library=false;
  }
  if(mode=="gga-small") {--layout.nao;forbid_probe=true;library=false;}
  if(mode=="gga-consumer") {
    probed_target.compute_capability_major=12;probed_target.fp32_fp64_ratio=64;library=false;
  }
  if(mode=="dense") {layout.local_ao=false;forbid_probe=true;library=false;}
  if(mode=="empty") {layout.ao_map_entries=0;forbid_probe=true;library=false;}
  if(mode=="response") {layout.response=true;forbid_probe=true;library=false;}
  if(mode=="mixed-ao") {layout.ao_precision=CudaXcAoPrecision::Fp32ComputeFp64Storage;forbid_probe=true;library=false;}
  if(mode=="mixed-contraction") {layout.fp32_contractions=true;forbid_probe=true;library=false;}
  if(mode=="no-budget") {budget=0;forbid_probe=true;library=false;}
  if(mode=="small-budget") {budget=96ULL<<20;library=false;}
  if(mode=="unsupported") {probe_error=cudaErrorNotSupported;library=false;}
  if(mode=="invalid-attribute") {probe_error=cudaErrorInvalidValue;library=false;}
  if(mode=="probe-failure") probe_error=77;
  if(mode=="pending-failure") {probe_error=cudaErrorNotSupported;pending_error=700;}
  if(mode=="wide-tile") layout.tile_points=1024;
  if(mode=="spin") layout.spins=2;
  if(mode=="zero-reuse") forbid_probe=true;
  if(mode=="device-oom" || mode=="sync-failure" || mode=="destroy-failure") {
    device_oom=true;library=false;
    failure=mode=="sync-failure"?1:mode=="destroy-failure"?2:0;
  }
  auto ledger=std::make_shared<DeviceResourceLedger>();
  ledger->device=device;ledger->limit=128ULL<<20;
  if(mode=="ledger") {ledger->limit=0;library=false;}
  active_device_resource_ledger=ledger;
  bool runtime_failed=false,invalid=false;
  try {
    std::unique_ptr<ProbePrepared> binding;
    if(mode.starts_with("cost-")) {
      std::uint64_t replays=mode=="cost-cold"?1:10;
      library=mode!="cost-cold";
      if(mode=="cost-pack") library=false;
      binding=prepare_cost_probe(layout,budget,replays,mode=="cost-pack",mode=="cost-unknown");
    } else {
      binding=prepare_probe(layout,stream,budget,mode=="zero-reuse"?0:40);
    }
    const auto& diagnostic=binding->diagnostic();
    assert((diagnostic.candidate.provider=="cublas")==library);
    assert(!library || diagnostic.candidate.algorithm==probe_algorithm);
    assert(diagnostic.provider_allowance==(library?CudaContractionContext::kProviderAllowance:0));
    assert(diagnostic.matrix_bytes==(library?layout.spins*layout.nao*layout.nao*sizeof(double):0));
    assert(ledger->live==diagnostic.matrix_bytes);
    assert(library || live_handles==0);
    if(mode=="cost-cold" || mode=="cost-pack") assert(!library_prepared);
    forbid_probe=false;
    const auto rejection=indexed_fp64_profile_rejection(layout,budget);
    if(mode=="consumer" || mode=="unknown" || mode=="old" || mode=="gga-consumer")
      assert(rejection.find("hardware")!=std::string_view::npos);
    if(mode=="small-ao" || mode=="small-grid" || mode=="gga" || mode=="gga-small" ||
       mode=="gga-rollback" || mode=="gga-uks" || mode=="gga-other")
      assert(rejection.find("workload")!=std::string_view::npos);
    if(mode=="mixed-ao" || mode=="mixed-contraction")
      assert(rejection.find("strict FP64")!=std::string_view::npos);
  } catch(const generativeqc_tensor::DeviceRuntimeError&) {runtime_failed=true;}
    catch(const std::invalid_argument&) {invalid=true;}
  assert(runtime_failed==(mode=="probe-failure" || mode=="pending-failure" ||
                          mode=="sync-failure" || mode=="destroy-failure"));
  assert(invalid==(mode=="zero-reuse" || mode=="gga-invalid"));
  assert(device==1 && live_handles==0 && ledger->live==0 && device_allocation_owners.empty());
}
"""


@pytest.fixture(scope="module", params=["potential", "density"])
def lowering_profile_probe(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    required_native_cxx: NativeCxx,
) -> Path:
    """Compile the generated adapter and actual context/executor preparation.

    Only CUDA launch syntax and dependency includes are removed from the
    executor; no selection or cleanup logic is replaced. Provider doubles must
    not execute numerical kernels. Launch-only locals/parameters may be unused
    after syntax removal; other compiler warnings remain errors.
    """
    folder = tmp_path_factory.mktemp("cuda-lowering-profile")
    cuda = CUDA.replace(
        "inline int cudaGetLastError(){return 0;}",
        "inline int pending_error=0;\n"
        "inline int cudaGetLastError(){int result=pending_error;pending_error=0;return result;}",
    )
    cuda += "\nconstexpr int cudaErrorNotSupported=801;\n"
    (folder / "cuda_runtime_api.h").write_text(cuda)
    context = (ROOT / "src/tensor/cuda_contraction.cuh").read_text()
    start = context.index("class CudaContractionContext {")
    context = context[start : context.index("\ntemplate <class T>", start)]
    potential = request.param == "potential"
    executor_name = "cuda_symmetric_product" if potential else "cuda_panel_product"
    executor = (ROOT / f"src/tensor/{executor_name}.cuh").read_text()
    executor = re.sub(
        r"^#include.*\n|^#pragma once.*\n", "", executor, flags=re.MULTILINE
    )
    executor = re.sub(r"<<<.*?>>>", "", executor, flags=re.DOTALL)
    (folder / "tensor").mkdir()
    (folder / f"tensor/{executor_name}.cuh").write_text(executor)
    prefix = PREFIX.replace(
        "int cublasCreate(void** h){",
        "bool library_prepared=false;\nint cublasCreate(void** h){library_prepared=true;",
    )
    if potential:
        adapter = emit_potential_portfolio("profile-test")
        prepared, owner = "PreparedSymmetricProduct", "CudaSymmetricProduct"
        metadata, alternative = "xc_potential", 3
        prepare = "prepare_potential(l,s,budget,replays)"
        arguments = "true,true,true,replays"
        algorithm = "indexed-symmetric-cross-gemm"
    else:
        metadata, alternative = "density_lowering", 5
        prepared, owner = "PreparedPanelProduct", "CudaPanelProduct"
        prepare = "prepare_density_provider(l,s,0,budget,replays)"
        arguments = "0,5,true,true,replays"
        algorithm = "bounded-matrix-panel-gemm"
        provider = re.sub(r"<<<.*?>>>", "", emit_density_provider(), flags=re.DOTALL)
        adapter = (
            '#include "tensor/cuda_panel_product.cuh"\n'
            + emit_indexed_fp64_profile()
            + "namespace generativeqc::dft::cuda_xc_detail {\n"
            + native_lowering_portfolio(
                *density_portfolio(16, "profile-test"), name=metadata
            )
            + "bool tiled_xc_admitted(std::size_t,std::size_t,std::size_t,std::size_t){return false;}\n"
            + "template<class... Args> void materialize_density_factor(Args...){std::abort();}\n"
            + "template<class... Args> void gather_density_factor(Args...){std::abort();}\n"
            + provider
            + "}\n"
        )
    probe = f"""
using ProbePrepared=generativeqc::tensor::{prepared};
constexpr std::string_view probe_algorithm="{algorithm}";
using namespace generativeqc::dft;
using namespace generativeqc::dft::cuda_xc_detail;
std::unique_ptr<ProbePrepared> prepare_probe(
    const CudaXcLayout& l,cudaStream_t s,std::size_t budget,std::uint64_t replays) {{
  return {prepare};
}}
std::unique_ptr<ProbePrepared> prepare_cost_probe(const CudaXcLayout& l,
    std::size_t budget,std::uint64_t replays,bool packing,bool unknown) {{
  auto offers={metadata}_candidates;
  offers[0].cost={{"host-test-evidence",false,0,20,0,0,0,0,0}};
  offers[{alternative}].cost={{"host-test-evidence",false,100,1,0,0,0,0,0}};
  if(packing) offers[{alternative}].cost.pack_ns=30;
  if(unknown) offers[{alternative}].cost.prepare_ns=std::nullopt;
  return std::make_unique<generativeqc::tensor::{owner}>(
      {metadata}_request,offers,{metadata}_target,{metadata}_compilation,l.nao,
      l.work_jets*l.tile_points,l.spins,stream,budget,{arguments});
}}
"""
    source = folder / "probe.cpp"
    source.write_text(
        prefix
        + SUPPORT
        + "namespace generativeqc::tensor {\n"
        + context
        + "}\n"
        + adapter
        + probe
        + DRIVER
    )
    return required_native_cxx.build_executable(
        [source],
        folder / "probe",
        compile_args=(
            "-std=c++20",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-variable",
            "-Wno-unused-parameter",
            "-pthread",
            "-I" + str(folder),
            "-I" + str(ROOT / "src"),
        ),
        link_args=("-pthread",),
    )


@pytest.mark.parametrize(
    "mode",
    [
        "qualified",
        "consumer",
        "unknown",
        "old",
        "small-ao",
        "small-grid",
        "gga",
        "gga-pbe",
        "gga-b3lyp",
        "gga-enabled",
        "gga-rollback",
        "gga-uks",
        "gga-other",
        "gga-consumer",
        "gga-small",
        "gga-invalid",
        "meta-rollback",
        "dense",
        "empty",
        "response",
        "mixed-ao",
        "mixed-contraction",
        "no-budget",
        "small-budget",
        "unsupported",
        "invalid-attribute",
        "probe-failure",
        "pending-failure",
        "wide-tile",
        "spin",
        "zero-reuse",
        "device-oom",
        "ledger",
        "sync-failure",
        "destroy-failure",
        "cost-cold",
        "cost-warm",
        "cost-pack",
        "cost-unknown",
    ],
)
def test_generated_lowering_admission_costs_and_fallback(
    lowering_profile_probe: Path, mode: str
) -> None:
    result = subprocess.run(
        [str(lowering_profile_probe), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

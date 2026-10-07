"""Execute the real KS controller on a host; this does not qualify CUDA numerics."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, signature: str) -> str:
    begin = source.index(signature)
    end = source.index("{", begin) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


@pytest.fixture(scope="module")
def controller_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    policy_start = source.index("    const bool range_incremental_eligible =")
    policy_end = source.index(
        "    if (incremental_direct_jk_policy.requested", policy_start
    )
    unit = r"""
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include "scf/types.hpp"
namespace scf = generativeqc::scf;
int cudaGetLastError() { return 0; }
void check(int) {}
// Host copies execute the same linear provider-buffer operations. No CUDA
// scheduling, kernel performance, or molecular numerical claim is implied.
void launch_prepare_incremental_density_kernel(unsigned,unsigned,unsigned,int,std::size_t n,
    const double* d,const double* a,double* out,double* maximum) {
  for(std::size_t i=0;i<n;++i) {out[i]=d[i]-a[i];*maximum=std::max(*maximum,std::abs(out[i]));}
}
void launch_add_matrix_kernel(unsigned,unsigned,unsigned,int,std::size_t n,
    const double* a,double* out) {for(std::size_t i=0;i<n;++i)out[i]+=a[i];}
void launch_copy_matrix_kernel(unsigned,unsigned,unsigned,int,std::size_t n,
    const double* a,double* out) {for(std::size_t i=0;i<n;++i)out[i]=a[i];}
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
struct Consumer {
  bool incremental_direct_jk=true, incremental_anchored=false, pending_incremental_delta=false;
  bool final_closure=false, strict_refinement=false, has_exchange=true, has_range_correction=false;
  unsigned incremental_delta_updates=0;
  std::size_t elements=1,matrix=1;
  int stream=0;
  scf::IncrementalDirectJkPolicy incremental_direct_jk_policy;
  scf::ScfResult output;
  double d=1,a=0,delta=0,maximum=0,jv=0,ja=0,kv=0,ka=0,rv=0,ra=0;
  double *density=&d,*incremental_anchor_density=&a,*incremental_delta_density=&delta,
      *incremental_max_abs_delta_density=&maximum,*j=&jv,*incremental_anchor_j=&ja,
      *exchange=&kv,*incremental_anchor_exchange=&ka,*range_exchange=&rv,
      *incremental_anchor_range_exchange=&ra;
  Consumer() {
    scf::ScfOptions options;
    options.incremental_direct_jk=true;
    options.screening_tolerance=0.0;
    options.incremental_direct_jk_rebuild_interval=0;
    scf::ResolvedFockBuild strategy;
    strategy.backend=scf::FockBackend::Cuda;
    strategy.screening_tolerance=1e-12;
    strategy.spec.coulomb.present=true;
    strategy.spec.coulomb.approximation=scf::FockApproximation::Exact;
    const bool fock_binding=true,fitted_coulomb=false;
    std::optional<scf::ResolvedFockBuild> range_correction;
    struct {bool any_lower_precision()const{return false;}} precision_schedule;
"""
    unit += source[policy_start:policy_end]
    unit += "\n}\n"
    for signature in (
        "bool incremental_delta_admitted()",
        "const double* prepare_incremental_jk_density()",
        "void finalize_incremental_jk_components()",
    ):
        unit += _definition(source, signature)
    unit += r"""
  void build(bool closure) {
    final_closure=closure;
    const auto* input=prepare_incremental_jk_density();
    // Independent linear J/K oracle for the actual reconstruction controller.
    jv=2*(*input); kv=3*(*input);
    finalize_incremental_jk_components();
    require(jv==2*d && kv==3*d,"full/delta raw J/K reconstruction changed the operator");
    ++output.iterations;
    d+=0.125;
  }
};
int main(int argc,char** argv) {
  if(argc!=2)return 99;
  try {
    const unsigned accepted=std::stoul(argv[1]);
    Consumer c;
    require(c.incremental_direct_jk && c.incremental_direct_jk_policy.active &&
                c.incremental_direct_jk_policy.effective_rebuild_interval==1,
            "native options bypassed the prepared provider screening cadence");
    for(unsigned i=0;i<accepted;++i)c.build(false);
    c.build(true);
    const auto& work=c.output.incremental_direct_jk;
    const auto full=(accepted+1)/2,delta=accepted/2;
    require(work.anchor_full_builds==full && work.delta_builds==delta,
            "screened controller did not alternate accepted full/delta builds");
    require(work.periodic_rebuilds==full-1 && work.post_scf_full_builds==1,
            "periodic and final full builds lost their distinct accounting");
    if(accepted==2) require(work.periodic_rebuilds==0,"early convergence invented a refresh");
    if(accepted>=3) require(work.periodic_rebuilds>0,"required refresh branch did not execute");
  } catch(const std::exception& error) {std::cerr<<error.what();return 1;}
  return 0;
}
"""
    directory = tmp_path_factory.mktemp("ks-incremental-controller")
    cpp, executable = directory / "controller.cpp", directory / "controller"
    cpp.write_text(unit)
    native_cxx.build_executable(
        [cpp],
        executable,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-fsanitize=undefined",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
        link_args=("-fsanitize=undefined",),
        compile_timeout=30,
    )
    return executable


@pytest.mark.parametrize("accepted_iterations", (1, 2, 3, 4, 7))
def test_actual_controller_counts_early_convergence_and_required_refresh(
    controller_probe: Path, accepted_iterations: int
) -> None:
    result = subprocess.run(
        [str(controller_probe), str(accepted_iterations)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr

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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include "scf/types.hpp"
#include "dft/ks_stopping_policy.hpp"
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
  bool fast_anchor_active=false, early_final_probe_requested=true;
  bool budget_stationarity=false, warm_energy_baseline=false;
  bool pending_nonlocal_seed=false, pending_nonlocal_frozen=false;
  unsigned nonlocal_seed_iterations=0, nonlocal_exact_epoch_iterations=0;
  const double* prepare_fast_anchor_jk_density() {
    throw std::runtime_error("generic incremental owner entered the Fast controller");
  }
  void finalize_fast_anchor_jk_components() {
    throw std::runtime_error("generic incremental owner entered the Fast controller");
  }
  bool incremental_direct_jk=true, incremental_anchored=false, pending_incremental_delta=false;
  bool incremental_energy_refinement=false;
  unsigned incremental_energy_full_builds=0;
  bool final_closure=false, strict_refinement=false, has_exchange=true, has_range_correction=false;
  unsigned incremental_delta_updates=0;
  std::size_t elements=1,matrix=1;
  int stream=0;
  scf::IncrementalDirectJkPolicy incremental_direct_jk_policy;
  scf::ScfOptions options;
  scf::ScfResult output;
  double d=1,a=0,delta=0,maximum=0,jv=0,ja=0,kv=0,ka=0,rv=0,ra=0;
  double *density=&d,*incremental_anchor_density=&a,*incremental_delta_density=&delta,
      *incremental_max_abs_delta_density=&maximum,*j=&jv,*incremental_anchor_j=&ja,
      *exchange=&kv,*incremental_anchor_exchange=&ka,*range_exchange=&rv,
      *incremental_anchor_range_exchange=&ra;
  Consumer() {
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
        "double physical_stationarity_tolerance()",
        "bool incremental_delta_admitted()",
        "void refine_incremental_energy(double density_change",
        "const double* prepare_incremental_jk_density()",
        "void account_incremental_full_energy_finalization()",
        "void finalize_incremental_jk_components()",
    ):
        unit += _definition(source, signature)
    # Exercise the same callable stopping policy as the CUDA owner, without
    # depending on the spelling or placement of its local assignments.
    unit += """
  bool stop(double change, double residual, double maximum) {
    refine_incremental_energy(change, residual, maximum);
    return generativeqc::dft::ordinary_ks_converged(
        {output.iterations, warm_energy_baseline,
         !incremental_direct_jk || incremental_energy_full_builds >= 2,
         !pending_nonlocal_seed && !pending_nonlocal_frozen &&
             (!nonlocal_seed_iterations || nonlocal_exact_epoch_iterations >= 2),
         output.energy_change, change, residual, maximum,
         options.energy_tolerance, options.density_tolerance,
         physical_stationarity_tolerance()});
  }
"""
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
    if(std::string(argv[1])=="gate") {
      for(bool tight:{false,true}) {
        Consumer probe;
        if(tight) {probe.options.energy_tolerance=1e-12; probe.options.density_tolerance=1e-10;}
        probe.output.iterations=2;
        const double density_tolerance=probe.options.density_tolerance;
        const double residual_tolerance=std::min(1e-9,density_tolerance);
        const double change=density_tolerance/2, residual=residual_tolerance/2;
        probe.output.energy_change=1e-5;
        require(!probe.stop(change,residual,residual),"large energy change passed the actual gate");
        require(probe.incremental_energy_refinement,
                "actual convergence block waited for energy before starting full refinement");
        probe.output.energy_change=probe.options.energy_tolerance/2;
        probe.incremental_energy_full_builds=1;
        require(!probe.stop(change,residual,residual),"one full build certified energy history");
        probe.incremental_energy_full_builds=2;
        require(probe.stop(change,residual,residual),"qualified full history did not converge");
        require(!probe.stop(density_tolerance,residual,residual),"density gate was relaxed");
        require(!probe.stop(change,residual_tolerance,residual),"RMS gate was relaxed");
        require(!probe.stop(change,residual,residual_tolerance),"AO maximum gate was relaxed");
        probe.output.energy_change=probe.options.energy_tolerance;
        require(!probe.stop(change,residual,residual),"energy gate was relaxed");
        probe.output.energy_change=probe.options.energy_tolerance/2;
        probe.output.iterations=1;
        require(!probe.stop(change,residual,residual),"cold first iteration invented history");
        probe.warm_energy_baseline=true;
        require(probe.stop(change,residual,residual),"validated warm history was ignored");
      }
      return 0;
    }
    if(std::string(argv[1])=="refinement") {
      Consumer probe;
      probe.build(false);
      probe.build(false);
      const double density_tolerance=probe.options.density_tolerance;
      const double residual_tolerance=std::min(1e-9,density_tolerance);
      probe.output.energy_change=1e-5;
      probe.refine_incremental_energy(density_tolerance,residual_tolerance/2,residual_tolerance/2);
      require(!probe.incremental_energy_refinement,"unstable density triggered energy refinement");
      probe.refine_incremental_energy(density_tolerance/2,residual_tolerance,residual_tolerance/2);
      require(!probe.incremental_energy_refinement,"unstable RMS residual triggered refinement");
      probe.refine_incremental_energy(density_tolerance/2,residual_tolerance/2,residual_tolerance);
      require(!probe.incremental_energy_refinement,"unstable maximum residual triggered refinement");
      probe.refine_incremental_energy(density_tolerance/2,residual_tolerance/2,residual_tolerance/2);
      require(probe.incremental_energy_refinement && !probe.incremental_delta_admitted(),
              "stationary density waited for the alternating incremental energy gate");
      require(probe.incremental_energy_full_builds==0,"refinement invented a full build");
      probe.refine_incremental_energy(density_tolerance*2,residual_tolerance*2,residual_tolerance*2);
      require(probe.incremental_energy_refinement && !probe.incremental_delta_admitted(),
              "a later full residual readmitted delta energy evaluations");
      probe.build(false);
      require(probe.incremental_energy_full_builds==1,
              "one full build was counted as a pair of full energy evaluations");
      probe.build(false);
      require(probe.incremental_energy_full_builds==2,
              "full energy qualification did not execute two target-operator builds");
      probe.account_incremental_full_energy_finalization();
      const auto& work=probe.output.incremental_direct_jk;
      require(work.delta_builds==1 && work.anchor_full_builds==2 && work.post_scf_full_builds==1 &&
                  work.anchor_full_builds+work.delta_builds+work.post_scf_full_builds==
                      probe.output.iterations,
              "full-energy finalization invented or lost a physical provider build");
      require(probe.output.energy_change>probe.options.energy_tolerance,
              "energy refinement relaxed the requested energy tolerance");
      return 0;
    }
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


def test_stationary_density_refines_full_energy_without_relaxing_its_gate(
    controller_probe: Path,
) -> None:
    """Blocked delta energy must not prevent full-operator convergence work."""
    result = subprocess.run(
        [str(controller_probe), "refinement"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_actual_stopping_keeps_energy_density_residual_and_full_history_gates(
    controller_probe: Path,
) -> None:
    """Evaluate ordinary/tight stop decisions instead of pinning source spelling."""
    result = subprocess.run(
        [str(controller_probe), "gate"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr

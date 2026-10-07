"""Execute the CUDA solver control flow with delayed, device-free events."""

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

ROOT = Path(__file__).resolve().parents[2]


def test_jacobi_refusal_continues_after_the_first_drained_trial(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if not compiler or not shutil.which("ccache"):
        pytest.skip("requires C++ compiler and ccache")
    source = (ROOT / "src/cc/cuda_solver.cu").read_text()
    # Retain the entire live solve loop, including convergence/replay, event
    # timing, carried outputs, error handling and publication. Numerical kernels
    # are replaced by deterministic delayed operations, not CPU CC equations.
    solve = source.split("SolverResult solve_cuda(", 1)[1].split(
        "}  // namespace generativeqc::cc", 1
    )[0]
    solve = "SolverResult solve_cuda(" + solve
    # Preserve the actual disabled-history early return. The active-history
    # stand-in drains like the real packing audit and controls its resource
    # outcome; constructor/replacement ownership has a separate live-source test.
    diis = source.split("bool run_diis(", 1)[1].split(
        "  // Appending to a full ring", 1
    )[0]
    diis = "bool run_diis(" + diis
    cpp = tmp_path / "jacobi.cpp"
    cpp.write_text(PREFIX + diis + DIIS_BODY + solve + MAIN)
    executable = tmp_path / "jacobi"
    compile_owner(compiler, tmp_path, [cpp], executable)
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include "cc/solver.hpp"
#include "cc/iteration_driver.hpp"
#include "solver/diis_ring.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
struct Event { unsigned sequence{}; };
using cudaEvent_t=Event*;
using cudaStream_t=int;
constexpr int cudaMemcpyDeviceToHost=1;
unsigned queued=0, completed=0, event_records=0, timing_reads=0, active_diis=0;
int scenario=0, failure=0;
int cudaEventRecord(cudaEvent_t event,cudaStream_t) {
  event->sequence=++queued; ++event_records; return 0;
}
int cudaEventElapsedTime(float* milliseconds,cudaEvent_t first,cudaEvent_t last) {
  ++timing_reads;
  if(first->sequence>completed || last->sequence>completed) return 600;
  *milliseconds=0.001F; return 0;
}
int cudaStreamSynchronize(cudaStream_t) { completed=queued; return 0; }
int cudaMemcpyAsync(void* to,const void* from,std::size_t size,int,cudaStream_t) {
  std::memcpy(to,from,size); ++queued; return 0;
}
void cuda_check(int error) {
  if(error) throw std::runtime_error("cudaErrorNotReady: unfinished trial events");
}
namespace generativeqc::cc {
namespace generated { struct DeviceIterationOutputs { unsigned step{}; }; }
void validate_problem(const Problem&,bool) {}
void validate_options(const SolverOptions&) {}
struct Owner {
  solver::DiisRing history;
  SolverDiagnostic diagnostic;
  unsigned restarts{}, step{};
  Event first_event,last_event;
  cudaEvent_t trial_begin=&first_event,trial_end=&last_event;
  cudaStream_t stream=0;
  std::size_t n1=1,n2=1;
  double current1=0,current2=0,previous1=0,previous2=0;
  double *last_t1=&previous1,*last_t2=&previous2;
  struct State { double *canonical_eps=nullptr,*t1,*t2; } state{nullptr,&current1,&current2};
  Owner(const Problem&,const SolverOptions& options,int) : history(options.diis_size) {
    diagnostic.packed_diis=scenario!=0;
  }
  generated::DeviceIterationOutputs iteration() {
    ++queued; return {step};
  }
  generated::DeviceIterationOutputs replay() { ++queued; return {step+100}; }
  std::array<double,3> read_status(generated::DeviceIterationOutputs output) {
    completed=queued;
    const bool replay=output.step>=100;
    const unsigned physical=replay ? output.step-100 : output.step;
    const double residual=(physical<3 || (replay && failure==1)) ? 1. : 0.;
    return {physical ? .5 : 1.,residual,residual};
  }
  void advance(generated::DeviceIterationOutputs,double) {
    previous1=current1; previous2=current2;
    current1=current2=double(++step); ++queued;
  }
  void check_generated_error() {
    completed=queued;
    if(step==2 && failure==2) throw std::runtime_error("nonfinite RCCSD injected update");
    if(step==2 && failure==3) throw std::runtime_error("injected CUDA driver error");
  }
};
"""

DIIS_BODY = r"""
  ++active_diis;
  completed=queued; // The real active packing/control path drains its stream.
  if(scenario>=2 && active_diis==1) {
    s.diagnostic.packed_diis=false;
    s.diagnostic.packed_diis_refused=true;
    ++s.restarts;
    if(scenario>=3) {
      // Budget refusal and full-history allocation OOM both disable the ring.
      s.history=solver::DiisRing(0);
      s.diagnostic.diis_disabled_after_packing_refusal=true;
    }
  }
  return false;
}
"""

MAIN = r"""
} // namespace generativeqc::cc
int main() {
  using namespace generativeqc::cc;
  // full, packed, full replacement, budget/Jacobi and allocation-OOM/Jacobi.
  for(scenario=0;scenario<5;++scenario) for(failure=0;failure<4;++failure) {
    queued=completed=event_records=timing_reads=active_diis=0;
    SolverOptions options; options.diis_size=8; options.max_iterations=4;
    options.packed_diis=scenario!=0;
    try {
      const auto result=solve_cuda(Problem{},options,0);
      if(failure==3) return 1;
      if(failure==0) {
        if(!result.converged() || result.diagnostic.iterations!=4 ||
           result.diagnostic.update_calls!=3 || result.diagnostic.iteration_graph_calls!=4 ||
           result.diagnostic.replay_graph_calls!=1 || result.t1[0]!=3.) return 2;
      } else if(failure==1) {
        if(result.status!=SolveStatus::NotConverged || result.diagnostic.replay_graph_calls!=2 ||
           result.diagnostic.iterations!=5 || result.t1[0]!=4.) return 3;
      } else {
        if(result.status!=SolveStatus::NumericalFailure || result.t1[0]!=1. ||
           result.reason!="nonfinite RCCSD injected update") return 4;
      }
      if(result.diagnostic.diis_disabled_after_packing_refusal!=(scenario>=3)) return 5;
      if(scenario>=3 && (active_diis!=1 || event_records!=2 || timing_reads!=1)) return 6;
      if(scenario<3 && (event_records!=2*active_diis || timing_reads!=active_diis)) return 7;
    } catch(const std::runtime_error& error) {
      if(failure!=3 || std::string(error.what())!="injected CUDA driver error") {
        std::cerr<<"scenario "<<scenario<<", failure "<<failure<<": "<<error.what()<<'\n';
        return 8;
      }
    }
  }
  std::cout<<"20 solver control scenarios passed, including subsequent Jacobi iterations\n";
}
"""

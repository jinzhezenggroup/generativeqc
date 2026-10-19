"""Execute the actual public capture dispatcher without a CUDA runtime."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
enum class PrecisionPhase : std::uint8_t { StrictAudit, Admitted };
struct CudaXcView {
 std::uint64_t generation;
 std::size_t nao;
 unsigned spins;
 double *potential,*totals;
 int* error;
 void* stream;
};
struct CudaXcPlan {
 struct { bool response=false; std::size_t nao=1; unsigned spins=1; } layout_;
 double *potential_=nullptr,*totals_=nullptr;
 int* error_=nullptr;
 void* stream_=nullptr;
 unsigned submissions=0;
 std::uint64_t seen_generation=99;
 PrecisionPhase seen_phase=PrecisionPhase::StrictAudit;
 bool seen_publication=true;
 const double* seen_density=nullptr;
 double *seen_rho=nullptr,*seen_gradient=nullptr;
 void enqueue(const double*,std::size_t,std::uint64_t,
              PrecisionPhase phase=PrecisionPhase::StrictAudit);
 void enqueue_density_features(const double*,std::size_t,std::uint64_t,double*,double*,
                               PrecisionPhase phase=PrecisionPhase::StrictAudit);
 CudaXcView enqueue_replay_density_features(const double*,std::size_t,double*,double*,
                                           PrecisionPhase phase=PrecisionPhase::StrictAudit);
 void enqueue_impl(const double* d,const double*,std::size_t,std::uint64_t generation,
                   PrecisionPhase phase,double* rho=nullptr,double* gradient=nullptr,
                   bool publish_generation=true) {
  // Ordinary physical enqueue intentionally allows both optional outputs absent.
  if((rho==nullptr)!=(gradient==nullptr)) throw std::invalid_argument("partial output");
  ++submissions;seen_density=d;seen_rho=rho;seen_gradient=gradient;
  seen_generation=generation;seen_publication=publish_generation;
  seen_phase=phase;
 }
};
"""
DRIVER = r"""
int main(int argc,char** argv) {
 if(argc!=5)return 99;
 const unsigned capture=std::atoi(argv[1]);
 const bool response=std::atoi(argv[2])!=0;
 const unsigned mask=std::atoi(argv[3]);
 const int phase_argument=std::atoi(argv[4]);
 const auto phase=phase_argument>0 ? PrecisionPhase::Admitted : PrecisionPhase::StrictAudit;
 CudaXcPlan plan;plan.layout_.response=response;
 double density=1,rho=0,gradient[3]{};
 double* rp=mask&1 ? &rho : nullptr;
 double* gp=mask&2 ? gradient : nullptr;
 bool rejected=false;
 try {
  if(capture==2) {
   const auto view=phase_argument<0 ? plan.enqueue_replay_density_features(&density,1,rp,gp)
                                  : plan.enqueue_replay_density_features(&density,1,rp,gp,phase);
   if(view.generation!=0||view.nao!=1||view.spins!=1)return 4;
  }
  else if(capture) {
   if(phase_argument<0) plan.enqueue_density_features(&density,1,1,rp,gp);
   else plan.enqueue_density_features(&density,1,1,rp,gp,phase);
  }
  else if(phase_argument<0) plan.enqueue(&density,1,1);
  else plan.enqueue(&density,1,1,phase);
 } catch(const std::invalid_argument&) {rejected=true;}
 const bool expected_rejection=response||(capture&&mask!=3);
 if(rejected!=expected_rejection)return 1;
 if(plan.submissions!=(expected_rejection?0u:1u))return 2;
 if(!rejected&&(plan.seen_density!=&density||
    plan.seen_rho!=(capture?rp:nullptr)||plan.seen_gradient!=(capture?gp:nullptr)))return 3;
 if(!rejected&&(plan.seen_generation!=(capture==2?0u:1u)||
    plan.seen_publication!=(capture!=2)))return 5;
 if(!rejected&&plan.seen_phase!=phase)return 6;
}
"""


@pytest.fixture(scope="module")
def capture_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/dft/cuda_xc.cpp").read_text()
    body = source.split("void CudaXcPlan::enqueue(", 1)[1]
    body = (
        "void CudaXcPlan::enqueue("
        + body.split("void CudaXcPlan::enqueue_response(", 1)[0]
    )
    folder = tmp_path_factory.mktemp("xc-capture-dispatch")
    unit, binary = folder / "probe.cpp", folder / "probe"
    unit.write_text(PREFIX + body + DRIVER)
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(unit), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize("response", (False, True))
@pytest.mark.parametrize("mask", range(4))
@pytest.mark.parametrize("capture", (1, 2), ids=("ordinary", "replay"))
@pytest.mark.parametrize("phase", (-1, 0, 1), ids=("default", "strict", "admitted"))
def test_capture_requires_both_outputs(
    capture_probe: Path, response: bool, mask: int, capture: int, phase: int
) -> None:
    result = subprocess.run(
        [str(capture_probe), str(capture), str(int(response)), str(mask), str(phase)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("response", (False, True))
@pytest.mark.parametrize("phase", (-1, 0, 1), ids=("default", "strict", "admitted"))
def test_ordinary_enqueue_keeps_optional_output_semantics(
    capture_probe: Path, response: bool, phase: int
) -> None:
    result = subprocess.run(
        [str(capture_probe), "0", str(int(response)), "0", str(phase)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

"""Fault-inject the production shell derivative wrapper without CUDA hardware."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def host_lifetime_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    symbol = "cudaError_t execute_generated_full_range_energy_derivatives("
    body = symbol + source.split(symbol, 1)[1].split("\ncudaError_t ", 1)[0]
    folder = tmp_path_factory.mktemp("direct-shell-host-lifetime")
    cpp, binary = folder / "probe.cpp", folder / "probe"
    cpp.write_text(PREFIX + body + SUFFIX)
    result = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(cpp), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize("throw_error", [False, True])
@pytest.mark.parametrize("failed_step", range(12))
def test_pending_downloads_outlive_early_returns_and_exceptions(
    host_lifetime_probe: Path, failed_step: int, throw_error: bool
) -> None:
    result = subprocess.run(
        [str(host_lifetime_probe), str(failed_step), str(int(throw_error))],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_preparation_staging_outlives_the_stream_draining_owner() -> None:
    source = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    body = source.split(
        "std::unique_ptr<GeneratedExchangePlan> prepare_generated_exchange(", 1
    )[1].split("namespace {", 1)[0]
    assert body.index("std::vector<std::uint32_t> bounded_pair_order;") < body.index(
        "auto plan = std::make_unique<GeneratedExchangePlan>();"
    )


PREFIX = r"""
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>
using cudaError_t = int;
using cudaStream_t = int;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaMemcpyDeviceToHost=2;
int step=0, fail_step=0, syncs=0; bool throw_error=false;
bool tracking=false, pending=false, freed_pending=false;
void* watched=nullptr;
void* operator new(std::size_t n) {
  void* p=std::malloc(n);
  if(!p) throw std::bad_alloc();
  if(tracking && n==6*sizeof(double)) watched=p;
  return p;
}
void operator delete(void* p) noexcept {
  if(p==watched && pending) freed_pending=true;
  std::free(p);
}
void operator delete(void* p,std::size_t) noexcept { ::operator delete(p); }
int operation() {
  if(++step!=fail_step) return cudaSuccess;
  if(throw_error) throw std::runtime_error("injected CUDA wrapper exception");
  return 7;
}
struct Copy {void* dst; double values[3];};
Copy copies[2]{}; unsigned copy_count=0;
int cudaMemsetAsync(void* dst,int value,std::size_t n,cudaStream_t) {
  int error=operation(); if(error) return error;
  std::memset(dst,value,n); return cudaSuccess;
}
int cudaGetLastError() { return operation(); }
int cudaMemcpyAsync(void* dst,const void* src,std::size_t n,int,cudaStream_t) {
  int error=operation(); if(error) return error;
  if(n!=3*sizeof(double) || copy_count>=2) throw std::runtime_error("bad copy");
  copies[copy_count].dst=dst;
  std::memcpy(copies[copy_count++].values,src,n);
  pending=true; return cudaSuccess;
}
int cudaStreamSynchronize(cudaStream_t) {
  ++syncs;
  int error=operation(); if(error) return error;
  for(unsigned i=0;i<copy_count;++i)
    std::memcpy(copies[i].dst,copies[i].values,3*sizeof(double));
  pending=false; copy_count=0; return cudaSuccess;
}
namespace generativeqc::scf::cuda_execution {
namespace detail { constexpr unsigned kDirectQuartetShellClassCount=1; }
struct Batch { int total_atoms=1; };
struct Shared {
  Batch batch; cudaStream_t stream=1; unsigned worker_blocks=1;
  double screening=0, *shell_bounds=nullptr, *schwarz=nullptr;
  std::uint8_t* active=nullptr;
};
struct GeneratedExchangePlan {
  Shared* shared; bool force_capability=true;
  std::uint32_t* bounded_pair_order;
  double *shell_pair_block_bounds, *force;
  unsigned long long* force_cursor;
  std::uint32_t* heads;
  double *shell_pair_density_bounds=nullptr, *system_density_bounds=nullptr;
  double* direct_spin=nullptr;
};
int prepare_generated_exchange_density(GeneratedExchangePlan&,bool,const double*,const double*) {
  return operation();
}
template<class... Args> void launch_bounded_shell_energy_derivative(Args&&...) {}
"""

SUFFIX = r"""
}
int main(int argc,char** argv) {
  fail_step=argc>1 ? std::atoi(argv[1]) : 0;
  throw_error=argc>2 && std::atoi(argv[2]);
  using namespace generativeqc::scf::cuda_execution;
  Shared shared;
  std::uint32_t pair=0,head=0; unsigned long long cursor=0;
  double force[3]{}, bound=1, density=1;
  GeneratedExchangePlan plan{&shared,true,&pair,&bound,force,&cursor,&head};
  std::vector<double> output{99.0};
  tracking=true;
  int status=0; bool threw=false;
  try { status=execute_generated_full_range_energy_derivatives(
      plan,false,&density,nullptr,1.0,-0.5,output); }
  catch(const std::exception&) { threw=true; }
  tracking=false;
  if(freed_pending) {std::cerr<<"result freed before queued D2H drained";return 2;}
  if(pending) {std::cerr<<"D2H still pending at API return";return 3;}
  if(fail_step) {
    if(throw_error ? !threw : status!=7) {std::cerr<<"lost injected failure";return 4;}
    if(output!=std::vector<double>{99.0}) {std::cerr<<"published partial result";return 5;}
  } else {
    if(threw || status || output!=std::vector<double>(6,0.0)) {
      std::cerr<<"successful result changed";return 6;
    }
    if(syncs!=1) {std::cerr<<"extra success-path synchronization";return 7;}
  }
  // Reuse the same retained owner after the failed call.
  step=0;fail_step=0;syncs=0;throw_error=false;
  if(execute_generated_full_range_energy_derivatives(
      plan,false,&density,nullptr,1.0,-0.5,output)!=cudaSuccess || pending || syncs!=1) {
    std::cerr<<"owner did not recover";return 8;
  }
}
"""

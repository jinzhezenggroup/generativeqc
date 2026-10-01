"""Host-execute the emitted kernel to test owner routing and its native declaration.

CUDA scheduling and scientific helper implementations are stand-ins here. This
checks the real emitted indexing/arithmetic, not GPU or physical qualification.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
using std::isfinite;
#define __device__
#define __global__
struct { size_t x{}; } threadIdx, blockIdx, blockDim;
constexpr int stationary_functional = 1, stationary_coefficients = 4, stationary_jets = 4;
constexpr size_t stationary_shift[4][3]{{1,2,3},{4,5,6},{5,7,8},{6,8,9}};
int atomicExch(int* p, int v) { const int old=*p; *p=v; return old; }
double finite(double v, int* error, double fallback) {
  if (isfinite(v)) return v;
  *error=1; return fallback;
}
namespace generativeqc::dft {
struct GridTaskView {
  size_t npoint{},nactive{},nao{};
  const double *features{},*ao{},*points{};
  const size_t* ao_ids{};
  const int* error{};
};
}
struct StationaryPointValue {
  bool valid=true;
  double energy=0.25,rho[2]{1,2},gradient[2][3]{},kinetic[2]{};
};
StationaryPointValue stationary_evaluate_point(const double*, const double (*)[3], const double*) {
  return {};
}
void ao_pullback(const double* c,const double* w,double* out) {
  for (size_t j=0;j<4;++j) out[j]=c[j]*w[j];
}
int local_norm=0,local_ratio=0,local_log=0,local_becke=0,local_ratio_prepared=0;
namespace generativeqc_grid_adjoint {
struct CenterPair {};
bool contract_point_prepared(const double*,const double*,size_t,int64_t owner,double scale,
                    double* grad,double*,double*,double*,double*,size_t*,
                    std::array<double,4>*,int,int,int,int,const CenterPair*,int) {
  for(size_t k=0;k<3;++k) grad[3*owner+k]+=scale*(k+1);
  return true;
}
}
"""

MAIN = r"""
int main(int argc,char**) {
  // Ambiguity here rejects a stale forward declaration, even though C++ would
  // otherwise accept a mismatching definition as a new overload.
  auto* kernel=&geometry_kernel;
  // Empty launches must not read stale partials or dereference task inputs.
  generativeqc::dft::GridTaskView empty{};
  double sentinel=123, zero_output[9]{};
  int zero_error=0;
  kernel(empty,nullptr,nullptr,nullptr,0,0,nullptr,1,nullptr,nullptr,nullptr,0,0,
         0,&sentinel,&sentinel,nullptr,&zero_error);
  for(size_t j=0;j<9;++j) {
    blockIdx.x=0; blockDim.x=128; threadIdx.x=j;
    geometry_reduce(&sentinel,1,0,zero_output,&zero_error);
    if(zero_output[j]!=0) return 6;
  }
  if(sentinel!=123 || zero_error) return 7;
  constexpr size_t na=3,ppa=701,total=na*ppa,n=2;
  const int64_t ao_atoms[n]{0,2};
  const double centers[3*na]{};
  const bool external=argc==2 || argc==4, arbitrary=argc>=3;
  std::vector<double> expected(9*na),reference;
  for (size_t capacity : {size_t(1),size_t(17),size_t(32),size_t(64),size_t(256),size_t(2048)}) {
  for (size_t tile : {size_t(17),size_t(257),total}) {
    for (bool implicit : {false,true}) {
      if (arbitrary && implicit) continue;
      std::vector<double> result(9*na);
      std::vector<double> partial(capacity*9*na+2,987654),scratch(capacity*9*na+2,987654);
      for(size_t begin=0;begin<total;begin+=tile) {
        const size_t np=std::min(tile,total-begin);
        std::vector<int64_t> owners(np);
        std::vector<double> points(3*np),features(10*np,1),ao(10*np*n,0.5);
        std::vector<double> work(8*np*n,external?0:0.75),weights(np,1),raw(np,1);
        // Include an offset inside a larger strided seed owner.
        std::vector<double> seeds(6*(total+7),0);
        for(size_t p=0;p<np;++p) {
          if(!external) { weights[p]=1+0.03*std::sin(begin+p); raw[p]=0.8+0.07*std::cos(begin+p); }
          owners[p]=arbitrary ? int64_t((begin+p)*7%na) : int64_t((begin+p)/ppa);
          for(size_t k=0;k<3;++k) {
            seeds[(2+k)*(total+7)+begin+p+3]=(begin+p+1)*(k+1);
            if(external && capacity==1 && tile==17 && !implicit)
              expected[3*na+3*owners[p]+k]+=(begin+p+1)*(k+1);
          }
        }
        int error=0,producer_error=0;
        generativeqc::dft::GridTaskView view{np,n,n,features.data(),ao.data(),points.data(),
                                           nullptr,&producer_error};
        const size_t lanes=std::min(capacity,np);
        const size_t threads=128, blocks=(lanes+threads-1)/threads;
        // Dirty unused capacity and guard values must never be reduced/touched.
        blockDim.x=threads;
        for(size_t lane=0;lane<blocks*threads;++lane) {
          threadIdx.x=lane%threads;
          blockIdx.x=lane/threads;
          kernel(view,work.data(),ao_atoms,implicit?nullptr:owners.data(),
                 implicit?begin:0,implicit?ppa:0,centers,na,weights.data(),raw.data(),
                 external?seeds.data():nullptr,total+7,begin+3,
                 lanes,partial.data()+1,scratch.data()+1,nullptr,&error);
        }
        if(error) return 1;
        if(partial.front()!=987654 || partial.back()!=987654 ||
           scratch.front()!=987654 || scratch.back()!=987654) return 4;
        for(size_t j=0;j<9*na;++j) {
          blockIdx.x=j/threads; threadIdx.x=j%threads;
          geometry_reduce(partial.data()+1,na,lanes,result.data(),&error);
        }
        if(error) return 5;
        // Sticky producer failure suppresses publication even with dirty partials.
        producer_error=1;
        blockIdx.x=threadIdx.x=0;
        const auto prior=result;
        kernel(view,work.data(),ao_atoms,owners.data(),0,0,centers,na,weights.data(),raw.data(),
               nullptr,0,0,lanes,partial.data()+1,scratch.data()+1,nullptr,&error);
        for(size_t j=0;j<9*na;++j) {
          blockIdx.x=j/threads; threadIdx.x=j%threads;
          geometry_reduce(partial.data()+1,na,lanes,result.data(),&error);
        }
        if(!error || result!=prior) return 8;
      }
      if(reference.empty()) reference=result;
      for(size_t j=0;j<result.size();++j)
        if(std::abs(result[j]-reference[j])>2e-10) return 2;
      if(external && result!=expected) return 3;
    }
  }
  }
  return 0;
}
"""


def _kernel_source() -> str:
    path = ROOT / "python/generativeqc_compiler/method/stationary_cuda.py"
    tree = ast.parse(path.read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_STATIONARY_SCIENTIFIC_KERNELS"
            for target in node.targets
        )
    )
    source = ast.literal_eval(assignment.value)
    begin = source.index("__global__ void geometry_kernel(")
    end = source.index("}  // namespace generativeqc_stationary_cuda", begin)
    ao_begin = source.index("__device__ bool geometry_point_ao(")
    ao_end = source.index("struct GeometryBlockControl", ao_begin)
    return source[ao_begin:ao_end] + source[begin:end]


@pytest.fixture(scope="module")
def geometry_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    header = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    declaration = re.search(r"__global__ void geometry_kernel\([^;]+;", header)
    assert declaration is not None
    directory = tmp_path_factory.mktemp("stationary-geometry")
    source, executable = directory / "probe.cpp", directory / "probe"
    source.write_text(f"{PREFIX}\n{declaration[0]}\n{_kernel_source()}\n{MAIN}")
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Werror=uninitialized",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


@pytest.mark.parametrize(
    "arguments",
    [(), ("external",), ("arbitrary", "owners"), ("arbitrary", "external", "owners")],
)
def test_emitted_geometry_owner_routes(
    geometry_probe: Path, arguments: tuple[str, ...]
) -> None:
    process = subprocess.run(
        [str(geometry_probe), *arguments],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert process.returncode == 0, (arguments, process.returncode, process.stderr)


def test_actual_becke_helper_inside_emitted_geometry_kernel(tmp_path: Path) -> None:
    """Compare cache/direct through the real emitted kernel, with AO/XC stubs."""
    from generativeqc_compiler.xc.grid_native import (
        emit_grid_adjoint,
        emit_grid_partials,
    )

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    prefix = PREFIX[: PREFIX.index("int local_norm=")]
    source = tmp_path / "becke_kernel.cpp"
    source.write_text(
        prefix
        + emit_grid_adjoint()
        + emit_grid_partials(3)
        + _kernel_source()
        + r"""
int main() {
  constexpr size_t na=3, np=267, n=2, ppa=89;
  double centers[3*na]{0.1,-0.2,0.3,1.1,0.5,-0.2,-0.7,1.2,0.4};
  const int64_t ao_atoms[n]{0,2};
  std::vector<generativeqc_grid_adjoint::CenterPair> pairs(3);
  for(int geometry=0;geometry<3;++geometry) {
    centers[4]+=geometry==1 ? 0.17 : geometry==2 ? -0.17 : 0;
    if(!generativeqc_grid_adjoint::prepare_center_geometry(
           centers,na,1e-12,pairs.data(),local_norm,local_ratio_geometry)) return 1;
    for(size_t lanes : {size_t(1),size_t(17),size_t(32),size_t(256)}) {
      for(size_t tile : {size_t(17),size_t(257),np}) {
        for(bool implicit : {false,true}) {
          std::vector<double> direct;
          for(bool cached : {false,true}) {
            std::vector<double> result(9*na),partial(lanes*9*na),scratch(lanes*9*na);
            for(size_t begin=0;begin<np;begin+=tile) {
              const size_t count=std::min(tile,np-begin),active=std::min(lanes,count);
              std::vector<double> xyz(3*count),features(10*count,1),ao(10*count*n,0.5);
              std::vector<double> work(8*count*n,0.75),weights(count),raw(count);
              std::vector<int64_t> owners(count);
              for(size_t p=0;p<count;++p) {
                owners[p]=(begin+p)/ppa;
                for(size_t k=0;k<3;++k)
                  xyz[3*p+k]=centers[3*owners[p]+k]+0.27+0.13*std::sin((begin+p)*(k+1)+k);
                weights[p]=1+0.03*std::sin(begin+p); raw[p]=0.8+0.07*std::cos(begin+p);
              }
              int error=0;
              generativeqc::dft::GridTaskView view{count,n,n,features.data(),ao.data(),xyz.data(),nullptr,nullptr};
              blockDim.x=32;
              for(size_t lane=0;lane<((active+31)/32)*32;++lane) {
                blockIdx.x=lane/32; threadIdx.x=lane%32;
                geometry_kernel(view,work.data(),ao_atoms,implicit?nullptr:owners.data(),
                                implicit?begin:0,implicit?ppa:0,centers,na,weights.data(),raw.data(),
                                nullptr,0,0,active,partial.data(),scratch.data(),
                                cached?pairs.data():nullptr,&error);
              }
              for(size_t j=0;j<9*na;++j) {
                blockIdx.x=j/32; threadIdx.x=j%32;
                geometry_reduce(partial.data(),na,active,result.data(),&error);
              }
              if(error) return 2;
            }
            if(!cached) direct=result;
            else if(result!=direct) return 3;
          }
        }
      }
    }
  }
  return 0;
}
"""
    )
    executable = tmp_path / "becke_kernel"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True, timeout=30)

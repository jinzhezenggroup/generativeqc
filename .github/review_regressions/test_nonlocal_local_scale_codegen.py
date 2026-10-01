"""Preserve local-scale backend math and the raw/preconditioned error boundary."""

from __future__ import annotations

import ctypes
import math
import random
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from tools.generate_nonlocal_pair_native import native_header

ROOT = Path(__file__).resolve().parents[2]


def test_generated_local_scales_compile_for_a_cuda_device(tmp_path: Path) -> None:
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires the Clang CUDA frontend")
    # Deliberately model CUDA's host/device math boundary: std overloads are
    # host-only, while global overloads are callable from device code. No GPU
    # kernel or numerical oracle is executed by this frontend-only test.
    (tmp_path / "cmath").write_text(
        "__host__ __device__ double sqrt(double);\n"
        "__host__ __device__ double pow(double, double);\n"
        "namespace std { __host__ double sqrt(double); "
        "__host__ double pow(double, double); }\n"
    )
    (tmp_path / "generated.hpp").write_text(native_header())
    source = tmp_path / "device.cu"
    source.write_text(
        '#define __host__ __attribute__((host))\n'
        '#define __device__ __attribute__((device))\n'
        '#define __global__ __attribute__((global))\n'
        '#define __CUDACC__ 1\n'
        'namespace generativeqc::dft::nlc {enum class Vv10Variant {vv10,rvv10};}\n'
        '#include "generated.hpp"\n'
        '__global__ void probe(double* out) {\n'
        'using namespace generativeqc::dft::nlc;\n'
        'auto value = generated::local_scales_cuda<Vv10Variant::rvv10, true>'
        '(0.5,0.1,6.0,0.01);\n'
        'generated::precondition_local_scales_cuda<Vv10Variant::rvv10>'
        '(value.omega,value.kappa);\n'
        'out[0]=value.omega;\n}\n'
    )
    subprocess.run(
        [compiler, "-x", "cuda", "--cuda-device-only", "-nocudainc",
         "-nocudalib", "--cuda-gpu-arch=sm_80", "-std=c++17", "-fsyntax-only",
         "-I", str(tmp_path), str(source)],
        check=True, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("compiler_name", ["g++", "clang++"])
def test_emitted_local_scales_preserve_raw_and_transformed_policies(
    tmp_path: Path, compiler_name: str
) -> None:
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"requires {compiler_name}")
    (tmp_path / "generated.hpp").write_text(native_header())
    native = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    start = native.index("template <Vv10Variant Variant, bool Features>\n__global__ void local_scales_kernel")
    stop = native.index("\n}\n", start) + 3
    kernel = native[start:stop].replace("__global__ ", "")
    source = tmp_path / "scales.cpp"
    source.write_text(r'''
#include <cmath>
#include <cstddef>
#include <numbers>
using std::isfinite;
namespace generativeqc::dft::nlc {enum class Vv10Variant {vv10,rvv10};}
#include "generated.hpp"
using namespace generativeqc::dft::nlc;
struct Dim {size_t x;};
Dim blockIdx{0},blockDim{1},threadIdx{0};
void atomicExch(int* target,int value) {*target=value;}
''' + kernel + r'''
extern "C" int actual(int mode, double rho, double sigma, double b, double c,
                       double* output) {
  const bool features=mode%2;
  if (mode<2) {
    const auto v=features? generated::local_scales_cpu<true>(rho,sigma,b,c):
                          generated::local_scales_cpu<false>(rho,sigma,b,c);
    output[0]=v.omega;output[1]=v.kappa;output[2]=v.domega_drho;
    output[3]=v.domega_dsigma;output[4]=v.dkappa_drho;
    return 0;
  }
  auto v=features?generated::local_scales_cuda<Vv10Variant::vv10,true>(rho,sigma,b,c):
                  generated::local_scales_cuda<Vv10Variant::vv10,false>(rho,sigma,b,c);
  int failed=!isfinite(v.omega)||!isfinite(v.kappa)||v.kappa<=0;
  if (features) failed|=!isfinite(v.domega_drho)||!isfinite(v.domega_dsigma)||!isfinite(v.dkappa_drho);
  if(mode>=4) {
    generated::precondition_local_scales_cuda<Vv10Variant::rvv10>(v.omega,v.kappa);
    failed|=!isfinite(v.omega)||!isfinite(v.kappa);
  }
  output[0]=v.omega;output[1]=v.kappa;output[2]=v.domega_drho;
  output[3]=v.domega_dsigma;output[4]=v.dkappa_drho;
  return failed;
}
extern "C" int original(int mode, double rho, double sigma, double b, double c,
                         double* output) {
  constexpr double pi=3.141592653589793238462643383279502884;
  const bool features=mode%2;
  double omega,kappa,drho=0,dsigma=0,dkappa=0;
  if(mode<2) {
    const auto rho2=rho*rho, rho4=rho2*rho2, sigma2=sigma*sigma;
    omega=std::sqrt(c*sigma2/rho4+(4*pi/3)*rho);
    kappa=b*1.5*pi*std::pow(rho/(9*pi),1./6);
    if(features) {
      const auto rho5=rho4*rho;
      drho=((4*pi/3)-4*c*sigma2/rho5)/(2*omega);
      dsigma=c*sigma/(omega*rho4);dkappa=kappa/(6*rho);
    }
  } else {
    const double ratio=sigma/(rho*rho);
    omega=sqrt(c*ratio*ratio+(4*pi/3)*rho);
    kappa=b*1.5*pi*pow(rho/(9*pi),1./6);
    if(features) {
      drho=((4*pi/3)-4*c*sigma*sigma/pow(rho,5.))/(2*omega);
      dsigma=c*sigma/(omega*pow(rho,4.));dkappa=kappa/(6*rho);
    }
  }
  int failed=!isfinite(omega)||!isfinite(kappa)||kappa<=0;
  if(features) failed|=!isfinite(drho)||!isfinite(dsigma)||!isfinite(dkappa);
  if(mode>=4) {omega/=kappa;kappa*=sqrt(kappa);failed|=!isfinite(omega)||!isfinite(kappa);}
  output[0]=omega;output[1]=kappa;output[2]=drho;output[3]=dsigma;output[4]=dkappa;
  return mode<2?0:failed;
}
extern "C" int actual_underflow_kernel(double* output) {
  const double rho=1,gradient[3]{},weight=0;
  int failed=0;
  local_scales_kernel<Vv10Variant::rvv10,false>(1,1e-220,0.01,&weight,&rho,gradient,
       output,output+1,output+2,output+3,output+4,output+5,&failed);
  return failed;
}
''')
    library = tmp_path / "scales.so"
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math",
         "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    lib = ctypes.CDLL(str(library))
    values = ctypes.c_double * 6
    for name in ("actual", "original"):
        function = getattr(lib, name)
        function.argtypes = [ctypes.c_int, *([ctypes.c_double] * 4), ctypes.POINTER(ctypes.c_double)]
        function.restype = ctypes.c_int
    rng = random.Random(1655)
    cases = [(10 ** rng.uniform(-60, 40), 10 ** rng.uniform(-60, 40),
              10 ** rng.uniform(-5, 3), 10 ** rng.uniform(-5, 1)) for _ in range(512)]
    cases.extend((rho, sigma, b, 0.01) for rho in (0.0, 1e-300, 1.0, 1e300)
                 for sigma in (0.0, 1e-300, 1.0, 1e300)
                 for b in (0.0, 1e-220, 6.0, 1e300))
    for mode in range(6):
        for case in cases:
            first, second = values(), values()
            assert lib.actual(mode, *case, first) == lib.original(mode, *case, second)
            for actual, original in zip(first, second):
                assert (math.isnan(actual) and math.isnan(original)) or (
                    struct.pack("d", actual) == struct.pack("d", original)
                ), (mode, case, actual, original)
    lib.actual_underflow_kernel.argtypes = [ctypes.POINTER(ctypes.c_double)]
    output = values()
    assert lib.actual_underflow_kernel(output) == 0
    assert output[1] == 0.0  # Positive raw kappa legitimately underflows after validation.

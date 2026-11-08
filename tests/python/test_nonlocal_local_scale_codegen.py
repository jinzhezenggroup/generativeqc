"""Preserve local-scale backend math and the raw/preconditioned error boundary."""

from __future__ import annotations

import ctypes
import math
import random
import shutil
import struct
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

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
        "__host__ __device__ bool signbit(double);\n"
        "__host__ __device__ bool isfinite(double);\n"
        "namespace std { __host__ double sqrt(double); "
        "__host__ double pow(double, double); }\n"
    )
    (tmp_path / "generated.hpp").write_text(native_header())
    source = tmp_path / "device.cu"
    source.write_text(
        "#define __host__ __attribute__((host))\n"
        "#define __device__ __attribute__((device))\n"
        "#define __global__ __attribute__((global))\n"
        "#define __CUDACC__ 1\n"
        # The header also declares pair-schedule kernels. Model their CUDA
        # dependencies without weakening the host-only std math boundary.
        "struct Dim { unsigned x, y, z; };\n"
        "extern __device__ Dim blockIdx, blockDim, threadIdx, gridDim;\n"
        "__device__ double __shfl_down_sync(unsigned, double, unsigned, int=32);\n"
        "__device__ int __shfl_sync(unsigned, int, int, int=32);\n"
        "__device__ double __shfl_sync(unsigned, double, int, int=32);\n"
        "__device__ double atomicAdd(double*, double);\n"
        "__device__ unsigned long long atomicAdd(unsigned long long*, unsigned long long);\n"
        "__device__ int atomicExch(int*, int);\n"
        "namespace generativeqc::dft::nlc {enum class Vv10Variant {vv10,rvv10};}\n"
        '#include "generated.hpp"\n'
        "__global__ void probe(double* out) {\n"
        "using namespace generativeqc::dft::nlc;\n"
        "auto value = generated::local_scales_cuda<Vv10Variant::rvv10, true>"
        "(0.5,0.1,6.0,0.01);\n"
        "generated::precondition_local_scales_cuda<Vv10Variant::rvv10>"
        "(value.omega,value.kappa);\n"
        "out[0]=value.omega;\n}\n"
    )
    command = [
        compiler,
        "-x",
        "cuda",
        "--cuda-device-only",
        "-nocudainc",
        "-nocudalib",
        "--cuda-gpu-arch=sm_80",
        "-std=c++17",
        "-fsyntax-only",
        "-I",
        str(tmp_path),
        str(source),
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr

    # Keep the modeled math boundary observable: a host-only std overload must
    # still fail when called by device code, even with scheduling declarations.
    source.write_text(
        source.read_text()
        + "__global__ void host_only_probe(double* out) { out[0]=std::sqrt(out[0]); }\n"
    )
    rejected = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=30
    )
    assert rejected.returncode != 0, "device code accepted a host-only math call"
    assert "sqrt" in rejected.stderr, rejected.stderr


@pytest.mark.parametrize("compiler_name", ["g++", "clang++"])
def test_emitted_local_scales_preserve_raw_and_transformed_policies(
    tmp_path: Path, compiler_name: str, native_cxx: Any
) -> None:
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"requires {compiler_name}")
    (tmp_path / "generated.hpp").write_text(native_header())
    native = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    start = native.index(
        "template <Vv10Variant Variant, bool Features>\n__global__ void local_scales_kernel"
    )
    stop = native.index("\n}\n", start) + 3
    kernel = native[start:stop].replace("__global__ ", "")
    source = tmp_path / "scales.cpp"
    source.write_text(
        r"""
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
"""
        + kernel
        + r"""
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
"""
    )
    library = tmp_path / "scales.so"
    replace(native_cxx, compiler=compiler).build_shared(
        (source,),
        library,
        compile_args=("-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math"),
    )
    lib = ctypes.CDLL(str(library))
    values = ctypes.c_double * 6
    for name in ("actual", "original"):
        function = getattr(lib, name)
        function.argtypes = [
            ctypes.c_int,
            *([ctypes.c_double] * 4),
            ctypes.POINTER(ctypes.c_double),
        ]
        function.restype = ctypes.c_int
    rng = random.Random(1655)
    cases = [
        (
            10 ** rng.uniform(-60, 40),
            10 ** rng.uniform(-60, 40),
            10 ** rng.uniform(-5, 3),
            10 ** rng.uniform(-5, 1),
        )
        for _ in range(512)
    ]
    cases.extend(
        (rho, sigma, b, 0.01)
        for rho in (0.0, 1e-300, 1.0, 1e300)
        for sigma in (0.0, 1e-300, 1.0, 1e300)
        for b in (0.0, 1e-220, 6.0, 1e300)
    )
    cases.extend(
        (
            10 ** rng.uniform(-100, 100),
            10 ** rng.uniform(-100, 100),
            10 ** rng.uniform(-200, 200),
            10 ** rng.uniform(-250, 250),
        )
        for _ in range(1024)
    )
    cases.extend(
        (1.0, sigma, 6.0, c)
        for sigma in (1e150, 5e153, 1e154)
        for c in (1e-300, 1e-150, 0.01)
    )
    for mode in range(6):
        for case in cases:
            first, second = values(), values()
            assert lib.actual(mode, *case, first) == lib.original(mode, *case, second)
            for actual, original in zip(first, second, strict=True):
                assert (math.isnan(actual) and math.isnan(original)) or (
                    struct.pack("d", actual) == struct.pack("d", original)
                ), (mode, case, actual, original)
    lib.actual_underflow_kernel.argtypes = [ctypes.POINTER(ctypes.c_double)]
    output = values()
    assert lib.actual_underflow_kernel(output) == 0
    assert (
        output[1] == 0.0
    )  # Positive raw kappa legitimately underflows after validation.

    import mpmath as mp

    with mp.workdps(90):
        pi = mp.mpf(float(math.pi))
        plasma = mp.mpf(float(4 * math.pi / 3))
        b, c = mp.mpf(6), mp.mpf(0.01)

        def omega(rho: Any, sigma: Any) -> Any:
            return mp.sqrt(c * sigma * sigma / rho**4 + plasma * rho)

        def kappa(rho: Any) -> Any:
            return b * mp.mpf(1.5) * pi * mp.root(rho / (9 * pi), 6)

        for rho in (1e-5, 0.13, 1.0, 17.0):
            for sigma in (0.0, 0.07, 12.0):
                reference_rho, reference_sigma = mp.mpf(rho), mp.mpf(sigma)
                expected = [
                    float(omega(reference_rho, reference_sigma)),
                    float(kappa(reference_rho)),
                    float(
                        mp.diff(
                            lambda value, reference_sigma=reference_sigma: omega(
                                value, reference_sigma
                            ),
                            reference_rho,
                        )
                    ),
                    float(
                        mp.diff(
                            lambda value, reference_rho=reference_rho: omega(
                                reference_rho, value
                            ),
                            reference_sigma,
                        )
                    ),
                    float(mp.diff(kappa, reference_rho)),
                ]
                for mode in (1, 3):
                    actual = values()
                    lib.actual(mode, rho, sigma, float(b), float(c), actual)
                    assert list(actual)[:5] == pytest.approx(expected, rel=3e-13, abs=0)

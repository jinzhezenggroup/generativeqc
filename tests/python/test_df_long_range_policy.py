"""Qualify the production erf DF policy, including auxiliary g, with libcint."""

from __future__ import annotations

import ctypes
import os
import shutil
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.common.compiler_cache import run_cached_compiler
from generativeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda
from generativeqc_compiler.integral.df_policy import emit_df_policy_cuda

from tools.generativeqc_validation.df_derivatives import make_df_derivative_fixture
from tools.generativeqc_validation.df_values import make_df_value_fixture


@pytest.fixture(scope="module", params=("cpu", "cuda"))
def range_policy(request: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Exercise the actual policy adapter, not a second integral implementation."""
    cuda = request.param == "cuda"
    if cuda and os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly allocated CUDA device")
    if cuda:
        assert os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")
    compiler = shutil.which(os.environ.get("CUDACXX", "nvcc") if cuda else "c++")
    if not compiler:
        pytest.skip("C++/CUDA compiler unavailable")
    directory = tmp_path_factory.mktemp("df-range-policy-" + request.param)
    (directory / "generated_df_derivatives.cuh").write_text(emit_df_derivatives_cuda())
    (directory / "policy.cuh").write_text(emit_df_policy_cuda(derivatives=True))
    if not cuda:
        (directory / "cuda_runtime.h").write_text("")
    source = r"""
#include <cstddef>
#include <cstdint>
#include "policy.cuh"
namespace df = generativeqc::scf::generated_df_policy;
using Vec3=df::LongRangeValue::Vec3;
using Angular=df::LongRangeValue::Angular;
struct Input {
  std::uint32_t count; Angular angular[3]; Vec3 centers[3];
  double exponents[3], weight;
};
static_assert(sizeof(Input)==144 && offsetof(Input,weight)==136);
template<unsigned Rank> QUALIFIER void value(const Input& in,double omega,double* out) {
  double e[3]{in.exponents[0],in.exponents[Rank==2?2:1],in.exponents[2]};
  Vec3 r[3]{in.centers[0],in.centers[Rank==2?2:1],in.centers[2]};
  Angular a[3]{in.angular[0],in.angular[Rank==2?2:1],in.angular[2]};
  double v=0;
  df::LongRangeValue{omega}.accumulate<Rank>(v,e,r,a,in.weight);
  df::LongRangeDerivative::Accumulator response{};
  df::LongRangeDerivative{omega}.accumulate<Rank>(response,e,r,a,in.weight);
  out[0]=v;
  for(unsigned slot=0;slot<3;++slot)
    for(unsigned axis=0;axis<3;++axis) out[1+3*slot+axis]=response.gradient[slot][axis];
}
QUALIFIER void evaluate(const Input& in,double omega,double* out) {
  if(in.count==2) value<2>(in,omega,out); else value<3>(in,omega,out);
}
"""
    if cuda:
        source += r"""
__global__ void values(const Input* in,std::size_t n,double omega,double* out) {
  const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) evaluate(in[i],omega,out+10*i);
}
extern "C" int probe(const Input* in,std::size_t n,double omega,double* out) {
  Input* d=nullptr; double* o=nullptr;
  auto status=cudaMalloc(&d,n*sizeof(Input));
  if(status==cudaSuccess) status=cudaMalloc(&o,10*n*sizeof(double));
  if(status==cudaSuccess) status=cudaMemcpy(d,in,n*sizeof(Input),cudaMemcpyHostToDevice);
  if(status==cudaSuccess) {
    values<<<(n+127)/128,128>>>(d,n,omega,o);
    status=cudaGetLastError();
  }
  if(status==cudaSuccess) status=cudaMemcpy(out,o,10*n*sizeof(double),cudaMemcpyDeviceToHost);
  cudaFree(o); cudaFree(d);
  return status;
}
"""
    else:
        source = (
            "#define __device__\n#define __forceinline__ inline\n"
            "#define __noinline__ __attribute__((noinline))\n" + source
        )
        source += r"""
extern "C" int probe(const Input* in,std::size_t n,double omega,double* out) {
  for(std::size_t i=0;i<n;++i) evaluate(in[i],omega,out+10*i);
  return 0;
}
"""
    path = directory / ("probe.cu" if cuda else "probe.cpp")
    path.write_text(source.replace("QUALIFIER", "__device__" if cuda else ""))
    library = directory / "probe.so"
    flags = ["-arch=native", "-Xcompiler=-fPIC"] if cuda else ["-fPIC"]
    result = run_cached_compiler(
        [
            compiler,
            "-O2",
            "-std=c++17",
            "-shared",
            *flags,
            str(path),
            "-I",
            str(directory),
            "-o",
            str(library),
        ],
        300,
        label="long-range auxiliary-g DF policy oracle",
    )
    assert result.returncode == 0, result.stderr
    dll = ctypes.CDLL(str(library))
    dll.probe.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_void_p,
    ]
    dll.probe.restype = ctypes.c_int

    def evaluate(records: np.ndarray, omega: float) -> np.ndarray:
        output = np.full((len(records), 10), np.nan)
        assert (
            dll.probe(records.ctypes.data, len(records), omega, output.ctypes.data) == 0
        )
        return output

    return evaluate


@pytest.mark.parametrize("omega", (1e-3, 0.3, 1.2, 1e3))
@pytest.mark.parametrize(
    "angular",
    ((1, 2), (0, 4), (4, 0), (4, 4), (1, 2, 3), (0, 0, 4), (1, 2, 4), (3, 3, 4)),
)
def test_long_range_policy_values_and_center_responses(
    range_policy: Any, angular: tuple[int, ...], omega: float
) -> None:
    """Check all Cartesian/spherical components and translation invariance."""
    fixture = make_df_derivative_fixture(angular, omega=omega)
    value_fixture = make_df_value_fixture(angular, omega=omega)
    output = range_policy(fixture.records, omega)
    values = value_fixture.contract(output[:, 0])
    response = output[:, 1:].reshape(-1, 3, 3)[:, : len(angular)]
    actual = fixture.contract(response)
    np.testing.assert_allclose(values, value_fixture.reference, atol=8e-11, rtol=8e-11)
    np.testing.assert_allclose(actual, fixture.reference, atol=8e-11, rtol=8e-11)
    np.testing.assert_allclose(
        fixture.spherical(actual), fixture.spherical_reference, atol=8e-11, rtol=8e-11
    )
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-12)

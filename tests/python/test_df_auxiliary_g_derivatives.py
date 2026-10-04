"""Independent generated auxiliary-g center-response qualification.

This tests an explicit compiler variant; production derivative dispatch is
unchanged until the native integration and its endpoint gates are qualified.
"""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess
import typing
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.df_derivatives import axis_polynomial
from generativeqc_compiler.integral.df_derivatives_cuda import (
    emit_df_derivatives_cpu,
    emit_df_derivatives_cuda,
)

from tools.generativeqc_validation.df_derivatives import make_df_derivative_fixture

ROOT = Path(__file__).resolve().parents[2]
SIGNATURES = [a for a in product(range(5), repeat=2) if 4 in a] + [
    (a, b, 4) for a, b in product(range(4), repeat=2)
]


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def derivative_probe(
    request: typing.Any, tmp_path_factory: pytest.TempPathFactory
) -> typing.Any:
    """Compile the same generated center response for host and allocated CUDA.

    Inputs use the validation fixture's 144-byte primitive ABI. CPU execution
    is an emission check; only the separate CUDA parameter exercises the GPU.
    """
    cuda = request.param == "cuda"
    if cuda and os.environ.get("GENERATIVEQC_DF_G_CUDA_TEST") != "1":
        pytest.skip("requires explicit finite Slurm allocation")
    if cuda:
        assert os.environ.get("SLURM_JOB_ID"), "allocate GPU through Slurm"
    compiler = shutil.which(os.environ.get("CUDACXX", "nvcc") if cuda else "c++")
    cache = shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("compiler and ccache required")
    directory = tmp_path_factory.mktemp("df-g-derivative-" + request.param)
    header = (
        emit_df_derivatives_cuda(auxiliary_g=True)
        if cuda
        else emit_df_derivatives_cpu(auxiliary_g=True)
    )
    (directory / "values.h").write_text(header)
    source = directory / ("probe.cu" if cuda else "probe.cpp")
    common = r"""
#include <cstddef>
#include <cstdint>
#include "values.h"
namespace df = generativeqc::scf::generated_df_auxiliary_g_derivatives;
struct Input {
  std::uint32_t count; df::Angular angular[3]; df::Vec3 centers[3];
  double exponents[3], weight;
};
static_assert(sizeof(Input)==144 && offsetof(Input,weight)==136);
QUALIFIER void value(const Input& in,double* output) {
  const auto* e=in.exponents; const auto* r=in.centers; const auto* a=in.angular;
  const auto response=in.count==2 ? df::metric(e[0],r[0],a[0],e[2],r[2],a[2])
      : df::three_center(e[0],r[0],a[0],e[1],r[1],a[1],e[2],r[2],a[2]);
  const df::Vec3 channels[]{response.first,response.second,response.third};
  for(unsigned center=0;center<3;++center){
    output[3*center]=in.weight*channels[center].x;
    output[3*center+1]=in.weight*channels[center].y;
    output[3*center+2]=in.weight*channels[center].z;
  }
}
"""
    gpu = r"""
__global__ void values(const Input* inputs,std::size_t n,double* output) {
  const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) value(inputs[i],output+9*i);
}
extern "C" int probe(const Input* inputs,std::size_t n,double* output) {
  Input* d=nullptr; double* o=nullptr;
  auto status=cudaMalloc(&d,n*sizeof(Input));
  if(status==cudaSuccess) status=cudaMalloc(&o,9*n*sizeof(double));
  if(status==cudaSuccess) status=cudaMemcpy(d,inputs,n*sizeof(Input),cudaMemcpyHostToDevice);
  if(status==cudaSuccess) {
    values<<<(n+127)/128,128>>>(d,n,o);
    status=cudaGetLastError();
  }
  if(status==cudaSuccess) status=cudaMemcpy(output,o,9*n*sizeof(double),cudaMemcpyDeviceToHost);
  cudaFree(o);cudaFree(d);
  return status;
}
"""
    cpu = r"""
extern "C" int probe(const Input* inputs,std::size_t n,double* output) {
  for(std::size_t i=0;i<n;++i) value(inputs[i],output+9*i);
  return 0;
}
"""
    source.write_text(
        common.replace("QUALIFIER", "__device__" if cuda else "")
        + (gpu if cuda else cpu)
    )
    obj, library = directory / "probe.o", directory / "probe.so"
    flags = ["-arch=sm_120", "-Xcompiler=-fPIC"] if cuda else ["-fPIC"]
    subprocess.run(
        [
            cache,
            compiler,
            "-O2",
            "-std=c++17",
            *flags,
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        check=True,
        capture_output=True,
        timeout=300,
    )
    subprocess.run(
        [compiler, "-shared", str(obj), "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    dll = ct.CDLL(str(library))
    dll.probe.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p]
    dll.probe.restype = ct.c_int

    def evaluate(records: np.ndarray) -> np.ndarray:
        output = np.full((len(records), 3, 3), np.nan)
        assert dll.probe(records.ctypes.data, len(records), output.ctypes.data) == 0
        return output

    return evaluate


def test_explicit_g_derivative_axis_domain_preserves_old_boundaries() -> None:
    for powers, count in (((4, 3, 4), 12), ((3, 4, 4), 12), ((5, 0, 4), 10)):
        assert len(axis_polynomial(*powers, auxiliary_g_derivative=True)[1]) == count
        with pytest.raises(ValueError):
            axis_polynomial(*powers)
        with pytest.raises(ValueError):
            axis_polynomial(*powers, auxiliary_g=True)
    for powers in ((4, 4, 4), (5, 1, 4), (0, 5, 0), (0, 0, 5)):
        with pytest.raises(ValueError):
            axis_polynomial(*powers, auxiliary_g_derivative=True)


@pytest.mark.parametrize("angular", SIGNATURES)
@pytest.mark.parametrize("variant", ("coincident", "asymmetric"))
def test_complete_cartesian_spherical_and_auxiliary_center_blocks(
    derivative_probe: typing.Any, angular: tuple[int, ...], variant: str
) -> None:
    pytest.importorskip("pyscf")
    fixture = make_df_derivative_fixture(angular, variant=variant)
    primitive = derivative_probe(fixture.records)
    if len(angular) == 2:
        primitive = primitive[:, [0, 2]]
    actual = fixture.contract(primitive)
    np.testing.assert_allclose(
        actual, fixture.reference, atol=8e-11, rtol=8e-11, err_msg=fixture.name
    )
    np.testing.assert_allclose(
        fixture.spherical(actual),
        fixture.spherical_reference,
        atol=8e-11,
        rtol=8e-11,
        err_msg=fixture.name,
    )
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-12, rtol=0)
    if variant == "asymmetric":
        assert np.max(np.abs(actual[-1])) > 1e-8

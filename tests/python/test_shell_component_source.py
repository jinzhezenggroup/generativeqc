"""Shared shell preparation against independent primitive integrals.

The emitted producer runs once for several Cartesian components, with successive
full/LR/SR/full radial overwrites. This checks mathematical reuse and lifetime
assumptions on the host; CUDA barriers, matrices and endpoint work have separate
native gates.
"""

import ctypes
import itertools
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)
from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)
from generativeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)
from test_hermite_convolution import independent_jet, normalization
from test_scalar_center_gradient import POSITIONS, powers_for

ROOT = Path(__file__).resolve().parents[2]
CLASSES = tuple(
    p
    for p in itertools.product(range(4), repeat=4)
    if 5 <= sum(p) <= 8 and p[0] >= p[1] and p[2] >= p[3] and p[:2] >= p[2:]
)


@pytest.fixture(scope="module")
def component_source(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    """Compile the actual emitted source with ccache, without loading CUDA."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    folder = tmp_path_factory.mktemp("component-source")
    (folder / "cuda_runtime.h").write_text(
        "#pragma once\n#define __host__\n#define __device__\n"
        "#define __forceinline__ inline\n#define __noinline__\n"
    )
    for name, content in {
        **emit_direct_cartesian_contraction_headers(),
        **emit_direct_pair_support_headers(),
    }.items():
        (folder / name).write_text(content)
    source = emit_direct_recurrence_headers()["generated_direct_shell_class.cuh"]
    begin = source.index("/** A bounded source shared")
    end = source.index("/** Evaluate the shared Hermite contraction", begin)
    (folder / "component.cuh").write_text(
        '#include "generated_direct_cartesian.cuh"\n'
        '#include "generated_direct_shell_pair_hermite.cuh"\n'
        "namespace generativeqc::scf::cuda_execution {\n" + source[begin:end] + "}\n"
    )
    cpp, obj, lib = folder / "probe.cpp", folder / "probe.o", folder / "probe.so"
    cpp.write_text(PROBE)
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-fPIC",
            "-I" + str(folder),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-c",
            str(cpp),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    subprocess.run(
        [compiler, "-shared", str(obj), "-o", str(lib)], check=True, timeout=30
    )
    owner = ctypes.CDLL(str(lib))
    owner.evaluate.argtypes = [
        ctypes.c_uint,
        np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    owner.evaluate.restype = ctypes.c_size_t
    return owner


@pytest.mark.parametrize("partition", CLASSES)
@pytest.mark.parametrize("case", ("distinct", "repeated", "diffuse", "translated"))
def test_many_components_share_independent_full_lr_sr_roots(
    component_source: ctypes.CDLL, partition: tuple[int, ...], case: str
) -> None:
    positions = np.asarray(POSITIONS)
    exponents = np.asarray((0.8, 0.6, 0.7, 0.9))
    omega = 0.3
    if case == "repeated":
        positions[1] = positions[0]
        exponents = np.asarray((0.05, 1.8, 0.09, 0.8))
    elif case == "diffuse":
        positions *= 4
        exponents = np.asarray((0.018, 0.041, 0.073, 0.012))
    elif case == "translated":
        positions += np.asarray((3.0, -7.0, 1.5))
        omega = 1.5
    powers = np.asarray(
        [powers_for(partition, orientation) for orientation in range(3)]
    )
    actual = np.empty((4, 3, 2))
    storage = component_source.evaluate(
        sum(partition), powers, positions, exponents, omega, actual
    )
    assert storage == 6 * 4 * 4 * 8 * 8 + math.comb(sum(partition) + 4, 4) * 8 + 64
    np.testing.assert_array_equal(
        actual[:, :, 0].view(np.uint64), actual[:, :, 1].view(np.uint64)
    )
    np.testing.assert_array_equal(actual[0], actual[3])
    for radial in range(3):
        for component, angular in enumerate(powers):
            norm = normalization(angular, exponents)
            expected = (
                independent_jet(angular, positions, exponents, radial, omega, 1)[0]
                * norm
            )
            np.testing.assert_allclose(
                actual[radial, component, 0] * norm, expected, rtol=3e-10, atol=2e-11
            )


PROBE = r"""
#include "component.cuh"
using namespace generativeqc::scf::cuda_execution;
template<unsigned Order>
std::size_t run(const unsigned* powers, const double* positions, const double* exponents,
                double omega, double* output) {
  CartesianComponentSource<Order> source;
  Vec3<double> centers[4]; unsigned shells[4];
  for(unsigned s=0;s<4;++s){
    centers[s]={positions[3*s],positions[3*s+1],positions[3*s+2]};
    shells[s]=powers[3*s]+powers[3*s+1]+powers[3*s+2];
  }
  prepare_cartesian_component_geometry(source,shells,centers,exponents);
  for(unsigned r=0;r<4;++r){
    const auto range=static_cast<generativeqc::integrals::CoulombRange>(r==3?0:r);
    if(!prepare_cartesian_component_radial(source,range,omega))return 0;
    for(unsigned c=0;c<3;++c){
      Angular angular[4];for(unsigned s=0;s<4;++s)
        angular[s]={powers[12*c+3*s],powers[12*c+3*s+1],powers[12*c+3*s+2]};
      output[6*r+2*c]=consume_cartesian_component(source,angular);
      output[6*r+2*c+1]=primitive_eri_cartesian<Order,double>(
        exponents[0],centers[0],angular[0],exponents[1],centers[1],angular[1],
        exponents[2],centers[2],angular[2],exponents[3],centers[3],angular[3],range,omega);
    }
  }
  return sizeof(source);
}
extern "C" std::size_t evaluate(unsigned order,const unsigned* powers,const double* positions,
                                const double* exponents,double omega,double* output){
  switch(order){
    case 5:return run<5>(powers,positions,exponents,omega,output);
    case 6:return run<6>(powers,positions,exponents,omega,output);
    case 7:return run<7>(powers,positions,exponents,omega,output);
    case 8:return run<8>(powers,positions,exponents,omega,output);
  }return 0;
}
"""

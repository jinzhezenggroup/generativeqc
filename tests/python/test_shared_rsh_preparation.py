"""Joint radial preparation against independent Gaussian-integral/jet oracles.

Compile the actual emitted shell-pair consumer and both retained separate
consumers. The quadrature/Wick oracle is independent of production recurrence.
These CPU checks do not qualify device scheduling, memory or endpoint timing.
"""

import ctypes
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
from test_coulomb_optional_allocation import compile_cached_probe
from test_hermite_convolution import POWERS, independent_jet, normalization

ROOT = Path(__file__).resolve().parents[2]
HIGH_POWERS = [powers for powers in POWERS if sum(map(sum, powers)) >= 5]


@pytest.fixture(scope="module")
def paired_primitive(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., int]:
    """Execute emitted strict-FP64 values/jets and exact shell-specific bounds."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    folder = tmp_path_factory.mktemp("shared-rsh")
    (folder / "cuda_runtime.h").write_text(
        "#pragma once\n#define __host__\n#define __device__\n"
        "#define __forceinline__ inline\n#define __noinline__\n"
    )
    for name, content in {
        **emit_direct_cartesian_contraction_headers(),
        **emit_direct_pair_support_headers(),
    }.items():
        (folder / name).write_text(content)
    emitted = emit_direct_recurrence_headers()["generated_direct_shell_class.cuh"]
    begin = emitted.index("/** Two independent radial values")
    end = emitted.index("/**\n * Evaluate one Cartesian primitive quartet", begin)
    (folder / "paired.cuh").write_text(
        '#include "generated_direct_cartesian.cuh"\n'
        '#include "generated_direct_shell_pair_hermite.cuh"\n'
        "namespace generativeqc::scf::cuda_execution {\n" + emitted[begin:end] + "}\n"
    )
    dispatch = "\n".join(
        f"case {index}: run<{','.join(str(sum(p)) for p in powers)},Scalar>("
        "powers,positions,exponents,omega,range,mask,selected,out); return 0;"
        for index, powers in enumerate(HIGH_POWERS)
    )
    cpp, obj, lib = folder / "probe.cpp", folder / "probe.o", folder / "probe.so"
    cpp.write_text(PROBE.replace("// DISPATCH", dispatch))
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
        [compiler, "-shared", str(obj), "-o", str(lib)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    library = ctypes.CDLL(str(lib))
    library.evaluate.argtypes = [
        ctypes.c_uint,
        np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_double,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    library.evaluate.restype = ctypes.c_int
    return library.evaluate


@pytest.mark.parametrize("index,powers", list(enumerate(HIGH_POWERS)))
@pytest.mark.parametrize("radial", (1, 2), ids=("long", "short"))
@pytest.mark.parametrize("case", ("distinct", "repeated", "diffuse", "translation"))
def test_shared_preparation_preserves_both_radial_values_and_jets(
    paired_primitive: Callable[..., int],
    index: int,
    powers: Any,
    radial: int,
    case: str,
) -> None:
    positions = np.asarray(
        ((0.1, -0.2, -0.8), (0.3, 0.1, 0.7), (-0.5, 0.6, 0.2), (0.8, -0.4, 0.3))
    )
    exponents = np.asarray((0.8, 0.6, 0.7, 0.9))
    mask, omega = 1, 0.3
    if case == "repeated":
        positions[1] = positions[0]
        exponents = np.asarray((0.05, 1.8, 0.09, 0.8))
        mask = 3
    elif case == "diffuse":
        positions *= 4
        exponents = np.asarray((0.018, 0.041, 0.073, 0.012))
        mask = 8
    elif case == "translation":
        mask, omega = 15, 1.5
    powers = np.asarray(powers, dtype=np.uint32)
    norm = normalization(powers, exponents)
    expected = (
        np.concatenate(
            [
                independent_jet(powers, positions, exponents, r, omega, mask)
                for r in (0, radial)
            ]
        )
        * norm
    )
    for selected in range(4):
        for kind in (0, 1):
            actual = np.empty(16)
            assert (
                paired_primitive(
                    index,
                    powers,
                    positions,
                    exponents,
                    omega,
                    radial,
                    mask,
                    selected,
                    kind,
                    actual,
                )
                == 0
            )
            # This host build disables FMA contraction. Sharing must preserve
            # every bit of each old consumer, in addition to the independent gate.
            np.testing.assert_array_equal(
                actual[:8].view(np.uint64), actual[8:].view(np.uint64)
            )
            if kind:
                np.testing.assert_allclose(
                    actual[:8] * norm, expected, rtol=3e-10, atol=2e-11
                )
            else:
                np.testing.assert_allclose(
                    actual[[0, 4]] * norm, expected[[0, 4]], rtol=3e-10, atol=2e-11
                )


@pytest.fixture(scope="module")
def shared_policy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the production parser without querying a device or runtime."""
    source = (ROOT / "src/scf/cuda/rhf_policy.cpp").read_text()
    begin = source.index("bool canonical_rsh_values_requested()")
    end = source.index("unsigned direct_coulomb_reachable_mode()", begin)
    folder = tmp_path_factory.mktemp("shared-rsh-policy")
    cpp, executable = folder / "probe.cpp", folder / "probe"
    cpp.write_text(
        "#include <cstdlib>\n#include <cstring>\n"
        + source[begin:end]
        + "int main(){return canonical_rsh_values_requested()?1:0;}\n"
    )
    compile_cached_probe(cpp, executable)
    return executable


@pytest.mark.parametrize(
    "value,expected",
    [(None, 0), ("0", 0), ("shared", 1), ("1", 1), ("forces", 0), ("invalid", 0)],
)
def test_shared_rsh_is_explicit_and_default_off(
    shared_policy: Path, value: str | None, expected: int
) -> None:
    env = dict(os.environ)
    env.pop("GENERATIVEQC_CANONICAL_RSH_VALUES", None)
    if value is not None:
        env["GENERATIVEQC_CANONICAL_RSH_VALUES"] = value
    result = subprocess.run([str(shared_policy)], env=env, check=False, timeout=10)
    assert result.returncode == expected


PROBE = r"""
#include "paired.cuh"
using namespace generativeqc::scf::cuda_execution;
template<typename Scalar> void write(Scalar v,double* out) {
  out[0]=scalar_value(v);out[1]=out[2]=out[3]=0;
  if constexpr(std::is_same_v<Scalar,Dual3>) {
    out[1]=v.derivative_x;out[2]=v.derivative_y;out[3]=v.derivative_z;
  }
}
template<unsigned A,unsigned B,unsigned C,unsigned D,typename Scalar>
void run(const unsigned* powers,const double* positions,const double* exponents,
    double omega,unsigned range,unsigned mask,unsigned selected,double* out) {
  Angular angular[4];Vec3<Scalar> centers[4];
  for(unsigned i=0;i<4;++i) {
    angular[i]={powers[3*i],powers[3*i+1],powers[3*i+2]};
    centers[i]={scalar<Scalar>(positions[3*i]),scalar<Scalar>(positions[3*i+1]),scalar<Scalar>(positions[3*i+2])};
    if constexpr(std::is_same_v<Scalar,Dual3>) {
      double seed=(mask&(1U<<i))?1.0:0.0;
      centers[i].x.derivative_x=seed;centers[i].y.derivative_y=seed;centers[i].z.derivative_z=seed;
    }
  }
#define INPUTS exponents[0],centers[0],angular[0],exponents[1],centers[1],angular[1], \
               exponents[2],centers[2],angular[2],exponents[3],centers[3],angular[3]
  auto radial=static_cast<generativeqc::integrals::CoulombRange>(range);
  const auto pair=primitive_eri_cartesian_shell_pairs<A,B,C,D,Scalar,true>(
      INPUTS,radial,omega,(selected&1)!=0,(selected&2)!=0);
  const auto full=primitive_eri_cartesian_shell_pairs<A,B,C,D>(
      INPUTS,generativeqc::integrals::CoulombRange::Full,0.0,(selected&1)!=0,(selected&2)!=0);
  const auto other=primitive_eri_cartesian_shell_pairs<A,B,C,D>(
      INPUTS,radial,omega,(selected&1)!=0,(selected&2)!=0);
#undef INPUTS
  write(pair.full,out);write(pair.selected,out+4);write(full,out+8);write(other,out+12);
}
template<typename Scalar> int dispatch(unsigned index,const unsigned* powers,
    const double* positions,const double* exponents,double omega,unsigned range,
    unsigned mask,unsigned selected,double* out) {
  switch(index) {
// DISPATCH
  }
  return -1;
}
extern "C" int evaluate(unsigned index,const unsigned* powers,const double* positions,
    const double* exponents,double omega,unsigned range,unsigned mask,unsigned selected,
    unsigned kind,double* out) {
  if(kind==0)return dispatch<double>(index,powers,positions,exponents,omega,range,mask,selected,out);
  return dispatch<Dual3>(index,powers,positions,exponents,omega,range,mask,selected,out);
}
"""

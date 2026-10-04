"""All-center scalar roots against independent primitive and contracted oracles.

Compile the emitted primitive, complete shell-class canonicalizer, and policy
freezer. Host execution establishes arithmetic and slot ownership, not CUDA
resource use, native queue coverage, or complete endpoint performance.
"""

import ctypes
import itertools
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)
from generativeqc_compiler.integral.direct_order2_shell_cuda import (
    emit_direct_order2_shell_header,
)
from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)
from generativeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)
from generativeqc_compiler.integral.direct_source_contraction_cuda import (
    emit_direct_source_contraction_header,
)
from numpy.typing import NDArray
from test_coulomb_optional_allocation import compile_cached_probe
from test_direct_jk_optional_allocation import _definition
from test_hermite_convolution import independent_jet, normalization

ROOT = Path(__file__).resolve().parents[2]
PARTITIONS = (
    (3, 3, 1, 0),
    (3, 2, 2, 0),
    (3, 2, 1, 1),
    (2, 2, 2, 1),
    (3, 3, 2, 0),
    (3, 3, 1, 1),
    (3, 2, 2, 1),
    (2, 2, 2, 2),
)
POSITIONS = ((0.1, -0.2, -0.8), (0.3, 0.1, 0.7), (-0.5, 0.6, 0.2), (0.8, -0.4, 0.3))
PERMUTATIONS = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


def powers_for(partition: tuple[int, ...], orientation: int) -> NDArray[np.uint32]:
    powers = np.zeros((4, 3), dtype=np.uint32)
    for center, order in enumerate(partition):
        for quantum in range(order):
            powers[center, (center + quantum + orientation * (quantum + 1)) % 3] += 1
    return powers


@pytest.fixture(scope="module")
def scalar_probe(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires a C++ compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    folder = tmp_path_factory.mktemp("scalar-center")
    headers = {
        **emit_direct_cartesian_contraction_headers(),
        **emit_direct_pair_support_headers(),
        **emit_direct_recurrence_headers(),
        "generated_direct_order2_shell.cuh": emit_direct_order2_shell_header(),
        "generated_direct_source_contraction.cuh": emit_direct_source_contraction_header(),
    }
    for name, source in headers.items():
        (folder / name).write_text(source)
    (folder / "cuda_runtime.h").write_text(
        "#pragma once\n#include <algorithm>\n#include <cmath>\n"
        "#define __device__\n#define __host__\n#define __forceinline__ inline\n#define __noinline__\n"
        "using std::min;using std::max;\ninline unsigned __popc(unsigned x){return __builtin_popcount(x);}\n"
    )
    dispatch = "\n".join(
        f"case {item}:return primitive<{','.join(map(str, partition))}>(powers,positions,exponents,radial,omega,output);"
        for item, partition in enumerate(PARTITIONS)
    )
    cpp, obj, library = folder / "probe.cpp", folder / "probe.o", folder / "probe.so"
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
        [compiler, "-shared", str(obj), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    owner = ctypes.CDLL(str(library))
    integers = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    owner.evaluate.argtypes = [
        ctypes.c_uint,
        integers,
        doubles,
        doubles,
        ctypes.c_uint,
        ctypes.c_double,
        doubles,
    ]
    owner.evaluate.restype = ctypes.c_int
    owner.contracted.argtypes = [
        integers,
        doubles,
        doubles,
        doubles,
        doubles,
        integers,
        ctypes.c_uint,
        ctypes.c_double,
        doubles,
    ]
    owner.contracted.restype = ctypes.c_int
    return owner


@pytest.mark.parametrize("item", range(8))
@pytest.mark.parametrize("orientation", range(4))
@pytest.mark.parametrize("case", ("distinct", "repeated", "diffuse", "translated"))
@pytest.mark.parametrize("radial", range(3), ids=("full", "long", "short"))
def test_scalar_primitive_all_centers(
    scalar_probe: ctypes.CDLL, item: int, orientation: int, case: str, radial: int
) -> None:
    powers = powers_for(PARTITIONS[item], orientation)
    positions = np.asarray(POSITIONS)
    exponents = np.asarray((0.8, 0.6, 0.7, 0.9))
    omega = 0.3
    if case == "repeated":
        positions[1] = positions[0]
    elif case == "diffuse":
        positions *= 4
        exponents = np.asarray((0.018, 0.041, 0.073, 0.012))
    elif case == "translated":
        positions += np.asarray((1.7, -0.6, 2.3))
        omega = 1.5
    actual = np.empty(24)
    assert (
        scalar_probe.evaluate(item, powers, positions, exponents, radial, omega, actual)
        == 0
    )
    norm = normalization(powers, exponents)
    expected = (
        np.asarray(
            [
                independent_jet(powers, positions, exponents, radial, omega, 1 << c)[1:]
                for c in range(4)
            ]
        )
        * norm
    )
    np.testing.assert_allclose(
        actual[:12].reshape((4, 3)) * norm, expected, rtol=3e-10, atol=2e-11
    )
    np.testing.assert_allclose(
        actual[12:].reshape((4, 3)) * norm, expected, rtol=3e-10, atol=2e-11
    )


@pytest.mark.parametrize("item", range(8))
@pytest.mark.parametrize("radial", range(3), ids=("full", "long", "short"))
def test_contracted_gradients_restore_every_slot(
    scalar_probe: ctypes.CDLL, item: int, radial: int
) -> None:
    powers = powers_for(PARTITIONS[item], 1)
    positions = np.asarray(POSITIONS)
    exponents = np.asarray(((0.8, 0.23), (0.6, 0.31), (0.7, 0.19), (0.9, 0.43)))
    coefficients = np.asarray(((0.7, -0.2), (-0.4, 0.3), (1.2, 0.1), (0.9, -0.15)))
    ao_coefficients = np.asarray((1.1, 0.8, 1.3, 0.6))
    expected = np.zeros((4, 3))
    for selected in itertools.product(range(2), repeat=4):
        current = np.asarray([exponents[c, selected[c]] for c in range(4)])
        weight = np.prod(
            [coefficients[c, selected[c]] * ao_coefficients[c] for c in range(4)]
        )
        expected += weight * np.asarray(
            [
                independent_jet(powers, positions, current, radial, 0.3, 1 << c)[1:]
                for c in range(4)
            ]
        )
    norm = normalization(powers, exponents[:, 0])
    for permutation in PERMUTATIONS:
        output = np.empty(12)
        indices = np.asarray(permutation, dtype=np.uint32)
        assert (
            scalar_probe.contracted(
                powers,
                positions,
                exponents,
                coefficients,
                ao_coefficients,
                indices,
                radial,
                0.3,
                output,
            )
            == 0
        )
        np.testing.assert_allclose(
            output.reshape((4, 3)) * norm,
            expected[list(permutation)] * norm,
            rtol=3e-10,
            atol=2e-11,
        )


@pytest.mark.parametrize("indices", ((0, 0, 1, 2), (0, 1, 0, 1), (0, 0, 0, 0)))
@pytest.mark.parametrize("radial", range(3), ids=("full", "long", "short"))
def test_equal_ao_ids_preserve_independent_slot_gradients(
    scalar_probe: ctypes.CDLL, indices: tuple[int, ...], radial: int
) -> None:
    """AO equality must not collapse gradient slots during canonicalization."""
    powers = powers_for(PARTITIONS[7], 1)
    positions = np.asarray(POSITIONS)
    exponents = np.asarray(((0.8, 0.23), (0.6, 0.31), (0.7, 0.19), (0.9, 0.43)))
    coefficients = np.asarray(((0.7, -0.2), (-0.4, 0.3), (1.2, 0.1), (0.9, -0.15)))
    ao_coefficients = np.asarray((1.1, 0.8, 1.3, 0.6))
    slots = list(indices)
    expected = np.zeros((4, 3))
    for selected in itertools.product(range(2), repeat=4):
        current = np.asarray([exponents[indices[c], selected[c]] for c in range(4)])
        weight = np.prod(
            [
                coefficients[indices[c], selected[c]] * ao_coefficients[indices[c]]
                for c in range(4)
            ]
        )
        expected += weight * np.asarray(
            [
                independent_jet(
                    powers[slots], positions[slots], current, radial, 0.3, 1 << c
                )[1:]
                for c in range(4)
            ]
        )
    output = np.empty(12)
    assert (
        scalar_probe.contracted(
            powers,
            positions,
            exponents,
            coefficients,
            ao_coefficients,
            np.asarray(indices, dtype=np.uint32),
            radial,
            0.3,
            output,
        )
        == 0
    )
    norm = normalization(powers[slots], exponents[slots, 0])
    np.testing.assert_allclose(
        output.reshape((4, 3)) * norm, expected * norm, rtol=3e-10, atol=2e-11
    )


@pytest.fixture(scope="module")
def scalar_policy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    policy = (ROOT / "src/scf/cuda/rhf_policy.cpp").read_text()
    setup = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    definitions = "\n".join(
        _definition(policy, marker)
        for marker in (
            "bool scalar_center_gradient_requested()",
            "unsigned direct_coulomb_reachable_mode()",
            "unsigned direct_hermite_convolution_mode()",
        )
    )
    folder = tmp_path_factory.mktemp("scalar-center-policy")
    cpp, executable = folder / "probe.cpp", folder / "probe"
    cpp.write_text(
        "#include <cstdlib>\n#include <cstring>\n#include <cassert>\n"
        "namespace cuda_policy {\n" + definitions + "}\n"
        "struct DeviceBatch { unsigned direct_coulomb_reachable=0, direct_hermite_convolution=0; bool direct_scalar_center_gradient=false; };\n"
        + _definition(setup, "void configure_direct_coulomb_recurrence(")
        + "\nint main(){DeviceBatch original;configure_direct_coulomb_recurrence(original);"
        "const bool selected=original.direct_scalar_center_gradient;"
        'setenv("GENERATIVEQC_DIRECT_SCALAR_CENTER_GRADIENT","0",1);'
        "DeviceBatch later;configure_direct_coulomb_recurrence(later);"
        "assert(!later.direct_scalar_center_gradient && original.direct_scalar_center_gradient==selected);return selected?1:0;}\n"
    )
    compile_cached_probe(cpp, executable)
    return executable


@pytest.mark.parametrize("selection", (None, "0", "scalar", "1", "invalid"))
@pytest.mark.parametrize("reachable", ("0", "values", "forces", "all"))
@pytest.mark.parametrize("convolution", ("0", "values", "forces", "all"))
def test_scalar_policy_is_frozen_and_other_force_consumers_retain_priority(
    scalar_policy: Path, selection: str | None, reachable: str, convolution: str
) -> None:
    env = {
        **os.environ,
        "GENERATIVEQC_DIRECT_COULOMB_REACHABLE": reachable,
        "GENERATIVEQC_DIRECT_HERMITE_CONVOLUTION": convolution,
    }
    env.pop("GENERATIVEQC_DIRECT_SCALAR_CENTER_GRADIENT", None)
    if selection is not None:
        env["GENERATIVEQC_DIRECT_SCALAR_CENTER_GRADIENT"] = selection
    result = subprocess.run([str(scalar_policy)], env=env, check=False, timeout=10)
    expected = (
        selection in ("scalar", "1")
        and reachable not in ("forces", "all")
        and convolution not in ("forces", "all")
    )
    assert result.returncode == int(expected)


PROBE = r"""
#include "generated_direct_source_contraction.cuh"
using namespace generativeqc::scf::cuda_execution;
template<unsigned A,unsigned B,unsigned C,unsigned D>
int primitive(const unsigned* powers,const double* positions,const double* exponents,
              unsigned radial,double omega,double* output) {
  Angular angular[4]; Vec3<double> center[4];
  for(unsigned c=0;c<4;++c) {
    angular[c]={powers[3*c],powers[3*c+1],powers[3*c+2]};
    center[c]={positions[3*c],positions[3*c+1],positions[3*c+2]};
  }
  const auto range=static_cast<generativeqc::integrals::CoulombRange>(radial);
  const auto actual=primitive_eri_scalar_center_gradient<A,B,C,D>(
      exponents[0],center[0],angular[0],exponents[1],center[1],angular[1],
      exponents[2],center[2],angular[2],exponents[3],center[3],angular[3],range,omega);
  for(unsigned c=0;c<4;++c) {
    for(unsigned axis=0;axis<3;++axis) output[3*c+axis]=actual.center[c][axis];
    Vec3<Dual3> seeded[4];
    for(unsigned j=0;j<4;++j) {
      const double s=c==j?1.:0.;
      seeded[j]={{center[j].x,s,0,0},{center[j].y,0,s,0},{center[j].z,0,0,s}};
    }
    const auto old=primitive_eri_cartesian_shell_pairs<A,B,C,D,Dual3>(
      exponents[0],seeded[0],angular[0],exponents[1],seeded[1],angular[1],
      exponents[2],seeded[2],angular[2],exponents[3],seeded[3],angular[3],range,omega);
    output[12+3*c]=old.derivative_x;output[13+3*c]=old.derivative_y;output[14+3*c]=old.derivative_z;
  }
  return 0;
}
extern "C" int evaluate(unsigned item,const unsigned* powers,const double* positions,
    const double* exponents,unsigned radial,double omega,double* output) {
  switch(item) {
    // DISPATCH
    default:return 1;
  }
}
extern "C" int contracted(const unsigned* powers,const double* positions,
    const double* exponents,const double* coefficients,const double* ao_coefficients,
    const unsigned* indices,unsigned radial,double omega,double* output) {
  const std::int32_t atoms[4]={0,1,2,3},ao_shells[4]={0,1,2,3};
  const std::int64_t offsets[5]={0,2,4,6,8};
  std::uint8_t shell_angular[4],ao_angular[12];
  for(unsigned c=0;c<4;++c) {
    shell_angular[c]=powers[3*c]+powers[3*c+1]+powers[3*c+2];
    for(unsigned axis=0;axis<3;++axis) ao_angular[3*c+axis]=powers[3*c+axis];
  }
  DeviceBatch batch{};
  batch.batch_size=1;batch.direct_nbf=4;batch.positions=positions;batch.shell_atoms=atoms;
  batch.direct_ao_shells=ao_shells;batch.shell_angular=shell_angular;
  batch.direct_ao_angular=ao_angular;batch.direct_ao_coefficients=ao_coefficients;
  batch.shell_primitive_offsets=offsets;batch.primitive_exponents=exponents;
  batch.primitive_coefficients=coefficients;
  const auto range=static_cast<generativeqc::integrals::CoulombRange>(radial);
  const auto klass=direct_quartet_shell_class_device(shell_angular[indices[0]],shell_angular[indices[1]],
                                                    shell_angular[indices[2]],shell_angular[indices[3]]);
  const auto order=direct_shell_class_angular_order(klass);
  CartesianQuartetGradient actual;
  if(order==7) actual=contracted_eri_scalar_center_gradient<7>(klass,batch,0,indices[0],indices[1],indices[2],indices[3],range,omega);
  else if(order==8) actual=contracted_eri_scalar_center_gradient<8>(klass,batch,0,indices[0],indices[1],indices[2],indices[3],range,omega);
  else return 1;
  for(unsigned c=0;c<4;++c)
    for(unsigned axis=0;axis<3;++axis) output[3*c+axis]=actual.center[c][axis];
  return 0;
}
"""

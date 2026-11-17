"""Execute coexisting angular families against independent integral blocks."""

import ctypes
import typing
from itertools import product

import numpy as np
import pytest
from generativeqc_compiler.integral.one_electron_cuda import (
    emit_one_electron_values_cuda,
)
from generativeqc_compiler.integral.one_electron_families_cuda import (
    emit_one_electron_derivative_family_cuda,
    emit_one_electron_family_metadata_cpp,
    emit_one_electron_value_family_cuda,
)
from test_one_electron_g_shared_values import (
    test_emitted_g_components_against_libcint as _check_values,
)


@pytest.fixture(scope="module")
def angular_families(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: typing.Any
) -> typing.Any:
    """Compile actual generated headers together through the verified cache."""
    directory = tmp_path_factory.mktemp("one-electron-angular-families")
    (directory / "cuda_runtime.h").write_text(
        "#pragma once\n#define __device__\n#define __forceinline__ inline\n"
        "#define __noinline__\n"
    )
    for name, source in (
        ("generated_one_electron_values.cuh", emit_one_electron_values_cuda(4)),
        ("value_family.cuh", emit_one_electron_value_family_cuda()),
        ("derivative_family.cuh", emit_one_electron_derivative_family_cuda()),
        ("family_metadata.cuh", emit_one_electron_family_metadata_cpp()),
    ):
        (directory / name).write_text(source)
    source = r"""
#include "generated_one_electron_values.cuh"
#include "value_family.cuh"
#include "family_metadata.cuh"
static_assert(generativeqc::scf::generated_one_electron_families::through_f_maximum_angular_momentum
    == generativeqc::scf::generated_one_electron_through_f::ValueFamily::maximum_angular_momentum);
struct NuclearView {
  const long long* atom_offsets;
  const double* atomic_numbers;
  const double* positions;
};
extern "C" void evaluate_f(const double* inputs, double* outputs, unsigned count) {
  using Policy = generativeqc::scf::generated_one_electron_through_f::ValueFamily;
  for (unsigned i = 0; i < count; ++i) {
    const double* p = inputs + 14*i;
    const long long offsets[2]{0,1};
    const NuclearView view{offsets, p+11, p+8};
    const auto pair = Policy::prepare(p[0],p[1],p+2,p+5);
    Policy::Accumulator result{};
    Policy::accumulate(result,pair,p[12],p[13],1.0,view,0);
    double values[Policy::channels];
    Policy::project(result,values);
    outputs[3*i] = values[0];
    outputs[3*i+1] = values[2];
    outputs[3*i+2] = values[3];
  }
}
extern "C" void evaluate_g(const double* inputs, double* outputs, unsigned count) {
  namespace one = generativeqc::scf::generated_one_electron;
  for (unsigned i = 0; i < count; ++i) {
    const double* p = inputs + 14*i;
    const auto pair = one::make_pair(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
    const auto st = one::overlap_kinetic(pair,p[12],p[13]);
    outputs[3*i] = st.overlap;
    outputs[3*i+1] = st.kinetic;
    outputs[3*i+2] = p[11]*one::attraction(pair,p[12],p[13],p[8],p[9],p[10]);
  }
}
"""
    from tools.generativeqc_validation.one_electron_derivatives_cuda import (
        derivative_evaluation_body,
    )

    derivative_source = (
        '#include "derivative_family.cuh"\n#include "family_metadata.cuh"\n'
        "static_assert(generativeqc::scf::generated_one_electron_families::through_f_maximum_angular_momentum\n"
        " == generativeqc::scf::generated_one_electron_through_f_derivatives::Program::maximum_angular_momentum);\n"
        'extern "C" void derivative_f(const double* inputs, double* outputs, unsigned count) {\n'
        "using one = generativeqc::scf::generated_one_electron_through_f_derivatives::Program;\n"
        "for (unsigned i=0;i<count;++i) { const double* p=inputs+14*i; double* out=outputs+27*i;\n"
        + derivative_evaluation_body()
        + "\n}}\n"
    )
    path = directory / "values.cpp"
    path.write_text(source)
    derivative_path = directory / "derivatives.cpp"
    derivative_path.write_text(derivative_source)
    library_path = directory / "probe.so"
    required_native_cxx.build_shared(
        [path, derivative_path],
        library_path,
        compile_args=["-std=c++17", "-O0", "-I", str(directory)],
        compile_timeout=180,
    )
    library = ctypes.CDLL(str(library_path))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    for name in ("evaluate_f", "evaluate_g", "derivative_f"):
        function = getattr(library, name)
        function.argtypes = [pointer, pointer, ctypes.c_uint]
        function.restype = None
    return library


@pytest.mark.parametrize(
    ("angular", "primitive_exponents"),
    [
        (angular, exponents)
        for angular in product(range(4), repeat=2)
        for exponents in ((0.8, 0.35), (0.03, 14.0), (18.0, 0.05))
        if max(angular) >= 3 or exponents == (0.8, 0.35)
    ],
)
@pytest.mark.parametrize("variant", ["asymmetric", "near", "boys_below", "distant"])
def test_through_f_value_policy_against_libcint(
    angular_families: typing.Any,
    angular: tuple[int, int],
    primitive_exponents: tuple[float, float],
    variant: str,
) -> None:
    _check_values(angular_families.evaluate_f, angular, primitive_exponents, variant)


@pytest.mark.parametrize("angular", [(4, 0), (1, 4), (4, 4)])
@pytest.mark.parametrize(
    "primitive_exponents", [(0.8, 0.35), (0.03, 14.0), (18.0, 0.05)]
)
def test_g_fallback_coexists_with_through_f(
    angular_families: typing.Any,
    angular: tuple[int, int],
    primitive_exponents: tuple[float, float],
) -> None:
    _check_values(
        angular_families.evaluate_g, angular, primitive_exponents, "asymmetric"
    )


def test_typed_derivative_bindings_against_independent_blocks(
    angular_families: typing.Any,
) -> None:
    from tools.generativeqc_validation.one_electron_derivatives import (
        one_electron_derivative_matrix,
    )

    evaluate = angular_families.derivative_f
    for fixture in one_electron_derivative_matrix():
        values = np.zeros((len(fixture.records), 27))
        evaluate(fixture.records, values, len(values))
        actual = fixture.contract(values)
        np.testing.assert_allclose(actual, fixture.reference, atol=3e-11, rtol=6e-12)
        np.testing.assert_allclose(
            fixture.spherical(actual),
            fixture.spherical_reference,
            atol=3e-11,
            rtol=6e-12,
        )

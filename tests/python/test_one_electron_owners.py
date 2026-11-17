"""Consumer-specific shell bounds must coexist without sharing C++ definitions."""

import ctypes
import typing

import numpy as np
import pytest
from generativeqc_compiler.integral.one_electron_cuda import (
    emit_one_electron_values_cuda,
)
from generativeqc_compiler.integral.one_electron_derivatives_cuda import (
    emit_one_electron_derivatives_cuda,
)

from tools.generativeqc_validation.one_electron_derivatives_cuda import (
    derivative_evaluation_body,
)


@pytest.mark.parametrize("owner", ["", "COSX", "bad::owner", "_reserved", "a__b", 3])
@pytest.mark.parametrize(
    "emit", [emit_one_electron_values_cuda, emit_one_electron_derivatives_cuda]
)
def test_emission_rejects_invalid_owner(emit: typing.Any, owner: typing.Any) -> None:
    with pytest.raises(ValueError, match="owner"):
        emit(0, owner=owner)


def test_distinct_shell_bounds_coexist(
    tmp_path: typing.Any, required_native_cxx: typing.Any
) -> None:
    """Execute both owners in one TU, including their different invalid-ID bounds.

    Low-angular artifacts keep this ownership test cheap; the existing complete
    through-f emitted-header tests qualify COSX against independent Libcint.
    """
    headers = [
        emit_one_electron_values_cuda(0),
        emit_one_electron_derivatives_cuda(1, owner="cosx"),
        emit_one_electron_derivatives_cuda(0),
        emit_one_electron_values_cuda(1, owner="cosx"),
    ]
    source = (
        "#define __device__\n#define __forceinline__ inline\n#define __noinline__\n"
    )
    source += "\n".join(headers).replace("#include <cuda_runtime.h>", "")
    for name, qualifier in (("default", ""), ("cosx", "_cosx")):
        namespace = f"generativeqc::scf::generated{qualifier}_one_electron"
        source += f"""
extern "C" void evaluate_{name}(const double* p, double* output) {{
  {{
    namespace one = {namespace};
    const auto pair = one::make_pair(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    const auto st = one::overlap_kinetic(pair, p[12], p[13]);
    output[0] = st.overlap;
    output[1] = st.kinetic;
    output[2] = p[11] * one::attraction(pair, p[12], p[13], p[8], p[9], p[10]);
  }}
  namespace one = {namespace}_derivatives;
  double* out = output + 3;
"""
        source += derivative_evaluation_body() + "\n}\n"
    path = tmp_path / "owners.cpp"
    path.write_text(source)
    library_path = tmp_path / "owners.so"
    required_native_cxx.build_shared(
        [path], library_path, compile_args=["-std=c++17", "-O0"], compile_timeout=120
    )
    library = ctypes.CDLL(str(library_path))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    default = library.evaluate_default
    scoped = library.evaluate_cosx
    for evaluate in (default, scoped):
        evaluate.argtypes = [pointer, pointer]
        evaluate.restype = None
    inputs = np.array(
        [0.7, 1.3, 0.2, -0.4, 0.1, -0.3, 0.6, -0.8, 0.5, 0.2, -0.1, 2.3, 0, 0]
    )
    ordinary = np.zeros(30)
    specialized = np.zeros(30)
    default(inputs, ordinary)
    scoped(inputs, specialized)
    np.testing.assert_allclose(specialized, ordinary, atol=2e-14, rtol=2e-14)
    inputs[12] = 1
    default(inputs, ordinary)
    scoped(inputs, specialized)
    assert np.isnan(ordinary[:3]).all()
    assert np.isnan(ordinary[[3, 12, 21]]).all()
    assert np.isfinite(specialized).all()

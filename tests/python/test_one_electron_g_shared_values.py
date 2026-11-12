"""Execute emitted high-angular S/T/V against independent Libcint integrals.

The host probe validates CUDA-source arithmetic and dispatch only. It does not
qualify device resources, compilation cost, or complete GPU endpoint speed.
"""

import ctypes
import math
import typing
from itertools import product

import numpy as np
import pytest
from generativeqc_compiler.integral.one_electron_cuda import (
    emit_one_electron_values_cuda,
)
from generativeqc_compiler.integral.shell_spec import cartesian_components


@pytest.fixture(scope="module")
def emitted_g_values(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: typing.Any
) -> typing.Any:
    """Compile the actual g-capable emitter with the mandatory host cache."""
    directory = tmp_path_factory.mktemp("one-electron-g-shared")
    source = emit_one_electron_values_cuda(4).replace("#include <cuda_runtime.h>", "")
    source = (
        source.replace("__device__", "")
        .replace("__forceinline__", "inline")
        .replace("__noinline__", "")
    )
    source += r"""
extern "C" void evaluate_g(const double* inputs, double* outputs, unsigned count) {
  namespace one = generativeqc::scf::generated_one_electron;
  for (unsigned i = 0; i < count; ++i) {
    const double* p = inputs + 14 * i;
    const auto pair = one::make_pair(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    const auto st = one::overlap_kinetic(pair, p[12], p[13]);
    outputs[3 * i] = st.overlap;
    outputs[3 * i + 1] = st.kinetic;
    outputs[3 * i + 2] = p[11] * one::attraction(pair, p[12], p[13], p[8], p[9], p[10]);
  }
}
"""
    path = directory / "probe.cpp"
    path.write_text(source)
    library_path = directory / "probe.so"
    required_native_cxx.build_shared(
        [path], library_path, compile_args=["-std=c++17", "-O0"], compile_timeout=180
    )
    library = ctypes.CDLL(str(library_path))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    library.evaluate_g.argtypes = [pointer, pointer, ctypes.c_uint]
    library.evaluate_g.restype = None
    return library.evaluate_g


def _geometry(variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.array([0.2, -0.3, 0.1])
    b = np.array([-0.4, 0.15, 0.5])
    c = np.array([0.17, -0.11, -0.4])
    if variant == "coincident":
        b, c = a.copy(), a.copy()
    elif variant == "near":
        b, c = a + [1e-10, 0, 0], a + [0, 0, 1e-9]
    elif variant in ("boys_below", "boys_above"):
        argument = 20 + (-1e-8 if variant == "boys_below" else 1e-8)
        c = (0.8 * a + 0.35 * b) / 1.15 + [math.sqrt(argument / 1.15), 0, 0]
    elif variant == "distant":
        c = np.array([50.0, 30.0, -40.0])
    elif variant == "translated":
        translation = np.array([128.0, -64.0, 256.0])
        a, b, c = a + translation, b + translation, c + translation
    return a, b, c


def _raw_norm(exponent: float, component: str) -> float:
    value = (math.pi / (2 * exponent)) ** 1.5
    for axis in "xyz":
        n = component.count(axis)
        value *= math.prod(range(1, 2 * n, 2)) / (4 * exponent) ** n
    return value


@pytest.mark.parametrize("angular", list(product(range(5), repeat=2)))
@pytest.mark.parametrize(
    "variant",
    [
        "asymmetric",
        "coincident",
        "near",
        "boys_below",
        "boys_above",
        "distant",
        "translated",
    ],
)
def test_emitted_g_components_against_libcint(
    emitted_g_values: typing.Any, angular: tuple[int, int], variant: str
) -> None:
    """Cover all 1,225 Cartesian pairs, not an interpreter of the original DAG."""
    gto = pytest.importorskip("pyscf.gto")
    alpha, beta, charge = 0.8, 0.35, 2.3
    a, b, c = _geometry(variant)
    components = tuple(cartesian_components(l) for l in angular)
    mol = gto.M(
        atom=[("ghost-H", a), ("ghost-He", b)],
        basis={"H": [[angular[0], [alpha, 1.0]]], "He": [[angular[1], [beta, 1.0]]]},
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    overlap = mol.intor("int1e_ovlp_cart")
    with mol.with_rinv_origin(c):
        reference = np.array(
            [
                overlap,
                mol.intor("int1e_kin_cart"),
                -charge * mol.intor("int1e_rinv_cart"),
            ]
        )
    exponents = [alpha] * len(components[0]) + [beta] * len(components[1])
    scales = np.sqrt(
        overlap.diagonal()
        / [
            _raw_norm(e, name)
            for e, name in zip(exponents, components[0] + components[1], strict=True)
        ]
    )
    reference /= scales[None, :, None] * scales[None, None, :]
    offsets = [sum(len(cartesian_components(k)) for k in range(l)) for l in angular]
    inputs = np.array(
        [
            [alpha, beta, *a, *b, *c, charge, offsets[0] + i, offsets[1] + j]
            for i, j in product(range(len(components[0])), range(len(components[1])))
        ]
    )
    values = np.empty((len(inputs), 3))
    emitted_g_values(inputs, values, len(inputs))
    actual = values.T.reshape(3, len(components[0]), len(components[1]))
    expected = reference[:, : len(components[0]), len(components[0]) :]
    # These are the incumbent independent gates, not relaxed for shared lowering.
    atol, rtol = (2e-9, 2e-11) if 4 in angular else (2e-11, 3e-12)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("invalid", [35, 256, 2**32 - 1])
@pytest.mark.parametrize("position", [12, 13])
def test_emitted_g_invalid_component_refuses(
    emitted_g_values: typing.Any, invalid: int, position: int
) -> None:
    """Out-of-domain Cartesian dispatch keeps explicit NaN refusal."""
    inputs = np.array(
        [[0.8, 0.35, 0.2, -0.3, 0.1, -0.4, 0.15, 0.5, 0.17, -0.11, -0.4, 2.3, 20, 20]]
    )
    inputs[0, position] = invalid
    outputs = np.empty((1, 3))
    emitted_g_values(inputs, outputs, 1)
    assert np.isnan(outputs).all()

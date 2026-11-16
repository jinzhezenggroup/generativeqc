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
    _emit_component_index,
    _emit_pair_geometry,
    _emit_support_cuda,
    emit_one_electron_values_cuda,
)
from generativeqc_compiler.integral.one_electron_derivatives_cuda import (
    _emit_axis_permutations,
    _emit_gradient_shell_helper,
)
from generativeqc_compiler.integral.shell_spec import cartesian_components

_GRADIENT_PAIRS = ((0, 4), (1, 4), (4, 0), (4, 1), (4, 4))


@pytest.fixture(scope="module")
def emitted_g_gradient_x(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: typing.Any
) -> typing.Any:
    """Exercise real g-bearing helpers, including both Gaussian-center orders.

    Canonical lower/g helpers precede their reversed g/lower callers. The shared
    covariance fixes x, so x derivatives cover the changed dispatch. Independent
    Boys inputs isolate it from the existing radial evaluator; full axis
    composition and device qualification have separate gates.
    """
    directory = tmp_path_factory.mktemp("one-electron-g-gradient")
    source = "\n".join(
        [
            _emit_support_cuda(9),
            "struct GradientAxis { double first, second; };",
            _emit_pair_geometry(),
            _emit_component_index(4),
            _emit_axis_permutations(4),
            *(
                _emit_gradient_shell_helper(attraction, angular)
                for angular in _GRADIENT_PAIRS
                for attraction in (False, True)
            ),
            "}",
            "#endif",
        ]
    ).replace("#include <cuda_runtime.h>", "")
    source = (
        source.replace("__device__", "")
        .replace("__forceinline__", "inline")
        .replace("__noinline__", "")
    )
    source += r"""
extern "C" void evaluate_gradient_x(const double* inputs, double* outputs,
                                    unsigned count) {
  namespace one = generativeqc::scf::generated_one_electron;
  for (unsigned index = 0; index < count; ++index) {
    const double* row = inputs + 24 * index;
    const auto pair = one::make_pair(row[0], row[1], row[2], row[3], row[4],
                                    row[5], row[6], row[7]);
    const auto component = static_cast<unsigned>(row[13]);
    one::GradientAxis st{NAN, NAN}, attraction{NAN, NAN};
    switch (static_cast<unsigned>(row[12])) {
"""
    for index, angular in enumerate(_GRADIENT_PAIRS):
        suffix = f"{angular[0]}{angular[1]}"
        source += f"""
      case {index}U:
        st = one::overlap_kinetic_gradient_x_{suffix}(pair, component);
        attraction = one::attraction_gradient_x_{suffix}(
            pair, component, row[8], row[9], row[10], row + 14);
        break;
"""
    source += r"""
    }
    double* result = outputs + 6 * index;
    result[0] = st.first;
    result[1] = -st.first;
    result[2] = st.second;
    result[3] = -st.second;
    result[4] = row[11] * attraction.first;
    result[5] = row[11] * attraction.second;
  }
}
"""
    path = directory / "probe.cpp"
    path.write_text(source)
    library_path = directory / "probe.so"
    required_native_cxx.build_shared(
        [path],
        library_path,
        compile_args=["-std=c++17", "-O0", "-ffp-contract=off"],
        compile_timeout=180,
    )
    library = ctypes.CDLL(str(library_path))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    library.evaluate_gradient_x.argtypes = [pointer, pointer, ctypes.c_uint]
    library.evaluate_gradient_x.restype = None
    return library.evaluate_gradient_x


@pytest.fixture(
    scope="module", params=[None, 1200], ids=["baseline-f", "sm120-shared-f"]
)
def emitted_g_values(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    required_native_cxx: typing.Any,
) -> typing.Any:
    """Execute both target lowerings; this is not CUDA hardware qualification."""
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
    arguments = ["-std=c++17", "-O0"]
    if request.param is not None:
        arguments.append(f"-D__CUDA_ARCH__={request.param}")
    required_native_cxx.build_shared(
        [path], library_path, compile_args=arguments, compile_timeout=180
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


@pytest.mark.parametrize(
    "angular,primitive_exponents",
    [
        (angular, exponents)
        for angular in product(range(5), repeat=2)
        for exponents in ((0.8, 0.35), (0.03, 14.0), (18.0, 0.05))
        if max(angular) >= 3 or exponents == (0.8, 0.35)
    ],
)
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
    emitted_g_values: typing.Any,
    angular: tuple[int, int],
    primitive_exponents: tuple[float, float],
    variant: str,
) -> None:
    """Check every pair and diffuse/tight f/g orders against Libcint.

    Scalar covariance permutes all axes and may exchange Gaussian centers, so
    unequal exponents and non-symmetric external positions must remain covered.
    The g-capable header may share f-bearing representatives too; s/p/d pairs
    retain their ordinary-exponent control cases and the same numerical gates.
    """
    gto = pytest.importorskip("pyscf.gto")
    alpha, beta = primitive_exponents
    charge = 2.3
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


@pytest.mark.parametrize("angular", _GRADIENT_PAIRS)
@pytest.mark.parametrize(
    "primitive_exponents", [(0.8, 0.35), (0.03, 14.0), (18.0, 0.05)]
)
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
def test_emitted_g_gradient_covariance_against_libcint(
    emitted_g_gradient_x: typing.Any,
    angular: tuple[int, int],
    variant: str,
    primitive_exponents: tuple[float, float],
) -> None:
    """All component orbits retain independently checked A/B S/T/V derivatives."""
    gto = pytest.importorskip("pyscf.gto")
    special = pytest.importorskip("scipy.special")
    (alpha, beta), charge = primitive_exponents, 2.3
    center_a, center_b, center_c = _geometry(variant)
    components = tuple(cartesian_components(value) for value in angular)
    molecule = gto.M(
        atom=[("ghost-H", center_a), ("ghost-He", center_b)],
        basis={"H": [[angular[0], [alpha, 1.0]]], "He": [[angular[1], [beta, 1.0]]]},
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    first_count = len(components[0])
    exponents = [alpha] * first_count + [beta] * len(components[1])
    scales = np.sqrt(
        molecule.intor("int1e_ovlp_cart").diagonal()
        / [
            _raw_norm(exponent, component)
            for exponent, component in zip(
                exponents, components[0] + components[1], strict=True
            )
        ]
    )
    reference = []
    with molecule.with_rinv_origin(center_c):
        for family, factor in (("ovlp", 1.0), ("kin", 1.0), ("rinv", -charge)):
            first = (
                -factor
                * molecule.intor_by_shell(f"int1e_ip{family}_cart", (0, 1), comp=3)[0]
            )
            second = (
                -factor
                * molecule.intor_by_shell(f"int1e_ip{family}_cart", (1, 0), comp=3)[0].T
            )
            reference.extend((first, second))
    expected = np.asarray(reference) / (
        scales[:first_count][None, :, None] * scales[first_count:][None, None, :]
    )
    product_center = (alpha * center_a + beta * center_b) / (alpha + beta)
    argument = (alpha + beta) * np.sum((product_center - center_c) ** 2)
    orders = np.arange(10)
    boys = special.hyp1f1(orders + 0.5, orders + 1.5, -argument) / (2 * orders + 1)
    inputs = np.array(
        [
            [
                alpha,
                beta,
                *center_a,
                *center_b,
                *center_c,
                charge,
                _GRADIENT_PAIRS.index(angular),
                component,
                *boys,
            ]
            for component in range(first_count * len(components[1]))
        ]
    )
    outputs = np.empty((len(inputs), 6))
    emitted_g_gradient_x(inputs, outputs, len(inputs))
    actual = outputs.T.reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-11)


@pytest.mark.parametrize("angular", _GRADIENT_PAIRS)
@pytest.mark.parametrize("invalid", [None, 225, 256, 2**32 - 1])
def test_emitted_g_gradient_invalid_component_refuses(
    emitted_g_gradient_x: typing.Any, angular: tuple[int, int], invalid: int | None
) -> None:
    """Canonicalization must not turn an invalid component into a valid orbit."""
    if invalid is None:
        invalid = math.prod(len(cartesian_components(value)) for value in angular)
    inputs = np.array(
        [
            [
                0.8,
                0.35,
                *_geometry("asymmetric")[0],
                *_geometry("asymmetric")[1],
                *_geometry("asymmetric")[2],
                2.3,
                _GRADIENT_PAIRS.index(angular),
                invalid,
                *np.ones(10),
            ]
        ]
    )
    outputs = np.empty((1, 6))
    emitted_g_gradient_x(inputs, outputs, 1)
    assert np.isnan(outputs).all()

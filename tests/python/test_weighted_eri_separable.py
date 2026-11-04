"""Independent science and compiled ABI gates for exact pair-weight projection."""

from __future__ import annotations

import ctypes
import math
from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.ir import (
    DerivativeSpec,
    NuclearCoordinates,
    TranslationInvariant,
)
from generativeqc_compiler.integral.weighted_eri import (
    WeightedEriKernel,
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)
from generativeqc_compiler.integral.weighted_eri_cuda import (
    emit_weighted_eri_function,
    emit_weighted_eri_header,
)

from tools.generate_validation_references import pyscf_molecule, quartet_data
from tools.generativeqc_validation.weighted_eri import primitive_variables

CLASSES = ((1, 1, 1, 1), (2, 1, 1, 1))
EXPONENTS = (0.6, 0.8, 1.1, 0.9)
CENTERS = np.array(
    [
        [0.13, -0.31, 0.24],
        [-0.43, 0.27, 0.51],
        [0.68, -0.14, -0.22],
        [-0.21, 0.48, -0.63],
    ]
)


@cache
def _kernel(angular: tuple[int, ...], basis: str = "relative") -> WeightedEriKernel:
    return build_weighted_eri_kernel(
        build_weighted_eri_ir(angular), derivative_basis=basis, pair_separable=True
    )


def _fixture(
    kernel: WeightedEriKernel, case: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    centers = CENTERS.copy()
    if case == "pair_coincident":
        centers[1] = centers[0]
    elif case == "all_coincident":
        centers[:] = centers[0]
    generator = np.random.default_rng(6317)
    pairs = [generator.normal(size=len(pair)) for pair in kernel.pair_components]
    if case == "zero":
        pairs[1][:] = 0
    return centers, pairs


def _evaluate(
    kernel: WeightedEriKernel, centers: np.ndarray, pairs: list[np.ndarray]
) -> np.ndarray:
    variables = primitive_variables(
        EXPONENTS, centers, kernel.integral.maximum_coulomb_order
    )
    variables.update(
        {
            f"pair_weight_{side}_{index}": weight
            for side, weights in enumerate(pairs)
            for index, weight in enumerate(weights)
        }
    )
    roots = (kernel.value, *(value for row in kernel.gradients for value in row))
    return np.array([kernel.graph.evaluate(root, variables) for root in roots])


def _primitive_norm(exponent: float, component: str) -> float:
    """Obtain monomial normalization from the analytic Gaussian self-overlap."""
    powers = tuple(component.count(axis) for axis in "xyz")
    odd_factors = math.prod(math.prod(range(1, 2 * power, 2)) for power in powers)
    return (2 * exponent / math.pi) ** 0.75 * math.sqrt(
        (4 * exponent) ** sum(powers) / odd_factors
    )


@cache
def _libcint(angular: tuple[int, ...], case: str) -> tuple[np.ndarray, np.ndarray]:
    """Use separately normalized libcint blocks, never the weighted factory."""
    pytest.importorskip("pyscf")
    kernel = _kernel(angular)
    centers, _ = _fixture(kernel, case)
    molecule, scales, _ = pyscf_molecule(
        {
            "atomic_numbers": [1] * 4,
            "coordinates": centers.tolist(),
            "shells": [
                {
                    "atom_index": center,
                    "angular_momentum": degree,
                    "primitives": [[exponent, 1.0]],
                }
                for center, (degree, exponent) in enumerate(
                    zip(angular, EXPONENTS, strict=True)
                )
            ],
            "basis_representation": "cartesian",
            "charge": 0,
            "multiplicity": 1,
        }
    )
    reference = quartet_data(molecule, scales)
    normalization = np.array(
        [
            math.prod(
                _primitive_norm(exponent, component)
                for exponent, component in zip(EXPONENTS, components, strict=True)
            )
            for components in kernel.spec.components
        ]
    )
    return (
        np.asarray(reference["eri"]).ravel() / normalization,
        np.asarray(reference["gradient"]).reshape(12, -1) / normalization,
    )


@pytest.mark.parametrize("angular", CLASSES)
@pytest.mark.parametrize("basis", ("nuclear", "relative"))
@pytest.mark.parametrize(
    "case", ("regular", "pair_coincident", "all_coincident", "zero")
)
def test_pair_projection_matches_libcint(
    angular: tuple[int, ...], basis: str, case: str
) -> None:
    """Check the complete shell beyond the arbitrary-weight partition bound."""
    kernel = _kernel(angular, basis)
    centers, pairs = _fixture(kernel, case)
    assert (
        tuple(
            bra + ket
            for bra in kernel.pair_components[0]
            for ket in kernel.pair_components[1]
        )
        == kernel.spec.components
    )
    weights = np.outer(*pairs).ravel()
    value, gradient = _libcint(angular, case)
    expected = np.r_[weights @ value, gradient @ weights]
    actual = _evaluate(kernel, centers, pairs)
    np.testing.assert_allclose(actual, expected, rtol=4e-11, atol=4e-12)
    np.testing.assert_allclose(actual[1:].reshape(4, 3).sum(axis=0), 0, atol=2e-13)


@pytest.mark.parametrize("angular", CLASSES)
def test_pair_projection_signed_scale_recovery_and_finite_difference(
    angular: tuple[int, ...],
) -> None:
    """Weights stay fixed under AD, including a nonfinal recovered shell slot."""
    integral = build_weighted_eri_ir(angular)
    invariant = TranslationInvariant(dependent_center=1)
    consumer = integral.contractions[0]
    changed = replace(
        integral,
        operator=replace(integral.operator, invariants=(invariant,)),
        derivative=DerivativeSpec(1, NuclearCoordinates(), (invariant,)),
        contractions=(
            replace(
                consumer,
                output_sign=-1,
                weights=replace(consumer.weights, prefactor=0.7),
            ),
        ),
    )
    candidate = build_weighted_eri_kernel(
        changed, derivative_basis="relative", pair_separable=True
    )
    base = _kernel(angular)
    centers, pairs = _fixture(base, "regular")
    actual = _evaluate(candidate, centers, pairs)
    np.testing.assert_allclose(
        actual, -0.7 * _evaluate(base, centers, pairs), atol=4e-12
    )
    direction = np.random.default_rng(917).normal(size=(4, 3))
    analytic = np.sum(actual[1:].reshape(4, 3) * direction)
    for step in (2e-5, 5e-6):
        numeric = (
            _evaluate(candidate, centers + step * direction, pairs)[0]
            - _evaluate(candidate, centers - step * direction, pairs)[0]
        ) / (2 * step)
        assert numeric == pytest.approx(analytic, rel=2e-8, abs=2e-8)


@pytest.mark.parametrize("flag", (None, 0, 1, "true"))
def test_pair_projection_requires_explicit_boolean(flag: Any) -> None:
    with pytest.raises(TypeError, match="boolean"):
        build_weighted_eri_kernel(
            build_weighted_eri_ir(CLASSES[1]), pair_separable=flag
        )


@pytest.mark.parametrize("indices", ((), (0,), tuple(range(162))))
def test_pair_projection_rejects_explicit_component_masks(
    indices: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="complete ordered pairs"):
        build_weighted_eri_kernel(
            build_weighted_eri_ir(CLASSES[1]), indices, pair_separable=True
        )


@pytest.mark.parametrize(
    "options", ({"packed_weights": True}, {"scalar_type": "float"})
)
def test_pair_projection_rejects_incompatible_emission(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="packed component|FP64"):
        emit_weighted_eri_function(_kernel(CLASSES[1]), "candidate", **options)


@pytest.fixture(scope="module")
def compiled_pairs(native_cxx: Any, tmp_path_factory: Any) -> Any:
    """Compile the actual two-pointer ABI, including reordered output centers."""
    folder = tmp_path_factory.mktemp("weighted-pair-projection")
    kernels = tuple(_kernel(angular) for angular in CLASSES)
    source = emit_weighted_eri_header(
        tuple((kernel, f"pair_{index}") for index, kernel in enumerate(kernels)),
        backend="cpu",
    )
    source += "\nnamespace generativeqc::scf::generated_weighted_eri {\n"
    for index, kernel in enumerate(kernels):
        source += emit_weighted_eri_function(
            kernel,
            f"selected_{index}",
            backend="cpu",
            include_value=False,
            gradient_centers=(3, 1),
        )
    source += "}\nusing namespace generativeqc::scf::generated_weighted_eri;\n"
    for index in range(len(kernels)):
        source += f"""
extern "C" void probe_{index}(const double* input, const double* bra,
                              const double* ket, double* output, bool selected) {{
  Geometry geometry{{}};
  int offset = 0;
  geometry.inverse_two_p = input[offset++];
  geometry.inverse_two_q = input[offset++];
  geometry.rho = input[offset++];
  for (double& value : geometry.difference) value = input[offset++];
  for (auto& row : geometry.shifts) for (double& value : row) value = input[offset++];
  for (double& value : geometry.product_scales) value = input[offset++];
  for (auto& row : geometry.decay) for (double& value : row) value = input[offset++];
  geometry.prefactor = input[offset++];
  for (double& value : geometry.boys) value = input[offset++];
  const auto result = selected ? selected_{index}(geometry, bra, ket) : pair_{index}(geometry, bra, ket);
  output[0] = result.value;
  for (int center = 0; center < 4; ++center)
    for (int axis = 0; axis < 3; ++axis) output[1 + 3 * center + axis] = result.center[center][axis];
}}
"""
    translation_unit = folder / "probe.cpp"
    translation_unit.write_text(source)
    library = native_cxx.build_shared(
        [translation_unit],
        folder / "probe.so",
        compile_args=("-std=c++17", "-O2"),
        compile_timeout=180,
    )
    loaded = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    for index in range(len(kernels)):
        function = getattr(loaded, f"probe_{index}")
        function.argtypes = (array, array, array, array, ctypes.c_bool)
        function.restype = None
    return loaded


@pytest.mark.parametrize("index", range(len(CLASSES)))
@pytest.mark.parametrize(
    "case", ("regular", "pair_coincident", "all_coincident", "zero")
)
def test_compiled_pair_abi_matches_libcint(
    compiled_pairs: Any, index: int, case: str
) -> None:
    kernel = _kernel(CLASSES[index])
    centers, pairs = _fixture(kernel, case)
    variables = primitive_variables(EXPONENTS, centers, 13)
    names = (
        "inverse_two_p",
        "inverse_two_q",
        "rho",
        *(f"difference_{axis}" for axis in "xyz"),
        *(f"{prefix}_{axis}" for prefix in ("pa", "pb", "qc", "qd") for axis in "xyz"),
        *(
            f"{prefix}_product_scale"
            for prefix in ("first", "second", "third", "fourth")
        ),
        *(
            f"decay_{prefix}_{axis}"
            for prefix in ("first", "second", "third", "fourth")
            for axis in "xyz"
        ),
        "prefactor",
        *(f"boys_{order}" for order in range(14)),
    )
    inputs = np.array([variables[name] for name in names])
    value, gradient = _libcint(CLASSES[index], case)
    weights = np.outer(*pairs).ravel()
    expected = np.r_[weights @ value, gradient @ weights]
    function = getattr(compiled_pairs, f"probe_{index}")
    for selected in (False, True):
        output = np.empty(13)
        function(inputs, *pairs, output, selected)
        if selected:
            selected_expected = np.zeros(13)
            selected_expected[1:7] = expected[1:].reshape(4, 3)[[3, 1]].ravel()
            np.testing.assert_allclose(
                output, selected_expected, rtol=4e-11, atol=4e-12
            )
        else:
            np.testing.assert_allclose(output, expected, rtol=4e-11, atol=4e-12)

"""Execute the generated through-f CPU owner against independent libcint jets."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.integral.range_exchange_cpu import (
    emit_full_range_primitive_derivative_cpu,
)
from generativeqc_compiler.integral.shell_spec import cartesian_components

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
CENTERS = np.array(
    [[0.1, -0.2, -0.7], [-0.3, 0.15, 0.7], [0.4, 0.6, -0.2], [-0.5, 0.3, 0.1]]
)


@pytest.fixture(scope="module")
def primitive_derivative(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> ctypes._CFuncPtr:
    """Compile the production emitter with the mandatory verified cache launcher."""
    directory = tmp_path_factory.mktemp("cpu-full-range-primitive")
    source = directory / "primitive.cpp"
    source.write_text(emit_full_range_primitive_derivative_cpu())
    binary = required_native_cxx.build_shared(
        [source],
        directory / "primitive.so",
        compile_args=(
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
        compile_timeout=240,
    )
    library = ctypes.CDLL(str(binary))
    evaluate = library.generativeqc_full_range_primitive_derivative_cpu
    evaluate.argtypes = (
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_double),
    )
    evaluate.restype = ctypes.c_int
    return evaluate


def _evaluate(
    function: ctypes._CFuncPtr,
    exponents: np.ndarray,
    centers: np.ndarray,
    labels: tuple[str, ...],
) -> np.ndarray:
    powers = np.array(
        [[label.count(axis) for axis in "xyz"] for label in labels], dtype=np.uint32
    )
    result = np.full((4, 3), 123.0)
    status = function(
        exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        centers.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        powers.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert status == 0
    return result


def _oracle(
    exponents: np.ndarray, centers: np.ndarray, labels: tuple[str, ...]
) -> tuple:
    """Normalize Cartesian libcint functions independently through self-overlap."""
    gto = pytest.importorskip("pyscf.gto")
    atoms = [(f"H{slot}", position) for slot, position in enumerate(centers)]
    basis = {
        symbol: [[len(label), [float(exponent), 1.0]]]
        for (symbol, _), label, exponent in zip(atoms, labels, exponents, strict=True)
    }
    molecule = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=True, verbose=0)
    offsets = molecule.ao_loc_nr()
    indices = tuple(cartesian_components(len(label)).index(label) for label in labels)
    overlap = molecule.intor("int1e_ovlp_cart")
    oracle_scale = math.sqrt(
        math.prod(
            overlap[offset + index, offset + index]
            for offset, index in zip(offsets[:-1], indices, strict=True)
        )
    )
    native_scale = 1.0
    for label, exponent in zip(labels, exponents, strict=True):
        denominator = math.prod(
            math.prod(range(1, 2 * label.count(axis), 2)) for axis in "xyz"
        )
        native_scale *= (2 * exponent / math.pi) ** 0.75 * math.sqrt(
            (4 * exponent) ** len(label) / denominator
        )
    expected = np.empty((4, 3))
    for slot, order in enumerate(
        ((0, 1, 2, 3), (1, 0, 2, 3), (2, 3, 0, 1), (3, 2, 0, 1))
    ):
        derivative = molecule.intor_by_shell("int2e_ip1_cart", order)
        expected[slot] = -derivative[
            (slice(None), *(indices[index] for index in order))
        ]
    return molecule, indices, expected / oracle_scale, native_scale, oracle_scale


@pytest.mark.parametrize(
    "labels",
    (
        ("", "", "", ""),
        ("xxx", "", "", ""),
        ("xxy", "xz", "y", "z"),
        ("xyz", "yyy", "xzz", "xxx"),
        ("zzz", "yyy", "xxx", "xyz"),
    ),
)
@pytest.mark.parametrize("exponents", ((0.8, 0.6, 1.1, 0.4), (20.0, 0.04, 5.0, 1.2)))
@pytest.mark.parametrize("geometry", ("separated", "coincident-pairs", "coincident"))
def test_generated_primitive_matches_independent_analytic_derivatives(
    primitive_derivative: ctypes._CFuncPtr,
    labels: tuple[str, ...],
    exponents: tuple[float, ...],
    geometry: str,
) -> None:
    """Cover every slot, f/f high order, cancellation and tight/diffuse products."""
    centers = CENTERS.copy()
    if geometry == "coincident-pairs":
        centers[1], centers[3] = centers[0], centers[2]
    elif geometry == "coincident":
        centers[:] = centers[0]
    alpha = np.array(exponents)
    _, _, expected, scale, _ = _oracle(alpha, centers, labels)
    actual = _evaluate(primitive_derivative, alpha, centers, labels) * scale
    np.testing.assert_allclose(actual, expected, atol=5e-11, rtol=5e-10)
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=5e-12, rtol=0)


def test_generated_f_primitive_multistep_independent_value_differences(
    primitive_derivative: ctypes._CFuncPtr,
) -> None:
    """Check positive nuclear signs using fresh independent integrals, not jets."""
    labels = ("xxy", "xz", "y", "z")
    alpha = np.array([0.8, 0.6, 1.1, 0.4])
    molecule, indices, _, scale, oracle_scale = _oracle(alpha, CENTERS, labels)
    actual = _evaluate(primitive_derivative, alpha, CENTERS, labels) * scale
    for step in (1e-3, 3e-4, 1e-4):
        independent = np.empty_like(actual)
        for slot in range(4):
            for axis in range(3):
                values = []
                for sign in (-1, 1):
                    centers = CENTERS.copy()
                    centers[slot, axis] += sign * step
                    moved = molecule.set_geom_(centers, unit="Bohr", inplace=False)
                    values.append(
                        moved.intor_by_shell("int2e_cart", (0, 1, 2, 3))[indices]
                    )
                independent[slot, axis] = (values[1] - values[0]) / (
                    2 * step * oracle_scale
                )
        np.testing.assert_allclose(actual, independent, atol=2e-7, rtol=0)


@pytest.mark.parametrize(
    "invalid", ("zero-exponent", "nan-center", "g-axis", "g-total", "overflow")
)
def test_primitive_failure_does_not_publish_partial_results(
    primitive_derivative: ctypes._CFuncPtr, invalid: str
) -> None:
    """Malformed metadata and numerical failure preserve caller-owned outputs."""
    alpha = np.ones(4)
    centers = CENTERS.copy()
    powers = np.zeros((4, 3), dtype=np.uint32)
    expected_status = 1
    if invalid == "zero-exponent":
        alpha[0] = 0
    elif invalid == "nan-center":
        centers[1, 0] = np.nan
    elif invalid == "g-axis":
        powers[0, 0] = 4
    elif invalid == "g-total":
        powers[2, :2] = 2
    else:
        alpha[:] = np.finfo(float).max
        expected_status = 2
    output = np.full(12, 123.0)
    status = primitive_derivative(
        alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        centers.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        powers.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert status == expected_status
    np.testing.assert_array_equal(output, 123.0)

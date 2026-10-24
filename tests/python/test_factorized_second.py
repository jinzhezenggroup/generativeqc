"""Independent physical f/g gates for the reusable compiler factor schedule."""

from __future__ import annotations

import ctypes
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.integral.factorized_second import emit_factorized_second
from generativeqc_compiler.integral.shell_spec import cartesian_components

from tools.generativeqc_validation.second_derivatives import _gaussian_squared_norm

ROOT = Path(__file__).resolve().parents[2]
POSITIONS = np.array(
    (
        (0.13, -0.31, 0.24),
        (-0.43, 0.27, 0.51),
        (0.68, -0.14, -0.22),
        (-0.21, 0.48, -0.63),
    )
)
EXPONENTS = np.array((0.6, 0.8, 1.1, 0.9))
DIRECTION = np.array(
    ((0.03, -0.05, 0.02), (-0.07, 0.01, 0.04), (0.02, 0.03, -0.06), (0.01, -0.02, 0.05))
)


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    """Build once through the verified compiler-cache adapter, without a GPU."""
    pytest.importorskip("pyscf")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    directory = tmp_path_factory.mktemp("factorized-second")
    source = directory / "factorized.cpp"
    source.write_text(
        emit_factorized_second()
        + r"""
extern "C" bool factorized(unsigned family, unsigned radial, const unsigned* angular,
    unsigned component, const double* record, const double* direction, double* output) {
  return generativeqc::scf::factorized_second::primitive(family,radial,angular,component,record,direction,output);
}
"""
    )
    output = directory / "factorized.so"
    result = CppCompilerAdapter(Path(compiler), compile_timeout=600).compile_shared(
        source, output, includes=(ROOT / "src",), options=("-ffp-contract=off",)
    )
    assert result.returncode == 0, result.stderr
    loaded = ctypes.CDLL(str(output))
    loaded.factorized.restype = ctypes.c_bool
    loaded.factorized.argtypes = (
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_uint,
        *(ctypes.POINTER(ctypes.c_double),) * 3,
    )
    return loaded


def evaluate(
    library: ctypes.CDLL,
    family: int,
    angular: tuple,
    component: int,
    radial: int = 0,
    *,
    exponents: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    direction: np.ndarray | None = None,
) -> tuple[bool, np.ndarray]:
    """Retain the physical record and independent-center packing of the ABI."""
    count = len(angular)
    orders = np.array(angular, dtype=np.uint32)
    record = np.r_[
        EXPONENTS[:count] if exponents is None else exponents,
        (POSITIONS[:count] if positions is None else positions).ravel(),
        1.0,
    ]
    motion = np.ascontiguousarray(DIRECTION[:count] if direction is None else direction)
    output = np.full(1 + 6 * count, np.nan)
    valid = library.factorized(
        family,
        radial,
        orders.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        component,
        *(
            array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            for array in (record, motion, output)
        ),
    )
    return valid, output


def reference(
    family: int, angular: tuple, positions: np.ndarray, radial: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Use independent libcint integrals/gradients, not generated factor graphs."""
    from pyscf import gto

    count = len(angular)
    names = ("H", "He", "Li", "Be")[:count]
    molecule = gto.M(
        atom=[
            (f"ghost-{name}", position)
            for name, position in zip(names, positions, strict=True)
        ],
        basis={
            name: [[order, [exponent, 1.0]]]
            for name, order, exponent in zip(names, angular, EXPONENTS, strict=False)
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    labels = tuple(cartesian_components(order) for order in angular)
    shape = tuple(map(len, labels))
    norms = [
        _gaussian_squared_norm(exponent, component)
        for exponent, components in zip(EXPONENTS, labels, strict=False)
        for component in components
    ]
    scales = np.split(
        np.sqrt(molecule.intor("int1e_ovlp_cart").diagonal() / norms),
        np.cumsum(shape)[:-1],
    )
    normalization = np.ones(shape)
    for axis, scale in enumerate(scales):
        dimensions = [1] * count
        dimensions[axis] = len(scale)
        normalization *= scale.reshape(dimensions)
    with (
        molecule.with_range_coulomb(0.3 if radial == 1 else -0.3)
        if radial
        else nullcontext()
    ):
        operator = ("int2e", "int3c2e", "int2c2e")[family]
        value = (
            molecule.intor_by_shell(operator + "_cart", tuple(range(count)))
            / normalization
        )
        if family == 0:
            permutations = ((0, 1, 2, 3), (1, 0, 2, 3), (2, 3, 0, 1), (3, 2, 0, 1))
            gradients = [
                -molecule.intor_by_shell("int2e_ip1_cart", order, comp=3).transpose(
                    0, *(1 + order.index(axis) for axis in range(4))
                )
                for order in permutations
            ]
        else:
            first = -molecule.intor_by_shell(
                operator + "_ip1_cart", tuple(range(count)), comp=3
            )
            if count == 2:
                second = -molecule.intor_by_shell(
                    "int2c2e_ip1_cart", (1, 0), comp=3
                ).transpose(0, 2, 1)
                gradients = [first, second]
            else:
                second = -molecule.intor_by_shell(
                    "int3c2e_ip1_cart", (1, 0, 2), comp=3
                ).transpose(0, 2, 1, 3)
                third = -molecule.intor_by_shell("int3c2e_ip2_cart", (0, 1, 2), comp=3)
                gradients = [first, second, third]
    return value.ravel(), (np.asarray(gradients) / normalization).reshape(3 * count, -1)


@pytest.mark.parametrize(
    "family,angular,radial",
    (
        (0, (0, 0, 0, 0), 0),
        (0, (1, 0, 1, 0), 0),
        (0, (2, 1, 0, 1), 0),
        (0, (3, 3, 3, 3), 0),
        (0, (3, 3, 3, 3), 1),
        (0, (3, 3, 3, 3), 2),
        (1, (0, 0, 0), 0),
        (1, (2, 1, 3), 0),
        (1, (3, 3, 4), 0),
        (2, (0, 0), 0),
        (2, (3, 4), 0),
        (2, (4, 4), 0),
    ),
)
def test_factorized_physical_value_gradient_jvp(
    library: ctypes.CDLL, family: int, angular: tuple, radial: int
) -> None:
    count = len(angular)
    value, gradient = reference(family, angular, POSITIONS[:count], radial)
    step = 2e-4
    displaced = [
        reference(
            family, angular, POSITIONS[:count] + sign * step * DIRECTION[:count], radial
        )[1]
        for sign in (-1, 1)
    ]
    coarse = (displaced[1] - displaced[0]) / (2 * step)
    displaced = [
        reference(
            family,
            angular,
            POSITIONS[:count] + sign * step / 2 * DIRECTION[:count],
            radial,
        )[1]
        for sign in (-1, 1)
    ]
    fine = (displaced[1] - displaced[0]) / step
    jvp = (4 * fine - coarse) / 3
    size = value.size
    for component in sorted({0, 1 if size > 1 else 0, size // 3, size // 2, size - 1}):
        valid, actual = evaluate(library, family, angular, component, radial)
        assert valid
        expected = np.r_[value[component], gradient[:, component], jvp[:, component]]
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=2e-9)
        np.testing.assert_allclose(
            actual[1 : 1 + 3 * count].reshape(count, 3).sum(axis=0), 0, atol=3e-12
        )
        np.testing.assert_allclose(
            actual[1 + 3 * count :].reshape(count, 3).sum(axis=0), 0, atol=3e-12
        )


@pytest.mark.parametrize(
    "family,angular", ((0, (3, 2, 1, 0)), (1, (3, 2, 4)), (2, (4, 3)))
)
def test_factorized_translation_and_real_exponents(
    library: ctypes.CDLL, family: int, angular: tuple
) -> None:
    count = len(angular)
    valid, actual = evaluate(library, family, angular, 0, direction=np.ones((count, 3)))
    assert valid
    np.testing.assert_allclose(actual[1 + 3 * count :], 0, atol=2e-13)
    for exponent in (0, -0.2, np.nan, np.inf):
        exponents = EXPONENTS[:count].copy()
        exponents[-1] = exponent
        valid, _ = evaluate(library, family, angular, 0, exponents=exponents)
        assert not valid


def test_factorized_emission_is_finite_and_host_only() -> None:
    source = emit_factorized_second()
    assert source == emit_factorized_second()
    assert len(source) < 16 * 1024**2
    assert "PySCF" not in source and "cupy" not in source
    with pytest.raises(ValueError, match="backend"):
        emit_factorized_second(backend="python")

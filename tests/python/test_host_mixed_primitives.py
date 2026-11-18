"""Executable independent-oracle gates for bounded native S/T/V mixed jets."""

from __future__ import annotations

import ctypes as ct
import os
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.shell_spec import cartesian_components

from tools.generativeqc_validation.second_derivatives import (
    _gaussian_squared_norm,
    libcint_one_electron_hessian,
)

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
UNSIGNED_POINTER = ct.POINTER(ct.c_uint)
CENTERS = np.array([[0.125, -0.25, 0.375], [-0.5, 0.25, 0.625], [0.75, -0.125, -0.25]])
LEFT = np.array([[0.25, -0.125, 0.375], [-0.25, 0.5, 0.125], [0.125, 0.25, -0.5]])
RIGHT = np.array([[-0.375, 0.25, 0.125], [0.5, -0.25, 0.375], [-0.125, 0.5, 0.25]])


def _pointer(values: np.ndarray) -> Any:
    return values.ctypes.data_as(
        UNSIGNED_POINTER if values.dtype == np.uint32 else POINTER
    )


@pytest.fixture(scope="module")
def native(tmp_path_factory: Any, required_native_cxx: Any) -> Any:
    """Compile the actual finite generator with strict IEEE host options."""
    directory = tmp_path_factory.mktemp("host-mixed-primitives")
    source = directory / "one_electron_mixed.cpp"
    script = (
        "import importlib.abc, runpy, sys\n"
        "class Guard(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname.split('.')[0] in {'generativeqc', 'pyscf', 'cupy', 'torch'}:\n"
        "   raise RuntimeError('forbidden build-time dependency: ' + fullname)\n"
        "sys.meta_path.insert(0, Guard())\n"
        f"sys.argv = ['generator', '--output', {str(source)!r}]\n"
        f"runpy.run_path({str(ROOT / 'tools/generate_one_electron_mixed.py')!r}, run_name='__main__')\n"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    arguments = [
        "-std=c++20",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fPIC",
        "-I" + str(ROOT / "src"),
    ]
    implementation = directory / "implementation.o"
    required_native_cxx.compile_object(
        source, implementation, args=arguments, timeout=600
    )
    probe = ROOT / "tests/native/host_mixed_primitives_probe.cpp"
    wrapper = directory / "probe.o"
    required_native_cxx.compile_object(probe, wrapper, args=arguments)
    library = directory / "mixed.so"
    required_native_cxx.link([implementation, wrapper], library, args=["-shared"])
    standalone = directory / "standalone.o"
    required_native_cxx.compile_object(
        probe, standalone, args=[*arguments, "-DGENERATIVEQC_HOST_MIXED_STANDALONE"]
    )
    executable = directory / "standalone"
    required_native_cxx.link([implementation, standalone], executable)
    loaded = ct.CDLL(str(library))
    for name in ("host_mixed_st", "host_mixed_v"):
        function = getattr(loaded, name)
        function.argtypes = [
            POINTER,
            POINTER,
            UNSIGNED_POINTER,
            POINTER,
            POINTER,
            POINTER,
        ]
        function.restype = ct.c_bool
    return loaded, executable


def _call(
    native: Any,
    family: str,
    angular: tuple[int, int],
    exponents: Any,
    centers: np.ndarray = CENTERS,
    left: np.ndarray = LEFT,
    right: np.ndarray = RIGHT,
) -> np.ndarray:
    """Run all Cartesian components, retaining only four scalars per call."""
    labels = tuple(tuple(cartesian_components(order)) for order in angular)
    exponents = np.ascontiguousarray(exponents, dtype=float)
    count = 3 if family == "nuclear_attraction" else 2
    positions, motion_left, motion_right = (
        np.ascontiguousarray(values[:count], dtype=float)
        for values in (centers, left, right)
    )
    output = np.empty(4 if count == 3 else 8)
    result = np.empty((4, len(labels[0]), len(labels[1])))
    function = native[0].host_mixed_v if count == 3 else native[0].host_mixed_st
    for first, second in product(range(len(labels[0])), range(len(labels[1]))):
        powers = np.array(
            [
                label.count(axis)
                for label in (labels[0][first], labels[1][second])
                for axis in "xyz"
            ],
            dtype=np.uint32,
        )
        assert function(
            *map(
                _pointer,
                (exponents, positions, powers, motion_left, motion_right, output),
            )
        )
        offset = 4 if family == "kinetic" else 0
        result[:, first, second] = output[offset : offset + 4]
    return result


def _oracle(
    family: str,
    angular: tuple[int, int],
    exponents: Any,
    centers: np.ndarray,
    left: np.ndarray = LEFT,
    right: np.ndarray = RIGHT,
) -> np.ndarray:
    """Use independent analytic libcint primal, bra gradients and Hessians."""
    gto = pytest.importorskip("pyscf.gto")
    molecule = gto.M(
        atom=[("ghost-H", centers[0]), ("ghost-He", centers[1])],
        basis={
            name: [[order, [exponent, 1]]]
            for name, order, exponent in zip(
                ("H", "He"), angular, exponents, strict=True
            )
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    labels = tuple(tuple(cartesian_components(order)) for order in angular)
    norms = np.array(
        [
            _gaussian_squared_norm(exponent, label)
            for exponent, components in zip(exponents, labels, strict=True)
            for label in components
        ]
    )
    scale = np.sqrt(molecule.intor("int1e_ovlp_cart").diagonal() / norms)
    size = len(labels[0])
    normalization = scale[:size, None] * scale[None, size:]
    operator, factor = {
        "overlap": ("ovlp", 1),
        "kinetic": ("kin", 1),
        "nuclear_attraction": ("rinv", -1),
    }[family]
    count = 3 if family == "nuclear_attraction" else 2
    with molecule.with_rinv_origin(centers[2] if count == 3 else (0, 0, 0)):
        value = (
            factor
            * molecule.intor_by_shell(f"int1e_{operator}_cart", (0, 1))
            / normalization
        )
        first = -factor * molecule.intor_by_shell(
            f"int1e_ip{operator}_cart", (0, 1), comp=3
        )
        second = -factor * molecule.intor_by_shell(
            f"int1e_ip{operator}_cart", (1, 0), comp=3
        ).transpose(0, 2, 1)
    gradients = (
        np.array([first, second] + ([-first - second] if count == 3 else []))
        / normalization
    )
    hessian = libcint_one_electron_hessian(family, angular, exponents, centers)
    return np.array(
        [
            value,
            np.einsum("ax,axij->ij", left[:count], gradients),
            np.einsum("ax,axij->ij", right[:count], gradients),
            np.einsum("ax,axbyij,by->ij", left[:count], hessian, right[:count]),
        ]
    )


@pytest.mark.parametrize("angular", tuple(product(range(4), repeat=2)))
@pytest.mark.parametrize("coincident", [False, True])
def test_all_components_against_independent_analytic_libcint(
    native: Any,
    angular: tuple[int, int],
    coincident: bool,
) -> None:
    """All 400 component pairs through f have independent four-field gates."""
    centers = CENTERS.copy()
    if coincident:
        centers[:] = centers[0]
    for family in ("overlap", "kinetic", "nuclear_attraction"):
        expected = _oracle(family, angular, (0.6, 0.8), centers)
        actual = _call(native, family, angular, (0.6, 0.8), centers)
        np.testing.assert_allclose(actual, expected, atol=4e-11, rtol=3e-11)


@pytest.mark.parametrize("angular", [(0, 0), (1, 3), (3, 2), (3, 3)])
@pytest.mark.parametrize("exponents", [(0.002, 0.007), (35.0, 50.0), (0.02, 35.0)])
def test_tight_diffuse_and_asymmetric_products(
    native: Any,
    angular: tuple[int, int],
    exponents: tuple[float, float],
) -> None:
    """Scale-aware independent gates cover both cancellation and wide magnitudes."""
    centers = CENTERS / np.sqrt(sum(exponents))
    for family in ("overlap", "kinetic", "nuclear_attraction"):
        expected = _oracle(family, angular, exponents, centers)
        actual = _call(native, family, angular, exponents, centers)
        scale = np.maximum(1.0, np.max(np.abs(expected), axis=(1, 2)))
        np.testing.assert_allclose(
            actual / scale[:, None, None],
            expected / scale[:, None, None],
            atol=4e-11,
            rtol=3e-11,
        )


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction"])
def test_symmetry_bilinearity_origin_and_translation(native: Any, family: str) -> None:
    """Independent nuclear slots commute and retain exact rigid-motion zeros."""
    angular, exponents = (3, 2), (0.6, 0.8)
    actual = _call(native, family, angular, exponents)
    reverse = _call(native, family, angular, exponents, left=RIGHT, right=LEFT)
    np.testing.assert_allclose(actual, reverse[[0, 2, 1, 3]], atol=5e-12, rtol=5e-12)
    combined = _call(native, family, angular, exponents, left=2 * LEFT - RIGHT)
    np.testing.assert_allclose(
        combined[3],
        2 * actual[3] - _call(native, family, angular, exponents, left=RIGHT)[3],
        atol=5e-12,
        rtol=5e-12,
    )
    translated = _call(
        native, family, angular, exponents, centers=CENTERS + [32, -64, 128]
    )
    np.testing.assert_array_equal(actual, translated)
    for moving_left in (True, False):
        motion = np.broadcast_to([0.25, -0.5, 0.125], (3, 3))
        arguments = {"left": motion} if moving_left else {"right": motion}
        rigid = _call(native, family, angular, exponents, **arguments)
        np.testing.assert_array_equal(rigid[[1 if moving_left else 2, 3]], 0)
    permutation = [1, 0, 2]
    exchanged = _call(
        native,
        family,
        angular[::-1],
        exponents[::-1],
        CENTERS[permutation],
        LEFT[permutation],
        RIGHT[permutation],
    )
    np.testing.assert_allclose(
        actual, exchanged.transpose(0, 2, 1), atol=5e-12, rtol=5e-12
    )


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction"])
def test_mixed_against_multistep_independent_gradient_difference(
    native: Any, family: str
) -> None:
    """Analytic mixed jets match converging differences of libcint gradients."""
    actual = _call(native, family, (3, 2), (0.6, 0.8))[3]
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        plus = _oracle(family, (3, 2), (0.6, 0.8), CENTERS + step * RIGHT)[1]
        minus = _oracle(family, (3, 2), (0.6, 0.8), CENTERS - step * RIGHT)[1]
        errors.append(float(np.max(np.abs(actual - (plus - minus) / (2 * step)))))
    scale = max(1.0, float(np.max(np.abs(actual))))
    assert errors[-1] <= 2e-7 * scale, errors
    assert errors[-1] <= max(5e-11 * scale, 0.03 * errors[0]), errors


@pytest.mark.parametrize("attraction", [False, True])
def test_refusal_publication_aliasing_and_recovery(
    native: Any, attraction: bool
) -> None:
    """Invalid arithmetic cannot partly publish, including aliased inputs."""
    function = native[0].host_mixed_v if attraction else native[0].host_mixed_st
    width = 4 if attraction else 8
    inputs = [
        np.array([0.6, 0.8]),
        CENTERS.copy(),
        np.array([2, 0, 1, 1, 1, 0], dtype=np.uint32),
        LEFT.copy(),
        RIGHT.copy(),
    ]
    expected = np.full(width, 17.0)
    assert function(*map(_pointer, inputs), _pointer(expected))
    for argument in range(6):
        output = np.full(width, 17.0)
        pointers = [*map(_pointer, inputs), _pointer(output)]
        pointers[argument] = None
        assert not function(*pointers)
        np.testing.assert_array_equal(output, 17.0)
    for argument, value in [
        (0, 0.0),
        (0, -1.0),
        (0, np.inf),
        (0, np.nan),
        (0, np.finfo(float).max),
        (1, np.inf),
        (1, np.nan),
        (3, np.inf),
        (4, np.nan),
        (2, 4),
        (2, np.iinfo(np.uint32).max),
    ]:
        changed = [values.copy() for values in inputs]
        changed[argument].flat[0] = value
        if argument == 0 and value == np.finfo(float).max:
            changed[0][:] = value
        output = np.full(width, 17.0)
        assert not function(*map(_pointer, changed), _pointer(output))
        np.testing.assert_array_equal(output, 17.0)
    changed = [values.copy() for values in inputs]
    changed[3] *= 1e200
    changed[4] *= 1e200
    output = np.full(width, 17.0)
    assert not function(*map(_pointer, changed), _pointer(output))
    np.testing.assert_array_equal(output, 17.0)
    for argument in (0, 1, 3, 4):
        changed = [values.copy() for values in inputs]
        buffer = np.zeros(max(width, changed[argument].size))
        buffer[: changed[argument].size] = changed[argument].ravel()
        changed[argument] = buffer
        assert function(*map(_pointer, changed), _pointer(buffer))
        np.testing.assert_array_equal(buffer[:width], expected)
    recovered = np.zeros(width)
    assert function(*map(_pointer, inputs), _pointer(recovered))
    np.testing.assert_array_equal(recovered, expected)


def test_python_free_native_execution(native: Any) -> None:
    """The linked executable evaluates nonzero mixed f-shell work without Python."""
    subprocess.run(
        [str(native[1])],
        env={**os.environ, "PATH": "", "PYTHONPATH": ""},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

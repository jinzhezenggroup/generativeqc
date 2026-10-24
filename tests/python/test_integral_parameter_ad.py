"""Independent high-angular-momentum tests of emitted primitive parameter AD."""

from __future__ import annotations

import ctypes as ct
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc_compiler.common.compiler_work import (
    compiler_work_budget,
)
from generativeqc_compiler.integral.one_electron_cpu import _hostify
from generativeqc_compiler.integral.one_electron_cuda import _emit_boys_support
from generativeqc_compiler.integral.parameter_ad import _emit_kernel, parameter_kernel
from generativeqc_compiler.integral.shell_spec import cartesian_components

if TYPE_CHECKING:
    from pathlib import Path


def _oracle(family: str, components: tuple[str, ...], inputs: np.ndarray) -> float:
    from pyscf import gto

    count = len(components)
    alpha = inputs[:count]
    centers = inputs[count:].reshape(-1, 3)
    labels = [f"H{i}" for i in range(count)]
    molecule = gto.M(
        atom=list(zip(labels, centers[:count], strict=True)),
        basis={
            label: [[len(component), [float(exponent), 1.0]]]
            for label, component, exponent in zip(
                labels, components, alpha, strict=True
            )
        },
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    indices = tuple(
        cartesian_components(len(component)).index(component)
        for component in components
    )
    if family == "eri":
        block = molecule.intor_by_shell("int2e_cart", tuple(range(4)))
    elif family == "nuclear_attraction":
        with molecule.with_rinv_origin(centers[2]):
            block = -molecule.intor_by_shell("int1e_rinv_cart", (0, 1))
    else:
        operator = {"overlap": "int1e_ovlp_cart", "kinetic": "int1e_kin_cart"}[family]
        block = molecule.intor_by_shell(operator, (0, 1))
    normalization = 1.0
    for shell, (component, exponent, index) in enumerate(
        zip(components, alpha, indices, strict=True)
    ):
        # Recover libcint's own radial/component factor from its self overlap;
        # compare the unnormalized primitive, not two normalization conventions.
        raw_norm = (math.pi / (2 * exponent)) ** 1.5 * math.prod(
            math.prod(range(1, 2 * component.count(axis), 2))
            / (4 * exponent) ** component.count(axis)
            for axis in "xyz"
        )
        diagonal = molecule.intor_by_shell("int1e_ovlp_cart", (shell, shell))[
            index, index
        ]
        normalization *= math.sqrt(diagonal / raw_norm)
    return float(block[indices] / normalization)


@pytest.mark.parametrize(
    "family,components",
    [
        ("overlap", ("xx", "xyz")),
        ("kinetic", ("xxxx", "zzzz")),
        ("nuclear_attraction", ("xxxx", "xxzz")),
        ("eri", ("xx", "y", "zz", "x")),
        ("eri", ("xxxx", "xxxx", "", "")),
    ],
)
def test_emitted_value_and_all_parameter_partials_match_libcint(
    family: str, components: tuple[str, ...], tmp_path: Path, native_cxx: Any
) -> None:
    with compiler_work_budget(5_000_000):
        kernel = parameter_kernel(family, components)
    source = tmp_path / "primitive.cpp"
    library_path = tmp_path / "primitive.so"
    source.write_text(
        "#include <cmath>\n"
        + _hostify(_emit_boys_support(17))
        + _emit_kernel("primitive", kernel)
        + '\nextern "C" void evaluate(const double* x, double* y) { primitive(x,y,true); }\n'
    )
    native_cxx.build_shared(
        (source,),
        library_path,
        compile_args=("-O1", "-ffp-contract=off"),
        compile_timeout=180,
    )
    library = ct.CDLL(str(library_path))
    evaluate = library.evaluate
    pointer = ct.POINTER(ct.c_double)
    evaluate.argtypes = [pointer, pointer]
    count = len(components)
    centers = 3 if family == "nuclear_attraction" else count
    rng = np.random.default_rng(614)
    inputs = np.concatenate(
        (np.linspace(0.65, 1.25, count), rng.normal(scale=0.4, size=3 * centers))
    )
    actual = np.empty(1 + inputs.size)
    evaluate(inputs.ctypes.data_as(pointer), actual.ctypes.data_as(pointer))
    np.testing.assert_allclose(
        actual[0], _oracle(family, components, inputs), atol=3e-10, rtol=3e-11
    )
    # Separate exponent and center directions catch a missing Boys/exponent
    # chain rule even if translation invariance alone happens to pass.
    for begin, end in ((0, count), (count, inputs.size)):
        direction = np.zeros_like(inputs)
        direction[begin:end] = rng.normal(size=end - begin)
        direction /= np.linalg.norm(direction)
        step = 2e-5
        numeric = (
            _oracle(family, components, inputs + step * direction)
            - _oracle(family, components, inputs - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(
            actual[1:] @ direction, numeric, atol=3e-7, rtol=3e-7
        )


def test_high_angular_compilation_obeys_symbolic_budget() -> None:
    """Mixed high-angular DAGs must fail finitely before native compilation."""
    from generativeqc_compiler.common.compiler_work import CompilerWorkLimit

    with (
        pytest.raises(CompilerWorkLimit, match="budget exhausted"),
        compiler_work_budget(100_000),
    ):
        parameter_kernel("eri", ("xxxx", "yyyy", "zzzz", "xxzz"))


@pytest.mark.parametrize(
    "family,components",
    [
        ("overlap", ("xx", "xyz")),
        ("kinetic", ("xxxx", "zzzz")),
        ("nuclear_attraction", ("xxxx", "xxzz")),
        ("eri", ("xx", "y", "zz", "x")),
        ("eri", ("xxxx", "yyyy", "zzzz", "xxzz")),
    ],
)
def test_tape_second_derivative_matches_independent_integrals(
    family: str, components: tuple[str, ...], tmp_path: Path, native_cxx: Any
) -> None:
    """Check mixed radial/center Hessians, including all-g ERIs and Boys F18."""
    from generativeqc_compiler.common.paths import asset_path
    from generativeqc_compiler.integral.parameter_ad import _directional_arithmetic
    from generativeqc_compiler.integral.parameter_tape import (
        emit_tape_kernel,
        emit_tape_support,
    )

    with compiler_work_budget(5_000_000):
        kernel = parameter_kernel(family, components, with_partials=False)
        support, codes, _ = emit_tape_support((kernel,))
        emitted = emit_tape_kernel("primitive", kernel, codes)
    size = len(kernel.parameters)
    source = tmp_path / "second.cpp"
    target = tmp_path / "second.so"
    source.write_text(
        "#include <cmath>\n#define GQC_AD_FN static\n"
        + _directional_arithmetic()
        + _hostify(_emit_boys_support(18))
        + support
        + '\n#include "parameter_ad_tape.hpp"\n'
        + emitted
        + f'\nextern "C" void evaluate(const double* x, const double* v, double* y) {{\n'
        f"Directional inputs[{size}], output[{size + 1}], scratch[2*parameter_tape_nodes];\n"
        f"for (int i=0; i<{size}; ++i) inputs[i]={{x[i],v[i]}};\n"
        "primitive(inputs, output, true, scratch);\n"
        f"for (int i=0; i<{size + 1}; ++i) y[i]=output[i].tangent;\n}}\n"
    )
    native_cxx.build_shared(
        (source,),
        target,
        compile_args=(
            "-O1",
            "-ffp-contract=off",
            f"-I{asset_path('src/integrals').resolve()}",
        ),
        compile_timeout=180,
    )
    library = ct.CDLL(str(target))
    evaluate = library.evaluate
    pointer = ct.POINTER(ct.c_double)
    evaluate.argtypes = [pointer, pointer, pointer]
    evaluate.restype = None
    count = len(components)
    centers = 3 if family == "nuclear_attraction" else count
    rng = np.random.default_rng(614)
    inputs = np.concatenate(
        (np.linspace(0.65, 1.25, count), rng.normal(scale=0.4, size=3 * centers))
    )
    u, v = rng.normal(size=(2, size))
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    actual = np.empty(size + 1)
    evaluate(*(a.ctypes.data_as(pointer) for a in (inputs, v, actual)))
    expected = u @ actual[1:]
    errors = []
    for step in (0.002, 0.0005):
        numeric = sum(
            a * b * _oracle(family, components, inputs + step * (a * u + b * v))
            for a in (-1, 1)
            for b in (-1, 1)
        ) / (4 * step**2)
        errors.append(abs(numeric - expected))
    assert errors[-1] < 2e-6 * max(1, abs(expected)), errors
    assert errors[-1] < max(errors[0] * 0.2, 2e-8), errors


def test_extended_boys_orders_match_independent_hypergeometric(
    tmp_path: Path, native_cxx: Any
) -> None:
    """Qualify every moment through F18 on both sides of the recurrence switch."""
    from scipy.special import hyp1f1

    source = tmp_path / "boys.cpp"
    target = tmp_path / "boys.so"
    source.write_text(
        "#include <cmath>\n"
        + _hostify(_emit_boys_support(18))
        + '\nextern "C" void evaluate(double x, double* y) { boys_values<18>(x,y); }\n'
    )
    native_cxx.build_shared(
        (source,), target, compile_args=("-O2", "-ffp-contract=off")
    )
    library = ct.CDLL(str(target))
    evaluate = library.evaluate
    pointer = ct.POINTER(ct.c_double)
    evaluate.argtypes = [ct.c_double, pointer]
    evaluate.restype = None
    orders = np.arange(19)
    actual = np.empty(19)
    for argument in (0.0, 1e-10, 0.3, 10.0, 19.999, 20.0, 20.001, 50.0, 500.0):
        evaluate(argument, actual.ctypes.data_as(pointer))
        expected = hyp1f1(orders + 0.5, orders + 1.5, -argument) / (2 * orders + 1)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)

"""Maple constants retain source precision before target arithmetic is emitted."""

import ctypes
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter
from generativeqc_compiler.xc.libxc_maple import import_maple_source


@pytest.mark.parametrize(
    "coefficient,expected",
    (
        ("1.0000000000000001 - 1", Fraction(1, 10**16)),
        ("1e-400 / 1e-399", Fraction(1, 10)),
        ("1e400 / 1e399", Fraction(10)),
        (".10000000000000001 - .1", Fraction(1, 10**17)),
        ("9007199254740993.0 - 9007199254740992", Fraction(1)),
        ("add((1.0000000000000001-1)*i^2,i=-1..1)", Fraction(2, 10**16)),
    ),
)
def test_decimal_constant_algebra_and_derivatives(
    coefficient: str, expected: Fraction, tmp_path: Path, native_cxx: object
) -> None:
    """Independent rational values qualify both Graph and compiled FP64 outputs.

    Check relative error without an absolute floor: losing the small coefficient
    entirely must not pass merely because the expected energy is small.
    """
    module = import_maple_source(f"a := [{coefficient}]: f := x -> a[1]*x^2:")
    graph = Graph()
    x = graph.variable("x")
    value = module.call(graph, "f", x)
    first = graph.differentiate(value, x)
    roots = (value, first, graph.differentiate(first, x))
    reference = np.array(
        [float(4 * expected), float(4 * expected), float(2 * expected)]
    )
    np.testing.assert_allclose(
        evaluate_array_graph(graph, roots, {"x": 2.0}), reference, rtol=2e-15, atol=0
    )

    emitter = ScalarCEmitter(graph, {"x": "x"})
    emitter.emit(roots)
    source = tmp_path / "decimal.cpp"
    source.write_text(
        "\n".join(
            (
                "#include <cmath>",
                'extern "C" void evaluate(double x, double* result) {',
                *emitter.lines,
                *(
                    f"result[{i}] = {emitter.reference(root)};"
                    for i, root in enumerate(roots)
                ),
                "}",
            )
        )
    )
    library = ctypes.CDLL(
        str(
            native_cxx.build_shared(
                [source], tmp_path / "decimal.so", compile_args=("-O2",)
            )
        )
    )
    library.evaluate.argtypes = (ctypes.c_double, ctypes.POINTER(ctypes.c_double))
    library.evaluate.restype = None
    result = np.empty(3)
    library.evaluate(2.0, result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    np.testing.assert_allclose(result, reference, rtol=2e-15, atol=0)


def test_source_and_external_decimal_bindings_agree() -> None:
    """Literal coefficients and existing exact external bindings use one algebra."""
    modules = (
        import_maple_source("a := 1.0000000000000001: f := x -> (a-1)*x:"),
        import_maple_source("f := x -> (a-1)*x:", bindings={"a": "1.0000000000000001"}),
    )
    for module in modules:
        graph = Graph()
        value = module.call(graph, "f", graph.variable("x"))
        assert evaluate_array_graph(graph, (value,), {"x": 1.0})[0] == 1e-16


def test_importer_bootstraps_without_its_generated_catalog() -> None:
    """Regeneration can load its tools while public resolution still fails closed."""
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import pathlib
import runpy
original = pathlib.Path.read_text
def missing_catalog(path, *args, **kwargs):
    if path.name == 'libxc_bulk_catalog.json':
        raise FileNotFoundError('test: generated catalog unavailable')
    return original(path, *args, **kwargs)
pathlib.Path.read_text = missing_catalog
tool = runpy.run_path('tools/import_libxc_bulk.py', run_name='import_probe')
assert callable(tool['make_catalog'])
try:
    from generativeqc_compiler.xc import FunctionalSpec
except FileNotFoundError:
    pass
else:
    raise AssertionError('public resolution accepted a missing catalog')
pathlib.Path.read_text = original
from generativeqc_compiler.xc import FunctionalSpec
from generativeqc_compiler.xc.spec import FunctionalSpec as canonical
assert FunctionalSpec is canonical
""",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=60,
    )

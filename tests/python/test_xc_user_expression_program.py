"""Independent polynomial oracles for the complete native CPU point/batch API."""

from __future__ import annotations

import copy
import ctypes as ct
import json
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from generativeqc._xc_user_expression import _validate_inputs as validate_inputs
from generativeqc.extensions import method, xc
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.xc.program import validate_features
from generativeqc_compiler.xc.spec import UnsupportedXC, functional
from generativeqc_compiler.xc.user_expression_native import emit_native
from generativeqc_compiler.xc.user_expression_program import (
    MAX_POINTS,
    UserExpressionProgram,
)

FORMULA = "-rho_a**4/7 + rho_b**3/5 + rho_a*rho_b/3 + rho_a/11 + 2"


def oracle(a: float, b: float) -> np.ndarray:
    # Hand derivatives, independently evaluated at 70-digit Decimal precision.
    with localcontext() as context:
        context.prec = 70
        a, b = Decimal(str(a)), Decimal(str(b))
        return np.array(
            list(
                map(
                    float,
                    (
                        -(a**4) / 7 + b**3 / 5 + a * b / 3 + a / 11 + 2,
                        -4 * a**3 / 7 + b / 3 + Decimal(1) / 11,
                        3 * b**2 / 5 + a / 3,
                        -12 * a**2 / 7,
                        Decimal(1) / 3,
                        6 * b / 5,
                    ),
                )
            )
        )


@pytest.fixture(scope="module")
def compiler() -> CppCompilerAdapter:
    executable = shutil.which("c++")
    cache = shutil.which("sccache") or shutil.which("ccache")
    if not executable or not cache:
        if sys.platform == "win32":
            pytest.skip(
                "native CPU expression requires C++ compiler and compiler cache"
            )
        pytest.fail("native CPU expression requires C++ compiler and compiler cache")
    return CppCompilerAdapter(executable)


@pytest.mark.parametrize("order,rows", [(0, 1), (1, 3), (2, 6)])
def test_complete_point_and_batch_decimal_oracle(
    compiler: CppCompilerAdapter, tmp_path: Path, order: int, rows: int
) -> None:
    program = xc.expression_program(xc.expression(FORMULA), order=order)
    native = xc.compile_expression(program, compiler=compiler, cache=tmp_path)
    x = np.array([[0, 1, 0, 0.4, 1e-12, 2], [0, 0, 1, 0.1, 1e-30, 3]])
    expected = np.stack([oracle(a, b)[:rows] for a, b in x.T], axis=1)
    np.testing.assert_allclose(native.evaluate(x), expected, rtol=4e-15, atol=1e-40)
    for i, (a, b) in enumerate(x.T):
        np.testing.assert_array_equal(
            native.point(rho_a=float(a), rho_b=float(b)), native.evaluate(x)[:, i]
        )
    assert native.evaluate(np.empty((2, 0))).shape == (rows, 0)
    assert not native.evaluate(x).flags.writeable
    with pytest.raises(ValueError):
        native.evaluate(x).setflags(write=True)
    np.testing.assert_allclose(
        native.evaluate(x[:, ::2]), expected[:, ::2], rtol=4e-15, atol=1e-40
    )


def test_native_finite_difference_and_negative_control(
    compiler: CppCompilerAdapter, tmp_path: Path
) -> None:
    native = xc.compile_expression(
        xc.expression_program(xc.expression(FORMULA)), compiler=compiler, cache=tmp_path
    )
    a, b, h = 0.4, 0.1, 1e-5
    actual = native.point(rho_a=a, rho_b=b)
    for index in range(2):
        plus, minus = [a, b], [a, b]
        plus[index] += h
        minus[index] -= h
        p = native.point(rho_a=plus[0], rho_b=plus[1])
        m = native.point(rho_a=minus[0], rho_b=minus[1])
        assert (p[0] - m[0]) / (2 * h) == pytest.approx(actual[1 + index], abs=2e-10)
        hessian = np.array([[actual[3], actual[4]], [actual[4], actual[5]]])
        np.testing.assert_allclose(
            (p[1:3] - m[1:3]) / (2 * h), hessian[:, index], atol=1e-9, rtol=1e-8
        )
    # Swapped spin derivatives and per-particle energy are rejected by this oracle.
    assert not np.allclose(actual[1:3], actual[1:3][::-1])
    assert not np.isclose(actual[0], oracle(a, b)[0] / (a + b))


@pytest.mark.parametrize(
    "source,value", [("7", 7), ("0*rho_a + rho_b**0", 1), ("0", 0)]
)
def test_constant_still_validates_both_declared_inputs(
    compiler: CppCompilerAdapter, tmp_path: Path, source: str, value: int
) -> None:
    native = xc.compile_expression(
        xc.expression_program(xc.expression(source)), compiler=compiler, cache=tmp_path
    )
    np.testing.assert_array_equal(
        native.point(rho_a=0, rho_b=0), [value, 0, 0, 0, 0, 0]
    )
    for bad in (-1.0, np.nan, np.inf):
        for index in range(2):
            inputs = np.zeros((2, 1))
            inputs[index, 0] = bad
            with pytest.raises(ValueError):
                native.evaluate(inputs)
            # Bypass Python validation to check the actual native ABI as well.
            result = np.full((6, 1), 123.0)
            status = native._function(
                inputs.ctypes.data_as(ct.POINTER(ct.c_double)),
                2,
                result.ctypes.data_as(ct.POINTER(ct.c_double)),
                6,
                1,
            )
            assert status == -2
            np.testing.assert_array_equal(result, 123.0)


def test_native_near_unity_exact_coefficients(
    compiler: CppCompilerAdapter, tmp_path: Path
) -> None:
    q = "9007199254740993/9007199254740992"
    formula = xc.expression(f"({q})*rho_a**2 + rho_b/3")
    program = xc.expression_program(formula)
    graph, roots = program.scalar_graph()
    from fractions import Fraction

    assert graph.node(roots[3]).payload == 2 * Fraction(
        9007199254740993, 9007199254740992
    )
    native = xc.compile_expression(program, compiler=compiler, cache=tmp_path)
    np.testing.assert_allclose(
        native.point(rho_a=0.5, rho_b=0.3), [0.35, 1, 1 / 3, 2, 0, 0], rtol=1e-15
    )


def test_overflow_does_not_publish_or_poison_previous_result(
    compiler: CppCompilerAdapter, tmp_path: Path
) -> None:
    native = xc.compile_expression(
        xc.expression_program(xc.expression("rho_a**4 + rho_b")),
        compiler=compiler,
        cache=tmp_path,
    )
    previous = native.point(rho_a=1, rho_b=2)
    before = previous.copy()
    with pytest.raises(ArithmeticError, match="no result published"):
        native.evaluate(np.array([[1.0, 1e200], [2.0, 0.0]]))
    np.testing.assert_array_equal(previous, before)
    np.testing.assert_array_equal(native.point(rho_a=1, rho_b=2), before)


def test_native_count_pointer_boundaries_and_maximum_batch(
    compiler: CppCompilerAdapter, tmp_path: Path
) -> None:
    native = xc.compile_expression(
        xc.expression_program(xc.expression("rho_a + 2*rho_b")),
        compiler=compiler,
        cache=tmp_path,
    )
    assert native._function(None, 0, None, 0, 0) == 0
    for counts in [(0, 6, 1), (2, 5, 1), (2, 6, 1), (0, 0, MAX_POINTS + 1)]:
        assert native._function(None, counts[0], None, counts[1], counts[2]) == -1
    x = np.zeros((2, MAX_POINTS))
    x[1] = 1
    values = native.evaluate(x)
    np.testing.assert_array_equal(
        values, np.repeat([[2], [1], [2], [0], [0], [0]], MAX_POINTS, axis=1)
    )
    for bad in [True, 1j, np.float32(1), "1"]:
        with pytest.raises(TypeError):
            native.point(rho_a=bad, rho_b=0)
    for bad in [-1.0, float("nan"), float("inf"), 10**1000]:
        with pytest.raises(ValueError):
            native.point(rho_a=0, rho_b=bad)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=int),
        np.zeros((2, 1), dtype=complex),
        [[0.0, 0.0]],
        np.zeros(2),
        np.zeros((3, 2)),
        np.zeros((2, 1, 1)),
        np.zeros((2, MAX_POINTS + 1)),
    ],
)
def test_input_shape_dtype_and_memory_bounds(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_inputs(bad)


def test_program_identity_detached_graph_and_admission() -> None:
    formula = xc.expression(FORMULA)
    program = xc.expression_program(formula)
    assert program.outputs == ((), (0,), (1,), (0, 0), (0, 1), (1, 1))
    original = program.identity
    graph, _roots = program.scalar_graph()
    graph.nodes.clear()
    payload = program.to_payload()
    payload["nodes"].clear()
    assert program.identity == original == copy.deepcopy(program).identity
    assert (
        xc.expression_program(xc.UserExpression.loads(formula.dumps())).identity
        == original
    )
    assert xc.expression_program(formula, order=1).identity != original
    with pytest.raises(FrozenInstanceError):
        program.order = 0
    with pytest.raises(TypeError):
        method.compose("custom", xc=formula)
    with pytest.raises(UnsupportedXC):
        validate_features(
            functional("LDA_X", spin="polarized"), np.zeros((7, 1)), order=2
        )


@pytest.mark.parametrize("order", [-1, 3, True, 1.0, "2"])
def test_order_rejection(order: int) -> None:
    with pytest.raises(UnsupportedXC):
        UserExpressionProgram(xc.expression("1"), order=order)


def test_budget_rejections(monkeypatch: pytest.MonkeyPatch) -> None:
    import generativeqc_compiler.xc.user_expression_native as native
    import generativeqc_compiler.xc.user_expression_program as programs

    monkeypatch.setattr(programs, "MAX_DERIVATIVE_GRAPH", 1)
    with pytest.raises(ValueError, match="node budget"):
        xc.expression_program(xc.expression(FORMULA))
    monkeypatch.undo()
    program = xc.expression_program(xc.expression(FORMULA))
    monkeypatch.setattr(native, "MAX_SOURCE_BYTES", 10)
    with pytest.raises(ValueError, match="source budget"):
        emit_native(program)


def test_source_identity_and_native_cache_replay(
    compiler: CppCompilerAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program = xc.expression_program(xc.expression(FORMULA))
    first = xc.compile_expression(program, compiler=compiler, cache=tmp_path)
    from hashlib import sha256

    source, metadata = emit_native(program)
    assert sha256(source.encode()).hexdigest() == metadata["source_sha256"]
    assert first.artifact.metadata["identity"]["source"] == metadata["source_sha256"]

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("artifact hit attempted compilation")

    monkeypatch.setattr(CppCompilerAdapter, "compile_shared", forbidden)
    second = xc.compile_expression(program, compiler=compiler, cache=tmp_path)
    assert second.artifact.library == first.artifact.library
    np.testing.assert_array_equal(
        second.point(rho_a=0.4, rho_b=0.1), first.point(rho_a=0.4, rho_b=0.1)
    )


def test_generation_never_imports_runtime_or_allocates_native_buffers() -> None:
    script = """
import builtins, ctypes, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "generativeqc" or name.startswith("generativeqc."):
        raise AssertionError("generation imported the public runtime")
    return original(name, *args, **kwargs)
def forbidden(*args, **kwargs):
    raise AssertionError("generation loaded a DSO or allocated runtime output")
builtins.__import__ = guarded
ctypes.CDLL = forbidden
np.empty = forbidden
from generativeqc_compiler.xc.user_expression import UserExpression
from generativeqc_compiler.xc.user_expression_program import UserExpressionProgram
from generativeqc_compiler.xc.user_expression_native import emit_native
source, metadata = emit_native(UserExpressionProgram(UserExpression("rho_a*rho_b")))
assert source and metadata["runtime_headers"]
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path(__file__).resolve().parents[2] / "python"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_compiler_artifact_does_not_load_library(
    compiler: CppCompilerAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program = xc.expression_program(xc.expression(FORMULA))

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("compiler operation attempted to load a runtime DSO")

    monkeypatch.setattr(ct, "CDLL", forbidden)
    artifact = program.compile(compiler=compiler, cache=tmp_path)
    assert artifact.metadata["identity"]["backend"] == "cpu"
    assert set(artifact.metadata["identity"]["headers"]) == {
        "xc_user_expression_point.hpp"
    }


def test_installed_asset_identity_execution_and_missing_asset(
    compiler: CppCompilerAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tomllib
    from generativeqc_compiler.common import paths
    from generativeqc_compiler.xc.user_expression_native import RUNTIME_HEADER

    root = Path(__file__).resolve().parents[2]
    program = xc.expression_program(xc.expression(FORMULA))
    source, metadata = emit_native(program)
    checkout = xc.compile_expression(
        program, compiler=compiler, cache=tmp_path / "checkout-cache"
    )
    config = tomllib.loads((root / "pyproject.toml").read_text())
    destination = config["tool"]["scikit-build"]["wheel"]["force-include"][
        RUNTIME_HEADER
    ]
    installed = tmp_path / "installed"
    header = installed / destination
    header.parent.mkdir(parents=True)
    shutil.copyfile(root / RUNTIME_HEADER, header)
    monkeypatch.setattr(paths, "PACKAGE", installed / "generativeqc_compiler")
    with pytest.raises(ValueError, match="source checkout"):
        paths.source_root()
    assert paths.asset_path(RUNTIME_HEADER) == header
    assert emit_native(program) == (source, metadata)
    runtime = xc.compile_expression(
        program, compiler=compiler, cache=tmp_path / "installed-cache"
    )
    assert (
        runtime.artifact.metadata["identity"] == checkout.artifact.metadata["identity"]
    )
    assert runtime.artifact.metadata["key"] == checkout.artifact.metadata["key"]
    np.testing.assert_allclose(
        runtime.point(rho_a=0.4, rho_b=0.1), oracle(0.4, 0.1), rtol=4e-15
    )
    header.unlink()
    with pytest.raises(FileNotFoundError, match="missing compiler input"):
        xc.compile_expression(
            program, compiler=compiler, cache=tmp_path / "installed-cache"
        )


def test_header_tamper_and_stale_artifact_are_rejected(
    compiler: CppCompilerAdapter, tmp_path: Path
) -> None:
    program = xc.expression_program(xc.expression(FORMULA))
    cache = tmp_path / "header"
    program.compile(compiler=compiler, cache=cache)
    (header,) = (cache / "source").glob("*/xc_user_expression_point.hpp")
    header.write_text("// altered staged runtime header\n")
    with pytest.raises(ValueError, match="source identity mismatch"):
        program.compile(compiler=compiler, cache=cache)
    artifact = program.compile(compiler=compiler, cache=tmp_path / "artifact")
    path = artifact.library.parent / "artifact.json"
    payload = json.loads(path.read_text())
    payload["identity"]["headers"]["xc_user_expression_point.hpp"] = "stale-header"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="runtime cache integrity failure"):
        program.compile(compiler=compiler, cache=tmp_path / "artifact")

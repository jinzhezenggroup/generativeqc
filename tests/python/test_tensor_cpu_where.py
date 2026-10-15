"""Executed finite FP64 CPU selection, independent oracles and byte ABI gates."""

from __future__ import annotations

import ctypes as ct
import os
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.tensor.cpu import (
    NativeTensorProgram,
    describe_cpu_compilation,
)
from generativeqc_compiler.tensor.optimize import prepare_for_backend
from generativeqc_compiler.tensor.program import Program


def native(program: Program, cache: Path, **options: object) -> NativeTensorProgram:
    return NativeTensorProgram(
        program,
        compiler=CppCompilerAdapter(Path(os.environ.get("CXX", "c++"))),
        cache=cache,
        **options,
    )


@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (0, 3)])
@pytest.mark.parametrize("selector", ["input", "constant", "comparison"])
def test_public_compiled_where_oracles_and_detached_replay(
    tmp_path: Path,
    shape: tuple[int, ...],
    selector: str,
) -> None:
    x = np.arange(np.prod(shape, dtype=int), dtype=np.float64).reshape(shape) - 2
    mask = x > 0
    y = np.asarray(-0.0)

    def expression(c: object, a: object, b: object) -> dict:
        predicate = (
            c if selector == "input" else True if selector == "constant" else a > 0
        )
        return {"out": xp.where(predicate, a, b), "mask": c, "other": a + 4}

    compiled = xp.compile(expression, backend="cpu", cache=tmp_path)
    condition = mask if selector != "constant" else True
    expected = np.where(condition, x, y)
    for _ in range(2):
        actual = compiled(mask, x, y)
        np.testing.assert_array_equal(actual["out"], expected)
        np.testing.assert_array_equal(np.signbit(actual["out"]), np.signbit(expected))
        np.testing.assert_array_equal(actual["mask"], mask)
        np.testing.assert_array_equal(actual["other"], x + 4)
        assert actual["out"].dtype == np.float64
        assert actual["mask"].dtype == np.bool_ and actual["mask"].itemsize == 1
        for value in actual.values():
            assert not np.shares_memory(value, x) and not np.shares_memory(value, mask)
            with pytest.raises(ValueError):
                value.flags.writeable = True


def test_broadcast_views_repeated_gather_and_decimal_oracle(tmp_path: Path) -> None:
    mask = np.asarray([[True], [False]])
    x = np.asarray([[1.25, -2.5, 4.0]])
    y = np.asarray([[3.5], [-1.0]])

    def expression(c: object, a: object, b: object) -> object:
        predicate = xp.take(xp.broadcast_to(c, (2, 3)), (2, 0, 2), axis=1)
        return xp.where(predicate, xp.take(a, (2, 0, 2), axis=1) * 2 + 1, b * 4 - 3)

    actual = xp.compile(expression, backend="cpu", cache=tmp_path)(mask, x, y)
    expected = np.where(mask, x[:, [2, 0, 2]] * 2 + 1, y * 4 - 3)
    np.testing.assert_array_equal(actual, expected)
    with localcontext() as context:
        context.prec = 80
        oracle = [
            [
                float(Decimal.from_float(x[0, k]) * 2 + 1)
                if mask[i, 0]
                else float(Decimal.from_float(y[i, 0]) * 4 - 3)
                for k in (2, 0, 2)
            ]
            for i in range(2)
        ]
    np.testing.assert_array_equal(actual, oracle)


@pytest.mark.parametrize(
    "bad",
    [
        lambda x: 1 / (x - x),
        lambda x: xp.log(-x),
        lambda x: xp.sqrt(-x),
        lambda x: xp.exp(x * 1e4),
    ],
)
def test_unselected_computed_branch_is_still_validated(
    tmp_path: Path, bad: object
) -> None:
    compiled = xp.compile(
        lambda x: xp.where(True, x, bad(x)), backend="cpu", cache=tmp_path
    )
    with pytest.raises(ValueError, match="evaluation failed"):
        compiled(np.ones(3))


def test_raw_typed_abi_poison_alignment_transaction_and_recovery(
    tmp_path: Path,
) -> None:
    feeds = {
        "c": np.asarray([True, False, True]),
        "x": np.asarray([1.0, 2.0, 3.0]),
        "y": np.asarray([4.0, 5.0, 6.0]),
    }
    program = xp.compile(
        lambda c, x, y: {"mask": c, "out": xp.where(c, x, y)}, backend="reference"
    ).lower(*feeds.values())
    executor = native(program, tmp_path)
    # One-byte Boolean storage and aligned FP64 fields, in the prepared ABI's
    # declared input order rather than an optimizer-specific traversal order.
    assert executor.resources["input_count"] == 56
    assert executor.resources["output_count"] == 32
    assert executor.resources["typed_bytes"] == 1
    packed = np.zeros(56, dtype=np.uint8)
    offsets, cursor = {}, 0
    for node in executor.inputs:
        name = node.attrs["name"]
        value = feeds[name]
        cursor = (cursor + value.itemsize - 1) // value.itemsize * value.itemsize
        offsets[name] = cursor
        packed[cursor : cursor + value.nbytes] = value.view(np.uint8)
        cursor += value.nbytes
    assert cursor == packed.size
    output = np.full(32, 77, dtype=np.uint8)
    pointer = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_ubyte))
    original = packed.copy()
    for poison in ("mask", "unselected_nan", "unselected_inf", "budget", "size"):
        packed[:] = original
        if poison == "mask":
            packed[offsets["c"]] = 2
        elif poison.startswith("unselected"):
            packed[offsets["y"] : offsets["y"] + 8] = np.asarray(
                [np.nan if poison.endswith("nan") else np.inf]
            ).view(np.uint8)
        code = executor.call(
            pointer(packed),
            55 if poison == "size" else 56,
            pointer(output),
            32,
            executor.resources["required_bytes"] - (poison == "budget"),
        )
        assert code != 0 and np.all(output == 77), poison
    packed[:] = original
    assert (
        executor.call(pointer(packed), 56, pointer(output), 32, executor.max_bytes) == 0
    )
    np.testing.assert_array_equal(output[8:32].view(np.float64), [1.0, 5.0, 3.0])
    replay = native(Program.loads(program.dumps()), tmp_path)
    assert replay.identity == executor.identity
    assert replay.artifact.library == executor.artifact.library
    np.testing.assert_array_equal(replay.execute(feeds)["out"], [1.0, 5.0, 3.0])


@pytest.mark.parametrize("dtypes", [("float32", "float32"), ("float32", "float64")])
def test_unsupported_dtype_rejects_before_compiler(
    tmp_path: Path, dtypes: tuple[str, str]
) -> None:
    program = xp.compile(lambda c, x, y: xp.where(c, x, y), backend="reference").lower(
        np.asarray([True, False]),
        np.ones(2, dtype=dtypes[0]),
        np.ones(2, dtype=dtypes[1]),
    )

    def forbidden() -> None:
        pytest.fail("unsupported dtype discovered a compiler")

    with pytest.raises(ValueError, match="float64|cast"):
        NativeTensorProgram(program, compiler=forbidden, cache=tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_where_budgets_and_materializing_owner_opt_in() -> None:
    program = xp.compile(lambda c, x: xp.where(c, x, 0), backend="reference").lower(
        np.asarray([True, False]), np.ones(2)
    )
    _, _, resources, _ = describe_cpu_compilation(program)
    for options in (
        {"max_bytes": resources["required_bytes"] - 1},
        {"max_work": resources["scalar_work"] - 1},
        {"max_nodes": 1},
    ):
        with pytest.raises(ValueError, match="budget"):
            describe_cpu_compilation(program, **options)
    for backend in ("cpu", "cuda", "portable", "scalar"):
        with pytest.raises(ValueError, match="where"):
            prepare_for_backend(program, backend)
    for backend in ("cuda", "portable", "scalar"):
        with pytest.raises(ValueError, match="where admission"):
            prepare_for_backend(program, backend, allow_cpu_where=True)
    assert (
        prepare_for_backend(
            program, "cpu", allow_cpu_boolean=True, allow_cpu_where=True
        ).logical_hash
        == program.logical_hash
    )


def test_failed_compile_recovery_and_closed_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.common.compiler_process import CompileResult

    mask, values = np.asarray([True, False]), np.asarray([1.0, 2.0])
    compiled = xp.compile(
        lambda c, x: xp.where(c, x, 7.5), backend="cpu", cache=tmp_path
    )

    def fail(*args: object, **kwargs: object) -> CompileResult:
        return CompileResult(1, False, 0.0, "", "injected compile failure")

    with monkeypatch.context() as patch:
        patch.setattr(CppCompilerAdapter, "compile_shared", fail)
        with pytest.raises(RuntimeError, match="injected compile failure"):
            compiled(mask, values)
    assert not list(tmp_path.rglob("artifact.json"))
    assert compiled.compilation_info()["native_executables"] == 0
    result = compiled(mask, values)
    np.testing.assert_array_equal(result, [1.0, 7.5])
    assert compiled.compilation_info()["native_executables"] == 1
    compiled.close()
    np.testing.assert_array_equal(result, [1.0, 7.5])
    assert compiled.compilation_info()["native_executables"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        compiled(mask, values)

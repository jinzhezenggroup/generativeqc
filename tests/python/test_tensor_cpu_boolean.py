"""Executed native Boolean ABI and public finite-FP64 comparison qualification."""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path

import array_api_strict as strict
import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.tensor import ir
from generativeqc_compiler.tensor.cpu import NativeTensorProgram, emit_cpu
from generativeqc_compiler.tensor.cpu_bundle import NativeTensorProgramBundle
from generativeqc_compiler.tensor.cpu_emit import emit_cpu as emit_small_cpu
from generativeqc_compiler.tensor.optimize import prepare_for_backend
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec

COMPARISONS = ("equal", "not_equal", "greater", "greater_equal", "less", "less_equal")


def native(program: Program, cache: Path, **options: object) -> NativeTensorProgram:
    return NativeTensorProgram(
        program,
        compiler=CppCompilerAdapter(Path(os.environ.get("CXX", "c++"))),
        cache=cache,
        **options,
    )


@pytest.mark.parametrize("name", COMPARISONS)
@pytest.mark.parametrize(
    "left,right",
    (
        (np.asarray(-0.0), np.asarray(0.0)),
        (np.asarray([[-0.0], [2.0]]), np.asarray([[0.0, -3.0, 2.0]])),
        (np.empty((0, 1)), np.ones((1, 3))),
        (np.asarray([1e-300, -1e300, 1e300]), np.asarray([0.0, -1e300, 1.0])),
    ),
)
def test_compiled_public_six_comparisons(
    tmp_path: Path, name: str, left: np.ndarray, right: np.ndarray
) -> None:
    operation = getattr(xp, name)
    compiled = xp.compile(lambda x, y: operation(x, y), backend="cpu", cache=tmp_path)
    before = (left.copy(), right.copy())
    actual = compiled(left, right)
    expected = getattr(np, name)(left, right)
    strict_expected = getattr(strict, name)(strict.asarray(left), strict.asarray(right))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, np.asarray(strict_expected))
    assert actual.dtype == np.bool_ and actual.itemsize == 1
    assert not np.shares_memory(actual, left) and not np.shares_memory(actual, right)
    with pytest.raises(ValueError):
        actual.flags.writeable = True
    np.testing.assert_array_equal(left, before[0])
    np.testing.assert_array_equal(right, before[1])
    np.testing.assert_array_equal(compiled(left, right), expected)


@pytest.mark.parametrize("shape", [(), (0,), (3, 4)])
def test_bool_inputs_constants_views_and_mixed_real_outputs(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    mask = np.arange(np.prod(shape, dtype=int)).reshape(shape) % 2 == 0
    real = np.asarray(2.0)

    def expression(m: object, x: object) -> dict:
        return {
            "mask": m,
            "view": xp.reshape(
                xp.permute_dims(m, tuple(reversed(range(len(shape))))), (-1,)
            ),
            "constant": xp.full_like(m, True),
            "real": x * x + 1,
        }

    compiled = xp.compile(expression, backend="cpu", cache=tmp_path)
    program = compiled.lower(mask, real)
    executor = native(program, tmp_path / "direct")
    result = compiled(mask, real)
    expected = {
        "mask": mask,
        "view": mask.transpose().reshape(-1),
        "constant": np.ones(shape, dtype=bool),
        "real": real * real + 1,
    }
    for name, value in expected.items():
        np.testing.assert_array_equal(result[name], value)
        assert result[name].dtype == value.dtype
        assert not np.shares_memory(result[name], mask)
    restored = native(Program.loads(program.dumps()), tmp_path / "direct")
    assert restored.identity == executor.identity
    assert restored.artifact.library == executor.artifact.library
    for name, value in executor.execute({"m": mask, "x": real}).items():
        np.testing.assert_array_equal(value, result[name])


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: xp.broadcast_to(x, (2, 3, 4)),
        lambda x: xp.flip(x, axis=1),
        lambda x: x[1:, ::-2],
        lambda x: xp.take(x, (3, 1, 3), axis=1),
        lambda x: xp.expand_dims(x, (0, 2)),
        lambda x: xp.squeeze(xp.expand_dims(x, 1), 1),
        lambda x: xp.moveaxis(x, 0, 1),
        lambda x: xp.matrix_transpose(x),
    ],
)
def test_public_boolean_views(tmp_path: Path, operation: object) -> None:
    value = np.asarray(
        [
            [True, False, False, True],
            [False, True, True, False],
            [True, True, False, False],
        ]
    )
    compiled = xp.compile(operation, backend="cpu", cache=tmp_path)
    actual = compiled(value)
    assert actual.dtype == np.bool_ and actual.itemsize == 1
    np.testing.assert_array_equal(actual, operation(value))


@pytest.mark.parametrize(
    "name", ("zeros", "ones", "full", "zeros_like", "ones_like", "full_like")
)
def test_public_boolean_creation(tmp_path: Path, name: str) -> None:
    value = np.asarray([True, False, True])

    def expression(x: object) -> object:
        argument = x if name.endswith("_like") else x.shape
        options = {"dtype": xp.bool} if not name.endswith("_like") else {}
        return (
            getattr(xp, name)(argument, True, **options)
            if name.startswith("full")
            else getattr(xp, name)(argument, **options)
        )

    compiled = xp.compile(expression, backend="cpu", cache=tmp_path)
    actual = compiled(value)
    assert actual.dtype == np.bool_ and actual.itemsize == 1
    np.testing.assert_array_equal(actual, expression(value))


def test_public_boolean_broadcast_arrays(tmp_path: Path) -> None:
    def expression(x: object, y: object) -> dict:
        a, b = xp.broadcast_arrays(x, y)
        return {"a": a, "b": b}

    left, right = np.asarray([[True], [False]]), np.asarray([[False, True, False]])
    actual = xp.compile(expression, backend="cpu", cache=tmp_path)(left, right)
    expected = np.broadcast_arrays(left, right)
    for name, value in zip(("a", "b"), expected, strict=True):
        assert actual[name].dtype == np.bool_ and actual[name].itemsize == 1
        np.testing.assert_array_equal(actual[name], value)


def test_hostile_boolean_byte_abi_and_recovery(tmp_path: Path) -> None:
    axis = Index("i", IndexSpace("mask", "matrix", 3))
    mask = ir.input_tensor("mask", TensorSpec((axis,), dtype="bool", role="input"))
    executor = native(Program({"out": mask}), tmp_path)
    assert executor.resources["input_count"] == executor.resources["output_count"] == 3
    ptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_ubyte))
    values = np.asarray([0, 1, 0], dtype=np.uint8)
    output = np.full(3, 73, dtype=np.uint8)
    budget = executor.resources["required_bytes"]
    before = values.copy()
    for input_ptr, ni, output_ptr, no, limit in (
        (ptr(values), 2, ptr(output), 3, budget),
        (ptr(values), 3, ptr(output), 2, budget),
        (ptr(values), 3, ptr(output), 3, budget - 1),
        (None, 3, ptr(output), 3, budget),
        (ptr(values), 3, None, 3, budget),
        (ptr(values), 3, ptr(values), 3, budget),
        (
            ptr(values),
            3,
            ct.cast(output.ctypes.data + 1, ct.POINTER(ct.c_ubyte)),
            3,
            budget,
        ),
        (
            ct.cast(values.ctypes.data + 1, ct.POINTER(ct.c_ubyte)),
            3,
            ptr(output),
            3,
            budget,
        ),
    ):
        assert executor.call(input_ptr, ni, output_ptr, no, limit) == -1
        np.testing.assert_array_equal(output, [73, 73, 73])
        np.testing.assert_array_equal(values, before)
    for malformed in (2, 255):
        values[1] = malformed
        assert executor.call(ptr(values), 3, ptr(output), 3, budget) == -2
        np.testing.assert_array_equal(output, [73, 73, 73])
    values[1] = 1
    assert executor.call(ptr(values), 3, ptr(output), 3, budget) == 0
    np.testing.assert_array_equal(output, values)
    malformed_bool = np.asarray([0, 2, 0], dtype=np.uint8).view(np.bool_)
    with pytest.raises(ValueError, match="failed"):
        executor.execute({"mask": malformed_bool})
    for wrong in (np.ones(3), np.ones(3, dtype=np.int64), np.ones(2, dtype=bool)):
        with pytest.raises(ValueError, match="invalid bool"):
            executor.execute({"mask": wrong})
    recovered = executor.execute({"mask": np.asarray([True, False, True])})["out"]
    assert recovered.dtype == np.bool_ and recovered.itemsize == 1
    np.testing.assert_array_equal(recovered, [True, False, True])


def test_nonfinite_native_failure_is_transactional(tmp_path: Path) -> None:
    real = ir.input_tensor("x", TensorSpec(role="input"))
    executor = native(Program({"mask": ir.compare("less", real, real)}), tmp_path)
    values = np.asarray([np.inf], dtype=np.float64)
    output = np.full(executor.resources["output_count"], 77, dtype=np.uint8)
    ptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_ubyte))
    for bad in (np.nan, np.inf, -np.inf):
        values[0] = bad
        assert (
            executor.call(
                ptr(values), values.nbytes, ptr(output), output.size, executor.max_bytes
            )
            == -2
        )
        assert np.all(output == 77)
        with pytest.raises(ValueError, match="invalid float64"):
            executor.execute({"x": values.reshape(())})
    assert executor.execute({"x": np.asarray(1.0)})["mask"] == np.bool_(False)


def test_budget_overflow_and_backend_admission(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Program"):
        emit_cpu(None)
    mask = ir.input_tensor("mask", TensorSpec(dtype="bool", role="input"))
    program = Program({"mask": mask})
    _, resources = emit_cpu(program)
    for options in (
        {"max_bytes": resources["required_bytes"] - 1},
        {"max_work": 0},
        {"max_nodes": 0},
    ):
        with pytest.raises(ValueError, match="budget"):
            emit_cpu(program, **options)
    for backend in ("cpu", "cuda", "portable", "scalar"):
        with pytest.raises(ValueError, match="does not support bool"):
            prepare_for_backend(program, backend)
    for backend in ("cuda", "portable", "scalar"):
        with pytest.raises(ValueError, match="limited to"):
            prepare_for_backend(program, backend, allow_cpu_boolean=True)
    with pytest.raises(ValueError, match="bool"):
        emit_small_cpu(program, function_name="small")
    with pytest.raises(ValueError, match="bool"):
        NativeTensorProgramBundle(
            (program,),
            compiler=CppCompilerAdapter(Path(os.environ.get("CXX", "c++"))),
            cache=tmp_path,
        )
    huge = ir.input_tensor(
        "huge",
        TensorSpec(
            (Index("i", IndexSpace("huge", "matrix", (1 << 60) - 1)),), role="input"
        ),
    )
    with pytest.raises(ValueError, match="storage extent|byte/work budget"):
        emit_cpu(Program({"out": ir.compare("equal", huge, huge)}))
    fp32 = ir.input_tensor("x", TensorSpec(dtype="float32", role="input"))
    with pytest.raises(ValueError, match="float64"):
        emit_cpu(Program({"mask": ir.compare("equal", fp32, fp32)}))

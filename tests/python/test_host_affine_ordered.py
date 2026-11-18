"""Compiled ordered host contractions, fallbacks and transactional admission."""

from __future__ import annotations

import ctypes as ct
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.host_affine import emit_affine_host

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)
EXTENTS = {"row": "rows", "column": "columns", "reduction": "reductions"}


class Array(ct.Structure):
    """Borrow an explicitly sized, contiguous FP64 buffer."""

    _fields_ = [("data", POINTER), ("elements", ct.c_size_t)]


def _programs() -> dict[str, Program]:
    spaces = {
        label: IndexSpace(name, "auxiliary", 3)
        for label, name in (("i", "row"), ("j", "column"), ("k", "reduction"))
    }

    def tensor(name: str, labels: str) -> Any:
        return input_tensor(
            name,
            TensorSpec(
                tuple(
                    Index(f"axis_{axis}", spaces[label])
                    for axis, label in enumerate(labels)
                ),
                role="input",
            ),
        )

    programs = {}
    for ordinal, equation in enumerate(
        ("ik,kj->ij", "ki,kj->ij", "ik,jk->ij", "ki,jk->ij", "ik,kj->ji")
    ):
        for suffix, coefficient in (
            ("unit", 1),
            ("fraction", Fraction(1, 3)),
            ("large", 1 << 40),
        ):
            left, right = equation.split("->")[0].split(",")
            programs[f"layout_{ordinal}_{suffix}"] = Program(
                {
                    "out": einsum(
                        equation,
                        tensor("left", left),
                        tensor("right", right),
                        coefficient=coefficient,
                    )
                }
            )
    programs["diagonal"] = Program(
        {"out": einsum("kk,ij->ij", tensor("left", "kk"), tensor("right", "ij"))}
    )
    programs["ternary"] = Program(
        {
            "out": einsum(
                "ik,kj,ij->ij",
                tensor("left", "ik"),
                tensor("right", "kj"),
                tensor("third", "ij"),
            )
        }
    )
    return programs


@pytest.fixture(scope="module", params=[False, True], ids=["native", "scalar-fallback"])
def native(tmp_path_factory: Any, required_native_cxx: Any, request: Any) -> Any:
    """Compile the same graph with serial and ordered execution schedules."""
    directory = tmp_path_factory.mktemp("host-affine-ordered")
    programs = _programs()
    sources = [ROOT / "tests/native/ordered_binary_contraction_probe.cpp"]
    for variant in ("serial", "ordered"):
        (directory / f"{variant}.hpp").write_text(
            emit_affine_host(
                programs,
                namespace=variant,
                extents=EXTENTS,
                ordered_binary_contractions=variant == "ordered",
            )
        )
        source = directory / f"{variant}.cpp"
        source.write_text(f'''
#include "{variant}.hpp"
namespace maps = {variant};
extern "C" int {variant}_query(const char* name, std::size_t rows, std::size_t columns,
                              std::size_t reductions, std::size_t* bytes, std::size_t* work) {{
  maps::Extents shape{{}}; shape.rows = rows; shape.columns = columns; shape.reductions = reductions;
  for (const auto& program : maps::programs()) if (program.name == name)
    return program.workspace(shape, *bytes) && program.work(shape, *work) ? 0 : -1;
  return -1;
}}
extern "C" int {variant}_apply(const char* name, std::size_t rows, std::size_t columns,
    std::size_t reductions, const maps::Input* inputs, std::size_t input_count,
    const maps::Output* outputs, std::size_t output_count, double* arena, std::size_t arena_count) {{
  maps::Extents shape{{}}; shape.rows = rows; shape.columns = columns; shape.reductions = reductions;
  for (const auto& program : maps::programs()) if (program.name == name)
    return static_cast<int>(program.evaluate(shape, {{inputs, input_count}}, {{outputs, output_count}},
                                            {{arena, arena_count}}));
  return -1;
}}
''')
        sources.append(source)
    library = required_native_cxx.build_shared(
        sources,
        directory / "ordered.so",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + str(ROOT / "src"),
            *(["-DGQC_ORDERED_BINARY_FORCE_SCALAR=1"] if request.param else []),
        ],
    )
    loaded = ct.CDLL(str(library))
    for variant in ("serial", "ordered"):
        getattr(loaded, variant + "_query").argtypes = (
            [ct.c_char_p] + [ct.c_size_t] * 3 + [ct.POINTER(ct.c_size_t)] * 2
        )
        getattr(loaded, variant + "_apply").argtypes = (
            [ct.c_char_p]
            + [ct.c_size_t] * 3
            + [ct.POINTER(Array), ct.c_size_t] * 2
            + [POINTER, ct.c_size_t]
        )
    return loaded, programs


def _call(
    native: Any,
    variant: str,
    name: str,
    shape: tuple[int, int, int],
    pattern: str = "random",
    failure: str = "",
) -> tuple[int, np.ndarray, int, int]:
    loaded, programs = native
    program = programs[name]
    extents = dict(zip(EXTENTS, shape, strict=True))
    rng = np.random.default_rng(28442848)
    arrays = []
    for node in sorted(
        (node for node in program.nodes if node.op == "input"),
        key=lambda node: node.attrs["name"],
    ):
        dimensions = tuple(extents[axis.space.name] for axis in node.spec.indices)
        values = rng.normal(size=dimensions)
        if pattern == "range":
            values = np.ldexp(values, rng.integers(-450, 451, size=dimensions))
        elif pattern == "coefficient_overflow":
            values.fill(
                float.fromhex("0x1p1000")
                if node.attrs["name"] == "left"
                else float.fromhex("0x1p-1000")
            )
        elif pattern == "unit_overflow":
            values.fill(1e200)
        elif pattern == "zeros":
            values = np.resize(
                np.array([0.0, -0.0, np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0)]),
                dimensions,
            )
        arrays.append(np.ascontiguousarray(values))
    node = program.outputs["out"]
    output = np.full(
        tuple(extents[axis.space.name] for axis in node.spec.indices), -781.25
    )
    bytes_, work = ct.c_size_t(), ct.c_size_t()
    assert (
        getattr(loaded, variant + "_query")(
            name.encode(), *shape, ct.byref(bytes_), ct.byref(work)
        )
        == 0
    )
    arena = np.empty(bytes_.value // 8)
    inputs = (Array * len(arrays))(
        *(Array(values.ctypes.data_as(POINTER), values.size) for values in arrays)
    )
    outputs = (Array * 1)(Array(output.ctypes.data_as(POINTER), output.size))
    arena_count = arena.size
    if failure == "short_arena":
        arena_count -= 1
    elif failure == "input_output_alias":
        outputs[0].data = inputs[0].data
    elif failure == "input_arena_alias":
        inputs[0].data = arena.ctypes.data_as(POINTER)
    elif failure == "output_arena_alias":
        outputs[0].data = arena.ctypes.data_as(POINTER)
    elif failure == "misaligned_output":
        outputs[0].data = ct.cast(ct.c_void_p(output.ctypes.data + 1), POINTER)
    elif failure == "null_input":
        inputs[0].data = POINTER()
    elif failure == "wrong_size":
        outputs[0].elements += 1
    elif failure == "nonfinite_input":
        arrays[0].flat[0] = np.nan
    snapshots = [array.copy() for array in arrays]
    status = getattr(loaded, variant + "_apply")(
        name.encode(),
        *shape,
        inputs,
        len(inputs),
        outputs,
        len(outputs),
        arena.ctypes.data_as(POINTER),
        arena_count,
    )
    for values, expected in zip(arrays, snapshots, strict=True):
        np.testing.assert_array_equal(values.view(np.uint64), expected.view(np.uint64))
    return status, output, bytes_.value, work.value


@pytest.mark.parametrize(
    "shape",
    [
        (0, 0, 0),
        (0, 5, 7),
        (4, 0, 7),
        (4, 9, 0),
        (5, 4, 0),
        (7, 9, 0),
        (1, 1, 1),
        (3, 7, 17),
        (4, 4, 5),
        (5, 5, 9),
        (8, 8, 17),
        (9, 9, 23),
        (17, 19, 23),
    ],
)
@pytest.mark.parametrize(
    "pattern", ["random", "range", "zeros", "coefficient_overflow", "unit_overflow"]
)
def test_ordered_contractions_preserve_rounding_and_nonunit_fallback(
    native: Any, shape: tuple[int, int, int], pattern: str
) -> None:
    for name in native[1]:
        serial, expected, bytes_, work = _call(native, "serial", name, shape, pattern)
        ordered, actual, scheduled_bytes, scheduled_work = _call(
            native, "ordered", name, shape, pattern
        )
        assert (serial, bytes_, work) == (ordered, scheduled_bytes, scheduled_work)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        if serial:
            np.testing.assert_array_equal(actual, -781.25)


@pytest.mark.parametrize(
    "failure",
    [
        "short_arena",
        "input_output_alias",
        "input_arena_alias",
        "output_arena_alias",
        "misaligned_output",
        "null_input",
        "wrong_size",
        "nonfinite_input",
    ],
)
def test_ordered_contractions_keep_transactional_binding_guards(
    native: Any, failure: str
) -> None:
    for variant in ("serial", "ordered"):
        status, output, _, _ = _call(
            native, variant, "layout_0_unit", (3, 7, 17), failure=failure
        )
        assert status == (
            2 if failure == "short_arena" else 3 if failure == "nonfinite_input" else 1
        )
        np.testing.assert_array_equal(output, -781.25)


def test_ordered_contractions_reject_overflow_before_allocation(native: Any) -> None:
    loaded, _ = native
    for variant in ("serial", "ordered"):
        bytes_, work = ct.c_size_t(), ct.c_size_t()
        assert (
            getattr(loaded, variant + "_query")(
                b"layout_0_unit",
                (1 << (8 * ct.sizeof(ct.c_size_t))) - 1,
                7,
                17,
                ct.byref(bytes_),
                ct.byref(work),
            )
            != 0
        )


def test_ordered_native_strides_tails_and_empty_shapes(native: Any) -> None:
    assert native[0].ordered_binary_qualification() > 0


@pytest.mark.parametrize("policy", [None, 0, 1, "ordered"])
def test_ordered_schedule_requires_an_explicit_boolean(policy: object) -> None:
    with pytest.raises(TypeError, match="ordered_binary_contractions"):
        emit_affine_host(
            _programs(),
            namespace="test",
            extents=EXTENTS,
            ordered_binary_contractions=policy,
        )

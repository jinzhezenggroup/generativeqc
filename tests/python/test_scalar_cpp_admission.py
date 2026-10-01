"""Compile scalar lowering with adversarial names and failed output publication."""

import ctypes as ct
import math
import shutil
import subprocess
import typing
from pathlib import Path

import pytest
from generativeqc_compiler.tensor import (
    Program,
    TensorSpec,
    add,
    constant,
    divide,
    exp,
    input_tensor,
    log,
    multiply,
    power,
    scaled_bilinear,
    sqrt,
)
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp


def _compiled(
    tmp_path: Path, name: str, *, output_dependency_order: bool = False
) -> typing.Any:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    x = input_tensor(name, TensorSpec((), role="input"))
    source = emit_scalar_cpp(
        Program({"a": x, "b": multiply(x, x)}),
        function_name="scalar_probe",
        check_intermediates=False,
        output_dependency_order=output_dependency_order,
    )
    path = tmp_path / "probe.cpp"
    path.write_text(
        "#include <cmath>\n#include <algorithm>\n"
        + source
        + '\nextern "C" int call(double x, double* a, double* b) { return scalar_probe(x,*a,*b); }\n'
    )
    library = tmp_path / "probe.so"
    subprocess.run(
        [compiler, "-std=c++17", "-shared", "-fPIC", str(path), "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    call = ct.CDLL(str(library)).call
    call.argtypes = [ct.c_double, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
    call.restype = ct.c_int
    return call


@pytest.mark.parametrize("name", ("v0", "out_a", "x"))
def test_scalar_input_names_cannot_collide_with_emitter(
    tmp_path: Path, name: str
) -> None:
    call = _compiled(tmp_path, name)
    a, b = ct.c_double(-7), ct.c_double(-9)
    assert call(2.0, ct.byref(a), ct.byref(b)) == 1
    assert (a.value, b.value) == (2.0, 4.0)


@pytest.mark.parametrize("output_dependency_order", (False, True))
def test_scalar_failure_preserves_all_output_references(
    tmp_path: Path, output_dependency_order: bool
) -> None:
    call = _compiled(tmp_path, "x", output_dependency_order=output_dependency_order)
    a, b = ct.c_double(-7), ct.c_double(-9)
    assert call(1e200, ct.byref(a), ct.byref(b)) == 0
    assert (a.value, b.value) == (-7.0, -9.0)


def test_default_scalar_schedule_retains_generated_bytes() -> None:
    spec = TensorSpec((), role="input")
    x, y = (input_tensor(name, spec) for name in ("x", "y"))
    shared = multiply(x, x)
    program = Program({"first": multiply(shared, x), "last": add(shared, y)})
    options = {
        "function_name": "probe",
        "input_order": ("x", "y"),
        "output_order": ("last", "first"),
    }
    # Frozen before output-dependent scheduling was introduced. Opt-in must
    # not change source identities for any existing default-mode consumer.
    expected = """inline bool probe(double tensor_input_0, double tensor_input_1, double& tensor_output_0, double& tensor_output_1) noexcept {
  if (!std::isfinite(tensor_input_0) || !std::isfinite(tensor_input_1)) return false;
  const double v0 = tensor_input_0;
  const double v1 = tensor_input_1;
  const double v2 = v0 * v0;
  if (!std::isfinite(v2)) return false;
  double v3 = 0.0;
  v3 += 0x1.0000000000000p+0 * v2;
  v3 += 0x1.0000000000000p+0 * v1;
  if (!std::isfinite(v3)) return false;
  const double v4 = v2 * v0;
  if (!std::isfinite(v4)) return false;
  if (!std::isfinite(v3)) return false;
  if (!std::isfinite(v4)) return false;
  tensor_output_0 = v3;
  tensor_output_1 = v4;
  return true;
}
"""
    assert emit_scalar_cpp(program, **options) == expected
    assert (
        emit_scalar_cpp(program, **options, output_dependency_order=False) == expected
    )


@pytest.mark.parametrize(
    "output_order",
    (("first", "last", "alias"), ("last", "first", "alias"), None),
)
def test_output_dependency_schedule_preserves_root_order_and_shared_values(
    output_order: tuple[str, ...] | None,
) -> None:
    spec = TensorSpec((), role="input")
    x, y = (input_tensor(name, spec) for name in ("x", "y"))
    shared = multiply(x, x)
    outputs = {"first": multiply(shared, x), "last": add(shared, y), "alias": shared}
    program = Program(outputs)
    original = program.dumps()
    options = {
        "function_name": "probe",
        "input_order": ("x", "y"),
        "output_order": output_order,
        "caller_owned_checks": True,
        "output_dependency_order": True,
    }
    source = emit_scalar_cpp(program, **options)
    if output_order is None or output_order[0] == "first":
        expected_body = """  const double v0 = tensor_input_0;
  const double v1 = v0 * v0;
  const double v2 = v1 * v0;
  const double v3 = tensor_input_1;
  double v4 = 0.0;
  v4 += 0x1.0000000000000p+0 * v1;
  v4 += 0x1.0000000000000p+0 * v3;
"""
        roots = {"first": "v2", "last": "v4", "alias": "v1"}
    else:
        expected_body = """  const double v0 = tensor_input_0;
  const double v1 = v0 * v0;
  const double v2 = tensor_input_1;
  double v3 = 0.0;
  v3 += 0x1.0000000000000p+0 * v1;
  v3 += 0x1.0000000000000p+0 * v2;
  const double v4 = v1 * v0;
"""
        roots = {"first": "v4", "last": "v3", "alias": "v1"}
    for index, name in enumerate(
        sorted(outputs) if output_order is None else output_order
    ):
        expected_body += f"  tensor_output_{index} = {roots[name]};\n"
    expected_body += "  return true;\n}\n"
    assert source.split("\n", 1)[1] == expected_body
    assert program.dumps() == original
    assert (
        emit_scalar_cpp(Program(dict(reversed(tuple(outputs.items())))), **options)
        == source
    )


@pytest.mark.parametrize("output_dependency_order", (False, True))
def test_fma_lowering_is_opt_in_and_does_not_hide_shared_products(
    output_dependency_order: bool,
) -> None:
    spec = TensorSpec((), role="input")
    a, b, c = (input_tensor(name, spec) for name in ("a", "b", "c"))
    product = multiply(a, b)
    program = Program({"out": add(c, product)})
    default = emit_scalar_cpp(
        program,
        function_name="default",
        output_dependency_order=output_dependency_order,
    )
    fused = emit_scalar_cpp(
        program,
        function_name="fused",
        fused_accumulation=True,
        output_dependency_order=output_dependency_order,
    )
    assert "std::fma" not in default
    assert "std::fma" in fused
    shared = Program({"out": add(c, product), "product": product})
    assert "std::fma" not in emit_scalar_cpp(
        shared,
        function_name="shared",
        fused_accumulation=True,
        output_dependency_order=output_dependency_order,
    )


def _compile_program(
    tmp_path: Path, program: Program, **options: typing.Any
) -> tuple[typing.Any, str]:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    inputs = sorted(
        {node.attrs["name"] for node in program.live_nodes if node.op == "input"}
    )
    outputs = sorted(program.outputs)
    source = emit_scalar_cpp(
        program,
        function_name="scalar_probe",
        input_order=inputs,
        output_order=outputs,
        **options,
    )
    arguments = [f"inputs[{i}]" for i in range(len(inputs))]
    arguments += [f"outputs[{i}]" for i in range(len(outputs))]
    path, library = tmp_path / "probe.cpp", tmp_path / "probe.so"
    path.write_text(
        "#include <cmath>\n"
        + source
        + '\nextern "C" int call(const double* inputs, double* outputs) { '
        + f"return scalar_probe({', '.join(arguments)}); }}\n"
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            str(path),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    call = ct.CDLL(str(library)).call
    call.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
    call.restype = ct.c_int
    return call, source


@pytest.mark.parametrize("output_dependency_order", (False, True))
@pytest.mark.parametrize("ordered", (False, True))
def test_ordered_native_sums_preserve_signed_zero_and_unary_minus(
    tmp_path: Path, ordered: bool, output_dependency_order: bool
) -> None:
    x = input_tensor("x", TensorSpec((), role="input"))
    call, source = _compile_program(
        tmp_path,
        Program({"negated": add(x, coefficients=(-1,)), "sum": add(x, x)}),
        ordered_native_sums=ordered,
        output_dependency_order=output_dependency_order,
    )
    for value in (-0.0, 0.0):
        outputs = (ct.c_double * 2)(-7.0, -9.0)
        assert call((ct.c_double * 1)(value), outputs) == 1
        assert list(outputs) == [0.0, 0.0]
        expected_negation_sign = (
            -1.0 if ordered and value == 0.0 and math.copysign(1.0, value) > 0 else 1.0
        )
        expected_sum_sign = -1.0 if ordered and math.copysign(1.0, value) < 0 else 1.0
        assert math.copysign(1.0, outputs[0]) == expected_negation_sign
        assert math.copysign(1.0, outputs[1]) == expected_sum_sign
    assert "std::fma" not in source
    if ordered:
        assert " = 0.0;" not in source


@pytest.mark.parametrize("output_dependency_order", (False, True))
def test_ordered_native_sums_keep_term_order(
    tmp_path: Path, output_dependency_order: bool
) -> None:
    spec = TensorSpec((), role="input")
    x, y, z = (input_tensor(name, spec) for name in ("x", "y", "z"))
    call, _ = _compile_program(
        tmp_path,
        Program(
            {"ordered": add(x, y, z), "scaled": add(x, y, z, coefficients=(1, -1, 2))}
        ),
        ordered_native_sums=True,
        output_dependency_order=output_dependency_order,
    )
    outputs = (ct.c_double * 2)()
    assert call((ct.c_double * 3)(1e16, -1e16, 1.0), outputs) == 1
    assert list(outputs) == [1.0, 2e16]


@pytest.mark.parametrize("output_dependency_order", (False, True))
@pytest.mark.parametrize("check_intermediates", (False, True))
@pytest.mark.parametrize(
    ("operation", "value", "expected"),
    (
        ("zero_product", math.inf, "nan"),
        ("zero_quotient", 0.0, "nan"),
        ("zero_sum_coefficient", math.inf, "nan"),
        ("overflow", 1e200, "inf"),
        ("recovered_overflow", 1e200, "negative_zero"),
        ("input_nan", math.nan, "nan"),
        ("sqrt", -1.0, "nan"),
        ("log", 0.0, "negative_inf"),
        ("power", -1.0, "nan"),
        ("exp", 1000.0, "inf"),
    ),
)
def test_caller_owned_checks_publish_native_exceptional_arithmetic(
    tmp_path: Path,
    check_intermediates: bool,
    operation: str,
    value: float,
    expected: str,
    output_dependency_order: bool,
) -> None:
    x = input_tensor("x", TensorSpec((), role="input"))
    expressions = {
        "zero_product": multiply(constant(0), x),
        "zero_quotient": divide(constant(0), x),
        "zero_sum_coefficient": add(x, coefficients=(0,)),
        "overflow": multiply(x, x),
        "recovered_overflow": divide(constant("-3/2"), multiply(x, x)),
        "input_nan": x,
        "sqrt": sqrt(x),
        "log": log(x),
        "power": power(x, "3/2"),
        "exp": exp(x),
    }
    call, source = _compile_program(
        tmp_path,
        Program({"out": expressions[operation]}),
        check_intermediates=check_intermediates,
        caller_owned_checks=True,
        ordered_native_sums=True,
        output_dependency_order=output_dependency_order,
    )
    output = (ct.c_double * 1)(-7.0)
    assert call((ct.c_double * 1)(value), output) == 1
    assert "return false" not in source
    assert "std::isfinite" not in source
    if expected == "nan":
        assert math.isnan(output[0])
    elif expected == "inf":
        assert output[0] == math.inf
    elif expected == "negative_inf":
        assert output[0] == -math.inf
    else:
        assert output[0] == 0.0 and math.copysign(1.0, output[0]) == -1.0


@pytest.mark.parametrize("output_dependency_order", (False, True))
@pytest.mark.parametrize("operation", ("divide", "sqrt", "log", "power"))
def test_default_domain_checks_preserve_output_references(
    tmp_path: Path, operation: str, output_dependency_order: bool
) -> None:
    x = input_tensor("x", TensorSpec((), role="input"))
    expressions = {
        "divide": divide(constant(0), x),
        "sqrt": sqrt(x),
        "log": log(x),
        "power": power(x, "3/2"),
    }
    call, _ = _compile_program(
        tmp_path,
        Program({"out": expressions[operation]}),
        output_dependency_order=output_dependency_order,
    )
    output = (ct.c_double * 1)(-7.0)
    value = 0.0 if operation == "divide" else -1.0
    assert call((ct.c_double * 1)(value), output) == 0
    assert output[0] == -7.0


def test_caller_owned_scaled_bilinear_requires_direct_lowering(tmp_path: Path) -> None:
    x = input_tensor("x", TensorSpec((), role="input"))
    zero = constant(0)
    program = Program({"out": scaled_bilinear(zero, x, zero, x, x, x)})
    with pytest.raises(ValueError, match="requires direct_scaled_bilinear"):
        emit_scalar_cpp(program, function_name="probe", caller_owned_checks=True)
    call, source = _compile_program(
        tmp_path,
        program,
        caller_owned_checks=True,
        direct_scaled_bilinear=True,
        check_intermediates=False,
    )
    for value in (0.0, math.inf):
        output = (ct.c_double * 1)(-7.0)
        assert call((ct.c_double * 1)(value), output) == 1
        assert math.isnan(output[0])
    assert "return false" not in source

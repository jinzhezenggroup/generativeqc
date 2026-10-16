"""Output-demand TensorIR lowering for generated CUDA consumers."""

from fractions import Fraction
from pathlib import Path

import pytest
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    einsum,
    exp,
    input_tensor,
    reduce_sum,
    reshape,
    slice_tensor,
)
from generativeqc_compiler.tensor.cuda_inline import (
    exact_cuda_literal,
    lower_inline_cuda_output,
)


def _optional_diagnostic_program() -> Program:
    spin = Index("s", IndexSpace("spin", "spin", 2))
    vector = TensorSpec((spin,), role="input")
    production = input_tensor("density", vector)
    diagnostic = input_tensor("diagnostic_only", vector)
    return Program(
        {
            "force": reduce_sum(production, axes=(0,)),
            "diagnostic": reduce_sum(exp(diagnostic), axes=(0,)),
        }
    )


def test_inline_spin_contraction_is_bounded_and_keeps_pairing() -> None:
    for size in (2, 65):
        spin = Index("s", IndexSpace("spin", "spin", size))
        spec = TensorSpec((spin,), role="input")
        left, right = input_tensor("left", spec), input_tensor("right", spec)
        program = Program(
            {"weight": einsum("s,s->", left, right, coefficient=Fraction(-3, 7))}
        )
        bindings = {
            "left": tuple(f"a{i}" for i in range(size)),
            "right": tuple(f"b{i}" for i in range(size)),
        }
        if size > 64:
            with pytest.raises(ValueError, match="64-term work bound"):
                lower_inline_cuda_output(program, output="weight", bindings=bindings)
        else:
            lowered = lower_inline_cuda_output(
                program, output="weight", bindings=bindings
            )
            # Only arithmetic emitted above from this fixed, trusted test graph.
            assert eval(  # noqa: S307
                lowered.expression,
                {"__builtins__": {}},
                {"a0": 2, "a1": 3, "b0": 5, "b1": 7},
            ) == pytest.approx(-93 / 7)


def test_inline_cuda_applies_output_demand_before_consumer_lowering() -> None:
    program = _optional_diagnostic_program()
    lowered = lower_inline_cuda_output(
        program, output="force", bindings={"density": ("d0", "d1")}
    )

    assert lowered.expression == "((d0) + (d1))"
    assert lowered.required_inputs == ("density",)
    assert lowered.original_logical_hash == program.logical_hash
    assert lowered.specialization_logical_hash != program.logical_hash
    assert lowered.optimized_logical_hash == lowered.specialization_logical_hash
    assert len(lowered.optimizer_identity) == 64
    assert lowered.pruning_diagnostics["requested_outputs"] == ["force"]
    assert lowered.pruning_diagnostics["removed_outputs"] == ["diagnostic"]
    assert lowered.pruning_diagnostics["removed_inputs"] == ["diagnostic_only"]


def test_inline_cuda_fails_closed_when_requested_branch_is_not_supported() -> None:
    program = _optional_diagnostic_program()
    with pytest.raises(ValueError, match="unsupported TensorIR inline CUDA op: exp"):
        lower_inline_cuda_output(
            program,
            output="diagnostic",
            bindings={"diagnostic_only": ("q0", "q1")},
        )


def test_inline_cuda_rejects_missing_live_input_binding() -> None:
    program = _optional_diagnostic_program()
    with pytest.raises(ValueError, match="missing inline CUDA input binding: density"):
        lower_inline_cuda_output(program, output="force", bindings={})


def test_inline_cuda_structural_maps_preserve_row_major_coordinates() -> None:
    """Permuted broadcast axes, offset slices and reshape compose without copies."""
    row = Index("r", IndexSpace("rows", "batch", 2))
    column = Index("c", IndexSpace("columns", "batch", 3))
    repeat = Index("b", IndexSpace("replicas", "batch", 4))
    flat = Index("f", IndexSpace("flattened", "batch", 8))
    source = input_tensor("source", TensorSpec((row, column), role="input"))
    expanded = broadcast(source, (column, repeat, row), axes=(2, 0))
    selected = slice_tensor(expanded, ((1, 3), (1, 3), (0, 2)))
    vector = reshape(selected, (flat,))
    combined = add(vector, vector, coefficients=(Fraction(-2, 3), Fraction(5, 7)))
    lowered = lower_inline_cuda_output(
        Program({"value": reduce_sum(combined, axes=(0,))}),
        output="value",
        bindings={"source": tuple(str(value) for value in (2, -3, 7, 11, 13, -17))},
    )
    actual = eval(lowered.expression, {"__builtins__": {}})  # noqa: S307
    assert actual == pytest.approx(2 * (-3 + 13 + 7 - 17) * (-2 / 3 + 5 / 7))


def test_inline_cuda_broadcasts_scalar_and_retains_zero_coefficient_inputs() -> None:
    index = Index("r", IndexSpace("rows", "batch", 3))
    scalar = input_tensor("scalar", TensorSpec(role="input"))
    expanded = broadcast(scalar, (index,), axes=())
    result = add(expanded, expanded, coefficients=(Fraction(1, 3), 0))
    lowered = lower_inline_cuda_output(
        Program({"value": reduce_sum(result, axes=(0,))}),
        output="value",
        bindings={"scalar": "7.0"},
    )
    assert eval(lowered.expression, {"__builtins__": {}}) == pytest.approx(7.0)  # noqa: S307


@pytest.mark.parametrize(
    ("coefficients", "input_value", "expected"),
    (
        ((1,), "-0", 0.0),
        ((Fraction(1, 3),), "-0x0.0000000000001p-1022", 0.0),
        ((Fraction(-2, 3), Fraction(5, 7)), "21", 1.0),
    ),
)
def test_inline_cuda_add_preserves_seeded_accumulation(
    tmp_path: Path,
    coefficients: tuple[int | Fraction, ...],
    input_value: str,
    expected: float,
) -> None:
    """Execute dynamic signed zeros, underflow, and an ordered rational sum."""
    import shutil
    import subprocess

    compiler = shutil.which("c++")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if not compiler or not launcher:
        pytest.skip("C++ compiler and verified compiler cache are required")
    subprocess.run([launcher, "--version"], check=True, capture_output=True)
    scalar = input_tensor("scalar", TensorSpec(role="input"))
    program = Program(
        {"value": add(*(scalar for _ in coefficients), coefficients=coefficients)}
    )
    lowered = lower_inline_cuda_output(
        program, output="value", bindings={"scalar": "x"}
    )
    source, executable = tmp_path / "seeded_add.cpp", tmp_path / "seeded_add"
    object_file = tmp_path / "seeded_add.o"
    source.write_text(
        "#include <cmath>\n#include <cstdlib>\n#include <iomanip>\n"
        "#include <iostream>\nint main(int argc, char** argv) {\n"
        "if (argc != 2) return 2;\n"
        "const double x = std::strtod(argv[1], nullptr);\n"
        f"const double value = {lowered.expression};\n"
        "std::cout << std::signbit(value) << ' ' << std::setprecision(17) << value;\n"
        "}\n"
    )
    subprocess.run(
        [
            launcher,
            compiler,
            "-std=c++17",
            "-ffp-contract=off",
            "-c",
            str(source),
            "-o",
            str(object_file),
        ],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [launcher, compiler, str(object_file), "-o", str(executable)],
        check=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable), input_value], text=True, capture_output=True, check=True
    )
    signbit, actual = result.stdout.split()
    assert float(actual) == pytest.approx(expected)
    assert signbit == "0"


def test_exact_cuda_literal_and_scalar_binding_contract() -> None:
    assert exact_cuda_literal(2) == "2.0"
    assert exact_cuda_literal(Fraction(1, 3)) == "(1.0/3.0)"
    scalar = input_tensor("scalar", TensorSpec(role="input"))
    program = Program({"value": scalar})
    lowered = lower_inline_cuda_output(
        program, output="value", bindings={"scalar": "x"}
    )
    assert lowered.expression == "(x)"
    with pytest.raises(ValueError, match="binding size"):
        lower_inline_cuda_output(
            program, output="value", bindings={"scalar": ("x", "y")}
        )


@pytest.mark.parametrize(
    ("left_binding", "right_binding", "expected"),
    (
        ("a + b", "c", 20.0),
        ("a - b", "c", -4.0),
        ("a > b ? a : b", "c + a", 18.0),
    ),
)
def test_inline_cuda_preserves_bound_expression_precedence(
    tmp_path: Path,
    left_binding: str,
    right_binding: str,
    expected: float,
) -> None:
    """Compile the emitted scalar, independently checking arbitrary bindings."""
    import shutil
    import subprocess

    from generativeqc_compiler.tensor import einsum

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    index = Index("t", IndexSpace("singleton", "batch", 1))
    spec = TensorSpec((index,), role="input")
    left, right = input_tensor("left", spec), input_tensor("right", spec)
    program = Program({"value": einsum("t,t->t", left, right)})
    lowered = lower_inline_cuda_output(
        program,
        output="value",
        bindings={"left": left_binding, "right": right_binding},
    )
    source, executable = tmp_path / "binding.cpp", tmp_path / "binding"
    source.write_text(
        "#include <iostream>\nint main() { const double a=2,b=3,c=4; "
        + "std::cout << ("
        + lowered.expression
        + "); }\n"
    )
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)], check=True
    )
    result = subprocess.run(
        [str(executable)], text=True, capture_output=True, check=True
    )
    assert float(result.stdout) == expected

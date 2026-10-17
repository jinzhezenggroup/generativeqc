"""Execute generated affine arithmetic with a serial host CUDA launch shim."""

import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    multiply,
)
from generativeqc_compiler.tensor.cuda_affine import emit_affine_cuda


@pytest.mark.parametrize(
    ("coefficients", "input_value", "expected"),
    (
        ((1,), "-0", 0.0),
        ((Fraction(1, 3),), "-0x0.0000000000001p-1022", 0.0),
        ((Fraction(-2, 3), Fraction(5, 7)), "21", 1.0),
    ),
)
def test_affine_cuda_add_preserves_seeded_accumulation(
    tmp_path: Path,
    coefficients: tuple[int | Fraction, ...],
    input_value: str,
    expected: float,
) -> None:
    """Retain signed-zero semantics without requiring a CUDA device/compiler."""
    compiler = shutil.which("c++")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if not compiler or not launcher:
        pytest.skip("C++ compiler and verified compiler cache are required")
    subprocess.run([launcher, "--version"], check=True, capture_output=True)
    scalar = input_tensor("scalar", TensorSpec(role="input"))
    program = Program(
        {"value": add(*(scalar for _ in coefficients), coefficients=coefficients)}
    )
    generated = emit_affine_cuda(
        {"seeded_add": program}, namespace="test_affine", extents={}
    )
    # One host thread executes each kernel's complete grid-stride loop. Only
    # launch syntax changes; the emitted arithmetic and publication are intact.
    generated = re.sub(r"<<<.*?>>>", "", generated)
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#include <cmath>\nusing std::isfinite;\n"
        "#define __global__\n"
        "struct Dim { unsigned x; };\n"
        "static Dim blockIdx{0}, threadIdx{0}, blockDim{1}, gridDim{1};\n"
        "using cudaError_t = int; using cudaStream_t = void*;\n"
        "constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1;\n"
        "inline int cudaGetLastError() { return 0; }\n"
        "inline int atomicAdd(int* p, int v) { int old = *p; *p += v; return old; }\n"
        "inline int atomicExch(int* p, int v) { int old = *p; *p = v; return old; }\n"
    )
    (tmp_path / "generated.hpp").write_text(generated)
    source = tmp_path / "seeded_add.cpp"
    executable, object_file = tmp_path / "seeded_add", tmp_path / "seeded_add.o"
    source.write_text(
        '#include "generated.hpp"\n#include <cstdlib>\n#include <iomanip>\n'
        "#include <iostream>\n#include <vector>\n"
        "int main(int argc, char** argv) {\n"
        "if (argc != 2) return 2;\n"
        "const double x = std::strtod(argv[1], nullptr);\n"
        "test_affine::Extents shape{}; std::size_t bytes = 0;\n"
        "if (test_affine::workspace_seeded_add(shape, bytes)) return 3;\n"
        "std::vector<double> workspace(bytes / sizeof(double));\n"
        "int status = 0; double value = 99.0;\n"
        "if (test_affine::launch_seeded_add(shape, &x, &value, workspace.data(),\n"
        "bytes, &status, nullptr) || status) return 4;\n"
        "std::cout << std::signbit(value) << ' ' << std::setprecision(17) << value;\n"
        "}\n"
    )
    subprocess.run(
        [
            launcher,
            compiler,
            "-std=c++17",
            "-ffp-contract=off",
            "-I",
            str(tmp_path),
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


@pytest.mark.parametrize("vector", (False, True))
def test_affine_cuda_fusion_preserves_checks_outputs_and_duplicate_operands(
    tmp_path: Path,
    vector: bool,
) -> None:
    """Execute nested fused maps and a repeated-operand contraction unchanged."""
    compiler = shutil.which("c++")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if not compiler or not launcher:
        pytest.skip("C++ compiler and verified compiler cache are required")
    subprocess.run([launcher, "--version"], check=True, capture_output=True)
    indices = (Index("sample", IndexSpace("sample", "batch", 3)),) if vector else ()
    first = input_tensor("first", TensorSpec(indices, role="input"))
    second = input_tensor("second", TensorSpec(indices, role="input"))
    retained = add(first, second, coefficients=(Fraction(-2, 3), Fraction(5, 7)))
    nested = multiply(add(first, second), add(retained, coefficients=(Fraction(1, 3),)))
    contracted = einsum("p,p->p" if vector else ",->", nested, nested)
    program = Program({"retained": retained, "square": add(contracted, first)})
    for fused in (False, True):
        generated = emit_affine_cuda(
            {"evaluate": program},
            namespace=f"affine_{int(fused)}",
            extents={"sample": "samples"} if vector else {},
            fuse_elementwise=fused,
            fuse_contraction_epilogues=fused,
        )
        (tmp_path / f"generated_{int(fused)}.hpp").write_text(
            re.sub(r"<<<.*?>>>", "", generated)
        )
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#include <cmath>\nusing std::isfinite;\n#define __global__\n"
        "struct Dim { unsigned x; };\n"
        "static Dim blockIdx{0}, threadIdx{0}, blockDim{1}, gridDim{1};\n"
        "using cudaError_t = int; using cudaStream_t = void*;\n"
        "constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1;\n"
        "inline int cudaGetLastError() { return 0; }\n"
        "inline int atomicAdd(int* pointer, int value) { auto old = *pointer; *pointer += value; return old; }\n"
        "inline int atomicExch(int* pointer, int value) { auto old = *pointer; *pointer = value; return old; }\n"
    )
    source, executable = tmp_path / "fusion.cpp", tmp_path / "fusion"
    source.write_text(
        '#include "generated_0.hpp"\n#include "generated_1.hpp"\n'
        "#include <array>\n#include <limits>\n#include <vector>\n"
        "int main() {\n"
        "for (double first : {0.0, -0.0, 21.0, -1e-310, std::numeric_limits<double>::max()}) {\n"
        " const double second = first;\n"
        " std::array<double, 3> first_input{first, first, first}, second_input{second, second, second};\n"
        " std::array<std::array<double, 6>, 2> result;\n"
        " for (auto& values : result) values.fill(77.);\n"
        " int status[2]{};\n"
        " for (int fused = 0; fused < 2; ++fused) {\n"
        "  std::size_t bytes = 0;\n"
        "  if (fused ? affine_1::workspace_evaluate(SHAPE, bytes) : affine_0::workspace_evaluate(SHAPE, bytes)) return 2;\n"
        "  std::vector<double> workspace(bytes / sizeof(double));\n"
        "  auto error = fused\n"
        "   ? affine_1::launch_evaluate(SHAPE, first_input.data(), second_input.data(), &result[fused][0], &result[fused][3], workspace.data(), bytes, &status[fused], nullptr)\n"
        "   : affine_0::launch_evaluate(SHAPE, first_input.data(), second_input.data(), &result[fused][0], &result[fused][3], workspace.data(), bytes, &status[fused], nullptr);\n"
        "  if (error) return 3;\n"
        " }\n"
        " if (status[0] != status[1]) return 4;\n"
        " for (int output = 0; output < 6; ++output)\n"
        "  if (result[0][output] != result[1][output] || std::signbit(result[0][output]) != std::signbit(result[1][output])) return 5;\n"
        " if (first == std::numeric_limits<double>::max() && (!status[1] || result[1][0] != 77. || result[1][3] != 77.)) return 6;\n"
        "}\n}\n".replace("SHAPE", "{3}" if vector else "{}")
    )
    subprocess.run(
        [
            launcher,
            compiler,
            "-std=c++17",
            "-ffp-contract=off",
            "-I",
            str(tmp_path),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=10)

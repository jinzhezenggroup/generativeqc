"""Canonical compiler requests survive runtime-shape native projection."""

from __future__ import annotations

import os
import shutil
import subprocess
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.native_lowering import contraction_initializer

from tools.generate_df_ccsd_hoisted import packed_programs
from tools.generate_df_lambda import matrix_programs
from tools.generate_rccsd_native import (
    REPRESENTATIVE,
    _dim,
    _fraction,
    _packed_batched_matrix_gemm,
    _packed_matrix_gemm,
    _prepare_production,
    iteration_program,
)
from tools.generate_rhf_frame_response import programs as rhf_frame_programs


def test_native_projection_validates_cc_and_rhf_recipes(tmp_path: Path) -> None:
    """Validate semantic modes against matrix recipes at nonrepresentative sizes.

    Equal element counts would miss many transpose/batch mistakes. Compile the
    production native validator against every DF iteration and response request,
    then deliberately corrupt otherwise valid descriptors to exercise rejection.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    requests = []
    conventional = _prepare_production(iteration_program(*REPRESENTATIVE), "cuda")
    for program in (
        conventional,
        *packed_programs().values(),
        *matrix_programs().values(),
        *rhf_frame_programs().values(),
    ):
        adapter = TensorLoweringAdapter(program)
        for node in program.live_nodes:
            matrix = _packed_matrix_gemm(node)
            batched = _packed_batched_matrix_gemm(node)
            if matrix is None and batched is None:
                continue
            if batched is not None:
                ta, tb, batch, m, n, k = batched
            else:
                assert matrix is not None
                ta, tb, m, n, k = matrix
                batch = "1"
            initializer = contraction_initializer(
                adapter,
                node,
                _dim,
                transpose=(ta, tb),
                extents=(batch, m, n, k),
                coefficient=_fraction(node.attrs["coefficient"]),
            )
            canonical = adapter.request(node, backend="cuda")
            assert canonical.semantic_identity in initializer
            assert canonical.scientific_identity is not None
            assert canonical.scientific_identity in initializer
            assert canonical.precisions[0].identity in initializer
            requests.append(initializer)
    assert len(requests) > 300
    source = tmp_path / "projection.cpp"
    source.write_text(
        '#include "tensor/native_contraction.hpp"\n'
        "#include <iostream>\n#include <vector>\n"
        "std::size_t checked_product(std::initializer_list<std::size_t> factors){"
        "std::size_t value=1; for(auto n:factors) value=generativeqc::tensor::contraction_product(value,n); return value;}\n"
        "int main(){\n"
        "for(std::size_t o:{1,2,4}) for(std::size_t v:{1,3,7}) for(std::size_t q:{1,2,5}){\n"
        "const auto n=o+v; const std::vector<generativeqc::tensor::ContractionRequest> requests{\n"
        + ",\n".join(requests)
        + "};\n"
        "for(const auto& request:requests) request.validate();\n"
        "auto reject=[](const auto& r){try{r.validate();}catch(const std::exception&){return true;}return false;};\n"
        "auto r=requests.front(); r.precision.compute_dtype=generativeqc::runtime::PrecisionDtype::Fp32; if(!reject(r)) return 1;\n"
        "r=requests.front(); r.operands[0].strides[0]+=1; if(!reject(r)) return 2;\n"
        "r=requests.front(); r.operands[2].modes[0]=1000; if(!reject(r)) return 3;\n"
        'r=requests.front(); r.precision.math_mode="tf32"; if(!reject(r)) return 4;\n'
        'r=requests.front(); r.scientific_identity="unknown"; if(!reject(r)) return 5;\n'
        "r=requests.front(); r.m=std::numeric_limits<std::size_t>::max(); if(!reject(r)) return 6;\n"
        "r=requests.front(); r.operands[0].rank=100; if(!reject(r)) return 7;\n"
        "}\n}\n"
    )
    binary = tmp_path / "projection"
    compile_owner(compiler, tmp_path, [source], binary)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_typed_cuda_execution(tmp_path: Path) -> None:
    """Real-device dtype, transpose, batch, failure and capture-boundary gates."""
    if os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1":
        pytest.skip("requires explicit finite Slurm real-device qualification")
    cache, compiler = shutil.which("ccache"), shutil.which("nvcc")
    if cache is None or compiler is None:
        pytest.skip("CUDA compiler and ccache required")
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    obj, binary = tmp_path / "binding.o", tmp_path / "binding"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-arch=sm_120",
            "-I" + str(root / "src"),
            "-c",
            str(root / "tests/native/test_native_contraction_cuda.cu"),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CCACHE_BASEDIR": str(root)},
    )
    subprocess.run(
        [compiler, str(obj), "-lcublas", "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

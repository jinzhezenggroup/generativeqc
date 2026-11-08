"""Execute generated retained schedules on host; this is not GPU qualification."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.tensor.history_combination import (
    emit_history_combination,
    history_combination_candidate,
)
from generativeqc_compiler.tensor.ir import einsum, input_tensor
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.scf import diis_extrapolation_program
from generativeqc_compiler.tensor.scf_cuda import emit_density_cuda
from generativeqc_compiler.tensor.weighted_gram import (
    canonical_regions,
    retained_candidate,
)

from tools.generate_scf_array_native import diis_cuda_header, native_header

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_history_backends_share_science_and_borrow_storage() -> None:
    program = diis_extrapolation_program(2, 4, 3, spin_count=2)
    cpu, cuda = (history_combination_candidate(program, b) for b in ("cpu", "cuda"))
    assert cpu.request.semantic_identity == cuda.request.semantic_identity
    assert cpu.request.scientific_identity == cuda.request.scientific_identity
    for candidate in (cpu, cuda):
        assert candidate.request.operation == "einsum"
        assert candidate.workspace_bytes == 0
        assert candidate.execution is not None
        assert candidate.execution.determinism == "exact-order"
        assert all(op.strides is None for op in candidate.request.operands)


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
@pytest.mark.parametrize("mutation", ("dtype", "orientation", "coefficient"))
def test_history_emission_rejects_a_changed_equation(
    backend: str, mutation: str
) -> None:
    root = diis_extrapolation_program(2, 4, 3, spin_count=2).outputs["effective_fock"]
    history, weights = root.inputs
    if mutation == "dtype":
        history, weights = (
            input_tensor(n.attrs["name"], replace(n.spec, dtype="float32"))
            for n in root.inputs
        )
    output = einsum(
        "bhspq,bh->bsqp" if mutation == "orientation" else "bhspq,bh->bspq",
        history,
        weights,
        coefficient=2 if mutation == "coefficient" else 1,
    )
    with pytest.raises(ValueError, match="canonical topology"):
        emit_history_combination(Program({"effective_fock": output}), backend)


def test_df_scalar_density_uses_the_shared_gram_request() -> None:
    plain, weighted = canonical_regions("cuda", orbital_count=3)
    candidate = retained_candidate(plain, "occupied-scalar-cuda")
    incumbent = retained_candidate(plain, "occupied-cuda")
    assert (
        candidate.request.scientific_identity == incumbent.request.scientific_identity
    )
    assert candidate.request.operands == incumbent.request.operands
    assert candidate.workspace_bytes == 0
    assert dict(candidate.request.semantics)["occupation_representation"] == (
        "occupied-prefix-uniform-scalar"
    )
    with pytest.raises(ValueError, match="plain density"):
        retained_candidate(weighted, "occupied-scalar-cuda")


def test_generated_density_and_ring_combination_contracts(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    # Launch-coordinate stubs execute the actual generated density arithmetic.
    # No CUDA runtime, synchronization or device execution is claimed here.
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#define __device__\n#define __global__\n"
        "struct Coord { unsigned x = 0; };\n"
        "inline Coord threadIdx{}, blockIdx{}, blockDim{}, gridDim{};\n"
        "inline void __syncwarp() {}\n"
    )
    for name, source in (
        ("generated_scf_density_cuda.cuh", emit_density_cuda()),
        ("generated_scf_diis_cuda.cuh", diis_cuda_header()),
        ("generated_scf_array_native.hpp", native_header()),
    ):
        (tmp_path / name).write_text(source)
    executable = tmp_path / "retained_scf"
    required_native_cxx.build_executable(
        [ROOT / "tests/native/scf_retained_lowering_host.cpp"],
        executable,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "src"),
        ),
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
    assert result.stdout == "retained density and history contracts passed\n"

"""Shared P/W ownership, strict graph admission, and executable numeric oracles."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.method.gfn2_density_lowering import (
    emit_gfn2_density_contract,
)
from generativeqc_compiler.tensor.ir import add, einsum, input_tensor, multiply
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.scf import (
    density_program,
    weighted_density_input_specs,
    weighted_density_program,
)
from generativeqc_compiler.tensor.weighted_gram import (
    canonical_regions,
    recognize_weighted_gram,
    require_pair,
    retained_candidate,
    scalar_programs,
)
from generativeqc_compiler.tensor.weighted_gram_emit import emit_occupied

from tools.generate_gfn2_electronic_cuda import cuda_header

ROOT = Path(__file__).resolve().parents[2]



def _mutation(case: str, weighted: bool) -> Program:
    specs = weighted_density_input_specs(1, 3, spin_count=2, orbital_count=3)
    if case == "dtype":
        specs = {name: replace(spec, dtype="float32") for name, spec in specs.items()}
    values = {name: input_tensor(name, spec) for name, spec in specs.items()}
    c, f, e = (
        values[name] for name in ("coefficients", "occupations", "orbital_energies")
    )
    weights = multiply(f, e) if weighted else f
    if case == "weight-add":
        weights = add(f, e)
    if case == "weight-self":
        weights = multiply(f, f)
    if case == "weight-cube":
        weights = multiply(multiply(f, e), e)
    right = (
        input_tensor("other_coefficients", specs["coefficients"])
        if case == "alias"
        else c
    )
    if case == "cross-domain":
        from generativeqc_compiler.tensor.types import Index, IndexSpace

        b, s, p, i = specs["coefficients"].indices
        other = replace(
            specs["coefficients"],
            indices=(b, s, p, Index(i.name, IndexSpace("other", "orbital", 3))),
        )
        right = input_tensor("other_coefficients", other)
    root = einsum(
        "bspi,bsi,bsqi->bspq",
        c,
        weights,
        right,
        coefficient=2 if case == "coefficient" else 1,
    )
    return Program({"weighted_density" if weighted else "density": root})


@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("case", ("dtype", "coefficient", "alias", "cross-domain"))
def test_changed_graph_cannot_retain_an_unrelated_emitter(
    case: str, weighted: bool
) -> None:
    with pytest.raises(ValueError):
        recognize_weighted_gram(
            _mutation(case, weighted), "weighted_density" if weighted else "density"
        )


@pytest.mark.parametrize("case", ("weight-add", "weight-self", "weight-cube"))
def test_weight_graph_mutation_is_rejected(case: str) -> None:
    with pytest.raises(ValueError):
        recognize_weighted_gram(_mutation(case, True), "weighted_density")


def test_wrong_pair_and_wrong_physical_schedule_fail_closed() -> None:
    plain, weighted = canonical_regions("cpu")
    other = recognize_weighted_gram(
        weighted_density_program(1, 3, spin_count=2, orbital_count=3),
        "weighted_density",
    )
    with pytest.raises(ValueError, match="matching"):
        require_pair(plain, other)
    for region in (plain, weighted):
        for schedule in ("occupied-cuda", "column-scaled-cpu", "checked-pair-cuda"):
            with pytest.raises(ValueError, match="square"):
                retained_candidate(region, schedule)
    with pytest.raises(ValueError, match="unqualified"):
        retained_candidate(plain, "cublas")  # type: ignore[arg-type]


def test_candidate_selection_drives_emission(monkeypatch: pytest.MonkeyPatch) -> None:
    from generativeqc_compiler.tensor import weighted_gram_emit as emitter

    plain, weighted = canonical_regions("cpu")
    selected = retained_candidate(plain, "occupied-cpu")
    monkeypatch.setattr(
        emitter,
        "retained_candidate",
        lambda *_: replace(selected, implementation="unqualified"),
    )
    with pytest.raises(ValueError, match="admitted"):
        emit_occupied(plain, weighted, backend="cpu")


def test_candidate_requests_bind_graph_layout_and_retained_provider() -> None:
    plain, weighted = canonical_regions("cpu", orbital_count=3)
    for region in (plain, weighted):
        cpu = retained_candidate(region, "column-scaled-cpu")
        cuda = retained_candidate(region, "checked-pair-cuda")
        assert cpu.request.operation == cuda.request.operation == "einsum"
        assert (
            dict(cpu.request.semantics)["node_hash"]
            == region.adapter.hashes[region.contraction]
        )
        assert (
            dict(cuda.request.semantics)["weight_node_hash"]
            == region.adapter.hashes[region.contraction.inputs[1]]
        )
        assert cpu.request.operands[0].strides[-2:] == (1, 3)
        assert cuda.request.operands[0].strides[-2:] == (3, 1)
        assert (
            cpu.request.operands[0].alias_group == cpu.request.operands[2].alias_group
        )
        assert [p.name for p in cpu.providers] == ["generated.cpu", "cblas-lp64"]
        assert [p.name for p in cuda.providers] == ["generated.cuda"]
        assert cpu.execution.determinism == "reproducible"
        assert cuda.execution.determinism == "exact-order"
        assert cpu.workspace_bytes == cuda.workspace_bytes == 0
    assert all(
        p.provenance["density_node"] == plain.adapter.hashes[plain.contraction]
        for p in scalar_programs(plain, weighted).values()
    )


def test_original_science_and_old_math_deletion() -> None:
    method = (
        ROOT / "python/generativeqc_compiler/method/gfn2_electronic_runtime.py"
    ).read_text()
    for name in (
        "energy_weight",
        "weighted_coefficient",
        "density_contribution",
        "density_update",
    ):
        assert "def build_gfn2_" + name + "_program" not in method
    gaussian = (ROOT / "tools/generate_scf_array_native.py").read_text()
    assert "inline void density_from_orbitals(" not in gaussian
    assert "value += occupation_weight" not in gaussian
    assert "emit_occupied(" in gaussian
    assert density_program(1, 3).outputs["density"].op == "einsum"


def test_generated_paired_loop_against_independent_order_and_math_oracles(
    tmp_path: Path,
) -> None:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("C++ compiler and compiler cache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True, timeout=10)
    (tmp_path / "generated_gfn2_electronic_native.cuh").write_text(cuda_header())
    (tmp_path / "generated_gfn2_density_contract.inc").write_text(
        emit_gfn2_density_contract()
    )
    binary = tmp_path / "order"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-I",
            str(tmp_path),
            str(ROOT / "tests/native/test_weighted_gram_order.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=20)



def test_native_executor_consumes_the_selected_candidate_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.tensor import weighted_gram_emit as emitter

    generated = emitter.emit_native_header()
    native = (ROOT / "src/tensor/weighted_gram.hpp").read_text()
    assert "using CpuExecution = ColumnScaleDgemmLp64;" in generated
    assert (
        "template <class Algorithm = generated::CpuExecution, "
        "class Provider = CblasDgemmLp64>" in native
    )
    assert "std::is_same_v<Algorithm, generated::ColumnScaleDgemmLp64>" in native
    original = emitter.retained_candidate
    monkeypatch.setattr(
        emitter,
        "retained_candidate",
        lambda region, schedule: replace(
            original(region, schedule), implementation="unqualified"
        ),
    )
    with pytest.raises(ValueError, match="admitted"):
        emitter.emit_native_header()


@pytest.mark.parametrize("weighted", (False, True))
def test_occupied_cuda_candidate_describes_full_square_eigenframes(
    weighted: bool,
) -> None:
    from generativeqc_compiler.tensor import scf_cuda

    # Inspect the actual production binding, not a separate square fixture.
    region = scf_cuda._region(weighted)
    candidate = retained_candidate(region, "occupied-cuda")
    batch, spins, n, orbitals = region.coefficients.spec.shape
    assert (batch, spins, n, orbitals) == (1, 2, 3, 3)
    for position in (0, 2, 3):
        operand = candidate.request.operands[position]
        assert operand.strides == (spins * n * n, n * n, 1, n)
    weights = candidate.request.operands[1]
    assert weights.shape == (batch, spins, n)
    assert weights.strides is None
    assert (
        dict(candidate.request.semantics)["occupation_representation"]
        == "occupied-prefix-weight-1-or-2"
    )

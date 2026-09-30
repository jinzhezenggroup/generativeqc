"""Production TensorIR preparation shared by CPU/CUDA generators."""

from pathlib import Path

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    analyze_complexity,
    einsum,
    input_tensor,
    prepare_for_backend,
)

ROOT = Path(__file__).resolve().parents[2]


def _rank2_chain() -> Program:
    ao = IndexSpace("prepare_ao", "ao", 7)
    mo = IndexSpace("prepare_mo", "orbital", 7)
    coefficients = input_tensor(
        "coefficients",
        TensorSpec((Index("mu", ao), Index("p", mo)), role="input"),
    )
    hcore = input_tensor(
        "hcore",
        TensorSpec((Index("mu_h", ao), Index("nu", ao)), role="input"),
    )
    return Program(
        {"hcore_mo": einsum("mp,mn,nq->pq", coefficients, hcore, coefficients)}
    )


def test_production_prepare_strictly_lowers_rank2_chain_on_both_backends() -> None:
    source = _rank2_chain()
    assert analyze_complexity(source).max_work_degree == 4

    for backend in ("cpu", "cuda"):
        prepared = prepare_for_backend(source, backend=backend)
        assert analyze_complexity(prepared).max_work_degree == 3
        record = prepared.provenance["production_preparation"]
        assert record["backend"] == backend
        assert record["source_logical_hash"] == source.logical_hash
        assert record["prepared_logical_hash"] == prepared.logical_hash
        assert record["strict_degree_reassociation"] is True
        assert record["precision_locked"] is False


def test_production_prepare_can_preserve_original_contraction_order() -> None:
    source = _rank2_chain()
    preserved = prepare_for_backend(
        source,
        backend="cuda",
        preserve_contraction_order=True,
    )
    assert analyze_complexity(preserved).max_work_degree == 4
    record = preserved.provenance["production_preparation"]
    assert record["preserve_contraction_order"] is True
    assert record["strict_degree_reassociation"] is False


def test_rccsd_generator_has_no_identity_optimizer_escape_hatch() -> None:
    source = (ROOT / "tools/generate_rccsd_native.py").read_text()
    assert '"optimize": lambda program: program' not in source
    assert '"optimize": tensor_optimize' in source
    assert source.count('_production_program(') >= 20
    assert '_production_program(iteration_program(*REPRESENTATIVE), "cpu")' in source
    assert '_production_program(iteration_program(*REPRESENTATIVE), "cuda")' in source


def test_mp2_cpu_and_cuda_generation_share_production_preparation() -> None:
    source = (ROOT / "tools/generate_mp2_native.py").read_text()
    assert source.count("prepare_for_backend(") >= 2
    assert 'backend="cpu"' in source
    assert 'backend="cuda"' in source
    # CUDA planning must not run a second private contraction rewrite after the
    # shared production boundary has already fixed the prepared equation.
    assert "reassociate_contractions=False" in source

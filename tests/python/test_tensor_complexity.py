"""Symbolic complexity diagnostics and opt-in contraction-tree rewrites."""

import numpy as np
import pytest
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    analyze_complexity,
    einsum,
    execute,
    input_tensor,
    node_complexity,
    optimize,
    reassociate_einsums,
)
from generativeqc_compiler.tensor.cuda_gemm import gemm_contract


def _matrix(name: str, space: IndexSpace, left: str, right: str) -> Node:
    return input_tensor(
        name,
        TensorSpec(
            (Index(left, space), Index(right, space)),
            role="input",
        ),
    )


def test_symbolic_degree_is_independent_of_full_population_fixture_size() -> None:
    space = IndexSpace("ao", "ao", 1)
    matrix = _matrix("matrix", space, "i", "j")
    complexity = node_complexity(matrix)
    assert complexity.storage.powers == (("N", 2),)
    assert complexity.storage.degree == 2


def test_symbolic_degree_tracks_scientific_dimension_families() -> None:
    occupied = IndexSpace("occ", "occupied", 3)
    virtual = IndexSpace("vir", "virtual", 5)
    auxiliary = IndexSpace("aux", "auxiliary", 7)
    left = input_tensor(
        "left",
        TensorSpec(
            (
                Index("i", occupied),
                Index("a", virtual),
                Index("P", auxiliary),
            ),
            role="input",
        ),
    )
    right = input_tensor(
        "right",
        TensorSpec(
            (Index("P", auxiliary), Index("b", virtual)),
            role="input",
        ),
    )
    contraction = einsum("iaP,Pb->iab", left, right)
    complexity = node_complexity(contraction)
    assert complexity.work.degree == 4
    assert complexity.work.powers == (("O", 1), ("V", 2), ("A", 1))
    assert complexity.storage.powers == (("O", 1), ("V", 2))


def test_quartic_nary_contraction_reassociates_to_cubic_gemms() -> None:
    space = IndexSpace("ao", "ao", 4)
    a = _matrix("a", space, "i", "k")
    b = _matrix("b", space, "k", "l")
    c = _matrix("c", space, "l", "j")
    direct = Program({"out": einsum("ik,kl,lj->ij", a, b, c)})

    before = analyze_complexity(direct)
    assert before.max_work_degree == 4

    rewritten = reassociate_einsums(direct)
    assert rewritten.logical_hash != direct.logical_hash
    assert analyze_complexity(rewritten).max_work_degree == 3

    root = rewritten.outputs["out"]
    assert root.op == "einsum" and len(root.inputs) == 2
    assert gemm_contract(root) is not None
    assert any(
        child.op == "einsum" and gemm_contract(child) is not None
        for child in root.inputs
    )

    values = np.arange(16, dtype=np.float64).reshape(4, 4) / 17.0
    feeds = {
        "a": values,
        "b": values[::-1].copy(),
        "c": values.T.copy(),
    }
    expected = execute(direct, feeds).outputs["out"]
    actual = execute(rewritten, feeds).outputs["out"]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_retained_n4_output_is_reported_not_rewritten_away() -> None:
    space = IndexSpace("ao", "ao", 3)
    a = _matrix("a", space, "i", "j")
    b = _matrix("b", space, "k", "l")
    program = Program({"quartic": einsum("ij,kl->ijkl", a, b)})
    report = analyze_complexity(program)
    payload = report.summary_payload()
    assert report.max_storage_degree == 4
    materializations = payload["high_degree_materializations"]
    assert materializations["total"] == 1
    item = materializations["items"][0]
    assert item["output"] is True
    assert item["storage"]["notation"] == "O(N^4)"
    assert reassociate_einsums(program).logical_hash == program.logical_hash


def test_optimizer_can_opt_in_to_lower_order_contraction_tree() -> None:
    space = IndexSpace("ao", "ao", 4)
    a = _matrix("a", space, "i", "k")
    b = _matrix("b", space, "k", "l")
    c = _matrix("c", space, "l", "j")
    program = Program({"out": einsum("ik,kl,lj->ij", a, b, c)})

    ordinary = optimize(program)
    assert ordinary.provenance.get("complexity_diagnostics") is None

    optimized = optimize(program, reassociate_contractions=True)
    diagnostics = optimized.provenance["complexity_diagnostics"]
    assert diagnostics["reassociation"]["enabled"] is True
    assert diagnostics["reassociation"]["changed"] is True
    assert diagnostics["requested"]["max_work_degree"] == 4
    assert diagnostics["reassociated"]["max_work_degree"] == 3
    assert diagnostics["optimized"]["max_work_degree"] == 3


def test_reassociation_rejects_explicit_precision_execution() -> None:
    space = IndexSpace("ao", "ao", 3)
    a = _matrix("a", space, "i", "k")
    b = _matrix("b", space, "k", "l")
    c = _matrix("c", space, "l", "j")
    program = Program(
        {"out": einsum("ik,kl,lj->ij", a, b, c)},
        provenance={"precision_execution": {"schema": "test-placeholder"}},
    )

    with pytest.raises(ValueError, match="precision execution"):
        reassociate_einsums(program)


def test_intermediate_axis_limit_retains_a_direct_fallback() -> None:
    """A forbidden pair is not permission to drop a mathematical contraction."""
    space = IndexSpace("vir", "virtual", 3)
    a = _matrix("a", space, "i", "k")
    b = _matrix("b", space, "k", "l")
    c = _matrix("c", space, "l", "j")
    program = Program({"out": einsum("ik,kl,lj->ij", a, b, c)})
    constrained = reassociate_einsums(program, max_intermediate_axes={"virtual": 1})
    assert constrained.logical_hash == program.logical_hash
    allowed = reassociate_einsums(program, max_intermediate_axes={"virtual": 2})
    assert analyze_complexity(allowed).max_work_degree == 3
    assert allowed.logical_hash == reassociate_einsums(program).logical_hash


@pytest.mark.parametrize("limits", [{"invalid": 2}, {"virtual": -1}, {"virtual": True}])
def test_intermediate_axis_limits_validate_domains(limits: dict[str, int]) -> None:
    space = IndexSpace("vir", "virtual", 3)
    matrix = _matrix("a", space, "i", "j")
    with pytest.raises(ValueError, match="axis limits"):
        reassociate_einsums(Program({"out": matrix}), max_intermediate_axes=limits)


@pytest.mark.parametrize("size", [2, 7])
def test_degree_neutral_contraction_is_explicit_bounded_and_numerical(
    size: int,
) -> None:
    """A constant-work schedule preserves signed bilinears and the default policy."""
    orbital = IndexSpace("ao", "ao", size)
    point = IndexSpace("point", "batch", 5)
    density = _matrix("density", orbital, "i", "j")
    panel = TensorSpec((Index("p", point), Index("i", orbital)), role="input")
    left, right = input_tensor("left", panel), input_tensor("right", panel)
    direct = Program({"out": einsum("pi,ij,pj->p", left, density, right)})
    assert reassociate_einsums(direct).logical_hash == direct.logical_hash
    limits = {"batch": 1, "ao": 1}
    scheduled = reassociate_einsums(
        direct, require_degree_reduction=False, max_intermediate_axes=limits
    )
    assert (
        analyze_complexity(scheduled).max_work_degree
        == analyze_complexity(direct).max_work_degree
    )
    for node in scheduled.nodes:
        if node.op == "input" or node in scheduled.outputs.values():
            continue
        assert sum(axis.space.kind == "ao" for axis in node.spec.indices) <= 1
        assert sum(axis.space.kind == "batch" for axis in node.spec.indices) <= 1
    rng = np.random.default_rng(2844 + size)
    feeds = {
        "left": rng.normal(size=(5, size)),
        "right": rng.normal(size=(5, size)),
        "density": rng.normal(size=(size, size)),
    }
    expected = np.einsum("pi,ij,pj->p", feeds["left"], feeds["density"], feeds["right"])
    np.testing.assert_allclose(
        execute(scheduled, feeds).outputs["out"], expected, atol=3e-12, rtol=3e-12
    )
    blocked = reassociate_einsums(
        direct,
        require_degree_reduction=False,
        max_intermediate_axes={"batch": 0, "ao": 0},
    )
    assert blocked.logical_hash == direct.logical_hash


@pytest.mark.parametrize("policy", [None, 0, 1, "false"])
def test_degree_reduction_policy_requires_an_explicit_boolean(policy: object) -> None:
    space = IndexSpace("ao", "ao", 2)
    matrix = _matrix("density", space, "i", "j")
    with pytest.raises(TypeError, match="require_degree_reduction"):
        reassociate_einsums(Program({"out": matrix}), require_degree_reduction=policy)

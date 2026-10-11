"""CPU-only qualification for the shared structured tensor route registry.

These are source/selection tests, not real CUDA or endpoint profitability claims.
"""

from __future__ import annotations

import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.specialization import (
    CompilationIdentity,
    TargetCapabilities,
)
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from generativeqc_compiler.tensor.scf import density_program
from generativeqc_compiler.tensor.structured_routes import (
    bind_structured_route,
    structured_cuda_routes,
)

SOURCE = "test-candidate-only-source-identity"


def _program(expression: str, sizes: dict[str, int]) -> Program:
    left, right = expression.split("->")
    spaces = {label: IndexSpace(label, "batch", size) for label, size in sizes.items()}
    operands = [
        input_tensor(
            f"operand_{index}",
            TensorSpec(
                tuple(
                    Index(f"axis_{axis}", spaces[label])
                    for axis, label in enumerate(row)
                ),
                role="input",
            ),
        )
        for index, row in enumerate(left.split(","))
    ]
    return Program({"result": einsum(expression, *operands)})


def _routes(program: Program):
    plan = plan_cuda(
        program,
        cuda_target_info("sm_80"),
        schedule=TensorSchedule(),
        reassociate_contractions=False,
    )
    return plan, structured_cuda_routes(plan, source_identity=SOURCE)


@pytest.mark.parametrize(
    ("equation", "family"),
    [
        ("ik,kj->ij", "gemm"),
        ("ik,k->i", "gemv"),
        ("i,i->", "dot"),
    ],
)
def test_binary_einsums_use_one_shared_provider_request(
    equation: str, family: str
) -> None:
    program = _program(equation, {"i": 3, "j": 4, "k": 5})
    original = program.dumps()
    _, routes = _routes(program)
    eligible = [route for route in routes if route.family == family]
    assert len(eligible) == 1
    route = eligible[0]
    assert route.request.scientific_identity == program.logical_hash
    assert len(route.candidates) == 3
    assert {candidate.request.identity for candidate in route.candidates} == {
        route.request.identity
    }
    assert {candidate.providers[0].name for candidate in route.candidates} == {
        "nvidia.cublaslt",
        "nvidia.cutensor",
        "nvidia.cutlass-aot",
    }
    # No guessed toolkit/resources and no implicit default promotion.
    assert all(candidate.status == "unsupported" for candidate in route.candidates)
    assert all(candidate.reason for candidate in route.candidates)
    assert program.dumps() == original


def test_shared_coefficient_gram_is_recognized_without_rewriting_science() -> None:
    ao = IndexSpace("ao", "ao", 3)
    orbital = IndexSpace("orbital", "orbital", 2)
    coefficients = input_tensor(
        "coefficients",
        TensorSpec(
            (Index("p", ao), Index("i", orbital)),
            role="input",
        ),
    )
    program = Program(
        {"gram": einsum("pi,qi->pq", coefficients, coefficients)}
    )
    _, routes = _routes(program)
    assert any(route.family == "gram" for route in routes)


def test_matrix_chain_keeps_both_ordered_binary_operations() -> None:
    spaces = {
        label: IndexSpace(label, "batch", size)
        for label, size in {"i": 3, "j": 5, "k": 4, "l": 2}.items()
    }

    def matrix(name: str, labels: str):
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

    a, b, c = matrix("a", "ij"), matrix("b", "jk"), matrix("c", "kl")
    first = einsum("ij,jk->ik", a, b)
    root = einsum("ik,kl->il", first, c)
    program = Program({"chain": root})
    _, routes = _routes(program)
    assert any(route.family == "gemm" for route in routes)
    assert any(route.family == "matrix-chain" for route in routes)
    # The pass emits no fused A^T B A equation or reassociation.
    assert program.outputs["chain"] is root
    assert root.inputs[0] is first


def test_weighted_gram_reuses_existing_strict_rank_k_portfolio() -> None:
    program = density_program(1, 3, spin_count=1, orbital_count=2)
    plan, routes = _routes(program)
    rank_k = [route for route in routes if route.family == "weighted-gram"]
    assert len(rank_k) == 1
    route = rank_k[0]
    assert route.output_name == "density"
    assert dict(route.request.semantics)["symmetric_rank_k"] is True
    assert {c.providers[0].name for c in route.candidates} == {
        "generated.cuda",
        "cublas",
    }
    target = TargetCapabilities(
        plan.target.target_info,
        features=tuple(
            (feature, True) for feature in plan.target.required_cuda_features
        ),
    )
    # A known qualified incumbent may be retained, but missing phase costs
    # cannot automatically promote a competitor.
    incumbent = next(
        c for c in route.candidates if c.providers[0].name == "generated.cuda"
    )
    binding = bind_structured_route(
        route,
        target,
        CompilationIdentity(route.request.scientific_identity, SOURCE),
        qualified_incumbent=incumbent.identity,
    )
    assert binding.retained_incumbent
    assert binding.selected == incumbent
    with pytest.raises(ValueError, match="complete cost"):
        bind_structured_route(
            route,
            target,
            CompilationIdentity(route.request.scientific_identity, SOURCE),
        )


def test_target_mismatch_and_missing_source_identity_fail_closed() -> None:
    program = _program("ik,kj->ij", {"i": 3, "j": 4, "k": 5})
    plan = plan_cuda(
        program,
        cuda_target_info("sm_80"),
        schedule=TensorSchedule(),
        reassociate_contractions=False,
    )
    with pytest.raises(ValueError, match="source identity"):
        structured_cuda_routes(plan, source_identity="")
    wrong = TargetCapabilities(cuda_target_info("sm_90").target_info)
    with pytest.raises(ValueError, match="plan target"):
        structured_cuda_routes(
            plan, source_identity=SOURCE, target_capabilities=wrong
        )


def test_invalid_ternary_structure_is_not_misrecognized_as_rank_k() -> None:
    i = IndexSpace("i", "ao", 3)
    j = IndexSpace("j", "ao", 3)
    k = IndexSpace("k", "orbital", 2)
    left = input_tensor(
        "left", TensorSpec((Index("i", i), Index("k", k)), role="input")
    )
    right = input_tensor(
        "right", TensorSpec((Index("j", j), Index("k", k)), role="input")
    )
    weight = input_tensor("weight", TensorSpec((Index("k", k),), role="input"))
    program = Program({"out": einsum("ik,k,jk->ij", left, weight, right)})
    _, routes = _routes(program)
    assert not any(route.family == "weighted-gram" for route in routes)

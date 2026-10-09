"""Fold the two free occupied ladder indices without new CC equations.

Each ordered occupied pair owns a dense virtual-by-virtual matrix. Restricting
the spectator pair to i <= j preserves matrix lowering, unlike fully packed
(i,a)/(j,b) coordinates. The native consumer must establish simultaneous pair
symmetry of tau, or a separately certified rounding bound, before execution.
This compiler transform does not establish that runtime content contract.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cache
from itertools import permutations, product
from math import prod
from string import ascii_letters

from generativeqc_compiler.tensor import Index, IndexSpace, Node, Program, TensorSpec
from generativeqc_compiler.tensor.ir import (
    add,
    constant,
    einsum,
    input_tensor,
    transpose,
)

from .df_hoist import _word
from .doubles import _expand

LADDER_OUTPUT = "df_D05_vv_ladder"
PAIRED_TAU_INPUT = "df_tau_occupied_pairs"


def _prove_ladder_pair_symmetry(root: Node) -> None:
    """Prove reflection from exact inventory algebra, assuming only tau symmetry.

    Free spectator labels alone do not prove that the output can be reflected:
    a one-sided virtual contraction is a counterexample. Expand with the shared
    CC polynomial utility, then compare exact rational monomials modulo dummy
    renaming, commutative scalar factors, and tau_ijab = tau_jiba. No symmetry
    of the factor matrices is assumed. Small explicit limits keep this proof a
    finite compiler process; larger future inventories require a new proof.
    """

    @cache
    def expand(node: Node) -> list:
        labels = ascii_letters[: len(node.spec.indices)]
        if node.op == "input":
            return [(Fraction(1), labels, ((node, labels),))]
        if node.op == "add":
            if sum(len(expand(child)) for child in node.inputs) > 64:
                raise ValueError("ladder symmetry proof exceeds its term limit")
            return [
                term
                for child, coefficient in zip(
                    node.inputs, node.attrs["coefficients"], strict=True
                )
                for term in _expand(
                    labels + "->" + labels,
                    [expand(child)],
                    Fraction(*coefficient),
                )
            ]
        if node.op == "transpose":
            output = "".join(labels[axis] for axis in node.attrs["axes"])
            return _expand(labels + "->" + output, [expand(node.inputs[0])], 1)
        if node.op == "einsum":
            children = [expand(child) for child in node.inputs]
            if prod(map(len, children)) > 64:
                raise ValueError("ladder symmetry proof exceeds its term limit")
            return _expand(
                _word(node),
                children,
                Fraction(*node.attrs["coefficient"]),
            )
        raise ValueError("unsupported ladder symmetry-proof primitive")

    def canonical(terms: list) -> Counter:
        result = Counter()
        for coefficient, output, operands in terms:
            if len(operands) > 6:
                raise ValueError("ladder symmetry proof exceeds its factor limit")
            tau = [
                position
                for position, (node, _) in enumerate(operands)
                if node.attrs["name"] == "df_tau"
            ]
            if len(tau) != 1:
                raise ValueError("ladder symmetry proof requires one tau per term")
            spaces = {
                label: index.space
                for node, labels in operands
                for label, index in zip(labels, node.spec.indices, strict=True)
            }
            dummy = sorted(set(spaces) - set(output))
            if len(dummy) > 4:
                raise ValueError("ladder symmetry proof exceeds its dummy limit")
            groups = [
                [label for label in dummy if spaces[label] == space]
                for space in dict.fromkeys(spaces[label] for label in dummy)
            ]
            keys = []
            for renamings in product(*(permutations(group) for group in groups)):
                mapping = {label: position for position, label in enumerate(output)}
                for group, renamed in zip(groups, renamings, strict=True):
                    mapping.update(
                        (label, len(output) + dummy.index(replacement))
                        for label, replacement in zip(group, renamed, strict=True)
                    )
                for reflected in (False, True):
                    factors = []
                    for position, (node, labels) in enumerate(operands):
                        factor_labels = (
                            "".join(labels[axis] for axis in (1, 0, 3, 2))
                            if position == tau[0] and reflected
                            else labels
                        )
                        factors.append(
                            (
                                node.attrs["name"],
                                tuple(mapping[label] for label in factor_labels),
                            )
                        )
                    keys.append(tuple(sorted(factors)))
            result[min(keys)] += coefficient
        return Counter({key: value for key, value in result.items() if value})

    original = _expand("abcd->abcd", [expand(root)], 1)
    reflected = _expand("badc->abcd", [original], 1)
    if canonical(original) != canonical(reflected):
        raise ValueError("ladder does not preserve simultaneous pair symmetry")


@dataclass(frozen=True)
class LadderPairMajorant:
    """Coefficients of a conservative tau-projection error bound.

    The original graph supplies every absolute coefficient and reduction axis.
    For each factor row the native owner computes max(sum(abs(factor), axes)).
    The amplitude norm is computed anew, from the current T1. With tau error
    delta, the real-arithmetic output error is bounded by
    delta * (coefficient_0 + coefficient_1 * current_t1_norm), accumulated over Q.
    Native admission must additionally certify floating-point range/rounding.
    """

    coefficients: Program
    factor_norms: tuple[tuple[str, str, tuple[int, ...]], ...]
    amplitude_norm: tuple[str, tuple[int, ...]]


def build_ladder_pair_majorant(program: Program) -> LadderPairMajorant:
    """Derive a positive bound from the same linear ladder contraction tree.

    A tau-dependent operand is bounded in max norm. Its other operand is
    bounded by the maximum absolute row sum over the complete contracted
    domain. Assigning all reduction axes to this operand avoids unjustified
    splitting of coupled sums. Compound factor rows use the same bound
    recursively, assigning every shared summed axis to one operand rather
    than incorrectly multiplying two independently split coupled sums.
    """
    norms: dict[str, tuple[str, tuple[int, ...]]] = {}

    @cache
    def tau_dependent(node: Node) -> bool:
        return (
            node.attrs["name"] == "df_tau"
            if node.op == "input"
            else any(tau_dependent(child) for child in node.inputs)
        )

    @cache
    def amplitude_dependent(node: Node) -> bool:
        return (
            node.attrs["name"] == "t1"
            if node.op == "input"
            else any(amplitude_dependent(child) for child in node.inputs)
        )

    def multiply_polynomials(first: Counter, second: Counter) -> Counter:
        result = Counter()
        for first_term, first_value in first.items():
            for second_term, second_value in second.items():
                result[tuple(sorted((*first_term, *second_term)))] += (
                    first_value * second_value
                )
        return result

    @cache
    def row_polynomial(node: Node, axes: tuple[int, ...]) -> Counter:
        if node.op == "input":
            name = node.attrs["name"]
            if name not in ("bov", "bvv", "t1"):
                raise ValueError("ladder row-norm proof requires known input factors")
            bound_name = name + "_sum_axes_" + "_".join(map(str, axes))
            norms[bound_name] = (name, axes)
            return Counter({(bound_name,): Fraction(1)})
        if node.op == "transpose":
            return row_polynomial(
                node.inputs[0], tuple(sorted(node.attrs["axes"][axis] for axis in axes))
            )
        if node.op == "add":
            result = Counter()
            for child, coefficient in zip(
                node.inputs, node.attrs["coefficients"], strict=True
            ):
                for monomial, value in row_polynomial(child, axes).items():
                    result[monomial] += abs(Fraction(*coefficient)) * value
            return result
        if node.op != "einsum" or len(node.inputs) != 2:
            raise ValueError("ladder row-norm proof requires binary contractions")
        labels, output = node.attrs["labels"], node.attrs["output"]
        if any(len(set(word)) != len(word) for word in labels):
            raise ValueError("ladder row-norm proof does not admit repeated labels")
        summed = (set().union(*(set(word) for word in labels)) - set(output)) | {
            output[axis] for axis in axes
        }
        shared = set(labels[0]) & set(labels[1]) & summed
        amplitudes = [
            position
            for position, child in enumerate(node.inputs)
            if amplitude_dependent(child)
        ]
        owner = amplitudes[0] if len(amplitudes) == 1 else 1
        polynomials = [
            row_polynomial(
                child,
                tuple(
                    axis
                    for axis, label in enumerate(labels[position])
                    if label in summed and (label not in shared or position == owner)
                ),
            )
            for position, child in enumerate(node.inputs)
        ]
        return Counter(
            {
                monomial: abs(Fraction(*node.attrs["coefficient"])) * value
                for monomial, value in multiply_polynomials(*polynomials).items()
            }
        )

    @cache
    def polynomial(node: Node) -> Counter[tuple[str, ...]]:
        if node.op == "input":
            if node.attrs["name"] != "df_tau":
                raise ValueError("ladder majorant must terminate at tau")
            return Counter({(): Fraction(1)})
        if node.op == "transpose":
            return polynomial(node.inputs[0])
        if node.op == "add":
            result: Counter[tuple[str, ...]] = Counter()
            for child, coefficient in zip(
                node.inputs, node.attrs["coefficients"], strict=True
            ):
                for monomial, value in polynomial(child).items():
                    result[monomial] += abs(Fraction(*coefficient)) * value
            return result
        if node.op != "einsum" or len(node.inputs) != 2:
            raise ValueError("ladder majorant requires binary linear contractions")
        dependent = [
            position
            for position, child in enumerate(node.inputs)
            if tau_dependent(child)
        ]
        if len(dependent) != 1:
            raise ValueError("ladder majorant requires exactly one tau operand")
        carrier = dependent[0]
        factor = 1 - carrier
        labels = node.attrs["labels"]
        reductions = set().union(*(set(word) for word in labels)) - set(
            node.attrs["output"]
        )
        if not reductions <= set(labels[factor]):
            raise ValueError("ladder majorant lost a contracted factor axis")
        axes = tuple(
            position
            for position, label in enumerate(labels[factor])
            if label in reductions
        )
        return Counter(
            {
                monomial: value * abs(Fraction(*node.attrs["coefficient"]))
                for monomial, value in multiply_polynomials(
                    polynomial(node.inputs[carrier]),
                    row_polynomial(node.inputs[factor], axes),
                ).items()
            }
        )

    positive = polynomial(program.outputs[LADDER_OUTPUT])
    amplitude = [name for name, (source, _) in norms.items() if source == "t1"]
    if len(amplitude) != 1:
        raise ValueError("ladder majorant requires one current-amplitude norm")
    amplitude_name = amplitude[0]
    scalar = TensorSpec((), role="parameter", differentiable=False)
    inputs = {
        name: input_tensor(name, scalar) for name in norms if name != amplitude_name
    }
    terms: dict[int, list[Node]] = {0: [], 1: []}
    for monomial, coefficient in sorted(positive.items()):
        degree = monomial.count(amplitude_name)
        if degree not in terms:
            raise ValueError("ladder majorant is no longer affine in T1 norm")
        geometry = tuple(name for name in monomial if name != amplitude_name)
        if not geometry:
            value = constant(coefficient, scalar)
        else:
            value = einsum(
                ",".join("" for _ in geometry) + "->",
                *(inputs[name] for name in geometry),
                coefficient=coefficient,
            )
        terms[degree].append(value)
    coefficients = Program(
        {
            f"coefficient_{degree}": add(*values) if values else constant(0.0, scalar)
            for degree, values in terms.items()
        },
        provenance={
            "original_ladder_graph": program.logical_hash,
            "majorant": "absolute coefficients and complete contracted-factor row norms",
        },
    )
    return LadderPairMajorant(
        coefficients,
        tuple(
            (name, source, axes)
            for name, (source, axes) in sorted(norms.items())
            if name != amplitude_name
        ),
        (amplitude_name, norms[amplitude_name][1]),
    )


def fold_occupied_ladder_pairs(program: Program) -> Program:
    """Replace only free i/j spectators of the existing virtual ladder graph.

    Every einsum must carry both free labels through exactly one operand;
    neither may be reduced or shared with another operand. The proof terminates
    at the existing df_tau input. All other auxiliary outputs remain unchanged.
    Reconstruction of j/i swaps virtual axes, using tau_ijab = tau_jiba.
    Diagonal occupied pairs retain complete virtual matrices.
    """
    root = program.outputs[LADDER_OUTPUT]
    if len(root.spec.indices) != 4 or tuple(
        index.space.kind for index in root.spec.indices
    ) != ("occupied", "occupied", "virtual", "virtual"):
        raise ValueError("ladder folding requires the original ijab output")
    occupied = root.spec.indices[0].space
    if root.spec.indices[1].space != occupied:
        raise ValueError("ladder spectator populations differ")
    pair = Index(
        "occupied_pair",
        IndexSpace("occupied_pairs", "pair", occupied.size * (occupied.size + 1) // 2),
    )

    def indices(node: Node, axes: tuple[int, int]) -> tuple[Index, ...]:
        return tuple(
            pair if position == axes[0] else index
            for position, index in enumerate(node.spec.indices)
            if position != axes[1]
        )

    @cache
    def fold(node: Node, axes: tuple[int, int]) -> Node:
        if axes[0] == axes[1] or any(
            node.spec.indices[axis].space != occupied for axis in axes
        ):
            raise ValueError("ladder folding lost a free occupied spectator")
        if node.op == "input":
            if node.attrs["name"] != "df_tau" or axes not in ((0, 1), (1, 0)):
                raise ValueError("ladder folding requires a tau-only spectator source")
            if tuple(index.space.kind for index in node.spec.indices) != (
                "occupied",
                "occupied",
                "virtual",
                "virtual",
            ):
                raise ValueError("ladder tau must retain both virtual axes")
            packed = input_tensor(
                PAIRED_TAU_INPUT,
                replace(
                    node.spec,
                    indices=(pair, *node.spec.indices[2:]),
                    symmetries=(),
                ),
            )
            return packed if axes == (0, 1) else transpose(packed, (0, 2, 1))
        if node.op == "add":
            return add(
                *(fold(child, axes) for child in node.inputs),
                coefficients=tuple(
                    Fraction(*coefficient) for coefficient in node.attrs["coefficients"]
                ),
            )
        if node.op == "transpose":
            permutation = node.attrs["axes"]
            child_axes = (permutation[axes[0]], permutation[axes[1]])
            folded_child = fold(node.inputs[0], child_axes)
            remaining = [
                axis for axis in range(len(permutation)) if axis != child_axes[1]
            ]
            return transpose(
                folded_child,
                tuple(
                    remaining.index(axis)
                    for position, axis in enumerate(permutation)
                    if position != axes[1]
                ),
            )
        if node.op != "einsum":
            raise ValueError(f"unsupported ladder spectator primitive: {node.op}")
        labels, output = node.attrs["labels"], node.attrs["output"]
        spectators = (output[axes[0]], output[axes[1]])
        carrier = [
            position
            for position, word in enumerate(labels)
            if set(spectators) & set(word)
        ]
        if len(carrier) != 1 or any(
            labels[carrier[0]].count(label) != 1 for label in spectators
        ):
            raise ValueError("ladder spectators must be free in one operand only")
        paired_label = max(label for word in labels for label in word) + 1

        def word(original: tuple[int, ...]) -> str:
            return "".join(
                ascii_letters[paired_label if label == spectators[0] else label]
                for label in original
                if label != spectators[1]
            )

        children = tuple(
            fold(child, tuple(labels[position].index(label) for label in spectators))
            if position == carrier[0]
            else child
            for position, child in enumerate(node.inputs)
        )
        result = einsum(
            ",".join(word(original) for original in labels) + "->" + word(output),
            *children,
            coefficient=Fraction(*node.attrs["coefficient"]),
        )
        if result.spec.indices != indices(node, axes):
            expected = indices(node, axes)
            if tuple(index.space for index in result.spec.indices) != tuple(
                index.space for index in expected
            ):
                raise ValueError("folded ladder changed a nonspectator population")
        return result

    folded = fold(root, (0, 1))
    _prove_ladder_pair_symmetry(root)
    return Program(
        {
            name: folded if name == LADDER_OUTPUT else node
            for name, node in program.outputs.items()
        },
        provenance={
            **program.provenance,
            "occupied_ladder_pairs": "i<=j; dense virtual blocks; reflected j/i transposes a/b",
            "occupied_pair_contract": "tau simultaneous pair symmetry must be admitted by the native owner",
            "occupied_pair_original_graph": program.logical_hash,
            "native_execution_order": "dependencies",
        },
    )

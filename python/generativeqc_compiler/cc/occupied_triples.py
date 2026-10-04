"""Occupied-domain DF triples moments and fused scalar energy algebra.

The audited virtual-triangle inventory is rewritten as a full virtual sum.
Its six virtual Z rows have equal sums after a dummy-index relabeling, so one
row suffices. An occupied triangle then groups *all six* occupied permutations
with their 6/2/1 multiplicity; an individual occupied permutation is not an
equivalent contribution. Only six virtual W cubes are required for a tile.
"""

from __future__ import annotations

import typing
from fractions import Fraction

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    PrecisionDirective,
    Program,
    TensorSpec,
    add,
    divide,
    einsum,
    input_tensor,
    lower_precision,
    multiply,
)

from .triples import _LABELS, INVENTORY_HASH, OP, R3, SLOW_TABLE, V_TERMS, VP, W_TERMS

if typing.TYPE_CHECKING:
    from generativeqc_compiler.tensor import Node

PERMUTATIONS = tuple(VP[label] for label in _LABELS)
DF_TRIPLES_W_FP32_QUALIFICATION = "issue1764/df-triples-w-fp32-candidate-v1"


def inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
    """Map a transpose's output coordinates back to its input coordinates."""
    return tuple(permutation.index(axis) for axis in range(len(permutation)))


def occupied_label(outer: tuple[int, ...], transpose: tuple[int, ...]) -> str:
    """Identify the stored W/V seed selected by an occupied transpose."""
    coordinates = tuple(outer[axis] for axis in inverse(transpose))
    return _LABELS[PERMUTATIONS.index(coordinates)]


def _input(name: str, *indices: Index) -> Node:
    return input_tensor(
        name, TensorSpec(tuple(indices), role="parameter", differentiable=True)
    )


def df_panel_program(virtuals: int, auxiliaries: int) -> Program:
    """One occupied index's [a,b,f] panel from its B_ov row and symmetric B_vv.

    B_vv pair symmetry makes this (ia|fb), matching the audited W1 seed.
    The native owner supplies a strided B_ov view; this graph does not retain
    the complete [i,a,f,b] integral tensor or introduce an auxiliary W loop.
    """
    if any(type(x) is not int or x <= 0 for x in (virtuals, auxiliaries)):
        raise ValueError("positive DF panel extents required")
    vir = IndexSpace("occupied_tile_virtual", "virtual", virtuals)
    aux = IndexSpace("occupied_tile_auxiliary", "auxiliary", auxiliaries)
    a, b, f = (Index(name, vir) for name in ("a", "b", "f"))
    Q = Index("Q", aux)
    bov = _input("bov_i", Q, a)
    bvv = _input("bvv", Q, b, f)
    return Program(
        {"panel": einsum("Qa,Qbf->abf", bov, bvv)},
        provenance={"inventory": INVENTORY_HASH, "pair_contract": "symmetric B_vv"},
    )


def moment_program(occupied: int, virtuals: int) -> Program:
    """Audited W/V seeds with (i,j,k) fixed and (a,b,c) free.

    Parameter suffixes specify views of the canonical inputs: t2_kj[c,f],
    ovoo_ij[a,m], t2_mk[m,b,c], ovov_ij[a,b], t1_k[c], t2_ij[a,b], fov_k[c].
    A panel [a,b,f] holds ovvv[i,a,f,b]. Both W contractions are direct GEMMs;
    the native epilogue evaluates the two V products without storing V cubes.
    """
    if any(type(x) is not int or x <= 0 for x in (occupied, virtuals)):
        raise ValueError("positive occupied/virtual extents required")
    occ = IndexSpace("occupied_tile_occupied", "occupied", occupied)
    vir = IndexSpace("occupied_tile_virtual", "virtual", virtuals)
    a, b, c, f = (Index(name, vir) for name in ("a", "b", "c", "f"))
    m = Index("m", occ)
    w1 = einsum("abf,cf->abc", _input("panel", a, b, f), _input("t2_kj", c, f))
    w2 = einsum("am,mbc->abc", _input("ovoo_ij", a, m), _input("t2_mk", m, b, c))
    v1 = einsum("ab,c->abc", _input("ovov_ij", a, b), _input("t1_k", c))
    v2 = einsum("ab,c->abc", _input("t2_ij", a, b), _input("fov_k", c))
    return Program(
        {
            "w": add(w1, w2, coefficients=tuple(term[0] for term in W_TERMS)),
            "v": add(v1, v2, coefficients=tuple(term[0] for term in V_TERMS)),
        },
        provenance={
            "inventory": INVENTORY_HASH,
            "domain": "one ordered occupied triple",
        },
    )


def w_fp32_candidate_program(occupied: int, virtuals: int) -> Program:
    """Lower only the two W GEMM reductions to an explicit FP32 candidate.

    Inputs, the W sum, every V term, denominator algebra and all published
    outputs remain FP64. This is a compiler-owned qualification candidate for
    #1764, not a production selection policy: a method owner must still compare
    complete endpoints against the strict-FP64 schedule before promotion.
    """
    strict = moment_program(occupied, virtuals)
    w = strict.outputs["w"]
    if (
        w.op != "add"
        or len(w.inputs) != 2
        or any(node.op != "einsum" for node in w.inputs)
    ):
        raise ValueError("occupied triples W candidate requires two direct reductions")
    names = strict.debug_names
    directives = {
        names[node]: PrecisionDirective(
            storage_dtype="float32",
            compute_dtype="float32",
            accumulation_dtype="float32",
            qualification=DF_TRIPLES_W_FP32_QUALIFICATION,
        )
        for node in w.inputs
    }
    return lower_precision(strict, directives)


def v_scalar_program() -> Program:
    """Scalar form of the same V seed, embedded in the energy epilogue."""
    return Program(
        {
            "v": add(
                multiply(_input("ovov"), _input("t1")),
                multiply(_input("t2"), _input("fov")),
                coefficients=tuple(term[0] for term in V_TERMS),
            )
        },
        provenance={"inventory": INVENTORY_HASH, "domain": "one V element"},
    )


def energy_scalar_program() -> Program:
    """Fused contribution at (a,b,c) for one triangular occupied tile.

    ``w_OCC_VIR`` selects a stored occupied seed and a virtual permutation;
    ``v_OCC`` is the scalar V seed at the unpermuted (a,b,c). The denominator
    includes the occupied 6/2/1 multiplicity. Every occupied permutation is
    retained, including repeats: division by that multiplicity counts each
    ordered occupied triple exactly once. The virtual sum covers the entire
    cube and is never independently folded to a triangle here.
    """
    ws = {(occ, vir): _input(f"w_{occ}_{vir}") for occ in _LABELS for vir in _LABELS}
    vs = {occ: _input(f"v_{occ}") for occ in _LABELS}
    products = []
    for outer in PERMUTATIONS:
        projected = add(
            *(
                add(
                    ws[occupied_label(outer, permutation), "abc"],
                    vs[occupied_label(outer, permutation)],
                    coefficients=(1, Fraction(1, 2)),
                )
                for _, permutation in R3
            ),
            coefficients=tuple(coefficient for coefficient, _ in R3),
        )
        left = add(
            *(
                ws[occupied_label(outer, OP[order]), label]
                for label, order in SLOW_TABLE["abc"]
            )
        )
        products.append(multiply(left, projected))
    energy = divide(
        add(*products, coefficients=(2,) * len(products)), _input("denominator")
    )
    return Program(
        {"energy": energy},
        provenance={
            "inventory": INVENTORY_HASH,
            "domain": "occupied triangle, all six occupied permutations, full virtual cube",
            "virtual_row_reduction": "equal full-domain sums by dummy-index relabeling",
        },
    )

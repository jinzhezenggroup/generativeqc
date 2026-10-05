"""Same-frame DF numerical preconditioner for an unchanged exact RHF action.

With K_ia=x and K_ai=-x, the canonical physical action is
gap*x + 4(ia|jb)x - (ij|ab)x - (ib|ja)x. Retain the Coulomb term
and the diagonal of both exchange terms. This approximate matrix is only a
solver accelerator; its factors never replace an exact response action.
"""

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    broadcast,
    einsum,
    input_tensor,
)


def build_rhf_df_preconditioner(nocc: int, nvir: int, naux: int) -> Program:
    """Return D and U for D+U.T@U, with U in Q-major ia order.

    No (ov)^2 intermediate is constructed. Positivity/Cholesky admission is
    owned by the bounded numerical inverse, not by clipping this expression.
    """
    if any(type(n) is not int or n < 1 for n in (nocc, nvir, naux)):
        raise ValueError("RHF DF preconditioner dimensions must be positive")
    occ = IndexSpace("rhf_preconditioner_occ", "occupied", nocc)
    vir = IndexSpace("rhf_preconditioner_vir", "virtual", nvir)
    aux = IndexSpace("rhf_preconditioner_aux", "batch", naux)
    i, j = (Index(label, occ) for label in "ij")
    a, b = (Index(label, vir) for label in "ab")
    q = Index("Q", aux)

    def value(name: str, indices: tuple[Index, ...]) -> Node:
        return input_tensor(name, TensorSpec(indices, role="input"))

    eo, ev = value("eps_o", (i,)), value("eps_v", (a,))
    boo = value("boo", (q, i, j))
    bov = value("bov", (q, i, a))
    bvv = value("bvv", (q, a, b))
    gap = add(
        broadcast(ev, (i, a), (1,)), broadcast(eo, (i, a), (0,)), coefficients=(1, -1)
    )
    exchange = add(einsum("Qii,Qaa->ia", boo, bvv), einsum("Qia,Qia->ia", bov, bov))
    return Program(
        {
            "diagonal": add(gap, exchange, coefficients=(1, -1)),
            "low_rank": add(bov, coefficients=(2,)),
        },
        provenance={
            "purpose": "numerical preconditioner only; exact physical RHF operator unchanged",
            "matrix": "diag(gap-(ii|aa)-(ia|ia)) + 4*(ia|jb)",
            "frame": "qualified DF factors in the exact response reference's canonical frame",
        },
    )

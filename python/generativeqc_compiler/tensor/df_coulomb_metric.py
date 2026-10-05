"""Streamed DF Coulomb metric contractions in the existing immutable TensorIR."""

from __future__ import annotations

from .ir import add, einsum, input_tensor
from .program import Program
from .types import Index, IndexSpace, TensorSpec


def metric_program(naux: int, panel: int, operation: str) -> Program:
    """Keep charge accumulation, eigendirections and padded panels explicit."""
    if any(type(n) is not int or n <= 0 for n in (naux, panel)) or panel > naux:
        raise ValueError("positive bounded metric panel dimensions required")
    q = Index("q", IndexSpace("auxiliary", "auxiliary", naux))
    p = Index("p", IndexSpace("panel", "auxiliary", panel))
    e = Index("e", IndexSpace("eigendirection", "auxiliary", naux))
    if operation == "metric_charge":
        axes, vector_axes, expression = (p, q), (p,), "pq,p->q"
    elif operation == "metric_potential":
        axes, vector_axes, expression = (q, p), (q,), "qp,q->p"
    elif operation == "metric_project":
        axes, vector_axes, expression = (e, q), (q,), "eq,q->e"
    elif operation == "metric_rotate":
        axes, vector_axes, expression = (e, q), (e,), "eq,e->q"
    else:
        raise ValueError("unknown Coulomb metric operation")
    matrix = input_tensor("matrix", TensorSpec(axes, role="input"))
    vector = input_tensor("vector", TensorSpec(vector_axes, role="input"))
    product = einsum(expression, matrix, vector)
    if operation == "metric_charge":
        seed = input_tensor("seed", TensorSpec(product.spec.indices, role="input"))
        return Program({"result": add(seed, product)})
    return Program({"result": product})

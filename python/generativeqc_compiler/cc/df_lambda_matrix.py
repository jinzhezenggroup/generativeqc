"""Bounded Q batches of the existing staged DF adjoint, without new equations."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from string import ascii_letters

from generativeqc_compiler.tensor import Index, IndexSpace, Node, Program, optimize
from generativeqc_compiler.tensor.ir import (
    add,
    broadcast,
    einsum,
    input_tensor,
    transpose,
)

from .df_gemm import pack_df_contractions


def batch_auxiliary_program(program: Program, batch_size: int) -> Program:
    """Lift only bov/bvv-dependent values to a leading, explicit Q batch axis.

    Amplitudes and response seeds stay shared. Every added array is a normal
    IR node, so packing/broadcast storage enters the existing liveness planner.
    Outputs retain Q: the generated consumer fuses their ordered Q sum with
    accumulation, while factor cotangents are published as separate Q rows.
    Unsupported primitives fail closed rather than guessing broadcast rules.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("DF auxiliary batch must be positive")
    axis = Index("Q", IndexSpace("df_auxiliary_batch", "batch", batch_size))
    mapped: dict[Node, Node] = {}
    batched: dict[Node, bool] = {}
    for node in program.dependency_order:
        inputs = tuple(mapped[x] for x in node.inputs)
        varying = (
            node.attrs["name"] in ("bov", "bvv")
            if node.op == "input"
            else any(batched[x] for x in node.inputs)
        )
        batched[node] = varying
        if not varying:
            mapped[node] = node
        elif node.op == "input":
            # Original symmetry axes describe an unbatched factor. No orbital
            # symmetry is required for batching or matrix-layout recognition.
            spec = replace(node.spec, indices=(axis, *node.spec.indices), symmetries=())
            mapped[node] = input_tensor(node.attrs["name"], spec)
        elif node.op == "einsum":
            labels = node.attrs["labels"]
            q = ascii_letters[max(i for word in labels for i in word) + 1]
            words = [
                (q if batched[source] else "") + "".join(ascii_letters[i] for i in word)
                for source, word in zip(node.inputs, labels, strict=True)
            ]
            output = q + "".join(ascii_letters[i] for i in node.attrs["output"])
            mapped[node] = einsum(
                ",".join(words) + "->" + output,
                *inputs,
                coefficient=Fraction(*node.attrs["coefficient"]),
            )
        elif node.op == "add":
            indices = next(
                value.spec.indices
                for old, value in zip(node.inputs, inputs, strict=True)
                if batched[old]
            )
            expanded = tuple(
                value
                if batched[old]
                else broadcast(value, indices, tuple(range(1, len(indices))))
                for old, value in zip(node.inputs, inputs, strict=True)
            )
            mapped[node] = add(
                *expanded,
                coefficients=tuple(Fraction(*c) for c in node.attrs["coefficients"]),
            )
        elif node.op == "transpose":
            mapped[node] = transpose(
                inputs[0], (0, *(i + 1 for i in node.attrs["axes"]))
            )
        else:
            raise ValueError(f"unsupported DF auxiliary batch primitive: {node.op}")
    result = optimize(
        Program({name: mapped[node] for name, node in program.outputs.items()})
    )
    return Program(
        result.outputs,
        provenance={
            **program.provenance,
            "df_auxiliary_batch": "explicit Q axis; shared amplitude/seed inputs",
            "native_execution_order": "dependencies",
        },
    )


def matrix_program(program: Program, *, batch_size: int | None = None) -> Program:
    """Pack scalar or Q-batched staged programs for ordinary/strided FP64 GEMM."""
    source = (
        program if batch_size is None else batch_auxiliary_program(program, batch_size)
    )
    return pack_df_contractions(source, allow_batch=True)

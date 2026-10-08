"""Pack DF binary contractions without changing the audited contraction tree.

The bounded residual already forbids three/four-virtual intermediates. Packing
only permutes existing operand/result axes, so that storage invariant survives.
All extra arrays are ordinary IR nodes charged by the native arena planner.
"""

from __future__ import annotations

from fractions import Fraction
from string import ascii_letters

from generativeqc_compiler.tensor import Node, Program, einsum, optimize
from generativeqc_compiler.tensor.cuda_gemm import gemm_contract
from generativeqc_compiler.tensor.ir import _infer, transpose


def _matrix_operand(
    value: Node,
    labels: tuple[int, ...],
    batch: tuple[int, ...],
    rows: tuple[int, ...],
    columns: tuple[int, ...],
) -> tuple[Node, tuple[int, ...]]:
    """Fold permutation views into labels before deciding whether to pack.

    Native GEMM accepts either row/column group order via its transpose flag.
    Only whole matrix groups may swap; the order within a flattened group and
    the leading batch axes must remain exact, even when extents happen to match.
    """
    while value.op == "transpose":
        axes = value.attrs["axes"]
        labels = tuple(labels[axes.index(axis)] for axis in range(len(axes)))
        value = value.inputs[0]
    order = batch + rows + columns
    if labels in (order, batch + columns + rows):
        return value, labels
    axes = tuple(labels.index(label) for label in order)
    return transpose(value, axes), order


def pack_df_contractions(program: Program, *, allow_batch: bool = False) -> Program:
    """Expose matrix products with transpose-aware, explicitly budgeted layouts.

    Repeated indices, one-sided reductions, multi-operand contractions and
    elementwise products retain their original scalar lowering. Never infer
    spin/orbital symmetry or remove a contraction label from shape equality.
    """
    mapped: dict[Node, Node] = {}
    for node in program.dependency_order:
        inputs = tuple(mapped[x] for x in node.inputs)
        # Packing preserves domains but alpha-renames contraction indices. Infer
        # dependent metadata just as the shared IR rewrite passes do.
        current = (
            node
            if inputs == node.inputs
            else Node(
                node.op,
                inputs,
                _infer(node.op, inputs, node.attrs, node.spec),
                node.attributes,
            )
        )
        contract = gemm_contract(current)
        if (
            contract is None
            or (contract.batch_labels and not allow_batch)
            or not contract.m_labels
            or not contract.n_labels
            or not contract.k_labels
        ):
            mapped[node] = current
            continue

        def packed(
            value: Node, original: tuple[int, ...], order: tuple[int, ...]
        ) -> Node:
            axes = tuple(original.index(label) for label in order)
            return value if axes == tuple(range(len(axes))) else transpose(value, axes)

        # Lambda AD introduces permutation nodes whose labels can be composed
        # into the GEMM binding instead of materializing another seed buffer.
        a, a_labels = _matrix_operand(
            current.inputs[0],
            contract.a_labels,
            contract.batch_labels,
            contract.m_labels,
            contract.k_labels,
        )
        b, b_labels = _matrix_operand(
            current.inputs[1],
            contract.b_labels,
            contract.batch_labels,
            contract.k_labels,
            contract.n_labels,
        )

        def word(labels: tuple[int, ...]) -> str:
            return "".join(ascii_letters[label] for label in labels)

        result = einsum(
            word(a_labels) + "," + word(b_labels) + "->" + word(contract.c_order),
            a,
            b,
            coefficient=Fraction(*current.attrs["coefficient"]),
        )
        mapped[node] = packed(result, contract.c_order, contract.output_labels)
    result = optimize(
        Program({name: mapped[value] for name, value in program.outputs.items()})
    )
    if any(
        sum(i.space.kind == "virtual" for i in n.spec.indices) > 2
        for n in result.live_nodes
    ):
        raise ValueError("DF GEMM packing reconstructed an omitted virtual block")
    return Program(
        result.outputs,
        provenance={
            **program.provenance,
            "native_execution_order": "dependencies",
            "df_binary_layout": "packed matrix axes with transpose-aware operands",
        },
    )

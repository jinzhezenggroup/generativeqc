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


def pack_df_contractions(program: Program, *, allow_batch: bool = False) -> Program:
    """Expose unbatched matrix products using explicit, budgeted transposes.

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

        a = packed(current.inputs[0], contract.a_labels, contract.a_order)
        b = packed(current.inputs[1], contract.b_labels, contract.b_order)

        def word(labels: tuple[int, ...]) -> str:
            return "".join(ascii_letters[label] for label in labels)

        result = einsum(
            word(contract.a_order)
            + ","
            + word(contract.b_order)
            + "->"
            + word(contract.c_order),
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
            "df_binary_layout": "explicit packed matrix axes",
        },
    )

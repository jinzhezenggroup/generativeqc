"""Explicit low-precision scratch for DF-CCSD preconvergence, not publication."""

from __future__ import annotations

from dataclasses import replace

from generativeqc_compiler.common.precision import PrecisionDirective
from generativeqc_compiler.tensor import Node, Program
from generativeqc_compiler.tensor.precision import lower_precision

DF_CC_WARMUP_QUALIFICATION = "df-ccsd/fp32-auxiliary-warmup-v1"


def auxiliary_warmup_program(source: Program) -> Program:
    """Retype the existing auxiliary graph without changing its contractions.

    Native staging owns FP64-to-FP32 input conversions. Ordered Q accumulation
    widens the outputs to FP64. This program must never supply a convergence
    certificate: the controller switches permanently to the original FP64
    graph, clears mixed DIIS history and performs independent physical replay.
    """
    if any(node.spec.dtype != "float64" for node in source.live_nodes):
        raise ValueError("DF auxiliary warmup requires a uniform FP64 source")
    mapping: dict[Node, Node] = {}
    for node in source.nodes:
        mapping[node] = Node(
            node.op,
            tuple(mapping[value] for value in node.inputs),
            replace(node.spec, dtype="float32"),
            node.attributes,
        )
    candidate = Program(
        {name: mapping[node] for name, node in source.outputs.items()},
        provenance={
            **source.provenance,
            "fp64_source_equation": source.logical_hash,
            "native_precision_boundary": DF_CC_WARMUP_QUALIFICATION,
        },
    )
    directive = PrecisionDirective(
        "float32", "float32", "float32", DF_CC_WARMUP_QUALIFICATION
    )
    return lower_precision(
        candidate,
        {
            candidate.debug_names[node]: directive
            for node in candidate.live_nodes
            if node.op not in ("input", "constant", "cast")
        },
    )

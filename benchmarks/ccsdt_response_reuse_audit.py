"""Inspect existing CCSD(T) response programs with the existing exact CSE pass.

This is an opt-in source audit, not an execution schedule or profitability
model. The groups below have explicitly reviewed native input bindings. Equal
input names in unrelated programs are not evidence of equal physical values.
Runtime profiles must establish which programs execute and bound their cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from generativeqc_compiler.cc.occupied_triples_fock import resolvent_scalar_program
from generativeqc_compiler.cc.occupied_triples_response import (
    fused_tile_program,
    moment_vjp,
    panel_vjp,
)
from generativeqc_compiler.tensor import Program, optimize
from generativeqc_compiler.tensor.precision import precision_execution_contracts
from generativeqc_compiler.tensor.program import node_hashes

from benchmarks._retention import raw_output_path
from tools.generate_df_lambda import staged_programs
from tools.generate_rccsd_native import REPRESENTATIVE

ROOT = Path(__file__).resolve().parents[1]


def inventory(program: Program, extents: dict[str, int]) -> dict:
    """Count logical work, retaining the code generator's contraction trees.

    Runtime extents scale an already selected template; they do not rerun
    contraction planning at a different representative shape. Summands count
    scalar product terms, not FLOPs, instructions or measured memory traffic.
    """
    operations = Counter(node.op for node in program.live_nodes if node.op != "input")
    summands = 0
    for node in program.live_nodes:
        if node.op != "einsum":
            continue
        dimensions = {}
        for operand, labels in zip(node.inputs, node.attrs["labels"], strict=True):
            for axis, label in zip(operand.spec.indices, labels, strict=True):
                size = extents.get(axis.space.kind, axis.space.size)
                if label in dimensions and dimensions[label] != size:
                    raise ValueError(
                        "inconsistent runtime extent for a contraction label"
                    )
                dimensions[label] = size
        summands += math.prod(dimensions.values())
    return {
        "operations": dict(sorted(operations.items())),
        "contraction_summands": summands,
        "outputs": sorted(program.outputs),
    }


def compare(programs: dict[str, Program], extents: dict[str, int]) -> dict:
    """Expose a reviewed multi-output region to ordinary TensorIR exact CSE.

    These frontends are strict FP64. Do not generalize this composition to
    precision schedules: their execution provenance must be composed too.
    The result only bounds sharing within the supplied immutable binding epoch;
    it does not establish a safe retained lifetime or a profitable CUDA plan.
    """
    prepared = {name: optimize(program) for name, program in programs.items()}
    if any(
        precision_execution_contracts(program) for program in prepared.values()
    ) or any(
        node.spec.dtype != "float64"
        for program in prepared.values()
        for node in program.live_nodes
    ):
        raise ValueError("this audit only admits the reviewed strict-FP64 frontends")
    joint = optimize(
        Program(
            {
                owner + "__" + name: node
                for owner, program in prepared.items()
                for name, node in program.outputs.items()
            }
        )
    )
    separate = {name: inventory(program, extents) for name, program in prepared.items()}
    after = inventory(joint, extents)
    before = sum((Counter(item["operations"]) for item in separate.values()), Counter())
    # Existing TensorIR content addresses include ordered operands, coefficients,
    # indices, dtype and input names. No symmetry/reassociation rule is inferred.
    occurrences = defaultdict(list)
    nodes = {}
    for name, program in prepared.items():
        for node, digest in node_hashes(program.live_nodes).items():
            if node.op in ("input", "constant"):
                continue
            occurrences[digest].append(name)
            nodes[digest] = node
    candidates = []
    for digest, consumers in occurrences.items():
        if len(set(consumers)) < 2:
            continue
        node = nodes[digest]
        candidates.append(
            {
                "identity": digest,
                "operation": node.op,
                "shape": [
                    extents.get(i.space.kind, i.space.size) for i in node.spec.indices
                ],
                "consumers": consumers,
            }
        )
    return {
        "individual": separate,
        "composed": after,
        "eliminated_operations": dict(
            sorted((before - Counter(after["operations"])).items())
        ),
        "eliminated_contraction_summands": sum(
            item["contraction_summands"] for item in separate.values()
        )
        - after["contraction_summands"],
        "exact_common_nodes": sorted(candidates, key=lambda item: item["identity"]),
        "program_hashes": {
            name: program.logical_hash for name, program in prepared.items()
        },
    }


def audit(occupied: int, virtuals: int, auxiliaries: int) -> dict:
    """Audit one runtime shape using the templates consumed by native emission."""
    if min(occupied, virtuals, auxiliaries) < 1:
        raise ValueError("positive occupied, virtual and auxiliary extents required")
    extents = {"occupied": occupied, "virtual": virtuals, "auxiliary": auxiliaries}
    scalar = {
        "scalar_response": fused_tile_program(),
        "resolvent": resolvent_scalar_program(),
    }
    staged = staged_programs()
    parameters = {
        name: program
        for name, program in staged.items()
        if name.startswith("staged_parameter_")
    }
    # The final factor publication recomputes core seeds after uploading the
    # converged Lambda vector. Intermediate Krylov seeds are different epochs.
    final_core = {**parameters, "final_core_seeds": staged["staged_core"]}
    common_v = set.intersection(
        *(
            {
                node.attrs["name"]
                for node in program.live_nodes
                if node.op == "input" and node.attrs["name"].startswith("v_")
            }
            for program in scalar.values()
        )
    )
    return {
        "schema": "generativeqc.ccsdt-response-reuse-audit.v1",
        "runtime_extents": extents,
        "codegen_representative": list(REPRESENTATIVE),
        "interpretation": (
            "Logical exact-CSE ceilings only. Native physical bindings, ordered accumulation, "
            "stream/lifetime and device measurements must qualify any implementation. "
            "No original-equation audit or Krylov iteration is included as reusable work."
        ),
        "local_pullbacks": compare(
            {
                "w": moment_vjp(2, 3, "w"),
                "v": moment_vjp(2, 3, "v"),
                "panel": panel_vjp(3, 4),
            },
            extents,
        ),
        "scalar_and_resolvent": {
            **compare(scalar, extents),
            "common_pointwise_v_inputs": sorted(common_v),
            "pointwise_v_binding": (
                "Each common v_OCC_VIR names the same ordered occupied/virtual views in "
                "fused_response_cuda_source and fock_cuda_source. Each evaluates the existing "
                "v_scalar_program separately (two products and one sum). Denominator/gap "
                "inputs remain distinct: ordered subtraction and multiplicity are unchanged."
            ),
        },
        "retained_parameters": compare(parameters, extents),
        "parameters_and_final_core": compare(final_core, extents),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occupied", type=int, required=True)
    parser.add_argument("--virtuals", type=int, required=True)
    parser.add_argument("--auxiliaries", type=int, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    result = audit(args.occupied, args.virtuals, args.auxiliaries)
    result["source_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    # These hashes are one-off evidence, never a required-CI source fingerprint.
    paths = [
        "src/cc/df_triples_cuda.cu",
        "src/cc/df_lambda_cuda.cu",
        "src/methods/df_ccsdt_force.cu",
        "tools/generate_df_occupied_triples.py",
        "tools/generate_df_lambda.py",
        "python/generativeqc_compiler/cc/occupied_triples_response.py",
        "python/generativeqc_compiler/cc/occupied_triples_fock.py",
        "python/generativeqc_compiler/cc/df_lambda_reduction.py",
        "python/generativeqc_compiler/tensor/optimize.py",
    ]
    result["source_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

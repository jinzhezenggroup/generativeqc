"""Report GPU-free CUDA cost evidence from static facts or PTXAS diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from generativeqc_compiler.common.cuda_cost_model import static_cuda_cost
from generativeqc_compiler.common.cuda_resources import (
    KernelResources,
    compiled_gpu_profitability,
    parse_resources,
)
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.gpu_profitability import GpuProfitability


def _ptxas_resources(
    diagnostics: str, architecture: str
) -> tuple[tuple[KernelResources, ...], bool]:
    """Reject incomplete rows and contradictory retained-log provenance."""

    resources = parse_resources(diagnostics)
    declared = re.findall(r"Function properties for (\S+)", diagnostics)
    if not resources:
        raise ValueError("CUDA cost analysis requires PTXAS resource rows")
    if sorted(row.function for row in resources) != sorted(declared):
        raise ValueError("PTXAS log contains incomplete or unsupported resource rows")

    entries = re.findall(
        r"Compiling entry function '([^']+)' for '([^']+)'", diagnostics
    )
    entry_counts = Counter(function for function, _ in entries)
    resource_counts = Counter(row.function for row in resources)
    if entry_counts - resource_counts:
        raise ValueError("PTXAS log contains incomplete or unsupported resource rows")
    architectures = {entry_architecture for _, entry_architecture in entries}
    if architectures - {architecture}:
        raise ValueError(
            f"PTXAS architecture declarations {sorted(architectures)} "
            f"do not match requested target {architecture}"
        )
    # Headerless resource snippets remain useful, but cannot certify which
    # architecture produced them. Every reported function must be accounted for.
    architecture_verified = bool(entries) and entry_counts == resource_counts
    return resources, architecture_verified


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CUDA resource/parallelism screening report without probing "
            "or executing a GPU. The output is not a runtime prediction."
        )
    )
    parser.add_argument("--arch", required=True, help="CUDA architecture, e.g. sm_120")
    parser.add_argument("--block-threads", required=True, type=int)
    parser.add_argument(
        "--ptxas",
        type=Path,
        help="optional PTXAS -v diagnostics; compilation may happen elsewhere",
    )
    parser.add_argument("--grid-blocks", type=int)
    parser.add_argument(
        "--sm-count",
        type=int,
        help="optional target GPU SM count for global-grid saturation",
    )
    parser.add_argument("--traffic-bytes", type=int)
    parser.add_argument("--operations", type=int)
    parser.add_argument("--launches", type=int)
    parser.add_argument("--source-bytes", type=int)
    parser.add_argument(
        "--estimated-registers",
        type=int,
        help="pre-compilation registers/thread estimate when --ptxas is absent",
    )
    parser.add_argument(
        "--estimated-occupancy",
        type=float,
        help="pre-compilation occupancy upper bound in [0, 1]",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    target = cuda_target_info(args.arch)
    ptxas_evidence = None
    if args.ptxas is not None:
        resources, architecture_verified = _ptxas_resources(
            args.ptxas.read_text(encoding="utf-8"), target.architecture
        )
        ptxas_evidence = {
            "architecture": target.architecture if architecture_verified else None,
            "architecture_verified": architecture_verified,
            "functions": [row.function for row in resources],
        }
        profitability = compiled_gpu_profitability(
            resources,
            target,
            args.block_threads,
        )
        profitability = replace(
            profitability,
            semantic_traffic_bytes=args.traffic_bytes,
            arithmetic_operation_count=args.operations,
            launch_count=args.launches,
            source_bytes=args.source_bytes,
        )
    else:
        profitability = GpuProfitability(
            semantic_traffic_bytes=args.traffic_bytes,
            arithmetic_operation_count=args.operations,
            estimated_registers_per_thread=args.estimated_registers,
            estimated_occupancy_upper_bound=args.estimated_occupancy,
            launch_count=args.launches,
            source_bytes=args.source_bytes,
        )

    report = static_cuda_cost(
        profitability,
        target,
        args.block_threads,
        grid_blocks=args.grid_blocks,
        sm_count=args.sm_count,
    )
    if ptxas_evidence is not None and not ptxas_evidence["architecture_verified"]:
        report = replace(
            report,
            diagnostics=(
                *report.diagnostics,
                (
                    "PTXAS architecture is unverified for one or more resource rows; "
                    "the requested target is a caller assumption"
                ),
            ),
        )
    payload = report.to_payload()
    payload["ptxas_evidence"] = ptxas_evidence
    payload["screening_priority"] = report.screening_priority(0)
    payload["profitability"] = profitability.to_payload()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

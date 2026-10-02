#!/usr/bin/env python3
"""Report GPU-free CUDA cost evidence from static facts or PTXAS diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from generativeqc_compiler.common.cuda_cost_model import static_cuda_cost
from generativeqc_compiler.common.cuda_resources import (
    compiled_gpu_profitability,
    parse_resources,
)
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.gpu_profitability import GpuProfitability


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
    if args.ptxas is not None:
        resources = parse_resources(args.ptxas.read_text(encoding="utf-8"))
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
    payload = report.to_payload()
    payload["screening_priority"] = report.screening_priority(0)
    payload["profitability"] = profitability.to_payload()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

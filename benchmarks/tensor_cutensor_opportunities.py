"""Report packed TensorIR contractions that cuTENSOR could execute without packing.

This is a planning benchmark, not a cuTENSOR timing claim.  It keeps the
currently selected CUDA plan unchanged and reports only compiler-proven
stride-addressable opportunities plus the semantic packing traffic that a
future cuTENSOR provider could avoid.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.cc.lambda_equations import build_lambda_programs
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor.cuda_cutensor import cutensor_opportunities
from generativeqc_compiler.tensor.cuda_plan import plan_cuda

from benchmarks._support import raw_output_path, write_result
from tools.generate_rccsd_native import with_jacobi_update

if TYPE_CHECKING:
    from generativeqc_compiler.tensor import Program


def program(kind: str, nocc: int, nvir: int) -> Program:
    if kind == "iteration":
        return with_jacobi_update(
            build_ccsd_program(nocc, nvir, form="shared", diagnostics=False)
        )
    if kind == "lambda-transpose":
        return build_lambda_programs(nocc, nvir, form="shared").residual_vjp.program
    raise ValueError(f"unknown program {kind!r}")


def run(args: argparse.Namespace) -> dict[str, object]:
    target = cuda_target_info(args.architecture)
    plan = plan_cuda(
        program(args.program, args.nocc, args.nvir),
        target,
        max_bytes=args.max_bytes,
    )
    opportunities = cutensor_opportunities(plan)
    avoided = sum(item.packing_bytes_avoided for item in opportunities)
    total_packing = int(plan.semantic_traffic["layout_conversion_bytes"])
    return {
        "schema": "generativeqc.tensor.cutensor-opportunity-benchmark.v1",
        "program": args.program,
        "nocc": args.nocc,
        "nvir": args.nvir,
        "architecture": args.architecture,
        "plan_identity": plan.identity,
        "selected_provider": "existing-cuBLAS/generated-CUDA-plan",
        "execution_changed": False,
        "cutensor_sites": len(opportunities),
        "cutensor_flops": sum(item.flops for item in opportunities),
        "packing_bytes_avoidable": avoided,
        "selected_plan_packing_bytes": total_packing,
        "packing_fraction_avoidable": (
            avoided / total_packing if total_packing else 0.0
        ),
        "sites": [item.to_payload() for item in opportunities],
        "scope": (
            "compiler-proven packed binary contractions only; "
            "no cuTENSOR execution, workspace query, or speedup claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--program",
        choices=("iteration", "lambda-transpose"),
        default="iteration",
    )
    parser.add_argument("--nocc", type=int, default=5)
    parser.add_argument("--nvir", type=int, default=2)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--max-bytes", type=int, default=8 << 30)
    parser.add_argument("--output", type=raw_output_path)
    args = parser.parse_args()
    if args.nocc < 1 or args.nvir < 1 or args.max_bytes < 1:
        parser.error("nocc, nvir and max-bytes must be positive")
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        write_result(args.output, result)


if __name__ == "__main__":
    main()

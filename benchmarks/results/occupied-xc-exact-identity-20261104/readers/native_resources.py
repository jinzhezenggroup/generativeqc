"""Assess changed and fallback launches from the final library's cuobjdump.

Input is ``cuobjdump --dump-resource-usage LIBRARY | c++filt``. Stack and declared
local bytes are preserved separately: this format does not report spill loads
or stores. Analytical launch legality is not measured occupancy, allocator peak,
or a complete force-region profitability qualification.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from generativeqc_compiler.common.cuda_launch import (
    CudaLaunchLimits,
    assess_cuda_launch,
)
from generativeqc_compiler.common.cuda_target import cuda_target_info


def main() -> None:
    """Fail closed if any strict-FP64 repair/fallback launch is missing/illegal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = {
        "occupied_density_products": 256,
        "occupied_density_features": 128,
        "batch_density_products<double, true>": 256,
        "batch_density_features<true>": 128,
        "batch_density_products<double, false>": 256,
        "batch_density_features<false>": 128,
        "density_features<true>": 128,
        "density_features<false>": 128,
    }
    pattern = re.compile(
        r" Function (?P<symbol>[^\n]+):\n\s+REG:(?P<registers>\d+) "
        r"STACK:(?P<stack>\d+) SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)"
    )
    target = cuda_target_info("sm_120")
    limits = CudaLaunchLimits.from_target(target)
    launches = {}
    for match in pattern.finditer(args.resource_dump.read_text()):
        symbol = match["symbol"]
        if "cuda_xc_detail::(anonymous namespace)::" not in symbol:
            continue
        leaf = symbol.split("cuda_xc_detail::(anonymous namespace)::", 1)[1]
        name = leaf.split("(", 1)[0]
        if name not in required:
            continue
        assessment = assess_cuda_launch(
            limits,
            block_threads=required[name],
            registers_per_thread=int(match["registers"]),
            static_shared_bytes=int(match["shared"]),
        )
        if not assessment.legal:
            raise ValueError(f"illegal native launch {name}: {assessment.rejections}")
        launches[name] = {
            "symbol": symbol,
            "threads": required[name],
            "registers": int(match["registers"]),
            "stack_bytes": int(match["stack"]),
            "static_shared_bytes": int(match["shared"]),
            "declared_local_bytes": int(match["local"]),
            "spill_load_bytes": None,
            "spill_store_bytes": None,
            "analytical_launch": asdict(assessment),
        }
    if set(launches) != set(required):
        raise ValueError(f"missing native launches: {set(required) - set(launches)}")
    report = {
        "schema": "generativeqc.occupied-xc-native-launches.v1",
        "target": target.to_payload(),
        "launches": launches,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

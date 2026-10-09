#!/usr/bin/env python3
"""Emit source-bound rank-k qualification portfolios without loading runtime."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from generativeqc_compiler.tensor.scf import density_program, weighted_density_program
from generativeqc_compiler.tensor.symmetric_rank_k import emit_symmetric_rank_k_portfolio
from generativeqc_compiler.tensor.weighted_gram_emit import emit_scalar_stages


def render() -> str:
    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256(
        (root / "src/tensor/cuda_symmetric_rank_k.cuh").read_bytes()
    ).hexdigest()
    bodies = ["#pragma once", "#include <cmath>", '#include "runtime/lowering_binding.hpp"',
              "namespace generativeqc::tensor::rank_k_generated {"]
    stages = emit_scalar_stages("cuda", {
        "energy_weight": "rank_k_energy_weight",
        "weighted_coefficient": "rank_k_scale",
        "contribution": "rank_k_contribution",
        "updated": "rank_k_update",
    })
    bodies.extend(stages.values())
    for name, builder in (
        ("density", density_program),
        ("weighted_density", weighted_density_program),
    ):
        program = builder(1, 3, spin_count=2, orbital_count=5)
        for suffix, order in (("row", "row-major"), ("column", "column-major")):
            bodies.append(
                emit_symmetric_rank_k_portfolio(
                    program, name, source, name=f"rank_k_{name}_{suffix}", order=order
                )
            )
    bodies.append("}  // namespace generativeqc::tensor::rank_k_generated")
    return "\n".join(bodies) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()

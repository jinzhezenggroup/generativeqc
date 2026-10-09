#!/usr/bin/env python3
"""Emit source-bound rank-k qualification portfolios without loading runtime."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp
from generativeqc_compiler.tensor.scf import density_program, weighted_density_program
from generativeqc_compiler.tensor.symmetric_rank_k import (
    emit_symmetric_rank_k_portfolio,
    symmetric_rank_k_scalar_update_program,
)
from generativeqc_compiler.tensor.weighted_gram_emit import emit_scalar_stages

from tools.generate_build_identity import _inventory, _source_identity

_TOOLCHAIN_FILES = (
    "bin/nvcc",
    "bin/ptxas",
    "lib64/libcublas.so.12",
    "lib64/libcublasLt.so.12",
    "lib64/libcudart.so.12",
)


def compiler_identity(root: Path, toolkit_root: Path, host_compiler: Path) -> str:
    """Bind inventoried inputs for the fixed recipe; not a hermetic build proof."""
    root = root.resolve()
    toolkit_root = toolkit_root.resolve()
    host_compiler = host_compiler.resolve()
    manifest = root / "cmake/GenerativeQCSourceIdentity.json"
    toolchain = {}
    for relative in _TOOLCHAIN_FILES:
        path = toolkit_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing rank-k toolchain input: {relative}")
        toolchain[relative] = file_hash(path)
    headers = {
        path.relative_to(toolkit_root).as_posix(): file_hash(path)
        for path in sorted((toolkit_root / "include").rglob("*"))
        if path.is_file()
    }
    if not headers:
        raise FileNotFoundError("missing rank-k toolkit headers")
    toolchain["host_compiler"] = file_hash(host_compiler)
    environment = {}
    for name in (
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "COMPILER_PATH",
        "GCC_EXEC_PREFIX",
    ):
        if os.environ.get(name, ""):
            raise ValueError(
                f"{name} is unsupported by the fixed rank-k qualification recipe; "
                "unset it or explicitly inventory its inputs in a reviewed recipe"
            )
        environment[name] = ""
    return canonical_hash(
        {
            "schema": "generativeqc.rank-k-compilation.v2",
            "source": _source_identity(root, _inventory(root, manifest)),
            "toolchain": toolchain,
            "toolkit_headers": headers,
            "environment": environment,
            "compile": [
                "-std=c++20",
                "-O2",
                "-arch=sm_90",
                "-DGENERATIVEQC_TEST_HOOKS",
                "-ccbin={host_compiler}",
                "-I{source}/src",
                "-I{generated}",
                "-c",
            ],
            "link": [
                "--cudart=shared",
                "-ccbin={host_compiler}",
                "-L{toolkit}/lib64",
                "-lcublas",
                "-Xlinker",
                "-rpath",
                "-Xlinker",
                "{toolkit}/lib64",
            ],
        }
    )


def render(source: str) -> str:
    bodies = [
        "#pragma once",
        "#include <cmath>",
        '#include "runtime/lowering_binding.hpp"',
        "namespace generativeqc::tensor::rank_k_generated {",
    ]
    stages = emit_scalar_stages(
        "cuda",
        {
            "energy_weight": "rank_k_energy_weight",
            "weighted_coefficient": "rank_k_scale",
            "contribution": "rank_k_contribution",
            "updated": "rank_k_update",
        },
    )
    bodies.extend(stages.values())
    update = emit_scalar_cpp(
        symmetric_rank_k_scalar_update_program(),
        function_name="rank_k_alpha_beta_update",
        input_order=("alpha", "product", "beta", "old_output"),
        output_order=("updated",),
    ).replace(
        "inline bool rank_k_alpha_beta_update(",
        "__device__ inline bool rank_k_alpha_beta_update(",
        1,
    )
    bodies.append(update)
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
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--host-compiler", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(compiler_identity(ROOT, args.toolkit_root, args.host_compiler)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

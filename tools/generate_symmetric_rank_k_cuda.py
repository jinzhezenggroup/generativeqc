#!/usr/bin/env python3
"""Emit source-bound rank-k qualification portfolios without loading runtime."""

from __future__ import annotations

import argparse
import json
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
    emit_symmetric_rank_k_old_output_binding,
    emit_symmetric_rank_k_portfolio,
    symmetric_rank_k_scalar_overwrite_program,
    symmetric_rank_k_scalar_update_program,
)
from generativeqc_compiler.tensor.weighted_gram_emit import emit_scalar_stages

from tools.generate_build_identity import _inventory, _source_identity

_COMPILATION_SOURCES = ("tests/native/test_symmetric_rank_k_cuda.cu",)

_TOOLCHAIN_FILES = (
    "bin/nvcc",
    "bin/nvcc.profile",
    "bin/cudafe++",
    "bin/fatbinary",
    "bin/nvlink",
    "bin/ptxas",
    "bin/crt/link.stub",
    "nvvm/bin/cicc",
    "nvvm/libdevice/libdevice.10.bc",
    "lib64/libcublas.so.12",
    "lib64/libcublasLt.so.12",
    "lib64/libcudadevrt.a",
    "lib64/libcudart.so.12",
)


def _host_toolchain_identity(host_compiler: Path, manifest: Path) -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != "generativeqc.rank-k-host-toolchain.v1":
        raise ValueError("unsupported rank-k host-toolchain manifest")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("rank-k host-toolchain manifest has no entries")
    normalized = []
    roles = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"role", "sha256"}:
            raise ValueError("invalid rank-k host-toolchain entry")
        role, digest = entry["role"], entry["sha256"]
        if (
            not isinstance(role, str)
            or not role
            or role in roles
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("invalid rank-k host-toolchain role or digest")
        roles.add(role)
        normalized.append({"role": role, "sha256": digest})
    required = {
        "program:driver",
        "program:cc1plus",
        "program:as",
        "program:collect2",
        "program:ld",
        "config:gcc-specs",
        "config:ld-default-script",
        "link-input:libstdc++.so",
        "link-input:libgcc.a",
        "link-input:crtbeginS.o",
    }
    if not required.issubset(roles) or not any(
        role.startswith("header:") for role in roles
    ):
        raise ValueError("incomplete rank-k host-toolchain manifest")
    compiler_digest = file_hash(host_compiler)
    if data.get("compiler_sha256") != compiler_digest:
        raise ValueError(
            "rank-k host-toolchain manifest names different compiler bytes"
        )
    target, version = data.get("target"), data.get("version")
    if (
        not isinstance(target, str)
        or not target
        or not isinstance(version, str)
        or not version
    ):
        raise ValueError("rank-k host-toolchain target/version is missing")
    return {
        "schema": data["schema"],
        "compiler_sha256": compiler_digest,
        "target": target,
        "version": version,
        "entries": sorted(normalized, key=lambda entry: entry["role"]),
    }


def compiler_identity(
    root: Path,
    toolkit_root: Path,
    host_compiler: Path,
    host_toolchain_manifest: Path,
) -> str:
    """Bind inventoried inputs for the fixed recipe; not a hermetic build proof."""
    root = root.resolve()
    toolkit_root = toolkit_root.resolve()
    host_compiler = host_compiler.resolve()
    host_toolchain_manifest = host_toolchain_manifest.resolve()
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
    toolchain["host"] = _host_toolchain_identity(host_compiler, host_toolchain_manifest)
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
        "GCC_COMPARE_DEBUG",
        "DEPENDENCIES_OUTPUT",
        "SUNPRO_DEPENDENCIES",
        "LD_PRELOAD",
    ):
        if os.environ.get(name, ""):
            raise ValueError(
                f"{name} is unsupported by the fixed rank-k qualification recipe; "
                "unset it or explicitly inventory its inputs in a reviewed recipe"
            )
        environment[name] = ""
    return canonical_hash(
        {
            "schema": "generativeqc.rank-k-compilation.v4",
            "source": _source_identity(root, _inventory(root, manifest)),
            "compilation_sources": {
                relative: file_hash(root / relative)
                for relative in _COMPILATION_SOURCES
            },
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
        "#include <cstddef>",
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
    bodies.append(emit_symmetric_rank_k_old_output_binding())
    overwrite = emit_scalar_cpp(
        symmetric_rank_k_scalar_overwrite_program(),
        function_name="rank_k_alpha_overwrite",
        input_order=("alpha", "product"),
        output_order=("updated",),
    ).replace(
        "inline bool rank_k_alpha_overwrite(",
        "__device__ inline bool rank_k_alpha_overwrite(",
        1,
    )
    bodies.append(overwrite)
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
    batch_domains = []
    for name, builder in (
        ("density", density_program),
        ("weighted_density", weighted_density_program),
    ):
        for n, k, shape_suffix in ((3, 5, ""), (17, 9, "_n17_k9")):
            program = builder(1, n, spin_count=2, orbital_count=k)
            batches = 1
            for extent in program.outputs[name].spec.shape[:-2]:
                batches *= extent
            for suffix, order in (
                ("row", "row-major"),
                ("column", "column-major"),
            ):
                for update in ("overwrite", "update"):
                    portfolio_name = f"rank_k_{name}{shape_suffix}_{suffix}_{update}"
                    bodies.append(
                        emit_symmetric_rank_k_portfolio(
                            program,
                            name,
                            source,
                            name=portfolio_name,
                            order=order,
                            update=update,
                        )
                    )
                    bodies.append(
                        f"inline constexpr std::size_t {portfolio_name}_n = {n};"
                    )
                    bodies.append(
                        f"inline constexpr std::size_t {portfolio_name}_k = {k};"
                    )
                    batch_domains.append(
                        f"  if (identity == {portfolio_name}_request.identity) return {batches};"
                    )
    bodies.append(
        "inline std::size_t rank_k_compiled_batches(std::string_view identity) noexcept {\n"
        + "\n".join(batch_domains)
        + "\n  return 0;\n}"
    )
    bodies.append("}  // namespace generativeqc::tensor::rank_k_generated")
    return "\n".join(bodies) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--host-compiler", type=Path, required=True)
    parser.add_argument("--host-toolchain-manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(
            compiler_identity(
                ROOT,
                args.toolkit_root,
                args.host_compiler,
                args.host_toolchain_manifest,
            )
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

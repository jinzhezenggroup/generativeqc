"""Census actual Stationary/DF mathematics and execution layouts without a GPU.

This is an opt-in engineering report, not a kernel admission rule or a CI source
fingerprint. Exact geometry source equivalence does not establish equivalence of
the surrounding stationary consumer; matching polynomial powers does not erase
DF shell ordering, cache layouts, precision policy or derivative-center slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.common.paths import source_hashes
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.integral.cuda import CudaEmitter
from generativeqc_compiler.integral.df_derivatives import axis_polynomial
from generativeqc_compiler.integral.df_rys_shell import (
    RYS_SHELL_CLASSES,
    shell_rys_roots,
)
from generativeqc_compiler.integral.df_shell_derivatives import (
    SUPPORTED_SHELL_CLASSES,
    axis_cache_layout,
    shell_schedule,
)
from generativeqc_compiler.method.ks_execution import compile_ks_execution_plan
from generativeqc_compiler.method.stationary_cuda import (
    QUALIFIED_PARTITION_ITERATIONS,
    QUALIFIED_SPD_AOT_SHARDS,
    QUALIFIED_STATIONARY_AOT_PROFILE_NAMES,
    STATIONARY_AOT_ASSETS,
    _qualified_aot_profile,
    stationary_aot_profile_weight_programs,
    stationary_runtime_sources,
)
from generativeqc_compiler.method.stationary_native_cuda import (
    NATIVE_STATIONARY_CUDA_PROFILES,
)
from generativeqc_compiler.xc.geometry_cuda import emit_geometry_cuda


def _sha(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def stationary_census() -> dict[str, Any]:
    """Compare exact staged geometry, then include scientific consumer contracts.

    The second grouping is necessary, not sufficient, for runtime sharing: a
    compatible external point producer, admission and measured endpoints would
    still be required. Primitive ABI (sp versus spd) remains a separate axis.
    """
    geometry_groups: dict[str, list[str]] = defaultdict(list)
    contract_groups: dict[str, list[str]] = defaultdict(list)
    records = []
    for name in QUALIFIED_STATIONARY_AOT_PROFILE_NAMES:
        selected = _qualified_aot_profile(name)
        plan = selected.plan
        semilocal = compile_ks_execution_plan(plan.method).semilocal.functional
        geometry = emit_geometry_cuda(
            functional=selected.functional,
            semilocal=semilocal,
            iterations=QUALIFIED_PARTITION_ITERATIONS,
            staged_point=True,
        )
        contract = {
            "geometry_sha256": _sha(geometry),
            "spin_blocks": plan.spin_blocks,
            "runtime_sources": stationary_runtime_sources(plan),
            "external_sources": plan.external_source_names,
            "weights": stationary_aot_profile_weight_programs(name),
            "reduction": plan.reduction_program(atoms=1).logical_hash,
            "partition_iterations": QUALIFIED_PARTITION_ITERATIONS,
            "precision": "FP64/strict/no-fmad",
            "point_layout": "physical-spin-DevicePointValue-v1",
        }
        # Compare the source itself, without stripping constants or names to
        # manufacture equivalence. Hashes in the report are evidence only.
        geometry_groups[geometry].append(name)
        contract_groups[canonical_hash(contract)].append(name)
        records.append(
            {
                "profile": name,
                "ingredients": semilocal.ingredients,
                "staged_geometry_bytes": len(geometry.encode("utf-8")),
                "contract": contract,
            }
        )
    profiles = len(QUALIFIED_STATIONARY_AOT_PROFILE_NAMES)
    inventory = {
        "shared_sp_primitives": 1,
        "shared_spd_primitive_shards": QUALIFIED_SPD_AOT_SHARDS,
        "legacy_sp_wrappers": profiles,
        "legacy_spd_wrappers": profiles,
        "native_curated_wrappers": sum(
            name in NATIVE_STATIONARY_CUDA_PROFILES
            for name in QUALIFIED_STATIONARY_AOT_PROFILE_NAMES
        ),
    }
    return {
        "scope": "curated profiles with native and legacy sp/spd AOT enabled",
        "cuda_translation_units": inventory,
        "total_cuda_translation_units": sum(inventory.values()),
        "staged_geometry_groups": list(geometry_groups.values()),
        "necessary_consumer_contract_groups": list(contract_groups.values()),
        "qualified_kernel_merges": 0,
        "profiles": records,
    }


def df_census() -> dict[str, Any]:
    """Count reused axis polynomials while retaining every actual shell layout."""
    records = []
    powers: set[tuple[int, int, int]] = set()
    requests = 0
    for angular in SUPPORTED_SHELL_CLASSES:
        layout, size = axis_cache_layout(angular)
        requests += len(layout)
        powers.update(layout)
        components = 1
        for degree in angular:
            components *= (degree + 1) * (degree + 2) // 2
        records.append(
            {
                "angular": angular,
                "components": components,
                "axis_entries": len(layout),
                "axis_coefficients": size,
                "axis_layout_sha256": canonical_hash(tuple(layout.items())),
                "boys_maximum_order": sum(angular) + 1,
                "rys_roots": shell_rys_roots(angular)
                if angular in RYS_SHELL_CLASSES
                else None,
                "schedules": [
                    {
                        "variant": variant,
                        "component_lanes": schedule.component_lanes,
                        "triples_per_block": schedule.triples_per_block,
                        "shared_bytes": schedule.shared_bytes,
                    }
                    for variant in range(3)
                    for schedule in (shell_schedule(angular, variant),)
                ],
            }
        )
    polynomials = []
    for power in sorted(powers):
        # Domain admission differs at auxiliary g, even though coefficients
        # continue to use the same Gaussian-moment algebra.
        graph, roots = axis_polynomial(*power, auxiliary_g_derivative=power[2] == 4)
        emitter = CudaEmitter(graph, {}, scalar_type="double")
        emitter.emit(roots)
        expression = "\n".join(
            [*emitter.lines, *(emitter.reference(root) for root in roots)]
        )
        polynomials.append(
            {
                "powers": power,
                "coefficients": len(roots),
                "expression_sha256": _sha(expression),
            }
        )
    return {
        "cuda_translation_units": len(records),
        "axis_layout_requests": requests,
        "unique_axis_powers": len(powers),
        "unique_axis_expressions": len({p["expression_sha256"] for p in polynomials}),
        "unique_ordered_cache_layouts": len({r["axis_layout_sha256"] for r in records}),
        "orbital_exchange_orbits": len(
            {(min(a, b), max(a, b), c) for a, b, c in SUPPORTED_SHELL_CLASSES}
        ),
        "qualified_kernel_merges": 0,
        "classes": records,
        "polynomials": polynomials,
    }


def main() -> None:
    """Write reproducible inventory evidence; never probe or execute a device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "generativeqc.aot-reuse-census.v1",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "audit_script_sha256": _sha(Path(__file__).read_text()),
        "compiler_inputs_sha256": canonical_hash(
            source_hashes(
                "common",
                "integral",
                "tensor",
                "xc",
                "dft",
                "method",
                assets=STATIONARY_AOT_ASSETS,
            )
        ),
        "stationary": stationary_census(),
        "df": df_census(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "stationary_objects": report["stationary"][
                    "total_cuda_translation_units"
                ],
                "geometry_groups": len(report["stationary"]["staged_geometry_groups"]),
                "consumer_contract_groups": len(
                    report["stationary"]["necessary_consumer_contract_groups"]
                ),
                "df_objects": report["df"]["cuda_translation_units"],
                "df_axis_requests": report["df"]["axis_layout_requests"],
                "df_unique_axis_powers": report["df"]["unique_axis_powers"],
                "qualified_kernel_merges": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

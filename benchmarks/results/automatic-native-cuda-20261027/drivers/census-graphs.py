"""Measure actual order-one worker Graphs; this does not grant endpoint admission."""

import json
from pathlib import Path

from generativeqc_compiler.xc.automatic_semilocal import (
    automatic_registry_entries,
)
from generativeqc_compiler.xc.bulk_runtime import build_bulk_runtime_program
from generativeqc_compiler.xc.density_work import DENSITY_WORK_DOMAIN
from generativeqc_compiler.xc.libxc_work import automatic_work_policy

rows = []
for entry in automatic_registry_entries():
    layouts = (
        ("unpolarized", "polarized")
        if automatic_work_policy(entry.name).family == "lda"
        else ("polarized",)
    )
    for spin in layouts:
        program = build_bulk_runtime_program(
            entry.name,
            spin=spin,
            order=1,
            **({"domain": DENSITY_WORK_DOMAIN} if len(layouts) == 2 else {}),
        )
        rows.append(
            {
                "name": entry.name,
                "worker_spin": spin,
                "energy_nodes": program.graph.analyze_ssa(
                    program.roots[:1]
                ).reachable_node_count,
                "derivative_roots": len(program.roots) - 1,
                "ssa": program.graph.analyze_ssa(program.roots).to_payload(),
            }
        )
output = Path(".cache/1118-cuda-forces/census/graphs.json")
output.parent.mkdir(exist_ok=True, parents=True)
output.write_text(json.dumps(rows, indent=2) + "\n")
print(f"Measured {len(rows)} distinct worker layouts")

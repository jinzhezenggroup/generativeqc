"""Offline structural source census; this does not grant endpoint admission."""

import hashlib
import json
import time
from pathlib import Path

from generativeqc_compiler.method.stationary_native_cuda import (
    emit_native_stationary_profile_cuda,
)
from generativeqc_compiler.xc.automatic_point import emit_automatic_point
from generativeqc_compiler.xc.automatic_semilocal import (
    automatic_functional_ingredients,
    automatic_registry_entries,
)

root = Path(".cache/1118-cuda-forces/census")
root.mkdir(parents=True, exist_ok=True)
rows = []
for index, entry in enumerate(automatic_registry_entries()):
    started = time.perf_counter()
    points = []
    forces = []
    for spin in ("unpolarized", "polarized"):
        for backend in ("cpu", "cuda"):
            layout = (
                spin
                if automatic_functional_ingredients(entry.code) == ("rho",)
                else "polarized"
            )
            point = emit_automatic_point(
                entry.name,
                spin=layout,
                backend=backend,
                equal_spin=(
                    backend == "cuda"
                    and spin == "unpolarized"
                    and layout == "polarized"
                ),
            )
            points.append(
                {
                    "spin": spin,
                    "worker_spin": layout,
                    "backend": backend,
                    "source_bytes": len(point.source.encode()),
                    "source_sha256": hashlib.sha256(point.source.encode()).hexdigest(),
                    "identity": point.source_identity,
                    "expression_identity": point.expression_identity,
                    "outputs": len(point.outputs),
                    "features": len(point.features),
                }
            )
        profile = f"libxc_{entry.code}_{'rks' if spin == 'unpolarized' else 'uks'}"
        source = emit_native_stationary_profile_cuda(profile)
        forces.append(
            {
                "profile": profile,
                "source_bytes": len(source.encode()),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
        )
    row = {
        "name": entry.name,
        "code": entry.code,
        "point": points,
        "force": forces,
        "seconds": time.perf_counter() - started,
    }
    rows.append(row)
    (root / "catalog.json").write_text(
        json.dumps(
            {"schema": "generativeqc.native-automatic-source-census.v1", "rows": rows},
            indent=2,
        )
        + "\n"
    )
    print(index + 1, entry.name, round(row["seconds"], 3), flush=True)

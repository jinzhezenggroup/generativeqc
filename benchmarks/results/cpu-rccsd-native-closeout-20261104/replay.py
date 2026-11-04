"""Replay the supplied-MO CPU RCCSD observations against pinned independent inputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

from tools.cc_endpoint_fixtures import load

HERE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    """Keep numerical evidence checks effective even when Python uses -O."""
    if not condition:
        raise ValueError(message)


def main() -> None:
    evidence = json.loads((HERE / "evidence.json").read_text())
    observations = 0
    for campaign in evidence["campaigns"]:
        record = campaign["record"]
        rows = record["observations"]
        require(
            len(rows) == 6, "each fixture requires three stages at two thread counts"
        )
        require(record["no_cuda"] is True, "CPU evidence cannot qualify CUDA")
        for threads in (1, 4):
            stages = {
                row["stage"]: row for row in rows if row["blas_threads"] == threads
            }
            require(
                set(stages) == {"reference", "native_cold", "native_warm"},
                "missing endpoint stage",
            )
            reference = stages["reference"]
            metadata, _arrays = load(reference["fixture"])
            for stage, row in stages.items():
                require(
                    row["fixture"] == reference["fixture"], "mixed fixture identity"
                )
                require(
                    row["pinned_reference_energy"] == metadata["total_energy"],
                    "independent reference was changed",
                )
                require(
                    math.isfinite(row["energy"])
                    and abs(row["energy"] - metadata["total_energy"]) <= 1e-8,
                    "independent energy gate failed",
                )
                require(
                    all(
                        math.isfinite(row[field]) and 0 <= row[field] <= 1e-10
                        for field in ("final_r1_max", "final_r2_max")
                    ),
                    "independent expanded-equation residual gate failed",
                )
                require(
                    row["iterations"] == reference["iterations"]
                    and row["equation_hash"] == reference["equation_hash"],
                    "backends performed different iterative scientific work",
                )
                require(
                    math.isfinite(row["elapsed_seconds"])
                    and row["elapsed_seconds"] > 0
                    and row["process_peak_rss_kib_linux"] > 0,
                    "invalid endpoint timing or solver-process RSS",
                )
                if campaign["identity_enriched"]:
                    require(
                        row["reference_inputs_hash"] == metadata["inputs_hash"]
                        and row["reference_arrays_hash"] == metadata["arrays_hash"],
                        "pinned input identity differs",
                    )
                    blas = [
                        pool
                        for pool in row["blas_threadpools"]
                        if pool["user_api"] == "blas"
                    ]
                    require(
                        bool(blas)
                        and all(pool["num_threads"] == threads for pool in blas),
                        "requested BLAS threads were not observed",
                    )
                    if stage != "reference":
                        require(
                            bool(row["native_artifacts"]), "missing JIT binary identity"
                        )
                observations += 1
            if campaign["identity_enriched"]:
                require(
                    stages["native_cold"]["native_artifacts"]
                    == stages["native_warm"]["native_artifacts"],
                    "cold and warm replay did not use identical native artifacts",
                )
    print(
        json.dumps(
            {
                "status": "PASS",
                "observations": observations,
                "scope": "supplied-MO CPU RCCSD; no ownership-audit or CUDA acceptance",
                "default_disposition": "retain explicit cpu-native opt-in",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Authenticate complete fleet rankings and retain unsuccessful tuning candidates.

Default-grid rankings, matched scheduling ablations and fixed-density component
probes have distinct scopes. Only complete, independently qualified populations
enter a timing comparison; failed allocations retain their original receipts.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import re
import runpy
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = runpy.run_path(
    str(ROOT / "benchmarks/results/bounded-packed-projection-20261101/report.py")
)
require = PREVIOUS["require"]
same_task = PREVIOUS["PRIOR"]["same_task"]


def observations(records: list[dict]) -> dict[str, list[float]]:
    """Keep every complete cold and frozen-owner warm observation."""
    result = {
        phase: [
            s["seconds"] for r in records for s in r["samples"] if s["phase"] == phase
        ]
        for phase in ("cold", "warm")
    }
    require(len(result["cold"]) == 2 and len(result["warm"]) == 6, "ABBA population")
    return result


def work(record: dict) -> list[dict]:
    """Keep semantic work separate from unlike backend iteration definitions."""
    return [
        {
            "phase": s["phase"],
            "iterations": s["iterations"],
            "focks_or_potentials": s.get(
                "fock_builds", s.get("effective_potential_builds")
            ),
            "nonlocal": (s.get("ks_diagnostic") or {}).get("nonlocal_work"),
        }
        for s in record["samples"]
    ]


def main() -> None:
    """Recompute independent gates, full timings, work and paired discrepancies."""
    bundle, files = PREVIOUS["load"](Path(__file__).parent)
    references = {}
    for path, digest in bundle["references"].items():
        raw = (ROOT / path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, "reference " + path)
        references[path] = json.loads(lzma.decompress(raw))
    oracles = references["benchmarks/results/readme-refresh-20261026/oracles.json.xz"]
    reuse = references[
        "benchmarks/results/independent-force-tiles-20261031/receipts.json.xz"
    ]["a100"]["analysis"]["reference_reuse"]
    results, gates, failures, components = [], [], [], []
    calls = 0
    for group in bundle["groups"]:
        folder, kind = group["path"], group["kind"]
        members = {
            Path(name).name: text
            for name, text in files.items()
            if Path(name).parent.as_posix() == folder
        }
        exit_code = int(members["exit-code.txt"].strip())
        if kind == "component":
            require(exit_code == 0, "component calculation failed")
            require(
                "scalar, capture, local-map CPU and resource gates passed"
                in members["indexed-potential.log"],
                "component independent gates",
            )
            for line in members["indexed-potential-benchmark.log"].splitlines():
                if not line.startswith("mapped_potential_endpoint "):
                    continue
                row = dict(item.split("=", 1) for item in line.split()[1:])
                require(int(row["evaluations"]) == 6, "component repeat count")
                require(float(row["max_v_parity_error"]) < 1e-12, "component parity")
                components.append({"allocation": folder, **row})
            continue
        if kind == "failed":
            require(exit_code != 0, "failed allocation relabelled")
            failures.append({"allocation": folder, "exit_code": exit_code})
        else:
            require(exit_code == 0, "allocation failed")
        records = {}
        for name, text in members.items():
            if not name.startswith(("conditioning-", "slot-")) or not name.endswith(
                ".json"
            ):
                continue
            record = json.loads(text)
            require(record["status"] == "complete", "incomplete endpoint")
            if record["arm"] == "standard":
                identity_name = "identity.json"
                if kind in ("matrix-tile", "full-target"):
                    variant = "baseline" if "-baseline-" in name else "binaries"
                    identity_name = variant + "-identity.json"
                identity = json.loads(members[identity_name])
                for key in ("source_identity", "library_sha256"):
                    require(
                        record["native_build"][key] == identity[key], "native " + key
                    )
                for key in ("LSB_JOBID", "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES"):
                    if key in identity["scheduler"]:
                        require(
                            record["scheduler"].get(key) == identity["scheduler"][key],
                            key,
                        )
                for sample in record["samples"]:
                    precision = sample["precision"]
                    for timeline, key in (
                        ("strict_fock", "strict_stage_fock_builds"),
                        ("mixed_fock", "mixed_stage_fock_builds"),
                        ("post_scf_fock", "post_scf_fock_builds"),
                    ):
                        require(
                            sum(
                                e["count"]
                                for e in precision["scf_fock_timeline"]
                                if e["kind"] == timeline
                            )
                            == precision[key],
                            "Fock timeline",
                        )
                    require(
                        sum(
                            precision[k]
                            for k in (
                                "strict_stage_fock_builds",
                                "mixed_stage_fock_builds",
                                "post_scf_fock_builds",
                            )
                        )
                        == sample["fock_builds"],
                        "Fock census",
                    )
            gate = qualify([record], oracles, reference_reuse=reuse)[0]
            require(gate["status"] == "qualified", str(gate))
            gates.append({"file": folder + "/" + name, "qualification": gate})
            calls += len(record["samples"])
            records[name] = record
        if kind == "failed":
            # Successful conditioning does not repair a missing reference arm.
            continue
        for method in sorted({r["method"] for r in records.values()}):
            population = {n: r for n, r in records.items() if r["method"] == method}
            first = next(iter(population.values()))
            require(
                len(population) == (9 if kind == "point-tiles" else 6),
                "complete method population",
            )
            for record in population.values():
                same_task(first, record)
                require(first["scheduler"] == record["scheduler"], "allocation changed")
            if kind == "ranking":
                arms = {
                    arm: [population[f"slot-{i}-{arm}-{method}.json"] for i in slots]
                    for arm, slots in (("standard", (1, 4)), ("gpu4pyscf", (2, 3)))
                }
                baseline = "gpu4pyscf"
            elif kind == "point-tiles":
                arms = {
                    str(tile): [
                        r
                        for n, r in population.items()
                        if n.startswith("slot-") and f"-tile{tile}-" in n
                    ]
                    for tile in (256, 512, 1024)
                }
                baseline = "256"
            else:
                arms = {
                    arm: [population[f"slot-{i}-{arm}-{method}.json"] for i in slots]
                    for arm, slots in (("baseline", (1, 4)), ("binaries", (2, 3)))
                }
                baseline = "baseline"
            errors = []
            if kind != "ranking":
                base_sample = first["samples"][0]
                for name, record in population.items():
                    for sample in record["samples"]:
                        left, right = (
                            dict(base_sample["settings"]),
                            dict(sample["settings"]),
                        )
                        if kind == "point-tiles":
                            tile = int(re.search(r"-tile(\d+)-", name)[1])
                            require(
                                right.pop("scf_tile_points") == tile,
                                "executed point tile",
                            )
                            left.pop("scf_tile_points")
                        require(left == right, "scientific controls changed")
                        energy = abs(base_sample["energy"] - sample["energy"])
                        force = float(
                            np.max(
                                np.abs(
                                    np.array(base_sample["forces"]) - sample["forces"]
                                )
                            )
                        )
                        require(energy < 1e-7 and force < 1e-7, "paired E/F")
                        errors.append((energy, force))
            seconds = {arm: observations(recs) for arm, recs in arms.items()}
            results.append(
                {
                    "allocation": folder,
                    "kind": kind,
                    "method": method,
                    "device": first["accelerator"]["name"],
                    "baseline": baseline,
                    "seconds": seconds,
                    "change_percent": {
                        arm: {
                            phase: 100
                            * (median(v) / median(seconds[baseline][phase]) - 1)
                            for phase, v in values.items()
                        }
                        for arm, values in seconds.items()
                    },
                    "paired_max_energy": max((e for e, _ in errors), default=None),
                    "paired_max_force": max((f for _, f in errors), default=None),
                    "work": {name: work(record) for name, record in population.items()},
                }
            )
    print(
        json.dumps(
            {
                "qualified_populations": len(gates),
                "energy_force_calls": calls,
                "results": results,
                "gates": gates,
                "failed_allocations": failures,
                "component_observations": components,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

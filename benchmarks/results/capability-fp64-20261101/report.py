"""Replay complete endpoints without mixing pilots, snapshots or source revisions.

Every original text and FP64 word is authenticated before scientific qualification.
Only explicitly declared complete cohorts enter comparisons; deployment failures,
intermediate recovery snapshots and instrumented calls remain visible separately.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import runpy
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks._exact_fp64_evidence import load, require
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[3]
PRIOR = runpy.run_path(
    str(ROOT / "benchmarks/results/portable-integration-20261101/report.py")
)


def statistics(values: list[float]) -> dict:
    """Expose all observations and variability, including slow calls."""
    middle = median(values)
    return {
        "values": values,
        "median": middle,
        "min": min(values),
        "max": max(values),
        "mad": median(abs(v - middle) for v in values),
    }


def native_work(record: dict, identity: dict) -> None:
    """Bind measured binaries, actual allocations and complete Fock work."""
    for key, expected in identity["scheduler"].items():
        require(record["scheduler"].get(key) == expected, "allocation " + key)
    if record["arm"] != "standard":
        return
    for key in ("source_identity", "library_sha256"):
        require(record["native_build"][key] == identity[key], "native " + key)
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


def compare(group: dict, records: dict[str, dict]) -> dict:
    """Compare two complete cold/six warm observations per declared alternative."""
    arms = {
        arm: [records[name] for name in names] for arm, names in group["arms"].items()
    }
    first = next(iter(arms.values()))[0]
    for populations in arms.values():
        for record in populations:
            PRIOR["same_task"](first, record)
            require(record["scheduler"] == first["scheduler"], "comparison allocation")
            require(
                record["accelerator"]["pci_bus_id"]
                == first["accelerator"]["pci_bus_id"],
                "comparison physical device",
            )
    seconds = {
        arm: PRIOR["observations"](populations) for arm, populations in arms.items()
    }
    errors = []
    if group["kind"] != "ranking":
        anchor = first["samples"][0]
        for populations in arms.values():
            for record in populations:
                for sample in record["samples"]:
                    energy = abs(anchor["energy"] - sample["energy"])
                    force = float(
                        np.max(np.abs(np.asarray(anchor["forces"]) - sample["forces"]))
                    )
                    require(energy < 1e-7 and force < 1e-7, "paired energy/full force")
                    errors.append((energy, force))
    baseline = group["baseline"]
    return {
        **group,
        "device": first["accelerator"]["name"],
        "source_identities": sorted(
            {
                r["native_build"]["source_identity"]
                for pops in arms.values()
                for r in pops
                if r["arm"] == "standard"
            }
        ),
        "seconds": {
            arm: {phase: statistics(v) for phase, v in phases.items()}
            for arm, phases in seconds.items()
        },
        "change_percent": {
            arm: {
                phase: 100 * (median(v) / median(seconds[baseline][phase]) - 1)
                for phase, v in phases.items()
            }
            for arm, phases in seconds.items()
        },
        "paired_max_energy": max((e for e, _ in errors), default=None),
        "paired_max_force": max((f for _, f in errors), default=None),
        "work": {
            name: PRIOR["work"](records[name])
            for names in group["arms"].values()
            for name in names
        },
    }


def main() -> None:
    """Authenticate the inventory, recompute all gates, then print scoped rankings."""
    bundle, files = load(Path(__file__).parent)
    references = {}
    for path, expected in bundle["references"].items():
        raw = (ROOT / path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, "reference " + path)
        references[path] = json.loads(lzma.decompress(raw))
    oracles = references["benchmarks/results/readme-refresh-20261026/oracles.json.xz"]
    reuses = references[
        "benchmarks/results/independent-force-tiles-20261031/receipts.json.xz"
    ]
    records, gates, incomplete, exits = {}, {}, {}, {}
    distinct = {}
    components = defaultdict(list)
    for name, text in files.items():
        if name.endswith("/exit-code.txt"):
            exits[name] = int(text.strip())
        if name.endswith("-components.jsonl"):
            for line in text.splitlines():
                event = json.loads(line)
                if event.get("operation") == "cuda_ks_xc_shared":
                    require(
                        event["valid"]
                        and event["cuda_error"] == 0
                        and event["dropped_regions"] == 0
                        and event["dropped_tiles"] == 0,
                        "incomplete component ledger " + name,
                    )
                    if "/batching-" in name:
                        counters = event["counters"]
                        requested = int(Path(name).name.split("-")[2])
                        require(
                            1 <= counters["xc_point_batch_tiles"] <= requested,
                            "admitted XC tile bound",
                        )
                        require(
                            counters["xc_point_batch_bytes"] <= 1 << 30,
                            "unchanged XC capacity ceiling",
                        )
                    components[name].append(event)
        if not name.endswith(".json"):
            continue
        try:
            record = json.loads(text)
        except ValueError:
            continue
        if (
            not isinstance(record, dict)
            or record.get("schema") != "generativeqc.readme-defaults.v2"
        ):
            continue
        if record["status"] != "complete":
            incomplete[name] = {
                "status": record["status"],
                "error": record.get("error"),
                "completed_samples": len(record.get("samples", [])),
            }
            continue
        identity_name = str(Path(name).parent / "identity.json")
        require(identity_name in files, "missing measured identity " + name)
        identity = json.loads(files[identity_name])
        native_work(record, identity)
        device = "a100" if "a100" in name else "pro6000" if "/n2/" in name else "5090"
        gate = qualify(
            [record],
            oracles,
            reference_reuse=reuses[device]["analysis"]["reference_reuse"],
        )[0]
        require(gate["status"] == "qualified", name + ": " + str(gate))
        records[name], gates[name] = record, gate
        distinct[hashlib.sha256(text.encode()).hexdigest()] = record
        if name.startswith("capability-fleet-") and record["arm"] == "standard":
            proof = json.loads(files["capability-fleet-20261101/source.json"])
            require(
                identity["source_identity"] == proof["source_identity"], "final source"
            )
            require(
                identity["library_sha256"] == proof["binaries"]["libgenerativeqc.so"],
                "final binary",
            )
            for sample in record["samples"]:
                settings = sample["settings"]
                require(
                    settings["scf_tile_points"] == 256
                    and settings["force_tile_points"] == 1024,
                    "unchanged public tiles",
                )
    require(
        set(records) | set(incomplete) == set(bundle["endpoint_inventory"]),
        "endpoint population inventory",
    )
    results = [compare(group, records) for group in bundle["comparisons"]]
    print(
        json.dumps(
            {
                "complete_receipt_paths": len(records),
                "distinct_complete_receipts": len(distinct),
                "distinct_energy_force_calls": sum(
                    len(r["samples"]) for r in distinct.values()
                ),
                "results": results,
                "gates": gates,
                "incomplete": incomplete,
                "allocation_exits": exits,
                "xc_component_events": dict(components),
                "unretrieved_allocations": bundle["unretrieved_allocations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Recheck deferred-anchor outputs and expose the cost of every warm call."""

from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[3]
FAMILY = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    """Keep integrity and numerical gates active under Python optimization."""
    if not condition:
        raise ValueError(message)


def authenticated(raw: bytes, digest: str, label: str) -> bytes:
    require(hashlib.sha256(raw).hexdigest() == digest, f"checksum: {label}")
    return raw


def work(sample: dict) -> None:
    """Preparation is real work even when it is not a physical SCF iteration."""
    precision = sample["precision"]
    count = 0
    for kind, field in (
        ("strict_fock", "strict_stage_fock_builds"),
        ("mixed_fock", "mixed_stage_fock_builds"),
        ("post_scf_fock", "post_scf_fock_builds"),
    ):
        observed = sum(
            event["count"]
            for event in precision["scf_fock_timeline"]
            if event["kind"] == kind
        )
        require(observed == precision[field], f"Fock timeline: {kind}")
        count += observed
    require(count == sample["fock_builds"], "incomplete public Fock census")


def errors(left: dict, right: dict, energy_limit: float = 1e-10) -> dict:
    energy = abs(left["energy"] - right["energy"])
    forces = np.asarray(left["forces"]) - np.asarray(right["forces"])
    force = float(np.max(np.abs(forces)))
    require(np.isfinite(energy) and energy <= energy_limit, "paired energy gate")
    require(np.isfinite(forces).all() and force <= 1e-7, "paired force gate")
    return {"energy": energy, "force": force}


def same_task(left: dict, right: dict) -> None:
    for key in (
        "method",
        "mode",
        "atoms",
        "geometry_bohr",
        "orbital_basis",
        "basis_representation",
        "charge",
        "spin",
        "auxiliary_input_sha256",
        "protocol",
        "thread_controls",
    ):
        require(left[key] == right[key], f"different task: {key}")
    require(
        left["environment"]["host"]["node"] == right["environment"]["host"]["node"],
        "different node",
    )
    for key in ("name", "pci_bus_id"):
        require(left["accelerator"][key] == right["accelerator"][key], key)
    uuids = [
        (record["accelerator"].get("nvidia_smi") or {}).get("uuid")
        for record in (left, right)
    ]
    if all(uuids):
        require(uuids[0] == uuids[1], "different GPU UUID")
    else:
        require(
            all(record["scheduler"].get("LSB_JOBID") for record in (left, right)),
            "missing device locator",
        )
        require(
            left["scheduler"]["CUDA_VISIBLE_DEVICES"]
            == right["scheduler"]["CUDA_VISIBLE_DEVICES"],
            "different LSF device visibility",
        )


def main() -> None:
    """Replay scientific gates before computing any complete-endpoint timing."""
    raw = (FAMILY / "receipts.json.xz").read_bytes()
    digest = (FAMILY / "sha256.txt").read_text().split()[0]
    bundle = json.loads(lzma.decompress(authenticated(raw, digest, "receipts")))
    files = {}
    for name, member in bundle["files"].items():
        files[name] = authenticated(
            member["text"].encode(), member["sha256"], name
        ).decode()
    references = {}
    for name, digest in bundle["references"].items():
        references[name] = json.loads(
            lzma.decompress(authenticated((ROOT / name).read_bytes(), digest, name))
        )
    oracles = references["benchmarks/results/readme-refresh-20261026/oracles.json.xz"]
    reuse = references[
        "benchmarks/results/independent-force-tiles-20261031/receipts.json.xz"
    ]["a100"]["analysis"]["reference_reuse"]
    summaries, gates = [], []
    for version, folders in bundle["populations"].items():
        for folder in folders:
            require(files[f"{folder}/exit-code.txt"].strip() == "0", folder)
            identity = json.loads(files[f"{folder}/identity.json"])
            require(
                identity["source_identity"] == bundle["identities"][version],
                "prototype identity mismatch",
            )
            records = {}
            for name, text in sorted(files.items()):
                if Path(name).parent.as_posix() != folder:
                    continue
                if not (
                    Path(name).name == "conditioning.json"
                    or (Path(name).name.startswith("slot-") and name.endswith(".json"))
                ):
                    continue
                record = json.loads(text)
                require(record["status"] == "complete", name)
                require(
                    record["native_build"]["source_identity"]
                    == identity["source_identity"],
                    name,
                )
                require(
                    record["native_build"]["library_sha256"]
                    == identity["binary_sha256"],
                    f"binary mismatch: {name}",
                )
                gate = qualify([record], oracles, reference_reuse=reuse)[0]
                require(gate["status"] == "qualified", str(gate))
                gates.append({"file": name, "qualification": gate})
                for sample in record["samples"]:
                    work(sample)
                records[Path(name).name] = record
            require(len(records) == 5, f"incomplete conditioning/ABBA: {folder}")
            groups = {
                lazy: [r for name, r in records.items() if f"lazy-{lazy}-" in name]
                for lazy in (0, 1)
            }
            timings, observations, paired = {}, {}, []
            for lazy, group in groups.items():
                require(len(group) == 2, "incomplete ABBA pair")
                samples = [r["samples"] for r in group]
                for call in samples:
                    require(
                        [s["phase"] for s in call] == ["cold", "warm", "warm", "warm"],
                        "incomplete cold/three-warm population",
                    )
                populations = {
                    "cold": [s[0]["seconds"] for s in samples],
                    "first_warm": [s[1]["seconds"] for s in samples],
                    "pooled_warm": [s["seconds"] for call in samples for s in call[1:]],
                    "lifecycle": [sum(s["seconds"] for s in call) for call in samples],
                }
                timings[lazy] = {
                    phase: {"median": median(values), "samples": values}
                    for phase, values in populations.items()
                }
                observations[lazy] = [
                    [[s["seconds"], s["iterations"], s["fock_builds"]] for s in call]
                    for call in samples
                ]
            for left, right in zip(groups[0], groups[1], strict=True):
                same_task(left, right)
                for a, b in zip(left["samples"], right["samples"], strict=True):
                    require(a["iterations"] == b["iterations"], "changed physical work")
                    paired.append(errors(a, b))
            summaries.append(
                {
                    "version": version,
                    "folder": folder,
                    "device": groups[0][0]["accelerator"]["name"],
                    "timings": timings,
                    "observations": observations,
                    "percent_change": {
                        phase: 100 * (timings[1][phase]["median"] / value["median"] - 1)
                        for phase, value in timings[0].items()
                    },
                    "paired_errors": paired,
                }
            )
    large = []
    for folder in bundle["large"]:
        complete = files[f"{folder}/exit-code.txt"].strip() == "0"
        identity = json.loads(files[f"{folder}/identity.json"])
        require(
            identity["source_identity"] in bundle["identities"].values(),
            "unknown larger-control source",
        )
        controls = [
            json.loads(files[name])
            for name in (f"{folder}/large-lazy-0.json", f"{folder}/large-lazy-1.json")
            if name in files
        ]
        summary = {"folder": folder, "complete": complete, "completed_calls": []}
        for control in controls:
            # A timed-out owner may have no completed sample or binary receipt.
            # Every returned endpoint must still bind the allocation's build.
            if control["samples"]:
                native = control["native_build"]
                require(
                    native["source_identity"] == identity["source_identity"]
                    and native["library_sha256"] == identity["binary_sha256"],
                    "larger source/binary identity",
                )
            for sample in control["samples"]:
                work(sample)
                summary["completed_calls"].append(
                    {
                        "lazy": control["lazy"],
                        **{
                            k: sample[k]
                            for k in ("phase", "seconds", "iterations", "fock_builds")
                        },
                    }
                )
        if complete:
            require(len(controls) == 2, "missing larger control arm")
            for control in controls:
                require(control["status"] == "complete", "partial larger lifecycle")
                require(
                    [s["phase"] for s in control["samples"]]
                    == [
                        "cold",
                        "retained",
                        "evicted-fallback",
                        "second",
                        "moved",
                        "moved-warm",
                    ],
                    "larger phase census",
                )
            for key in (
                "geometry_bohr",
                "moved_positions_bohr",
                "density_fitting_memory_budget_bytes",
                "method",
                "atoms",
                "scheduler",
            ):
                require(controls[0][key] == controls[1][key], f"larger task: {key}")
            for a, b in zip(
                controls[0]["samples"], controls[1]["samples"], strict=True
            ):
                require(
                    a["evaluated_positions_bohr"] == b["evaluated_positions_bohr"],
                    "different geometry",
                )
                require(a["iterations"] == b["iterations"], "larger physical work")
                errors(a, b, 1e-9)
        large.append(summary)
    print(
        json.dumps(
            {
                "summaries": summaries,
                "gates": gates,
                "large": large,
                "checks": bundle["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

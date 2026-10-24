"""Verify frozen solver experiments and recompute complete endpoint summaries.

This opt-in receipt replay needs only Python's standard library. It executes no
archived code and requires no GPU. Source and binary identities authenticate the
measured variants; independent reference forces qualify each returned endpoint.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GATES = {"energy_per_atom": 2.5e-6, "force_max": 2.5e-5, "force_rms": 7.5e-6}
GROUPS = ("pilot", "prototypes", "handoff", "midpoint", "confirm")


def require(condition: bool, message: str) -> None:
    """Keep scientific and provenance checks effective under Python -O."""
    if not condition:
        raise ValueError(message)


def digest(value: bytes) -> str:
    """Use the same byte identity for retained files and measured native images."""
    return hashlib.sha256(value).hexdigest()


def load_packet() -> dict:
    """Validate the archive and every exact retained file before interpretation."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    archive = (ROOT / "evidence.json.xz").read_bytes()
    require(digest(archive) == manifest["archive_sha256"], "archive hash mismatch")
    require(len(archive) == manifest["archive_bytes"], "archive size mismatch")
    packet = json.loads(lzma.decompress(archive))
    require(packet["base"] == manifest["base"], "source base mismatch")
    files = packet["files"]
    require(len(files) == manifest["retained_files"], "retained file count mismatch")
    require(set(files) == set(packet["file_sha256"]), "file inventory mismatch")
    for name, contents in files.items():
        require(digest(contents.encode()) == packet["file_sha256"][name], name)
    return packet


def verify_sources(packet: dict) -> dict:
    """Reconstruct each input-map digest from the master map and actual source."""
    files = packet["files"]
    master = json.loads(files["n1/master-inputs.json"])
    require(master["base"] == packet["base"], "input map base mismatch")
    source = "src/dft/cuda_ks.cpp"
    require(
        digest(files["reconstruction/master-cuda_ks.cpp"].encode())
        == master["inputs"][source],
        "master source bytes mismatch",
    )
    images = {}
    for label in ("master", "prototype", "prototype2", "prototype3"):
        identity = json.loads(files[f"n1/{label}-identities.json"])
        require(identity["base"] == packet["base"], f"{label}: base mismatch")
        changes = identity["input_delta"]
        require(
            set(changes) == (set() if label == "master" else {source}),
            f"{label}: unexpected source change",
        )
        if changes:
            require(
                digest(files[f"n1/cuda_ks-{label}.cpp"].encode()) == changes[source],
                f"{label}: source bytes mismatch",
            )
        inputs = {**master["inputs"], **changes}
        require(
            digest(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode())
            == identity["production_inputs_sha256"],
            f"{label}: source map mismatch",
        )
        for name, value in identity["artifacts"].items():
            if name == f"libgenerativeqc-{label}.so":
                images[value] = label
    require(len(images) == 4, "missing measured native images")
    return images


def verify_sample(record: dict, sample: dict, reference: dict) -> dict:
    """Recompute independent errors and count failed as well as returned solves.

    The precision report owns rejected molecular Fock work. The initial-guess
    report describes the returned solve only: atomic preparation for an earlier
    failed precision attempt is unavailable, and must not be invented.
    """
    phase = "original" if sample["phase"] in ("cold", "warm") else "moved"
    oracle = next(row for row in reference["samples"] if row["phase"] == phase)
    # The frozen measurement driver checks its generated native coordinates
    # against this reference before constructing the calculator. Receipts bind
    # that exact driver and reference; they do not duplicate coordinate arrays.
    atoms = len(reference["atoms"])
    require(atoms == record["args"]["atoms"], "oracle atom count mismatch")
    forces, expected = sample["forces"], oracle["forces"]
    require(len(forces) == len(expected) == atoms, "force atom count mismatch")
    require(all(len(row) == 3 for row in forces + expected), "force shape mismatch")
    delta = [a - b for x, y in zip(forces, expected) for a, b in zip(x, y)]
    values = delta + [sample["energy"], oracle["energy"], sample["seconds"]]
    require(all(math.isfinite(value) for value in values), "nonfinite endpoint")
    require(sample["seconds"] > 0, "nonpositive endpoint time")
    errors = {
        "energy_per_atom": abs(sample["energy"] - oracle["energy"]) / atoms,
        "force_max": max(map(abs, delta)),
        "force_rms": math.sqrt(math.fsum(value * value for value in delta) / len(delta)),
    }
    require(record["gates"] == GATES, "changed scientific acceptance gates")
    for key, value in errors.items():
        require(
            math.isclose(value, sample["errors"][key], rel_tol=1e-12, abs_tol=1e-15),
            f"{key}: recorded error differs from independent recomputation",
        )
    passed = all(value <= GATES[key] for key, value in errors.items())
    require(sample["qualified"] == passed, "qualification label mismatch")
    require(passed, "returned endpoint fails independent solver gates")
    validation = sample["precision_validation"]
    solves = validation["solves"] if validation else []
    if validation:
        require(validation["solve_count"] == len(solves), "missing precision attempt")
        require(
            sum(row["fock_builds"] for row in solves) == sample["fock_builds"],
            "complete endpoint Fock census mismatch",
        )
        require(
            sample["batch_precision_validation"]["candidate_fock_builds"]
            == [solves[0]["fock_builds"]],
            "candidate Fock census mismatch",
        )
    guess = sample["initial_guess"]
    require(guess["work_counters_complete"], "incomplete returned-solve guess census")
    require(guess["discarded_target_fock_builds"] == 0, "uncovered initial-guess retry work")
    target_focks = (
        sum(row["fock_builds"] for row in solves) if solves else sample["fock_builds"]
    )
    return {
        "target_focks": target_focks,
        "rejected_precision_solves": sum(bool(row.get("error")) for row in solves),
        "returned_atomic_focks": guess["preliminary_fock_builds"],
        "all_atomic_work_available": len(solves) <= 1,
    }


def replay(packet: dict) -> dict:
    """Validate every population and derive fresh-confirmation medians only."""
    files, images = packet["files"], verify_sources(packet)
    records = {}
    total_samples = total_focks = 0
    fallback_samples = []
    for name, content in files.items():
        parts = name.split("/")
        selected = (
            len(parts) == 3 and parts[:1] == ["n4"] and parts[1] in GROUPS
        ) or name == "n4/smoke.json"
        if not selected or not name.endswith(".json"):
            continue
        record = json.loads(content)
        require("error" not in record, f"unfinished endpoint: {name}")
        require(len(record["samples"]) == 2, f"incomplete cold/warm pair: {name}")
        require(record["base"] == packet["base"], f"{name}: source mismatch")
        require(record["library_sha256"] in images, f"{name}: unknown native image")
        label = images[record["library_sha256"]]
        expected_label = {
            "pilot": "master",
            "prototypes": "prototype",
            "handoff": "prototype2",
            "midpoint": "prototype3",
            "confirm": "master" if parts[-1].startswith("master-") else "prototype2",
            "smoke.json": "master",
        }[parts[1]]
        require(label == expected_label, f"{name}: native arm mismatch")
        require(
            Path(record["library"]).name == f"libgenerativeqc-{label}.so",
            f"{name}: native library path mismatch",
        )
        require(record["allocation"]["SLURM_JOB_ID"], f"{name}: missing Slurm job")
        require("CUDA_VISIBLE_DEVICES" in record["allocation"], "missing visibility")
        driver = "probe2.py" if parts[1] in ("handoff", "midpoint", "confirm") else "probe.py"
        require(
            record["driver_sha256"] == digest(files[f"n4/{driver}"].encode()),
            f"{name}: driver mismatch",
        )
        args = record["args"]
        ref_name = f"references/{args['method']}-{args['mode']}-{args['atoms']}.json"
        require(digest(files[ref_name].encode()) == record["reference_sha256"], ref_name)
        reference = json.loads(files[ref_name])
        require(reference["method"] == args["method"], "oracle method mismatch")
        require(reference["mode"] == args["mode"], "oracle fitted operator mismatch")
        for sample in record["samples"]:
            work = verify_sample(record, sample, reference)
            total_samples += 1
            total_focks += work["target_focks"]
            if work["rejected_precision_solves"]:
                fallback_samples.append({"record": name, "phase": sample["phase"], **work})
        records[name] = record
    require(len(records) == 76, "missing experiment population")
    rows = []
    for atoms in (12, 48, 96):
        row = {"atoms": atoms}
        for arm in ("master", "candidate"):
            runs = [
                records[f"n4/confirm/{arm}-fast-direct-{atoms}-r{repeat}.json"]
                for repeat in (1, 2, 3)
            ]
            for run in runs:
                controls = run["prototype_controls"]
                require(controls["GENERATIVEQC_KS_ORTHOGONAL_DIIS"] == "0", "confirmation metric")
                require(
                    controls["GENERATIVEQC_KS_PROBE_EARLY_REFINEMENT"]
                    == ("0" if arm == "master" else "2"),
                    "confirmation handoff mismatch",
                )
                require(
                    controls["GENERATIVEQC_KS_PROBE_KEEP_MIXED_DIIS"]
                    == ("0" if arm == "master" else "1"),
                    "confirmation history mismatch",
                )
                for sample in run["samples"]:
                    require(len(sample["precision_validation"]["solves"]) == 1, "confirmation fallback")
            for index, phase in enumerate(("cold", "warm")):
                seconds = [run["samples"][index]["seconds"] for run in runs]
                median = statistics.median(seconds)
                row[f"{arm}_{phase}"] = {
                    "seconds": seconds,
                    "median_seconds": median,
                    "mad_seconds": statistics.median(abs(value - median) for value in seconds),
                    "target_focks": [run["samples"][index]["fock_builds"] for run in runs],
                }
        row["cold_reduction_percent"] = 100 * (
            1 - row["candidate_cold"]["median_seconds"] / row["master_cold"]["median_seconds"]
        )
        rows.append(row)
    return {
        "base": packet["base"],
        "populations": len(records),
        "independently_qualified_returned_endpoints": total_samples,
        "executed_target_focks_including_rejected_solves": total_focks,
        "guarded_fallback_samples": fallback_samples,
        "confirmation": rows,
        "minimum_cold_improvement_gate_met": any(
            row["cold_reduction_percent"] >= packet["confirmation_minimum_cold_improvement_percent"]
            for row in rows
        ),
    }


if __name__ == "__main__":
    print(json.dumps(replay(load_packet()), indent=2))

"""Audit a historical negative-promotion capsule without CUDA.

This is not a fresh qualification or a standard accepted validation envelope.
The reduced payload retains one new-campaign 384-AO RHF pair, seven final
unsuccessful retry dispositions, and old source/library receipts. Failure
causes, individual first attempts, and the old measured checkout cannot be
reconstructed from these files.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
SCHEMA = "generativeqc.historical-benchmark-capsule.v1"
CAMPAIGNS = {
    "old": {
        "revision": "96570fd06243de418e2eec4efb570d5510506d03",
        "library_sha256": "2998014a18547b813149521c51193cd8913602b5d7e4551e326a4d5c91053a77",
    },
    "new": {
        "revision": "d83aa86585a2f72ed852872c533723648b2799c3",
        "dirty": False,
        "library_sha256": "30734a0fc741da60389fb817b7429acd4a5164c840728308822fe215eae90a63",
        "native_source_identity": "09847dbbf578bf414e45dfe02d996573662119f86c98cd21287db79bffb496eb",
    },
}
SIZES = {"old": (12, 24, 48, 96), "new": (24, 48, 96)}
FINAL_UNSUCCESSFUL_RETRIES = {
    "old": {(24, 1), (24, 2), (96, 2)},
    "new": {(24, 1), (24, 2), (96, 1), (96, 2)},
}
SELECTED_PAIRS = (("new", 48, 1),)
EVIDENCE_SCOPE = {
    "selected_records": 2,
    "selected_on_off_pairs": 1,
    "final_unsuccessful_retry_outcomes": 7,
    "old_source_reconstruction_available": False,
    "failure_causes_verified": False,
    "individual_attempt_records_retained": False,
    "cold_native_numerical_arrays_retained": False,
    "fresh_qualification": False,
}
WORKLOAD_SHA256 = "e6c54e1f733f030d824fda87aaee06676c992fbe7ef187324bcc358c44c7e2f4"
DEVICE_UUID = "0639f74dc42b3c2b4a573795decc0888"
DEVICE_NAME = "NVIDIA GeForce RTX 4090"
MAX_ENERGY_DIFFERENCE = 1.0e-8
MAX_FORCE_DIFFERENCE = 1.0e-7


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "non-finite retained value")


def load_json(raw: bytes) -> dict:
    value = json.loads(raw)
    require(isinstance(value, dict), "record must be an object")
    finite(value)
    return value


def load_gz(path: Path) -> dict:
    return load_json(gzip.decompress(path.read_bytes()))


def expected_files() -> set[str]:
    """Keep the reduced inventory fixed, even if the manifest is re-signed."""
    paths = {
        "README.md",
        "verify.py",
        "old/source-sha.txt.gz",
        "old/library-sha.txt.gz",
    }
    for name, atoms, round_number in SELECTED_PAIRS:
        for mode in ("baseline", "incremental"):
            stem = f"{name}/rhf/direct-{atoms}-{mode}-{round_number}"
            paths.update((stem + ".json.gz", stem + ".outcome.gz"))
    for name, points in FINAL_UNSUCCESSFUL_RETRIES.items():
        paths.update(
            f"{name}/rhf/direct-{atoms}-incremental-{round_number}.outcome.gz"
            for atoms, round_number in points
        )
    return paths


def check_manifest() -> None:
    manifest = load_json((ROOT / "capsule.json").read_bytes())
    require(manifest.get("schema") == SCHEMA, "wrong historical capsule schema")
    require(manifest.get("campaigns") == CAMPAIGNS, "campaign identities changed")
    require(manifest.get("evidence_scope") == EVIDENCE_SCOPE, "evidence scope changed")
    require(
        not (ROOT / "publication.json").exists(),
        "standard publication envelope is not historical evidence",
    )
    decision = manifest["decision"]
    require(
        decision.get("status") == "rejected"
        and decision.get("scope") == "default-promotion",
        "capsule must reject default promotion",
    )
    entries = manifest["files"]
    paths = [entry["path"] for entry in entries]
    require(len(paths) == len(set(paths)), "duplicate manifest member")
    require(set(paths) == expected_files(), "retained member inventory changed")
    for entry in entries:
        path = ROOT / entry["path"]
        require(path.is_file(), f"missing retained member: {entry['path']}")
        raw = path.read_bytes()
        require(
            type(entry["bytes"]) is int
            and len(raw) == entry["bytes"]
            and hashlib.sha256(raw).hexdigest() == entry["sha256"],
            f"retained checksum/size mismatch: {entry['path']}",
        )
    require(
        {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.gz")}
        == {p for p in paths if p.endswith(".gz")},
        "unlisted compressed records",
    )


def check_receipts() -> None:
    source = (
        gzip.decompress((ROOT / "old/source-sha.txt.gz").read_bytes()).decode().strip()
    )
    library = (
        gzip.decompress((ROOT / "old/library-sha.txt.gz").read_bytes())
        .decode()
        .split()[0]
    )
    require(source == CAMPAIGNS["old"]["revision"], "source receipt mismatch")
    require(library == CAMPAIGNS["old"]["library_sha256"], "library receipt mismatch")


def maximum_abs_difference(first: object, second: object) -> float:
    if isinstance(first, list) or isinstance(second, list):
        require(
            isinstance(first, list) and isinstance(second, list), "result shapes differ"
        )
        require(len(first) == len(second) and bool(first), "result shapes differ")
        return max(
            maximum_abs_difference(a, b) for a, b in zip(first, second, strict=True)
        )
    require(
        type(first) in (int, float) and type(second) in (int, float),
        "nonnumeric result",
    )
    finite(first)
    finite(second)
    return abs(float(first) - float(second))


def compare(first: dict, second: dict) -> None:
    for field, limit in (
        ("energies_hartree", MAX_ENERGY_DIFFERENCE),
        ("forces_hartree_per_bohr", MAX_FORCE_DIFFERENCE),
    ):
        require(
            maximum_abs_difference(first[field], second[field]) <= limit,
            f"{field} gate failed",
        )


def check_record(record: dict, name: str, atoms: int, mode: str) -> None:
    expected = CAMPAIGNS[name]
    require(
        record.get("benchmark") == "compare_gpu4pyscf_batch", "not a comparator record"
    )
    require(
        type(record.get("schema_version")) is int and record["schema_version"] == 3,
        "unexpected comparator schema",
    )
    source = record["environment"]["git"]
    require(
        source["commit"] == expected["revision"]
        and source["dirty"] is expected["dirty"],
        "source identity mismatch",
    )
    native = record["native_build"]
    require(
        native["library_sha256"] == expected["library_sha256"],
        "library identity mismatch",
    )
    require(
        native["probe"]["source_identity"] == expected["native_source_identity"],
        "native source identity mismatch",
    )
    require(
        record["environment"]["accelerator"]["name"] == DEVICE_NAME
        and record["environment"]["accelerator"]["backend"] == "cuda"
        and record["environment"]["accelerator"]["compute_capability"] == [8, 9]
        and native["probe"]["device"]["name"] == DEVICE_NAME
        and native["probe"]["device"]["major"] == 8
        and native["probe"]["device"]["minor"] == 9,
        "device identity mismatch",
    )
    diagnostics = record["generativeqc"]["eigensolver_diagnostics"]
    require(
        len(diagnostics) == 1
        and diagnostics[0]["device_uuid"] == DEVICE_UUID
        and diagnostics[0]["device_name"] == DEVICE_NAME
        and diagnostics[0]["compute_capability"] == [8, 9],
        "device UUID mismatch",
    )
    workload = record["workload"]
    require(
        hashlib.sha256(
            json.dumps(workload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == WORKLOAD_SHA256,
        "measured workload identity changed",
    )
    require(
        workload["case"] == f"water-{atoms}"
        and workload["ao_count"] == 8 * atoms
        and workload["method"] == "rhf"
        and workload["batch_size"] == 1
        and workload["properties"] == ["energy", "forces"],
        "workload identity mismatch",
    )
    require(
        record["gate"]["passed"] is True and record["gate"]["failures"] == [],
        "recorded gate failed",
    )
    settings = record["settings"]
    require(
        type(settings["repeats_per_engine"]) is int
        and settings["repeats_per_engine"] == 3,
        "warm repeat count changed",
    )
    require(
        settings["interleave_policy"] == "deterministic ABBA"
        and settings["measurement_order"]
        == [
            "generativeqc",
            "gpu4pyscf",
            "gpu4pyscf",
            "generativeqc",
            "generativeqc",
            "gpu4pyscf",
        ],
        "measurement order changed",
    )
    require(
        settings["gates"]["maximum_energy_error_hartree"] == MAX_ENERGY_DIFFERENCE,
        "energy tolerance changed",
    )
    require(
        settings["gates"]["maximum_force_error_hartree_per_bohr"]
        == MAX_FORCE_DIFFERENCE,
        "force tolerance changed",
    )
    for engine, sequence in (("generativeqc", [0, 3, 4]), ("gpu4pyscf", [1, 2, 5])):
        result = record[engine]
        require(
            type(result["cold_seconds"]) in (int, float) and result["cold_seconds"] > 0,
            "invalid cold timing",
        )
        samples = result["warm_samples"]
        require(
            [sample["sequence_index"] for sample in samples] == sequence
            and all(type(sample["sequence_index"]) is int for sample in samples),
            "warm pair inventory changed",
        )
        seconds = [sample["seconds"] for sample in samples]
        require(
            all(type(t) in (int, float) and t > 0 for t in seconds),
            "invalid warm timing",
        )
        require(
            result["warm_seconds"] == seconds
            and result["warm_median_seconds"] == median(seconds),
            "warm timing summary mismatch",
        )
        for field in ("energies_hartree", "forces_hartree_per_bohr"):
            require(result[field] == samples[-1][field], "last-sample summary mismatch")
        require(
            result["convergence"] == samples[-1]["convergence"],
            "last-convergence summary mismatch",
        )
        for result_row in [result, *samples]:
            energy = result_row["energies_hartree"]
            force = result_row["forces_hartree_per_bohr"]
            require(
                len(energy) == 1
                and len(force) == 1
                and len(force[0]) == atoms
                and all(len(x) == 3 for x in force[0]),
                "physical result shape mismatch",
            )
        for convergence in [
            result["cold_convergence"],
            result["convergence"],
            *[s["convergence"] for s in samples],
        ]:
            require(
                len(convergence) == 1 and convergence[0]["converged"] is True,
                "successful record did not converge",
            )
            require(
                type(convergence[0]["iterations"]) is int
                and convergence[0]["iterations"] > 0,
                "invalid iteration count",
            )
            if engine == "generativeqc":
                diagnostic = convergence[0]["incremental_direct_jk"]
                require(
                    diagnostic["active"] is (mode == "incremental"),
                    "incremental activation mismatch",
                )
                for field, value in diagnostic.items():
                    if field.endswith(
                        ("_builds", "_shell_quartets", "_quartet_tiles")
                    ) or field in ("anchor_updates", "periodic_rebuilds"):
                        require(
                            type(value) is int and value >= 0,
                            "invalid incremental work count",
                        )
                require(
                    convergence[0]["incremental_direct_jk"]["requested"]
                    is (mode == "incremental"),
                    "incremental mode mismatch",
                )
    compare(record["generativeqc"], record["gpu4pyscf"])
    for native_sample, reference in zip(
        record["generativeqc"]["warm_samples"],
        record["gpu4pyscf"]["warm_samples"],
        strict=True,
    ):
        compare(native_sample, reference)


def check_outcome(stem: str, *, unsuccessful: bool) -> None:
    outcome = load_gz(ROOT / (stem + ".outcome.gz"))
    expected = (
        {"exit_code": 1, "attempt": 2, "retried": True}
        if unsuccessful
        else {"exit_code": 0, "attempt": 1}
    )
    require(
        outcome == expected
        and type(outcome["exit_code"]) is int
        and type(outcome["attempt"]) is int
        and (not unsuccessful or type(outcome["retried"]) is bool),
        f"retry disposition mismatch: {stem}",
    )


def check_campaign(name: str, sizes: tuple[int, ...]) -> dict:
    summary = {
        "points": [],
        "final_unsuccessful_retries": [],
        "pairs": [],
        "not_retained": [],
    }
    retained = set()
    for atoms, round_number in sorted(FINAL_UNSUCCESSFUL_RETRIES[name]):
        stem = f"direct-{atoms}-incremental-{round_number}"
        check_outcome(f"{name}/rhf/{stem}", unsuccessful=True)
        summary["final_unsuccessful_retries"].append(stem)
        retained.add(stem)
    for campaign, atoms, round_number in SELECTED_PAIRS:
        if campaign != name:
            continue
        records = {}
        for mode in ("baseline", "incremental"):
            stem = f"direct-{atoms}-{mode}-{round_number}"
            check_outcome(f"{name}/rhf/{stem}", unsuccessful=False)
            record = load_gz(ROOT / name / "rhf" / (stem + ".json.gz"))
            check_record(record, name, atoms, mode)
            records[mode] = record
            summary["points"].append(stem)
            retained.add(stem)
        base, inc = records["baseline"], records["incremental"]
        require(base["workload"] == inc["workload"], "paired workload mismatch")
        compare(base["generativeqc"], inc["generativeqc"])
        for a, b in zip(
            base["generativeqc"]["warm_samples"],
            inc["generativeqc"]["warm_samples"],
            strict=True,
        ):
            compare(a, b)
        timings = {
            phase: inc["generativeqc"][field] / base["generativeqc"][field]
            for phase, field in (
                ("cold", "cold_seconds"),
                ("warm", "warm_median_seconds"),
            )
        }
        require(timings["warm"] > 1, "claimed warm regression missing")
        require(timings["cold"] > 1, "claimed cold regression missing")
        summary["pairs"].append(
            {"atoms": atoms, "round": round_number, "ratios": timings}
        )
    # This is the historically attempted RHF matrix, not additional retained proof.
    summary["not_retained"] = [
        stem
        for atoms in sizes
        for mode in ("baseline", "incremental")
        for round_number in (1, 2)
        if (stem := f"direct-{atoms}-{mode}-{round_number}") not in retained
    ]
    return summary


def main() -> None:
    check_manifest()
    check_receipts()
    summaries = {name: check_campaign(name, sizes) for name, sizes in SIZES.items()}
    print(
        json.dumps(
            {
                "status": "verified-reduced-historical-negative-evidence",
                "evidence_scope": EVIDENCE_SCOPE,
                "campaigns": summaries,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

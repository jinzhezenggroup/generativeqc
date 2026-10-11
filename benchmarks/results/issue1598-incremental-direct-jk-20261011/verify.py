"""Audit a historical negative-promotion capsule without CUDA.

This is not a fresh qualification or a standard accepted validation envelope.
Failed RHF attempts retain only their final retry disposition, not exception
logs. The old dirty checkout cannot be reconstructed from the retained files.
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
        "dirty": True,
        "library_sha256": "2998014a18547b813149521c51193cd8913602b5d7e4551e326a4d5c91053a77",
        "native_source_identity": "54981b932caa9b88a4bc334634fed21ffc92ee9659b9128552a7462c806530f3",
    },
    "new": {
        "revision": "d83aa86585a2f72ed852872c533723648b2799c3",
        "dirty": False,
        "library_sha256": "30734a0fc741da60389fb817b7429acd4a5164c840728308822fe215eae90a63",
        "native_source_identity": "09847dbbf578bf414e45dfe02d996573662119f86c98cd21287db79bffb496eb",
    },
}
SIZES = {"old": (12, 24, 48, 96), "new": (24, 48, 96)}
FAILED = {
    "old": {(24, 1), (24, 2), (96, 2)},
    "new": {(24, 1), (24, 2), (96, 1), (96, 2)},
}
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
    paths = {"README.md", "verify.py"}
    paths.update(
        f"old/{name}.txt.gz" for name in ("device", "source-sha", "library-sha")
    )
    for name, sizes in SIZES.items():
        for atoms in sizes:
            for mode in ("baseline", "incremental"):
                for round_number in (1, 2):
                    stem = f"{name}/rhf/direct-{atoms}-{mode}-{round_number}"
                    paths.add(stem + ".outcome.gz")
                    if mode == "baseline" or (atoms, round_number) not in FAILED[name]:
                        paths.update((stem + ".json.gz", stem + "-attempt1.json.gz"))
    for mode in ("off", "on"):
        for round_number in (1, 2):
            stem = f"new/ks/pbe0-96-{mode}-{round_number}"
            paths.add(stem + ".outcome.gz")
            paths.update(f"{stem}-attempt{attempt}.json.gz" for attempt in (1, 2))
    return paths


def check_manifest() -> None:
    manifest = load_json((ROOT / "capsule.json").read_bytes())
    require(manifest.get("schema") == SCHEMA, "wrong historical capsule schema")
    require(manifest.get("campaigns") == CAMPAIGNS, "campaign identities changed")
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
    device = gzip.decompress((ROOT / "old/device.txt.gz").read_bytes()).decode()
    require(source == CAMPAIGNS["old"]["revision"], "source receipt mismatch")
    require(library == CAMPAIGNS["old"]["library_sha256"], "library receipt mismatch")
    require(
        DEVICE_NAME in device and DEVICE_UUID in device.replace("-", ""),
        "device receipt mismatch",
    )


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
    require(record.get("schema_version") == 3, "unexpected comparator schema")
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
        record["environment"]["accelerator"]["name"] == DEVICE_NAME,
        "device identity mismatch",
    )
    diagnostics = record["generativeqc"]["eigensolver_diagnostics"]
    require(
        len(diagnostics) == 1 and diagnostics[0]["device_uuid"] == DEVICE_UUID,
        "device UUID mismatch",
    )
    workload = record["workload"]
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
    require(settings["repeats_per_engine"] == 3, "warm repeat count changed")
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
            [sample["sequence_index"] for sample in samples] == sequence,
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


def check_campaign(name: str, sizes: tuple[int, ...]) -> dict:
    root = ROOT / name / "rhf"
    summary = {"points": [], "failures": [], "pairs": []}
    for atoms in sizes:
        records = {}
        for mode in ("baseline", "incremental"):
            for round_number in (1, 2):
                stem = f"direct-{atoms}-{mode}-{round_number}"
                outcome = load_gz(root / (stem + ".outcome.gz"))
                failed = mode == "incremental" and (atoms, round_number) in FAILED[name]
                expected = (
                    {"exit_code": 1, "attempt": 2, "retried": True}
                    if failed
                    else {"exit_code": 0, "attempt": 1}
                )
                require(
                    outcome == expected
                    and type(outcome["exit_code"]) is int
                    and type(outcome["attempt"]) is int
                    and (not failed or type(outcome["retried"]) is bool),
                    f"retry disposition mismatch: {stem}",
                )
                path = root / (stem + ".json.gz")
                if failed:
                    require(
                        not path.exists(), f"failed point has a selected record: {stem}"
                    )
                    summary["failures"].append(stem)
                    continue
                record = load_gz(path)
                require(
                    record == load_gz(root / (stem + "-attempt1.json.gz")),
                    f"selected/attempt mismatch: {stem}",
                )
                check_record(record, name, atoms, mode)
                if name == "old" and atoms == 96 and mode == "incremental":
                    require(
                        record["generativeqc"]["cold_convergence"][0]["iterations"]
                        == 30,
                        "claimed old 768-AO cold iteration count changed",
                    )
                records[mode, round_number] = record
                summary["points"].append(stem)
        for round_number in (1, 2):
            if ("incremental", round_number) not in records:
                continue
            base = records["baseline", round_number]
            inc = records["incremental", round_number]
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
            require(
                (timings["cold"] < 1) if atoms == 12 else (timings["cold"] > 1),
                "claimed cold timing direction changed",
            )
            summary["pairs"].append(
                {"atoms": atoms, "round": round_number, "ratios": timings}
            )
    return summary


def main() -> None:
    check_manifest()
    check_receipts()
    summaries = {name: check_campaign(name, sizes) for name, sizes in SIZES.items()}
    print(
        json.dumps(
            {"status": "verified-historical-negative-evidence", "campaigns": summaries},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

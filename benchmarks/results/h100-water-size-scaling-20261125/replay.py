"""Verify the archived raw records and reconstruct the paired cold summary."""

import argparse
import csv
import hashlib
import io
import json
import lzma
import math
import statistics
from itertools import product
from pathlib import Path

from acquire import ARMS, METHODS, MODES, REPEATS

from geometry import SIZES

GATES = {
    "energy_per_atom_Eh": 1e-5,
    "force_max_Eh_per_bohr": 1e-4,
    "force_rms_Eh_per_bohr": 3e-5,
}


def checksum(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def read_bundle(path):
    with lzma.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def errors(sample, reference, atoms):
    if len(sample["forces"]) != atoms or len(reference["forces"]) != atoms:
        raise ValueError("force vectors do not match atom count")
    delta = [
        left - right
        for a, b in zip(sample["forces"], reference["forces"], strict=True)
        for left, right in zip(a, b, strict=True)
    ]
    if len(delta) != 3 * atoms:
        raise ValueError("expected three Cartesian force components per atom")
    return {
        "energy_per_atom_Eh": abs(sample["energy"] - reference["energy"]) / atoms,
        "force_max_Eh_per_bohr": max(map(abs, delta)),
        "force_rms_Eh_per_bohr": math.sqrt(
            math.fsum(value * value for value in delta) / len(delta)
        ),
    }


def within(difference, fraction=1.0):
    return all(
        math.isfinite(difference[k]) and difference[k] <= fraction * v
        for k, v in GATES.items()
    )


def oracle(files, atoms, method, source):
    raw = files.get(f"oracle-{atoms}-{method}.json")
    if raw is None:
        return None, "no_oracle"
    record = json.loads(raw)
    if record["status"] != "complete":
        return None, "incomplete_oracle"
    if (
        record["source_revision"] != source["revision"]
        or record["geometry_sha256"] != source["geometry_sha256"]
        or record["packages"]["pyscf"] != "2.14.0"
        or "9462" not in record["cpu"]["model"]
    ):
        raise ValueError("oracle provenance differs from the accepted contract")
    if (
        "cpu_oracle_driver_sha256" in record
        and record["cpu_oracle_driver_sha256"] != source["cpu_oracle_driver_sha256"]
    ):
        raise ValueError("CPU oracle driver differs from the source pin")
    samples = record["samples"]
    if method != "hf":
        if [sample["level"] for sample in samples] != [4, 5]:
            raise ValueError("wrong oracle grid sequence")
        if not within(errors(*samples, atoms), 0.25):
            return None, "unstable_oracle_grid"
    elif len(samples) != 1:
        raise ValueError("wrong HF oracle count")
    return samples[-1], "available"


def read_attempt(files, formal, source, geometry_digest, build_text=None):
    campaign = json.loads(files["campaign.json"])
    hardware = campaign["hardware"]
    if hardware["native_source_identity"] != source["native_source_identity"]:
        raise ValueError("campaign native source identity differs")
    if formal:
        if build_text is None or hardware["build_receipt_sha256"] != checksum(
            build_text
        ):
            raise ValueError("campaign build receipt checksum differs")
        build = json.loads(build_text)
        if (
            build["status"] != "complete"
            or build["source_revision"] != source["revision"]
            or build["source_archive_sha256"] != source["source_archive_sha256"]
            or build["native_source_identity"] != source["native_source_identity"]
        ):
            raise ValueError("build receipt disagrees with the source pin")
        library_sha = build["native_library_sha256"]
    else:
        library_sha = source["native_library_sha256"]
    if hardware["native_library_sha256"] != library_sha:
        raise ValueError("campaign native library differs from the build")
    gpu_uuid = hardware["gpu"].split(",")[1].strip()
    seen = set()
    observations = {}
    for row in campaign["rows"]:
        name = row["name"]
        if name in seen:
            raise ValueError(f"duplicate attempt {name}")
        seen.add(name)
        for suffix, key in ((".json", "raw_sha256"), (".txt", "log_sha256")):
            if row.get(key) and checksum(files[name + suffix]) != row[key]:
                raise ValueError(f"incorrect {key} for {name}")
        if row["status"] != "complete":
            continue
        raw = json.loads(files[name + ".json"])
        if (
            raw["status"] != "complete"
            or len(raw["samples"]) != 1
            or not raw["samples"][0]["converged"]
            or raw["samples"][0]["phase"] != "cold"
            or raw["source_revision"] != source["revision"]
            or raw["paper_driver_sha256"] != source["paper_runner_sha256"]
            or raw["geometry_sha256"] != geometry_digest
            or raw["accelerator"]["nvidia_smi"]["uuid"] != gpu_uuid
            or any(raw[k] != row[k] for k in ("atoms", "method", "mode", "arm"))
        ):
            raise ValueError(f"invalid completed outcome {name}")
        if formal and raw["mode"] == "df":
            auxiliary = f"inputs/{raw['method']}-auxiliary.json"
            if raw["auxiliary_input_sha256"] != source["inputs_sha256"][auxiliary]:
                raise ValueError(
                    f"DF auxiliary basis differs from the source pin: {name}"
                )
        if raw["arm"] == "gpu4pyscf":
            if (
                raw["packages"]["gpu4pyscf"] != "1.9.0"
                or raw["packages"]["cupy"] != "13.4.1"
            ):
                raise ValueError("unexpected reference version")
        elif (
            raw["native_build"]["library_sha256"] != library_sha
            or raw["native_build"]["source_identity"]
            != source["native_source_identity"]
        ):
            raise ValueError(f"unexpected native build in {name}")
        sample = raw["samples"][0]
        if not (math.isfinite(sample["seconds"]) and sample["seconds"] > 0):
            raise ValueError(f"invalid endpoint seconds in {name}")
        if row["memory"]["peak_used_bytes"] < row["memory"]["baseline_bytes"]:
            raise ValueError(f"invalid peak memory in {name}")
        if formal and row["stage"] == "formal":
            key = (row["atoms"], row["method"], row["mode"], row["arm"], row["repeat"])
            if key in observations:
                raise ValueError(f"duplicate formal population {key}")
            observations[key] = ({**row, "gpu_uuid": gpu_uuid}, raw["samples"][0])
    return observations


def verify_population(campaign, sizes):
    stages = [("prewarm", 0), *(("formal", repeat) for repeat in range(REPEATS))]
    expected = set(product(sizes, METHODS, MODES, ARMS, stages))
    rows = campaign["rows"]
    actual = {
        (
            row["atoms"],
            row["method"],
            row["mode"],
            row["arm"],
            (row["stage"], row["repeat"]),
        )
        for row in rows
    }
    if (
        campaign["status"] != "finished"
        or len(rows) != len(expected)
        or actual != expected
    ):
        raise ValueError(f"incomplete acquisition population for sizes {sizes}")
    if any(row["status"] == "running" for row in rows):
        raise ValueError(f"unfinished outcome for sizes {sizes}")


def summarize(observations, oracles, source):
    rows = []
    details = {}
    for atoms in SIZES:
        for method in METHODS:
            reference, available = oracle(oracles, atoms, method, source)
            for mode in MODES:
                item = {
                    "atoms": atoms,
                    "method": method,
                    "mode": mode,
                    "oracle": available,
                }
                key = (atoms, method, mode)
                repeats = [
                    repeat
                    for repeat in range(REPEATS)
                    if all((*key, arm, repeat) in observations for arm in ARMS)
                ]
                devices = {
                    observations[(*key, arm, repeat)][0]["gpu_uuid"]
                    for repeat in repeats
                    for arm in ARMS
                }
                if len(devices) > 1:
                    raise ValueError(
                        f"paired observations span multiple GPUs for {key}"
                    )
                item["paired_repeats"] = len(repeats)
                cross_engine = [
                    errors(
                        observations[(*key, "standard", repeat)][1],
                        observations[(*key, "gpu4pyscf", repeat)][1],
                        atoms,
                    )
                    for repeat in repeats
                ]
                for metric in GATES:
                    item[f"cross_engine_max_{metric}"] = max(
                        (difference[metric] for difference in cross_engine),
                        default=None,
                    )
                qualified = []
                for arm in ARMS:
                    records = {
                        r: observations[(*key, arm, r)]
                        for r in range(REPEATS)
                        if (*key, arm, r) in observations
                    }
                    item[f"{arm}_complete"] = len(records)
                    item[f"{arm}_complete_median_seconds"] = (
                        statistics.median(
                            sample["seconds"] for _, sample in records.values()
                        )
                        if records
                        else None
                    )
                    item[f"{arm}_peak_mib"] = max(
                        (
                            row["memory"]["peak_used_bytes"] / 2**20
                            for row, _ in records.values()
                        ),
                        default=None,
                    )
                    item[f"{arm}_paired_median_seconds"] = (
                        statistics.median(records[r][1]["seconds"] for r in repeats)
                        if repeats
                        else None
                    )
                    for row, sample in records.values():
                        detail = {"oracle_status": available}
                        if reference is not None:
                            difference = errors(sample, reference, atoms)
                            detail.update(errors=difference, passed=within(difference))
                            if row["repeat"] in repeats:
                                qualified.append(detail["passed"])
                        details[row["name"]] = detail
                item["gpu4pyscf_over_standard"] = (
                    item["gpu4pyscf_paired_median_seconds"]
                    / item["standard_paired_median_seconds"]
                    if repeats
                    else None
                )
                item["all_paired_observations_pass"] = (
                    all(qualified) if qualified else None
                )
                rows.append(item)
    return rows, details


def as_csv(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    directory = args.directory
    source = json.loads((directory / "source.json").read_text())
    geometry_digest = hashlib.sha256(
        lzma.decompress((directory / "geometries.json.xz").read_bytes())
    ).hexdigest()
    if geometry_digest != source["geometry_sha256"]:
        raise ValueError("geometry archive differs from the frozen source manifest")
    legacy_source = json.loads((directory / "source-legacy.json").read_text())
    for name in ("gpu-pilot.json.xz", "gpu-legacy.json.xz"):
        path = directory / name
        if path.is_file():
            read_attempt(read_bundle(path), False, legacy_source, geometry_digest)
    observations = {}
    failures = []
    build_path = directory / "build-raw.json.xz"
    build_text = read_bundle(build_path)["build.json"] if build_path.is_file() else None
    for shard, sizes in (("main", SIZES[:3]), ("large", SIZES[3:])):
        raw_files = {}
        for path in sorted(directory.glob(f"gpu-{shard}-*.json.xz")):
            for name, contents in read_bundle(path).items():
                if name in raw_files:
                    raise ValueError(f"duplicate raw file {name}")
                raw_files[name] = contents
        if not raw_files:
            raise ValueError(f"missing H100 acquisition shard {shard}")
        shard_observations = read_attempt(
            raw_files, True, source, geometry_digest, build_text
        )
        campaign = json.loads(raw_files["campaign.json"])
        verify_population(campaign, sizes)
        for row in campaign["rows"]:
            if row["status"] == "complete":
                continue
            raw = json.loads(raw_files.get(row["name"] + ".json", "{}"))
            failures.append(
                {
                    "shard": shard,
                    **{
                        key: row[key]
                        for key in (
                            "name",
                            "atoms",
                            "method",
                            "mode",
                            "arm",
                            "stage",
                            "status",
                        )
                    },
                    "last_stage": row.get("last_stage"),
                    "error": raw.get("error", row.get("error")),
                    "subprocess_seconds": row.get("subprocess_seconds"),
                    "peak_mib": row["memory"]["peak_used_bytes"] / 2**20
                    if row.get("memory")
                    else None,
                }
            )
        if overlap := observations.keys() & shard_observations.keys():
            raise ValueError(f"duplicate observations across H100 shards: {overlap}")
        observations.update(shard_observations)
    cpu_path = directory / "cpu-raw.json.xz"
    oracles = read_bundle(cpu_path) if cpu_path.is_file() else {}
    long_pbe_path = directory / "cpu-pbe-long.json.xz"
    if long_pbe_path.is_file():
        name = "oracle-96-pbe.json"
        pbe_raw = read_bundle(long_pbe_path).get(name)
        if pbe_raw and json.loads(pbe_raw)["status"] == "complete":
            oracles[name] = pbe_raw
    rows, checks = summarize(observations, oracles, source)
    artifacts = {
        "summary.csv": as_csv(rows),
        "gates.json": json.dumps({"limits": GATES, "checks": checks}, indent=2) + "\n",
    }
    if failures:
        artifacts["failures.csv"] = as_csv(failures)
    for name, expected in artifacts.items():
        path = directory / name
        if args.write_summary:
            path.write_text(expected)
        elif path.read_bytes() != expected.encode():
            raise ValueError(f"summary differs from replay: {name}")
    print(
        f"{len(observations)} completed formal endpoints, {sum(row['paired_repeats'] for row in rows)} pairs"
    )


if __name__ == "__main__":
    main()

"""CPU-only replay of H100 CBS measurements and the original 5090 gates."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

SOURCE_PACKET = "9afbe5aba7cdb946e3148a40ca1c1b8b8877a6c1"
SOURCE_DIRECTORY = "benchmarks/results/cbs-reference-closeout-20261104"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_bytes(repository: Path, revision: str, name: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repository), "show", revision + ":" + name]
    )


def property_contract(result: dict, atoms: int, *, energy_only: bool = False) -> None:
    """Refuse missing force vectors before classifying a receipt as E+F work."""
    require(
        result["converged"] is True and result["executed_backend"] == "cuda",
        "incomplete CUDA endpoint",
    )
    require(
        len(result["components"]) == 5 and len(result["execution_groups"]) == 2,
        "incomplete CBS work",
    )
    require(
        np.isfinite(result["complete_seconds"]) and result["complete_seconds"] > 0,
        "invalid endpoint time",
    )
    for value in [result] + [row["result"] for row in result["components"]]:
        require(np.isfinite(value["energy"]), "nonfinite endpoint energy")
        if energy_only:
            require(
                value["forces"] is None,
                "energy-only endpoint unexpectedly retains forces",
            )
        else:
            require(
                value["forces"] is not None, "missing complete-endpoint force vector"
            )
            forces = np.asarray(value["forces"], dtype=float)
            require(
                forces.shape == (atoms, 3) and bool(np.isfinite(forces).all()),
                "invalid physical force vector",
            )


def force_field(result: dict, field: str) -> list:
    if field == "composite":
        return result["forces"]
    for component in result["components"]:
        level = (
            "high"
            if any(role.endswith("_high") for role in component["roles"])
            else "low"
        )
        if level + "/" + component["method"] == field:
            return component["result"]["forces"]
    raise ValueError("missing directional constituent")


def dot_roundoff_bound(forces: list, direction: list) -> float:
    """Bound two FP64 dot reductions using the standard gamma(2*n) bound."""
    products = np.abs(
        np.asarray(forces, dtype=float) * np.asarray(direction, dtype=float)
    )
    epsilon = np.finfo(float).eps
    gamma = 2 * products.size * epsilon / (1 - 2 * products.size * epsilon)
    return float(2 * gamma * np.sum(products))


def compare_routine_reports(
    actual: dict, expected: dict, task: Path, original: dict
) -> None:
    """Keep every gate/observation exact; permit bounded C2 dot-reduction roundoff."""
    require(set(actual) == set(expected), "routine report fields changed")
    for field, value in actual.items():
        if field != "errors":
            require(value == expected[field], "routine report differs: " + field)
    require(
        set(actual["errors"]) == set(expected["errors"]),
        "routine observable set changed",
    )
    native = json.loads((task / "run-v1/qm7-C2.json").read_text())
    direction = original["qm7_reference"]["direction"]
    for key, row in actual["errors"].items():
        target = expected["errors"][key]
        require(set(row) == set(target), "routine observable fields changed")
        parts = key.split("/")
        directional = (
            len(parts) >= 5
            and parts[0] == "qm7-C2"
            and parts[2] == "independent-C2-direction"
        )
        bound = 0.0
        if directional:
            sample = native["samples"][int(parts[1])]
            bound = dot_roundoff_bound(
                force_field(sample, "/".join(parts[4:])), direction
            )
        for field in row:
            if directional and field in ("max_absolute_error", "rms_error"):
                require(
                    np.isfinite(row[field])
                    and np.isfinite(target[field])
                    and abs(row[field] - target[field]) <= bound,
                    "C2 report exceeds FP64 reduction bound: " + key,
                )
            else:
                require(
                    row[field] == target[field],
                    "routine observable differs: " + key + "/" + field,
                )


def compare_supplemental_reports(
    actual: dict, expected: dict, task: Path, repository: Path
) -> None:
    """Allow only FP64 reduction roundoff in C4/C7 directional scalars."""
    require(set(actual) == set(expected), "supplemental report fields changed")
    for field, value in actual.items():
        if field != "errors":
            require(
                value == expected[field],
                "supplemental report differs: " + field,
            )
    require(
        set(actual["errors"]) == set(expected["errors"]),
        "supplemental observable set changed",
    )
    raw = git_bytes(
        repository,
        "4370d6c4b1bf4033818f31a9fcc622d3045a33bf",
        "benchmarks/results/cbs-independent-forces-20261024/inputs.json.xz",
    )
    inputs = json.loads(lzma.decompress(raw))
    for key, row in actual["errors"].items():
        target = expected["errors"][key]
        require(set(row) == set(target), "supplemental observable fields changed")
        parts = key.split("/")
        directional = len(parts) >= 6 and parts[3] == "direction"
        bound = 0.0
        if directional:
            carbon = int(parts[0][1:])
            native = json.loads((task / f"run-v1/qm7-C{carbon}.json").read_text())
            sample = next(
                r
                for r in native["samples"]
                if (r["policy"], r["repeat"]) == (parts[1], int(parts[2]))
            )
            direction = inputs["tasks"][f"C{carbon}-h0-minus"]["direction"]
            bound = dot_roundoff_bound(
                force_field(sample, "/".join(parts[5:])), direction
            )
        for field in row:
            if directional and field in ("native", "max_absolute_error"):
                require(
                    np.isfinite(row[field])
                    and np.isfinite(target[field])
                    and abs(row[field] - target[field]) <= bound,
                    "supplemental report exceeds FP64 reduction bound: " + key,
                )
            else:
                require(
                    row[field] == target[field],
                    "supplemental observable differs: " + key + "/" + field,
                )


def restore(capsules: Iterable[Path], destination: Path) -> set[str]:
    seen = set()
    for capsule in capsules:
        for row in json.loads(lzma.decompress(capsule.read_bytes())):
            name = row["path"]
            path = PurePosixPath(name)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and str(path) == name
                and "\\" not in name
                and ":" not in name,
                "unsafe retained path",
            )
            require(name not in seen, "duplicate retained member")
            seen.add(name)
            require(("text" in row) != ("base64" in row), "ambiguous retained encoding")
            raw = (
                row["text"].encode("utf-8")
                if "text" in row
                else base64.b64decode(row["base64"], validate=True)
            )
            require(
                len(raw) == row["bytes"] and digest(raw) == row["sha256"],
                "retained member differs: " + name,
            )
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
            if os.name == "posix":
                target.chmod(row["mode"] & 0o777)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--recover", type=Path)
    args = parser.parse_args()
    require(
        not sys.flags.optimize and os.environ.get("PYTHONOPTIMIZE") in (None, "", "0"),
        "unoptimized Python required by frozen readers",
    )
    package = Path(__file__).resolve().parent
    repository = args.repository or package.parents[2]
    publication = json.loads((package / "publication.json").read_text())
    require(publication["source_packet"] == SOURCE_PACKET, "source packet pin changed")
    require(
        publication["decision"]["status"] == "accepted", "publication is not accepted"
    )
    entries = publication["files"]
    require(
        len({row["path"] for row in entries}) == len(entries),
        "duplicate publication path",
    )
    for row in entries:
        path = PurePosixPath(row["path"])
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and str(path) == row["path"],
            "unsafe publication path",
        )
        raw = (package / row["path"]).read_bytes()
        require(
            len(raw) == row["bytes"] and digest(raw) == row["sha256"],
            "publication file differs: " + row["path"],
        )
    capsules = sorted(package.glob("records-*.json.xz"))
    require(
        capsules
        and {c.name for c in capsules}
        == {r["path"] for r in entries if r["path"].startswith("records-")},
        "retained capsule inventory changed",
    )
    with tempfile.TemporaryDirectory(prefix="gqc-h100-cbs-") as temporary_name:
        temporary = Path(temporary_name)
        task = args.recover or temporary / "measurements"
        require(not task.exists(), "recovery destination must not exist")
        task.mkdir(parents=True)
        members = restore(capsules, task)
        require(members == set(publication["members"]), "retained member set changed")
        for name, identity in publication["members"].items():
            path = task / name
            require(
                path.stat().st_size == identity["bytes"]
                and digest(path.read_bytes()) == identity["sha256"],
                "member binding changed",
            )
        # Replay the already-published SM90 build/source/reference capsule in full.
        source_package = temporary / "source-packet"
        source_package.mkdir()
        raw_manifest = git_bytes(
            repository, SOURCE_PACKET, SOURCE_DIRECTORY + "/publication.json"
        )
        require(
            digest(raw_manifest) == publication["source_publication_sha256"],
            "source publication changed",
        )
        source_manifest = json.loads(raw_manifest)
        (source_package / "publication.json").write_bytes(raw_manifest)
        for row in source_manifest["files"]:
            raw = git_bytes(
                repository, SOURCE_PACKET, SOURCE_DIRECTORY + "/" + row["path"]
            )
            require(
                len(raw) == row["bytes"] and digest(raw) == row["sha256"],
                "source packet file changed",
            )
            (source_package / row["path"]).write_bytes(raw)
        subprocess.run(
            [
                sys.executable,
                str(source_package / "replay.py"),
                "--repository",
                str(repository),
                "--recover",
                str(temporary / "source-recovery"),
            ],
            check=True,
        )
        inputs = json.loads((task / "inputs.json").read_text())
        require(
            inputs["historical_archive"] == "4370d6c4b1bf4033818f31a9fcc622d3045a33bf"
            and inputs["historical_records_path"]
            == "benchmarks/results/cbs-routine-standard-20261025/records.json.xz"
            and inputs["historical_records_sha256"]
            == "03b2722d016c13daaf6025a021de18e0ac333ebbb6efe94943c57748a57f5d3a",
            "original 5090 scientific reference pin changed",
        )
        historical = git_bytes(
            repository, inputs["historical_archive"], inputs["historical_records_path"]
        )
        require(
            digest(historical) == inputs["historical_records_sha256"],
            "historical independent reference changed",
        )
        (task / "historical-records.json.xz").write_bytes(historical)
        original = json.loads(lzma.decompress(historical))
        expected_cohort = {}
        for group in original["reference"].values():
            for name, row in group["cases"].items():
                expected_cohort[row["protocol"]["model"] + "--" + name] = {
                    "kind": "validation",
                    "name": name,
                    "protocol": row["protocol"],
                }
        for run in original["qm7_runs"].values():
            for carbon, protocol in run["protocols"].items():
                expected_cohort["qm7-C" + carbon] = {
                    "kind": "timing",
                    "carbon": int(carbon),
                    "protocol": protocol,
                }
        require(
            len(expected_cohort) == 20 and inputs["cohort"] == expected_cohort,
            "original 13 populations or seven QM7 sizes changed",
        )
        plan = json.loads((task / "plan.json").read_text())
        supervisor = json.loads((task / "run-v1/supervisor.json").read_text())
        require(
            supervisor["status"] == "COMPLETED_UNVERIFIED_OUTCOMES"
            and supervisor["aggregate_verification_returncode"] == 0,
            "incomplete supervisor",
        )
        require(
            len(supervisor["rows"]) == 20
            and {r["name"] for r in supervisor["rows"]} == set(inputs["cohort"]),
            "incomplete cohort dispatch",
        )
        require(
            all(
                r["returncode"] == r["verification_returncode"] == 0
                and r["status"] == "NUMERICAL_PASS"
                for r in supervisor["rows"]
            ),
            "failed population retained",
        )
        require(
            supervisor["device_before"] == supervisor["device_after"]
            and "H100" in supervisor["device_before"]
            and supervisor["device_before"].endswith("9.0"),
            "device identity changed",
        )
        expected_source = json.loads(
            (temporary / "source-recovery/source.json").read_text()
        )
        source_build = json.loads(
            (temporary / "source-recovery/build-receipts/build.json").read_text()
        )
        source_native = json.loads(
            (temporary / "source-recovery/source-native.json").read_text()
        )
        require(
            supervisor["source"] == expected_source
            and supervisor["library_sha256"] == source_build["library_sha256"],
            "source/core provenance differs",
        )
        require(
            set(supervisor["producer_sha256"])
            == {"native.py", "run.py", "verify.py", "process-owner.py", "prepare.py"},
            "executed producer inventory changed",
        )
        require(
            set(supervisor["input_sha256"])
            == {
                "inputs.json",
                "basis-manifest.json",
                "historical-records.json.xz",
                "plan.json",
            },
            "executed input inventory changed",
        )
        for field, name in [
            ("source_runner_sha256", "native.py"),
            ("supervisor_sha256", "run.py"),
            ("process_owner_sha256", "process-owner.py"),
        ]:
            require(
                supervisor[field] == digest((task / name).read_bytes()),
                "executed producer identity changed: " + name,
            )
        require(
            set(plan["producer_hashes"])
            == {
                "native.py",
                "run.py",
                "verify.py",
                "process-owner.py",
                "prepare.py",
                "release.py",
            },
            "planned producer inventory changed",
        )
        for group in ("producer_sha256", "input_sha256"):
            for name, checksum in supervisor[group].items():
                require(
                    digest((task / name).read_bytes()) == checksum,
                    "executed producer/input differs: " + name,
                )
        for name, checksum in plan["producer_hashes"].items():
            require(
                digest((task / name).read_bytes()) == checksum,
                "planned producer differs: " + name,
            )
        preflight_raw = (task / "run-v1/preflight.json").read_bytes()
        preflight = json.loads(preflight_raw)
        expected_preflight = {
            "job_name": supervisor["qz_job_name"],
            "hostname": supervisor["hostname"],
            "device": supervisor["device_before"],
            "library_sha256": supervisor["library_sha256"],
            "source": supervisor["source"],
            "source_runner_sha256": supervisor["source_runner_sha256"],
            "supervisor_sha256": supervisor["supervisor_sha256"],
            "process_owner_sha256": supervisor["process_owner_sha256"],
            "producer_sha256": supervisor["producer_sha256"],
            "input_sha256": supervisor["input_sha256"],
        }
        require(
            preflight == expected_preflight,
            "authenticated preflight differs from executed supervisor",
        )
        release_raw = (task / "cohort-v1-release.json").read_bytes()
        release = json.loads(release_raw)
        platform_raw = (task / "cohort-v1-platform-running.json").read_bytes()
        platform_logs = (task / "cohort-v1-platform-preflight-logs.txt").read_bytes()
        require(
            supervisor["dispatch_release"] == release
            and supervisor["dispatch_release_sha256"] == digest(release_raw)
            and release["preflight"] == preflight
            and release["platform_status"] == "RUNNING"
            and release["raw_platform_sha256"] == digest(platform_raw)
            and release["raw_platform_log_sha256"] == digest(platform_logs),
            "actual dispatch release differs",
        )
        require(
            ("GQC_CBS_PREFLIGHT_SHA256 " + digest(preflight_raw)).encode()
            in platform_logs,
            "platform does not bind this preflight",
        )
        platform = json.loads(platform_raw)
        require(
            platform["success"]
            and platform["data"]["name"] == supervisor["qz_job_name"]
            and platform["data"]["status"] == "RUNNING",
            "platform did not attest running allocation",
        )
        resource = platform["data"]["resource"]
        require(
            resource["gpu"] == 1
            and resource["cpu"] == 10
            and resource["memory_gib"] == 200
            and resource["nodes"] == 1
            and resource["shared_memory_gib"] == 8
            and "H100" in resource["gpu_type"],
            "allocation differs",
        )
        terminal = json.loads((task / "platform-terminal.json").read_text())
        require(
            terminal["success"]
            and terminal["data"]["name"] == supervisor["qz_job_name"]
            and terminal["data"]["status"] == "SUCCEEDED",
            "terminal allocation not released successfully",
        )
        instances = json.loads((task / "platform-terminal-instances.json").read_text())
        require(
            instances["success"]
            and instances["data"]["name"] == supervisor["qz_job_name"]
            and len(instances["data"]["items"]) == 1
            and instances["data"]["items"][0]["status"] == "instance_success",
            "successful worker termination not confirmed",
        )
        submission = json.loads((task / "submit-argv.json").read_text())
        require(
            submission[:5]
            == ["inspire", "--no-env-file", "--account", "default", "job"],
            "submission account/environment contract changed",
        )
        for option, value in [
            ("-n", supervisor["qz_job_name"]),
            ("--quota", "1,10,200"),
            ("--nodes", "1"),
            ("--image", "generativeqc-science-py311-cuda129:20261015"),
            ("--max-time", "2"),
            ("--priority", "4"),
            ("--shm-size", "8"),
            ("--workspace", "分布式训练空间"),
        ]:
            require(
                submission.count(option) == 1
                and submission[submission.index(option) + 1] == value,
                "submission contract changed: " + option,
            )
        command = json.loads((task / "platform-command.json").read_text())
        require(
            "--no-auto-fault-tolerance" in submission
            and "--no-enable-notification" in submission,
            "retry/notification submission contract changed",
        )
        require(
            command["success"]
            and command["data"]["command"]
            == "bash -c " + shlex.quote(submission[submission.index("--command") + 1]),
            "platform did not execute the recorded launch command",
        )
        basis_manifest = json.loads((task / "basis-manifest.json").read_text())
        require(set(basis_manifest) == set(inputs["cohort"]), "basis cohort changed")
        ef_calls = eo_calls = startups = 0
        for key, entry in inputs["cohort"].items():
            native = json.loads((task / "run-v1" / (key + ".json")).read_text())
            atoms = len(entry["protocol"]["atoms_bohr"])
            for result in native["samples"] + [native["startup"]]:
                property_contract(result, atoms)
            if entry["kind"] == "validation":
                property_contract(native["energy_only"], atoms, energy_only=True)
            require(
                native["source"] == expected_source
                and native["library_sha256"] == source_build["library_sha256"]
                and native["native_source_identity"]
                == source_native["native_source_identity"],
                "population source differs",
            )
            require(
                native["driver_sha256"] == digest((task / "native.py").read_bytes())
                and native["basis_manifest_sha256"]
                == digest((task / "basis-manifest.json").read_bytes()),
                "population driver/basis provenance differs",
            )
            require(
                native["startup"]["converged"]
                and native["startup"]["executed_backend"] == "cuda"
                and native["startup"]["forces"] is not None,
                "missing per-process startup",
            )
            startups += 1
            ef_calls += len(native["samples"]) + 1
            eo_calls += int("energy_only" in native)
            for role in ("orbital", "auxiliary"):
                rows = basis_manifest[key][role]
                require(
                    [r["identity"] for r in rows]
                    == entry["protocol"][role + "_identities"],
                    "prepared basis identities differ",
                )
                for row in rows:
                    require(
                        digest((task / row["path"]).read_bytes()) == row["sha256"],
                        "prepared basis bytes differ",
                    )
        require(
            (ef_calls, eo_calls, startups) == (93, 13, 20),
            "complete endpoint work set changed",
        )
        require(
            plan["cohort"]["full_energy_force_calls"] == ef_calls
            and plan["cohort"]["energy_only_calls"] == eo_calls,
            "declared work differs",
        )
        # Execute only the frozen numerical reader, never native.py/run.py/prepare.py.
        output = temporary / "recomputed.json"
        subprocess.run(
            [
                sys.executable,
                str(task / "verify.py"),
                "--records",
                str(task / "run-v1"),
                "--output",
                str(output),
            ],
            check=True,
        )
        actual = json.loads(output.read_text())
        require(
            actual["status"] == "PASS" and len(actual["cases"]) == 20,
            "original numerical gates failed",
        )
        aggregate = json.loads(
            (task / "run-v1/aggregate-verification.json").read_text()
        )
        compare_routine_reports(actual, aggregate, task, original)
        require(
            aggregate
            == json.loads(lzma.decompress((package / "summary.json.xz").read_bytes())),
            "published numerical summary differs",
        )
        supplemental_output = temporary / "supplemental.json"
        subprocess.run(
            [
                sys.executable,
                str(task / "verify-supplemental.py"),
                "--repository",
                str(repository),
                "--inputs",
                str(task / "inputs.json"),
                "--records",
                str(task / "run-v1"),
                "--output",
                str(supplemental_output),
            ],
            check=True,
        )
        supplemental = json.loads(supplemental_output.read_text())
        require(
            supplemental["status"] == "PASS" and len(supplemental["errors"]) == 180,
            "carried independent alkane reference gates failed",
        )
        compare_supplemental_reports(
            supplemental,
            json.loads(
                lzma.decompress((package / "supplemental-summary.json.xz").read_bytes())
            ),
            task,
            repository,
        )
        print(
            "PASS: 13 independent full-Cartesian populations, C1-C7, 93 E+F / 13 energy-only calls;",
            len(actual["errors"]),
            "routine-cohort observable checks; no solver/device execution",
        )


if __name__ == "__main__":
    main()

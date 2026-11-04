"""Recompute frozen DB36 H100 safety proofs from original observations."""

from __future__ import annotations

import hashlib
import json
import lzma
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks import h100_production_cohort as support

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "benchmarks/results/h100-sm90-db36-safety-20261103"
SOURCE = "db3657f08d983791c45b7d949cb9028fe0055e3f"
TREE = "681cefe8bd066a3fb973ab38e0ef856280625ed1"
CORE = "472bd0f9bceaa7917f7137c2cb543f0ebfc6c0b728ed024aeacc81b92c8c83a4"
REFERENCE = "15ab66a68a061355fda833a93e88484dbfee6360"
PACKETS = {
    "production-build-v1": "production-build-v1-metadata.tar.xz",
    "native-qualification-v1": "native-qualification-v1-completed.tar.xz",
    "public-direct-v1": "public-direct-v1-completed.tar.xz",
    "spatial-v1": "spatial-v1-completed.tar.xz",
    "spatial-racecheck-v2": "spatial-racecheck-v2-completed.tar.xz",
}
READERS = {
    "recover-build-v2.py",
    "unpack-native-v2.py",
    "replay-native-v3.py",
    "replay-native-rsh-stencils-v2.py",
    "unpack-public-direct-v2.py",
    "replay-public-direct-v3.py",
    "unpack-spatial-v2.py",
    "replay-spatial-v2.py",
    "replay-spatial-components-v2.py",
    "unpack-spatial-racecheck-v1.py",
    "replay-spatial-racecheck-v3.py",
    "validate-spatial-components-v1.py",
    "validate-spatial-racecheck-v1.py",
    "replay-spatial-composite-v1.py",
}
PROOFS = {
    "production-build-v1-local-replay.json": "PASS_RECEIVER_SM90_PRODUCTION33_BUILD_METADATA_ONLY",
    "native-qualification-v1-local-replay-v3.json": "PASS_CURRENT_SOURCE_NATIVE_FOUR_UNFILTERED_MEMCHECK",
    "native-rsh-stencils-v1-local-replay.json": "PASS_CURRENT_SOURCE_INDEPENDENT_RSH_STENCIL_REPLAY",
    "public-direct-v1-local-replay-v3.json": "PASS_CURRENT_SOURCE_AFFECTED_PUBLIC_DIRECT_EF_AND_FAULT_TESTS",
    "spatial-v1-components-local-replay-v1.json": "PASS_CURRENT_SOURCE_NORMAL6_AND_UNFILTERED_MEMCHECK6_COMPONENTS_ONLY",
    "spatial-racecheck-v2-local-replay-v1.json": "PASS_CURRENT_SOURCE_UNFILTERED_RACECHECK6_AND_POSITIVE_WORK",
    "spatial-composite-v1-local-replay.json": "PASS_CURRENT_SOURCE_GROUPED_DIRECT_SIX_ALL_MODES_AND_POSITIVE_WORK",
}


def check(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decoded_manifest(binding: dict) -> bytes:
    data = lzma.decompress((DIRECTORY / binding["path"]).read_bytes())
    check(
        len(data) == binding["decoded_bytes"]
        and digest(data) == binding["decoded_sha256"],
        "Original decoded manifest differs",
    )
    return data


def checked_publication() -> dict:
    publication = json.loads((DIRECTORY / "cohort.json").read_text())
    check(
        publication["schema"] == "generativeqc.h100-db36-safety.v1"
        and publication["source_revision"] == SOURCE
        and publication["source_tree"] == TREE
        and publication["core_sha256"] == CORE
        and publication["reference_source_revision"] == REFERENCE
        and publication["majority_accepted"] is False
        and publication["clean_timing_votes"] == 0
        and publication["original_failed_spatial_race_accepted_cases"] == 0
        and publication["complete_EF"] == 36
        and publication["spatial_energy_only_SCF"] == 18
        and publication["spatial_source_exports"] == 54,
        "Frozen safety scope differs",
    )
    required = {"readers/" + name for name in READERS} | set(PACKETS.values())
    required |= {name + "-download-manifest.json.xz" for name in PACKETS}
    required.add("actual-integration-audit.json.xz")
    entries = publication["files"]
    check(
        len(entries) == len(required) and {b["path"] for b in entries} == required,
        "Required original safety inventory differs",
    )
    by_name = {b["path"]: b for b in entries}
    for binding in entries:
        relative = Path(binding["path"])
        path = DIRECTORY / relative
        check(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.name != ".env"
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == binding["bytes"]
            and digest(path.read_bytes()) == binding["sha256"],
            "Original safety bytes differ: " + str(relative),
        )
    packets = publication["packets"]
    check(
        [b["cohort"] for b in packets] == list(PACKETS),
        "Original safety packet sequence differs",
    )
    for packet in packets:
        name = packet["cohort"]
        archive, manifest = packet["archive"], packet["manifest"]
        check(
            archive == by_name[PACKETS[name]]
            and manifest["decoded_name"] == name + "-download-manifest.json"
            and manifest["path"] == manifest["decoded_name"] + ".xz"
            and all(
                manifest[k] == by_name[manifest["path"]][k]
                for k in ["path", "bytes", "sha256"]
            ),
            "Archive/manifest identity differs",
        )
        decoded_manifest(manifest)
    check(
        set(publication["proofs"]) == set(PROOFS)
        and all(publication["proofs"][n]["status"] == s for n, s in PROOFS.items()),
        "Original proof inventory differs",
    )
    binding = publication["integration_review"]
    check(binding["path"] == "actual-integration-audit.json.xz", "Review name differs")
    raw = lzma.decompress((DIRECTORY / binding["path"]).read_bytes())
    check(
        len(raw) == binding["decoded_bytes"]
        and digest(raw) == binding["decoded_sha256"],
        "Actual integration review bytes differ",
    )
    report = json.loads(raw)
    check(
        report["source_revision"] == SOURCE
        and report["source_tree"] == TREE
        and report["current_gates_verdict"] == "PASS_ACTUAL_AFFECTED_SOURCE_BOUND_GATES"
        and report["dispatch_blockers"] == []
        and report["review_completed"] is True
        and report["review_scope_complete"] is True
        and report["review_positive"] is True
        and report["current_majority_accepted"] is False,
        "Actual source-bound review incomplete",
    )
    for name, proof in publication["proofs"].items():
        check(
            any(
                Path(p).name == name and h.removeprefix("sha256:") == proof["sha256"]
                for p, h in report["audited_input_hashes"].items()
            ),
            "Published safety proof was not audited: " + name,
        )
    return publication


def install(task: Path, cohort: str) -> None:
    producers = task / (cohort + "-executed-producers")
    if producers.exists():
        for source in sorted(producers.iterdir()):
            if source.is_file():
                support.copy_exact(source, task / source.name)
    admitted = task / (cohort + "-available-admissions")
    if admitted.exists():
        for source in admitted.rglob("*"):
            if source.is_file():
                relative = source.relative_to(admitted)
                if (
                    relative.as_posix() not in PROOFS
                    and relative.parts[0] != "external"
                ):
                    support.copy_exact(source, task / relative)


def reader(task: Path, name: str, *arguments: str) -> None:
    result = subprocess.run(
        [sys.executable, str(task / name), *arguments],
        cwd=task,
        env=support.ENV,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(name + " failed:\n" + result.stdout + "\n" + result.stderr)


def build_producers(task: Path) -> None:
    manifest = json.loads(
        (task / "production-build-v1-download-manifest.json").read_text()
    )
    with tarfile.open(task / PACKETS["production-build-v1"]) as archive:
        for name in ["build-integration-v1.py", "build-integration-v1-job.sh"]:
            original = "production-build-v1-executed-producers/" + name
            member = archive.getmember(original)
            check(member.isfile(), "Build producer is not a regular file")
            data = archive.extractfile(member).read()
            binding = manifest["files"][original]
            check(
                len(data) == binding["bytes"] and digest(data) == binding["sha256"],
                "Build producer differs",
            )
            with (task / name).open("xb") as stream:
                stream.write(data)


def replay(*, extension: Callable[[Path, Path], dict] | None = None) -> dict:
    check(
        sys.version_info >= (3, 12) and not sys.flags.optimize,
        "Use unoptimized Python3.12+",
    )
    publication = checked_publication()
    additional = None
    with tempfile.TemporaryDirectory(prefix="gqc-db36-safety-") as temporary:
        base = Path(temporary)
        task = base / "tasks/cohort-db36"
        task.mkdir(parents=True)
        for name, revision in [
            ("generativeqc-h100-current-evidence", SOURCE),
            ("generativeqc-h100-measurement-db36", SOURCE),
            ("generativeqc-h100-production", REFERENCE),
        ]:
            support.snapshot(base / name, revision, git_metadata=True)
        for name in READERS:
            support.copy_exact(DIRECTORY / "readers" / name, task / name)
        for packet in publication["packets"]:
            name = packet["archive"]["path"]
            support.copy_exact(DIRECTORY / name, task / name)
            manifest = packet["manifest"]
            with (task / manifest["decoded_name"]).open("xb") as stream:
                stream.write(decoded_manifest(manifest))
        build_producers(task)
        reader(task, "recover-build-v2.py")
        install(task, "production-build-v1")
        for cohort, unpacker, receivers in [
            (
                "native-qualification-v1",
                "unpack-native-v2.py",
                ["replay-native-v3.py", "replay-native-rsh-stencils-v2.py"],
            ),
            (
                "public-direct-v1",
                "unpack-public-direct-v2.py",
                ["replay-public-direct-v3.py"],
            ),
            ("spatial-v1", "unpack-spatial-v2.py", ["replay-spatial-components-v2.py"]),
            (
                "spatial-racecheck-v2",
                "unpack-spatial-racecheck-v1.py",
                ["replay-spatial-racecheck-v3.py", "replay-spatial-composite-v1.py"],
            ),
        ]:
            reader(task, unpacker)
            install(task, cohort)
            for name in receivers:
                reader(task, name)
        proofs = {}
        values = {}
        for name, status in PROOFS.items():
            data = (task / name).read_bytes()
            value = json.loads(data)
            check(
                value["status"] == status
                and value["source_revision"] == SOURCE
                and digest(data) == publication["proofs"][name]["sha256"],
                "Recomputed original proof differs: " + name,
            )
            proofs[name], values[name] = digest(data), value
        for admitted in task.glob("*-available-admissions"):
            for source in admitted.rglob("*.json"):
                name = source.relative_to(admitted).as_posix()
                if name in PROOFS:
                    check(
                        source.read_bytes() == (task / name).read_bytes(),
                        "Consumed proof differs: " + name,
                    )
        public = values["public-direct-v1-local-replay-v3.json"]["summaries"]
        spatial = values["spatial-composite-v1-local-replay.json"]
        check(
            public["production"]["tests"] == 8
            and public["production"]["full_EF"] == 33
            and public["fixture"]["tests"] == 3
            and public["fixture"]["full_EF"] == 3
            and spatial["energy_only_scf"] == 18
            and spatial["source_exports"] == 54
            and spatial["full_EF"] == spatial["clean_timing_votes"] == 0,
            "Public/spatial population differs",
        )
        if extension is not None:
            additional = extension(base, task)
    result = {
        "status": "PASS_PUBLIC_DB36_SEVEN_SAFETY_PROOFS_ONLY",
        "source_revision": SOURCE,
        "source_tree": TREE,
        "core_sha256": CORE,
        "proofs": proofs,
        "complete_EF": 36,
        "spatial_energy_only_SCF": 18,
        "spatial_source_exports": 54,
        "original_failed_race_accepted_cases": 0,
        "clean_timing_votes": 0,
        "majority_accepted": False,
    }
    if additional is not None:
        result["additional"] = additional
    return result


def main() -> None:
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(replay(), indent=2))


if __name__ == "__main__":
    main()

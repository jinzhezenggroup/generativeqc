"""CPU replay of the frozen cd2b H100 build, DF regression and screening evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "benchmarks/results/h100-sm90-cd2b-20261102"
SOURCE = "cd2b4514be0a38c4920934481b1f02d157a694b4"
TREE = "c76f5bd3cfa62416255c5d6914cf701d34e10a8d"
REFERENCE = "15ab66a68a061355fda833a93e88484dbfee6360"
REQUIRED_FILES = {
    "production-build-v1-metadata.tar.xz",
    "production-build-v1-download-manifest.json",
    "df-molecular-v1-completed.tar.xz",
    "df-molecular-v1-download-manifest.json",
    "screening-v1-completed.tar.xz",
    "screening-v1-download-manifest.json",
    "readers/replay_build.py",
    "readers/unpack_terminal.py",
    "readers/replay_df.py",
    "readers/replay_screening.py",
    "audit/EXPERIMENT_AUDIT.md",
    "audit/EXPERIMENT_AUDIT.json",
    "audit/artifact-manifest.json",
    "audit/preservation.json",
}
LOCAL_GIT_ENV = {
    **os.environ,
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_ALLOW_PROTOCOL": "",
    "GIT_TERMINAL_PROMPT": "0",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_publication(directory: Path) -> dict:
    publication = json.loads((directory / "cohort.json").read_text())
    if (
        publication.get("schema") != "generativeqc.h100-current-cohort.v1"
        or publication.get("source_revision") != SOURCE
        or publication.get("source_tree") != TREE
        or publication.get("reference_source_revision") != REFERENCE
    ):
        raise ValueError("Publication source/schema differs from the frozen cohort")
    entries = publication["files"]
    names = [entry["path"] for entry in entries]
    if len(names) != len(set(names)) or set(names) != REQUIRED_FILES:
        raise ValueError("Publication must bind every required reader and artifact")
    for binding in entries:
        name = binding["path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe publication member")
        path = directory / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked publication member: {name}")
        if path.stat().st_size != binding["bytes"] or digest(path) != binding["sha256"]:
            raise ValueError(f"Publication bytes differ: {name}")
    return publication


def _reader(path: Path, workspace: Path, reference: Path, *arguments: str) -> None:
    # Each receiver gets a clean module table. A prior NumPy/benchmark import
    # must not change which frozen helper files the next receiver authenticates.
    launcher = """
import importlib.util, pathlib, sys
path, workspace, repository, reference, *arguments = sys.argv[1:]
spec=importlib.util.spec_from_file_location('_retained_receiver',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.TASK=pathlib.Path(workspace)
module.REPO=pathlib.Path(repository)
module.ROOT=pathlib.Path(reference)
sys.path[:0]=[workspace,str(pathlib.Path(reference)/'python')]
sys.argv=[path,*arguments]
module.main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            launcher,
            str(path),
            str(workspace),
            str(ROOT),
            str(reference),
            *arguments,
        ],
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
        env=LOCAL_GIT_ENV,
    )
    if result.returncode:
        raise RuntimeError(f"{path.name} failed:\n{result.stdout}\n{result.stderr}")


def _install_producers(workspace: Path, cohort: str) -> None:
    for path in (workspace / (cohort + "-executed-producers")).iterdir():
        if not path.is_file():
            continue
        target = workspace / path.name
        if target.exists():
            if target.read_bytes() != path.read_bytes():
                raise ValueError(f"Conflicting frozen producer: {path.name}")
        else:
            shutil.copyfile(path, target)


def replay(directory: Path = DIRECTORY) -> dict:
    if sys.version_info < (3, 12):
        raise RuntimeError("Replay requires Python 3.12 or later")
    publication = checked_publication(directory)
    for revision in [
        publication["source_revision"],
        publication["reference_source_revision"],
    ]:
        if subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", revision + "^{commit}"],
            capture_output=True,
            check=False,
            env=LOCAL_GIT_ENV,
        ).returncode:
            raise RuntimeError(
                f"Frozen Git object unavailable; fetch it with: git fetch origin {revision}"
            )
    with tempfile.TemporaryDirectory(prefix="gqc-h100-cd2b-") as temporary:
        workspace = Path(temporary) / "packet"
        workspace.mkdir()
        for entry in publication["files"]:
            name = entry["path"]
            if Path(name).parent == Path("."):
                shutil.copyfile(directory / name, workspace / name)
        readers = directory / "readers"
        build_manifest = json.loads(
            (workspace / "production-build-v1-download-manifest.json").read_text()
        )
        with tarfile.open(workspace / build_manifest["archive"]) as stream:
            for member in stream.getmembers():
                relative = Path(member.name)
                if relative.parent == Path("production-build-v1-executed-producers"):
                    if not member.isfile():
                        raise ValueError("Linked build producer")
                    (workspace / relative.name).write_bytes(
                        stream.extractfile(member).read()
                    )
        _reader(readers / "replay_build.py", workspace, ROOT)
        _install_producers(workspace, "production-build-v1")
        for cohort in ["df-molecular-v1", "screening-v1"]:
            _reader(readers / "unpack_terminal.py", workspace, ROOT, "--cohort", cohort)
            _install_producers(workspace, cohort)
        _reader(readers / "replay_df.py", workspace, ROOT)
        reference = Path(temporary) / "reference"
        reference.mkdir()
        archive = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "archive",
                publication["reference_source_revision"],
                "--",
                "benchmarks/*.py",
                "python",
                "manifests",
            ],
            env=LOCAL_GIT_ENV,
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(reference, filter="data")
        oracle = Path("benchmarks/results/readme-repaired-20261021/oracles.json.xz")
        (reference / oracle).parent.mkdir(parents=True)
        shutil.copyfile(ROOT / oracle, reference / oracle)
        _reader(readers / "replay_screening.py", workspace, reference)
        build = json.loads(
            (workspace / "production-build-v1-local-replay.json").read_text()
        )
        df = json.loads((workspace / "df-molecular-v1-local-replay.json").read_text())
        screening = json.loads(
            (workspace / "screening-v1-local-replay-v3.json").read_text()
        )
        if not (
            build["source_revision"]
            == df["source_revision"]
            == screening["source_revision"]
            == SOURCE
            and build["source_tree"] == TREE
        ):
            raise ValueError("Receiver-returned source identities differ")
        policy = screening["global_policy_summaries"][0]
        return {
            "schema": "generativeqc.h100-current-cohort-replay.v1",
            "source_revision": publication["source_revision"],
            "build_status": build["status"],
            "df_status": df["status"],
            "df_tests": df["tests"],
            "df_public_full_ef": df["public_full_ef"],
            "screening_status": screening["status"],
            "screening_processes": len(screening["rows"]),
            "primary_inputs": policy["fixed_primary_inputs"],
            "qualified_primary_inputs": policy["qualified_native_reference_pairs"],
            "exploratory_ratios_gt_1_05": policy["single_process_ratios_above_1_05"],
            "complete_primary_geomean": policy["complete_primary_geomean"],
            "final_three_pair_acceptance": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(replay(), indent=2))


if __name__ == "__main__":
    main()

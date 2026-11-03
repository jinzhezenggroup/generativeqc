"""Prepare CPU-only public delivery from exact completed cohort artifacts."""

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
DIRECTORY = ROOT / "benchmarks/results/h100-sm90-cd2b-guards-20261102"
BASE = ROOT / "benchmarks/results/h100-sm90-cd2b-20261102"
ENV = {
    **os.environ,
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_ALLOW_PROTOCOL": "",
    "GIT_TERMINAL_PROMPT": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}
SOURCE = "cd2b4514be0a38c4920934481b1f02d157a694b4"
REFERENCE = "15ab66a68a061355fda833a93e88484dbfee6360"
PROOFS = {
    "production-build-v1-local-replay.json",
    "df-molecular-v1-local-replay.json",
    "screening-v1-local-replay.json",
    "screening-v1-local-replay-v3.json",
    "native-qualification-v1-local-replay.json",
    "native-qualification-v1-local-replay-v2.json",
    "standard-guards-v1-local-replay.json",
    "uks-fit-v1-local-replay.json",
    "reference-six-v5-local-replay.json",
    "nsys-v1-local-replay-v2.json",
    "public-budget-v7-local-replay.json",
    "facade-guards-v1-local-replay.json",
    "final-pairs-v1-local-replay.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


READERS = {
    "cohort-replay-v1.py",
    "replay-nsys-v2.py",
    "budget-capacity-replay-v1.py",
    "replay-native-v2.py",
    "replay-standard-guards-v2.py",
    "unpack-cohort-v2.py",
    "replay-uks-fit-v1.py",
    "replay-reference-six-v6.py",
}
REQUIRED_FILES = {
    "reference-six-v5-completed.tar.xz",
    "reference-six-v5-download-manifest.json",
    "nsys-v1-analysis-download.json",
    "nsys-cpu-tool-observation-v1.json",
    "readers/cohort-replay-v1.py",
    "readers/replay-nsys-v2.py",
    "readers/replay-reference-six-v6.py",
    "nsys-tool-identity-v1.json",
    "native-qualification-v1-completed.tar.xz",
    "readers/replay-native-v2.py",
    "readers/budget-capacity-replay-v1.py",
    "uks-fit-v1-completed.tar.xz",
    "nsys-terminal-status-v1.json",
    "observe-nsys-cpu-tools-v1.py",
    "readers/replay-uks-fit-v1.py",
    "standard-guards-v1-completed.tar.xz",
    "nsys-v1-analysis.tar.xz.part00",
    "uks-fit-v1-download-manifest.json",
    "native-qualification-v1-download-manifest.json",
    "readers/replay-standard-guards-v2.py",
    "readers/unpack-cohort-v2.py",
    "nsys-v1-analysis.tar.xz.part01",
    "standard-guards-v1-download-manifest.json",
}
PHASES = [
    [
        "native-qualification-v1",
        "replay-native-v2.py",
        "native-qualification-v1-local-replay-v2.json",
    ],
    ["nsys-v1", "replay-nsys-v2.py", "nsys-v1-local-replay-v2.json"],
    [
        "standard-guards-v1",
        "replay-standard-guards-v2.py",
        "standard-guards-v1-local-replay.json",
    ],
    ["uks-fit-v1", "replay-uks-fit-v1.py", "uks-fit-v1-local-replay.json"],
    [
        "reference-six-v5",
        "replay-reference-six-v6.py",
        "reference-six-v5-local-replay.json",
    ],
]
EXPECTED_STATUSES = {
    "native-qualification-v1": "PASS_CURRENT_SOURCE_NATIVE_FOUR_UNFILTERED_MEMCHECK",
    "nsys-v1": "PASS_CURRENT_SOURCE_3_CAPTURES_12_FULL_EF_AND_ACTUAL_SQLITE_REPLAY",
    "standard-guards-v1": "NATIVE_GUARDS_FAILED_OR_INCOMPLETE_WITH_ALL_ORIGINALS_RETAINED",
    "uks-fit-v1": "PASS_EXPLICIT_REFINED_UKS_DF_NINE_EF_ORIGINAL_DEFAULT_FAILURE_RETAINED",
    "reference-six-v5": "PASS6_PRIMARY_KNOWN_SOURCE_REPLICATIONS12_FULL_EF_AND_DUAL_GRID_QUARTER_GATES",
}


def artifact(name: str) -> Path:
    return DIRECTORY / ("readers/" + name if name in READERS else name)


def checked_publication() -> dict:
    publication = json.loads((DIRECTORY / "cohort.json").read_text())
    if (
        publication.get("schema") != "generativeqc.h100-guard-cohort.v1"
        or publication.get("source_revision") != SOURCE
        or publication.get("source_tree") != "c76f5bd3cfa62416255c5d6914cf701d34e10a8d"
        or publication.get("reference_source_revision") != REFERENCE
        or publication.get("phases") != PHASES
        or publication.get("expected_statuses") != EXPECTED_STATUSES
    ):
        raise ValueError("Fixed support cohort/source/status inventory differs")
    entries = publication["files"]
    names = [x["path"] for x in entries]
    if len(names) != len(set(names)) or set(names) != REQUIRED_FILES:
        raise ValueError("Support cohort must bind all mandatory artifacts and readers")
    for entry in entries:
        relative = Path(entry["path"])
        path = DIRECTORY / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha(path) != entry["sha256"]
        ):
            raise ValueError("Support cohort bytes differ: " + entry["path"])
    parts = publication.get("nsys_archive_parts", [])
    if [part.get("path") for part in parts] != [
        "nsys-v1-analysis.tar.xz.part00",
        "nsys-v1-analysis.tar.xz.part01",
    ] or any(
        part != next(entry for entry in entries if entry["path"] == part["path"])
        for part in parts
    ):
        raise ValueError("Nsight archive part inventory differs")
    return publication


def copy_exact(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError("Missing/linked source: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise ValueError("Conflicting original: " + str(target))
    else:
        shutil.copyfile(source, target)


def snapshot(target: Path, revision: str, git_metadata: bool = False) -> None:
    target.mkdir()
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "archive",
            revision,
            "--",
            "benchmarks",
            "python",
            "manifests",
        ],
        env=ENV,
    )
    with tarfile.open(fileobj=io.BytesIO(raw)) as stream:
        stream.extractall(target, filter="data")
    if git_metadata:
        # Only this temporary source view is writable. Its object reads stay local.
        subprocess.run(["git", "init", "-q", str(target)], check=True, env=ENV)
        objects = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO),
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "objects",
            ],
            text=True,
            env=ENV,
        ).strip()
        (target / ".git/objects/info/alternates").write_text(objects + "\n")
        (target / ".git/HEAD").write_text(revision + "\n")
        subprocess.run(
            ["git", "-C", str(target), "cat-file", "-e", "HEAD^{commit}"],
            check=True,
            env=ENV,
        )


def reader(
    path: Path,
    task: Path,
    native: Path,
    reference: Path,
    *args: str,
    reference_corpus: bool = False,
) -> None:
    launcher = """
import importlib.util,pathlib,sys
path,task,native,reference,corpus,*args=sys.argv[1:]
sys.path[:0]=[task,reference,str(pathlib.Path(reference)/'python')]
spec=importlib.util.spec_from_file_location('_cpu_retained_reader',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.TASK=pathlib.Path(task);module.REPO=pathlib.Path(native);module.REFERENCE=pathlib.Path(reference)
module.ROOT=pathlib.Path(task)/'reference-six-v5-reference-inputs/source-helper' if corpus=='yes' else pathlib.Path(native if pathlib.Path(path).name=='replay_df.py' else reference)
sys.argv=[path,*args];module.main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            launcher,
            str(path),
            str(task),
            str(native),
            str(reference),
            "yes" if reference_corpus else "no",
            *args,
        ],
        env=ENV,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            path.name + " failed:\n" + result.stdout + "\n" + result.stderr
        )


def install(task: Path, cohort: str) -> None:
    root = task / (cohort + "-executed-producers")
    for source in sorted(root.iterdir()):
        if source.is_file():
            copy_exact(source, task / source.name)
    admitted = task / (cohort + "-available-admissions")
    if admitted.exists():
        for source in admitted.rglob("*"):
            if source.is_file():
                relative = source.relative_to(admitted)
                if (
                    relative.as_posix() not in PROOFS
                    and relative.parts[0] != "external"
                ):
                    copy_exact(source, task / relative)


def verify_consumed_proofs(task: Path) -> None:
    for admitted in task.glob("*-available-admissions"):
        for source in admitted.rglob("*.json"):
            relative = source.relative_to(admitted)
            if (
                relative.as_posix() in PROOFS
                and (task / relative).exists()
                and source.read_bytes() != (task / relative).read_bytes()
            ):
                raise ValueError("Recomputed admission differs: " + str(relative))


def replay(
    *, extension: Callable[[Path, Path, Path], list[dict]] | None = None
) -> list[dict]:
    if sys.version_info < (3, 12) or sys.flags.optimize:
        raise RuntimeError("Replay requires unoptimized Python 3.12 or later")
    publication = checked_publication()
    phases = publication["phases"]
    with tempfile.TemporaryDirectory(prefix="gqc-public-current-") as tmp:
        temp = Path(tmp)
        task = temp / "packet"
        task.mkdir()
        native = temp / "native"
        reference = temp / "reference"
        snapshot(native, SOURCE, True)
        snapshot(reference, REFERENCE)
        oracle = Path("benchmarks/results/readme-repaired-20261021/oracles.json.xz")
        copy_exact(REPO / oracle, reference / oracle)
        spec = importlib.util.spec_from_file_location(
            "base_delivery", REPO / "benchmarks/h100_current_cohort.py"
        )
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
        base_publication = base.checked_publication(BASE)
        for binding in base_publication["files"]:
            name = binding["path"]
            if Path(name).parent == Path("."):
                copy_exact(BASE / name, task / name)
        build = json.loads(
            (task / "production-build-v1-download-manifest.json").read_text()
        )
        with tarfile.open(task / build["archive"]) as stream:
            for member in stream.getmembers():
                relative = Path(member.name)
                if relative.parent == Path("production-build-v1-executed-producers"):
                    if not member.isfile():
                        raise ValueError("Unsafe build producer")
                    (task / relative.name).write_bytes(
                        stream.extractfile(member).read()
                    )
        reader(BASE / "readers/replay_build.py", task, native, reference)
        install(task, "production-build-v1")
        for cohort in ["df-molecular-v1", "screening-v1"]:
            reader(
                BASE / "readers/unpack_terminal.py",
                task,
                native,
                reference,
                "--cohort",
                cohort,
            )
            install(task, cohort)
        reader(BASE / "readers/replay_df.py", task, native, reference)
        reader(BASE / "readers/replay_screening.py", task, native, reference)
        copy_exact(
            task / "screening-v1-local-replay-v3.json",
            task / "screening-v1-local-replay.json",
        )
        outputs = []
        for cohort, script, proof in phases:
            if cohort == "nsys-v1":
                manifest_name = "nsys-v1-analysis-download.json"
                m = json.loads((artifact(manifest_name)).read_text())
                if m["archive"]["name"] != "nsys-v1-analysis.tar.xz":
                    raise ValueError("Nsight archive name differs")
                copy_exact(artifact(manifest_name), task / manifest_name)
                archive = task / m["archive"]["name"]
                with archive.open("xb") as output:
                    for part in publication["nsys_archive_parts"]:
                        output.write(artifact(part["path"]).read_bytes())
                if (
                    archive.stat().st_size != m["archive"]["bytes"]
                    or sha(archive) != m["archive"]["sha256"]
                ):
                    raise ValueError("Nsight archive differs")
                with tarfile.open(archive) as stream:
                    members = stream.getmembers()
                    if len(members) != len(m["files"]) or {
                        x.name for x in members
                    } != set(m["files"]):
                        raise ValueError("Nsight member inventory differs")
                    for member in members:
                        relative = Path(member.name)
                        if (
                            not member.isfile()
                            or relative.is_absolute()
                            or ".." in relative.parts
                            or relative.parts[0]
                            not in ["nsys-v1", "nsys-v1-executed-producers"]
                            or (task / relative).exists()
                        ):
                            raise ValueError("Unsafe Nsight member")
                    stream.extractall(task, filter="data")
                for n, b in m["files"].items():
                    q = task / n
                    if q.stat().st_size != b["bytes"] or sha(q) != b["sha256"]:
                        raise ValueError("Nsight member bytes differ")
                for name in [
                    "nsys-terminal-status-v1.json",
                    "nsys-tool-identity-v1.json",
                    "nsys-cpu-tool-observation-v1.json",
                    "observe-nsys-cpu-tools-v1.py",
                ]:
                    copy_exact(artifact(name), task / name)
            else:
                for suffix in ["-completed.tar.xz", "-download-manifest.json"]:
                    copy_exact(artifact(cohort + suffix), task / (cohort + suffix))
                reader(
                    artifact("unpack-cohort-v2.py"),
                    task,
                    native,
                    reference,
                    "--cohort",
                    cohort,
                )
            install(task, cohort)
            for name in [
                script,
                "replay-standard-guards-v2.py",
                "cohort-replay-v1.py",
                "budget-capacity-replay-v1.py",
            ]:
                copy_exact(artifact(name), task / name)
            reader(
                task / script,
                task,
                native,
                reference,
                reference_corpus=(cohort == "reference-six-v5"),
            )
            if cohort == "native-qualification-v1":
                copy_exact(
                    task / proof, task / "native-qualification-v1-local-replay.json"
                )
            verify_consumed_proofs(task)
            result = json.loads((task / proof).read_text())
            outputs.append(
                {
                    "cohort": cohort,
                    "status": result["status"],
                    "proof_sha256": sha(task / proof),
                }
            )
            if result["status"] != publication["expected_statuses"][cohort]:
                raise ValueError("Receiver returned a different phase status")
        if extension is not None:
            outputs.extend(extension(task, native, reference))
        return outputs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(
        json.dumps(
            {
                "schema": "generativeqc.h100-guard-cohort-replay.v1",
                "source_revision": SOURCE,
                "phases": replay(),
                "final_three_pair_acceptance": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

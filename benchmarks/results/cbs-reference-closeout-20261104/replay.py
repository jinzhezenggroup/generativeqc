"""Offline replay of the frozen CBS publication; never invoke a native solver."""

from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import hashlib
import importlib.util
import io
import json
import lzma
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    require(
        "\\" not in name
        and ":" not in name
        and not path.is_absolute()
        and ".." not in path.parts
        and str(path) == name,
        "unsafe retained path",
    )
    return path


def restore(capsules: Iterable[Path], destination: Path) -> set[str]:
    seen = set()
    for capsule in capsules:
        records = json.loads(lzma.decompress(capsule.read_bytes()))
        for row in records:
            name = str(safe_name(row["path"]))
            require(name not in seen, "duplicate retained member")
            seen.add(name)
            raw = base64.b64decode(row["base64"], validate=True)
            require(
                digest(raw) == row["sha256"] and len(raw) == row["bytes"],
                "retained member checksum/size mismatch",
            )
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
            if os.name == "posix":
                target.chmod(row["mode"] & 0o777)
    return seen


def git_blobs(
    root: Path, revision: str, names: Iterable[str]
) -> Iterator[tuple[str, bytes]]:
    """Read the measured Git objects, independent of the current worktree."""
    process = subprocess.Popen(
        ["git", "-C", str(root), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        for name in names:
            safe_name(name)
            process.stdin.write((revision + ":" + name + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            require(
                len(header) == 3 and header[1] == b"blob",
                "missing frozen Git blob: " + name,
            )
            raw = process.stdout.read(int(header[2]))
            require(process.stdout.read(1) == b"\n", "invalid Git object framing")
            yield name, raw
    finally:
        process.stdin.close()
        process.stdout.close()
        require(process.wait(timeout=30) == 0, "Git object reader failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--recover", type=Path)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    root = args.repository or package.parents[2]
    manifest = json.loads((package / "publication.json").read_text())
    for row in manifest["files"]:
        raw = (package / str(safe_name(row["path"]))).read_bytes()
        require(
            len(raw) == row["bytes"] and digest(raw) == row["sha256"],
            "publication member changed: " + row["path"],
        )
    capsules = sorted(package.glob("records-*.json.xz"))
    require(capsules, "missing retained records")
    require(
        {x.name for x in capsules}
        == {
            x["path"]
            for x in manifest["files"]
            if x["path"].startswith("records-") and x["path"].endswith(".json.xz")
        },
        "unlisted retained capsule",
    )
    with tempfile.TemporaryDirectory(prefix="gqc-cbs-replay-") as temporary:
        task = Path(temporary)
        restore(capsules, task)
        revision = json.loads((task / "source.json").read_text())["revision"]
        runtime = json.loads((task / "runtime-source.json").read_text())
        require(runtime["revision"] == revision, "runtime revision mismatch")
        native = json.loads((task / "source-native.json").read_text())
        tree = subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "-r", "--name-only", revision],
            text=True,
        ).splitlines()
        specification = json.loads(
            next(git_blobs(root, revision, ["cmake/GenerativeQCSourceIdentity.json"]))[
                1
            ]
        )
        expected = {"cmake/GenerativeQCSourceIdentity.json"} | set(
            specification["files"]
        )
        for group in specification["recursive_groups"]:
            prefix = group["root"] + "/"
            expected |= {
                name
                for name in tree
                if name.startswith(prefix)
                and any(
                    fnmatch.fnmatch(name[len(prefix) :], pattern)
                    for pattern in group["patterns"]
                )
            }
        require(
            set(native["files"]) == expected and native["input_count"] == len(expected),
            "native source inventory membership changed",
        )
        require(
            digest(
                "".join(
                    name + ":" + native["files"][name] + "\n"
                    for name in sorted(expected)
                ).encode()
            )
            == native["native_source_identity"],
            "native source identity changed",
        )
        expected |= {
            name
            for name in tree
            if (
                name.startswith(
                    ("python/generativeqc/", "python/generativeqc_compiler/")
                )
                and name.endswith((".py", ".json"))
            )
            or (name.startswith("tools/") and name.endswith(".py"))
        }
        require(
            set(runtime["files"]) == expected, "runtime inventory membership changed"
        )
        require(
            all(runtime["files"][name] == sha for name, sha in native["files"].items()),
            "native/runtime inventory disagreement",
        )
        for name, raw in git_blobs(root, revision, sorted(runtime["files"])):
            require(
                digest(raw) == runtime["files"][name],
                "frozen runtime blob changed: " + name,
            )
        required = [
            "tests/reference_data/cbs/h2-dz-tz.json",
            "tests/reference_data/cbs/water-algebra.json",
            "benchmarks/results/cbs-routine-standard-20261025/records.json.xz",
            "benchmarks/results/cbs-reference-errors-cpu-20261023/summary.json",
        ]
        cpu_prefix = "benchmarks/results/cbs-reference-errors-cpu-20261023/"
        required = sorted(
            set(required) | {name for name in tree if name.startswith(cpu_prefix)}
        )
        source = task / "source"
        for name, raw in git_blobs(root, revision, required):
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)

        # Replay the frozen CPU capsule itself, rather than trusting a saved PASS.
        cpu_path = source / cpu_prefix / "replay.py"
        spec = importlib.util.spec_from_file_location("gqc_frozen_cpu_replay", cpu_path)
        cpu_reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cpu_reader)
        cpu_reader.ROOT = root
        cpu_actual = cpu_reader.verify_publication(cpu_path.parent)
        require(
            cpu_actual
            == json.loads((task / "cpu-historical-replay-v2.json").read_text()),
            "historical CPU capsule replay differs from retained result",
        )
        supervisor = json.loads(
            (task / "native-attempt-v3/supervisor.json").read_text()
        )
        for row in supervisor["rows"]:
            record = json.loads(
                (task / "native-attempt-v3" / (row["name"] + ".json")).read_text()
            )
            setting = record["semantic_environment"]["CUBLAS_WORKSPACE_CONFIG"]
            require(
                type(setting["present"]) is bool
                and setting["present"] is True
                and setting["value"] == ":16:8",
                "workspace presence/value type mismatch",
            )

        def verify(version: str, allocation: str) -> dict[str, Any]:
            report = task / ("recomputed-summary-" + version + ".json")
            result = subprocess.run(
                [
                    sys.executable,
                    str(task / ("verify-" + version + ".py")),
                    "--source",
                    str(source),
                    "--native",
                    str(task / ("native-attempt-" + version)),
                    "--oracle",
                    str(task / "oracle-completed-v1"),
                    "--task",
                    str(task),
                    "--cpu-replay",
                    str(task / "cpu-historical-replay-v2.json"),
                    "--allocation",
                    str(task / allocation),
                    "--output",
                    str(report),
                ],
                check=False,
            )
            actual = json.loads(report.read_text())
            require(
                result.returncode == (0 if version == "v3" else 1),
                "unexpected attempt exit",
            )
            require(
                actual
                == json.loads((task / ("summary-" + version + ".json")).read_text()),
                "recomputed attempt differs from retained summary",
            )
            return actual

        failed = verify("v2", "allocation-v2.json")
        require(
            failed["status"] == "FAIL_OR_INCOMPLETE"
            and len(failed["rows"]) == 4
            and len(failed["failures"]) == 2,
            "earlier refusals were reclassified",
        )
        actual = verify("v3", "allocation.json")
        require(
            actual == json.loads((package / "summary.json").read_text()),
            "recomputed science differs from published summary",
        )
        physical = json.loads((task / "physical-water-reference.json").read_text())
        locality = max(
            abs(value - physical["central"]["energies"][key])
            for derivative in physical["derivatives"]
            for point in derivative["points"]
            for key, value in point["energies"].items()
        )
        require(locality < 0.01, "physical-reference constituent locality gate failed")
        rows = []
        for row in actual["rows"]:
            for component, value in row["errors"].items():
                for step, force in value["forces"].items():
                    rows.append(
                        [
                            row["case"],
                            row["model"],
                            "fixture",
                            component,
                            step,
                            value["energy"]["maximum"],
                            force["maximum"],
                            force["rms"],
                        ]
                    )
        for row in actual["physical_water"]["calls"]:
            for component, value in row["errors"].items():
                for step, force in value["forces"].items():
                    rows.append(
                        [
                            "physical-water",
                            "full-df",
                            row["phase"],
                            component,
                            step,
                            value["energy"]["maximum"],
                            force["maximum"],
                            force["rms"],
                        ]
                    )
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "case",
                "model",
                "phase",
                "observable",
                "force_reference",
                "energy_absolute_error_Eh",
                "force_maximum_error_Eh_per_bohr",
                "force_RMS_error_Eh_per_bohr",
            ]
        )
        writer.writerows(rows)
        require(
            stream.getvalue().encode() == (package / "errors.csv").read_bytes(),
            "published CSV differs from recomputed errors",
        )
        if args.recover:
            args.recover.mkdir(parents=True, exist_ok=False)
            restore(capsules, args.recover)
        print(
            json.dumps(
                {
                    "status": actual["status"],
                    "cuda_fixture_rows": len(actual["rows"]),
                    "unsupported_cpu_rows": len(actual["unsupported_cpu"]),
                    "physical_water": actual["physical_water"]["status"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

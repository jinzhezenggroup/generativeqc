"""Bounded same-runner CPU diagnosis; never publish a CodSpeed/master result.

Raw counters are observations, not the private CodSpeed backend cost formula.
The unchanged endpoint suite and its numerical assertions run in four separate
processes, in baseline/head/head/baseline order. No retry or performance waiver.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

import codspeed_baseline as baseline

BENCHMARK = "benchmarks/test_cpu_codspeed.py"
URIS = (
    f"{BENCHMARK}::test_cpu_warm_endpoint_walltime[water-rhf-sto3g]",
    f"{BENCHMARK}::test_cpu_warm_endpoint_walltime[water-pbe-sto3g]",
    f"{BENCHMARK}::test_cpu_warm_endpoint_walltime[water-wb97mv-smallgrid-sto3g]",
    f"{BENCHMARK}::test_cpu_pbe_force_walltime",
    f"{BENCHMARK}::test_cpu_rhf_changed_geometry_pair_walltime",
)
ARMS = ("base-1", "head-1", "head-2", "base-2")
REQUIRED_EVENTS = frozenset(
    ("Ir", "Dr", "Dw", "I1mr", "D1mr", "D1mw", "ILmr", "DLmr", "DLmw", "Ct", "Cl")
)
BUILD_OPTIONS = {
    "GENERATIVEQC_ENABLE_CUDA": "OFF",
    "GENERATIVEQC_BUILD_TESTS": "OFF",
    "CMAKE_BUILD_TYPE": "Release",
    "GENERATIVEQC_COMPILER_CACHE": "ccache",
}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
MAX_PROFILE_BYTES = 512 * 1024 * 1024
TOTAL_SECONDS = 33 * 60
ARM_SECONDS = 8 * 60


def run_bounded(
    command: list[str],
    *,
    cwd: Path,
    stdout: TextIO,
    env: dict[str, str] | None = None,
    timeout: float,
) -> None:
    """Bound this invocation's process group, never a shared cache/service."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The CLI may exit before its benchmark/compiler children. Always
        # terminate any remaining members of this invocation's own group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def counters(events: list[str], value: str) -> dict[str, int]:
    tokens = value.split()
    if len(tokens) > len(events) or any(not re.fullmatch(r"[0-9]+", x) for x in tokens):
        raise ValueError("malformed Callgrind counter row")
    # Valgrind omits trailing zero counters. Totals are per-part, not cumulative.
    values = [int(x) for x in tokens] + [0] * (len(events) - len(tokens))
    return dict(zip(events, values, strict=True))


def parse_profile(path: Path) -> list[dict[str, object]]:
    measured: list[dict[str, object]] = []
    part: dict[str, str] = {}

    def finish() -> None:
        trigger = part.get("trigger", "")
        if not part or trigger in (
            "Program termination",
            "Client Request: Metadata: pytest-codspeed 5.0.3",
        ):
            return
        if not trigger.startswith("Client Request: "):
            raise ValueError(f"unexpected Callgrind trigger: {trigger}")
        uri = trigger.removeprefix("Client Request: ")
        if uri not in URIS:
            raise ValueError(f"unexpected measured URI: {uri}")
        events = part.get("events", "").split()
        if len(events) != len(set(events)) or not REQUIRED_EVENTS.issubset(events):
            raise ValueError("missing or duplicate Callgrind events")
        if "totals" not in part:
            raise ValueError("missing per-part Callgrind totals")
        totals = counters(events, part["totals"])
        if any(totals[x] <= 0 for x in ("Ir", "Ct", "Cl")):
            raise ValueError("empty instruction/cycle measurement")
        measured.append(
            {
                "uri": uri,
                "part": part.get("part"),
                "events": events,
                "totals": totals,
                "summary": counters(events, part["summary"])
                if "summary" in part
                else None,
            }
        )

    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("part:"):
                finish()
                part = {"part": line.split(":", 1)[1].strip()}
            elif line.startswith("desc: Trigger: "):
                if "trigger" in part:
                    raise ValueError("duplicate Callgrind trigger")
                part["trigger"] = line.removeprefix("desc: Trigger: ").strip()
            else:
                for field in ("events", "totals", "summary"):
                    if line.startswith(field + ":"):
                        if field in part:
                            raise ValueError(f"duplicate Callgrind {field}")
                        part[field] = line.split(":", 1)[1].strip()
    finish()
    return measured


def parse_arm(folder: Path) -> list[dict[str, object]]:
    processes = []
    for path in sorted(folder.iterdir()):
        if re.fullmatch(r"[0-9]+\.out", path.name):
            if path.is_symlink() or not path.is_file():
                raise ValueError("unexpected profile file type")
            records = parse_profile(path)
            if records:
                processes.append(records)
    # A single pytest process must retain the full identical selection/history.
    if len(processes) != 1 or [x["uri"] for x in processes[0]] != list(URIS):
        raise ValueError("expected exactly five ordered benchmarks in one process")
    return processes[0]


def comparison(arms: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    if set(arms) != set(ARMS) or any(
        len(records) != len(URIS) for records in arms.values()
    ):
        raise ValueError("incomplete ABBA observation set")
    result = {}
    for index, uri in enumerate(URIS):
        values = {arm: records[index]["totals"] for arm, records in arms.items()}
        if any(records[index]["uri"] != uri for records in arms.values()):
            raise ValueError("arm benchmark identity mismatch")
        events = set(values["base-1"])
        if any(set(x) != events for x in values.values()):
            raise ValueError("counter schema changed between arms")
        metrics = {}
        for event in sorted(events):
            observations = []
            for pair in (1, 2):
                control, head = (
                    values[f"base-{pair}"][event],
                    values[f"head-{pair}"][event],
                )
                observations.append(
                    {
                        "pair": pair,
                        "base": control,
                        "head": head,
                        "cost_change_percent": 100 * (head / control - 1)
                        if control
                        else None,
                        "inverse_cost_change_percent": 100 * (control / head - 1)
                        if head
                        else None,
                    }
                )
            spreads = {}
            for role in ("base", "head"):
                low, high = sorted(
                    (values[f"{role}-1"][event], values[f"{role}-2"][event])
                )
                spreads[role] = 100 * (high / low - 1) if low else None
            metrics[event] = {"pairs": observations, "repeat_spread_percent": spreads}
        result[uri] = metrics
    return result


def verify_build(checkout: Path) -> dict[str, object]:
    build = checkout / "build-benchmark"
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line and ":" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    if any(cache.get(key) != value for key, value in BUILD_OPTIONS.items()):
        raise ValueError("native build options differ from the standard CPU job")
    if Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != checkout.resolve():
        raise ValueError("native build is bound to a different checkout")
    if not re.search(
        r"(?:^|[\s/])ccache(?:\s|$)", (build / "CMakeFiles/rules.ninja").read_text()
    ):
        raise ValueError("native compiler command does not use ccache")
    library = build / "libgenerativeqc.so"
    resolved_library = library.resolve(strict=True)
    # CMake's VERSION/SOVERSION create a normal in-build SONAME symlink chain.
    if not resolved_library.is_file() or resolved_library.parent != build.resolve():
        raise ValueError("missing native CPU library")
    return {
        "library": str(library),
        "resolved_library": str(resolved_library),
        "library_sha256": sha256(library),
        "cmake_cache_sha256": sha256(build / "CMakeCache.txt"),
        "options": {key: cache[key] for key in BUILD_OPTIONS},
        "compiler": {
            key: cache.get(key)
            for key in (
                "CMAKE_CXX_COMPILER",
                "CMAKE_CXX_FLAGS",
                "CMAKE_CXX_FLAGS_RELEASE",
                "CMAKE_GENERATOR",
            )
        },
    }


def arm_command(python: Path, checkout: Path, profile: Path) -> list[str]:
    pytest_command = shlex.join(
        [str(python), "-m", "pytest", BENCHMARK, "--codspeed", "-q"]
    )
    return [
        "codspeed",
        "run",
        "--mode=simulation",
        "--simulation-tool=callgrind",
        "--cycle-estimation=true",
        "--exclude-allocations=false",
        "--skip-upload",
        "--skip-setup",
        "--profile-folder",
        str(profile),
        "--working-directory",
        str(checkout),
        "--",
        pytest_command,
    ]


def preserve_profiles(
    source: Path, destination: Path, remaining: int
) -> tuple[list[dict[str, object]], int]:
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(source.iterdir()):
        # Exclude arbitrary runner metadata/environment dumps and perf maps.
        if not re.fullmatch(r"(?:[0-9]+\.out|valgrind\.[0-9]+\.log)", path.name):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("unexpected raw profile file type")
        size = path.stat().st_size
        remaining -= size
        if remaining < 0:
            raise ValueError("raw profile retention exceeds bounded 512 MiB")
        target = destination / (path.name + ".gz")
        with (
            path.open("rb") as src,
            target.open("wb") as dst,
            gzip.GzipFile(filename="", fileobj=dst, mode="wb", mtime=0) as packed,
        ):
            shutil.copyfileobj(src, packed)
        files.append(
            {
                "file": target.name,
                "bytes": size,
                "sha256": sha256(path),
                "gzip_sha256": sha256(target),
            }
        )
    return files, remaining


def execute(args: argparse.Namespace) -> int:
    root = Path.cwd().resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "artifact"
    artifact.mkdir()
    report = {
        "schema": "generativeqc.codspeed-paired-diagnostic.v1",
        "status": "incomplete",
        "origin": "same PR job; not a published master baseline",
        "backend_score_available": False,
        "performance_clearance": False,
        "metric_notes": "Ir instructions; Ct/Cl centi-cycle instruction estimates; cache/syscall counters separate. No backend score, cache weights, pooled statistic or threshold waiver.",
        "order": list(ARMS),
        "arms": {},
    }
    deadline = time.monotonic() + TOTAL_SECONDS
    restore_checkout = False
    exit_code = 1

    def run(
        command: list[str],
        log: str,
        *,
        cwd: Path = root,
        env: dict[str, str] | None = None,
        seconds: int = 30,
    ) -> None:
        timeout = min(seconds, deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("paired diagnostic exhausted its 33-minute budget")
        with (artifact / log).open("w") as handle:
            run_bounded(
                command,
                cwd=cwd,
                env=env,
                stdout=handle,
                timeout=timeout,
            )

    try:
        if os.environ.get("CODSPEED_SKIP_UPLOAD") != "true":
            raise ValueError("artifact-only diagnostic requires explicit skip-upload")
        if any(os.environ.get(key) != "1" for key in THREAD_VARIABLES):
            raise ValueError(
                "paired diagnostic requires unchanged single-thread settings"
            )
        selection = baseline.benchmark_selection()
        if selection["tier"] != "pr" or selection["extra_cases"] != ["wb97mv"]:
            raise ValueError("paired diagnostic requires the fixed five-case selector")
        actual_base = baseline.tested_master_base(
            args.merge_sha, args.head_sha, args.base_ref
        )
        report["provenance"] = {
            "tested_merge": args.merge_sha,
            "pr_head": args.head_sha,
            "actual_base": actual_base,
            "selection": selection,
            "workflow": {
                key: os.environ.get(key)
                for key in (
                    "GITHUB_RUN_ID",
                    "GITHUB_RUN_ATTEMPT",
                    "GITHUB_REPOSITORY",
                    "GITHUB_JOB",
                )
            },
        }
        environment = baseline.environment_fingerprint()
        if not all(environment["cpu"].get(key) for key in ("vendor", "model", "flags")):
            raise ValueError("CPU identity unavailable")
        report["environment"] = environment
        run(["ccache", "--version"], "ccache-version.log")
        if not re.search(
            r"\b4\.14(?:\s|$)", (artifact / "ccache-version.log").read_text()
        ):
            raise ValueError("expected the existing verified ccache 4.14")
        run(["codspeed", "--version"], "codspeed-version.log")
        if not re.search(
            r"\b5\.0\.1(?:\s|$)", (artifact / "codspeed-version.log").read_text()
        ):
            raise ValueError("expected pinned CodSpeed runner 5.0.1")
        run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            "initial-status.log",
        )
        if (artifact / "initial-status.log").read_text().strip():
            raise ValueError("paired diagnostic requires a clean tested checkout")
        head_build = verify_build(root)
        run(["readelf", "-d", "-n", head_build["library"]], "head-elf.log")
        libraries = output / "libraries"
        libraries.mkdir()
        shutil.copyfile(head_build["library"], libraries / "head.so")
        if sha256(libraries / "head.so") != head_build["library_sha256"]:
            raise ValueError("head library snapshot identity mismatch")
        # Reuse one physical source/build/import path: separate worktree names
        # would introduce an avoidable path-length/allocation-layout confound.
        restore_checkout = True
        run(["git", "checkout", "--detach", actual_base], "baseline-checkout.log")
        run(["git", "rev-parse", "HEAD"], "baseline-head.log")
        if (artifact / "baseline-head.log").read_text().strip() != actual_base:
            raise ValueError("baseline worktree identity mismatch")
        if sha256(root / BENCHMARK) != selection["source_sha256"]:
            raise ValueError("baseline benchmark source differs from tested head")
        run(["ccache", "--show-stats"], "ccache-before.log")
        run(
            [
                "cmake",
                "-S",
                ".",
                "-B",
                "build-benchmark",
                "-G",
                "Ninja",
                *[f"-D{key}={value}" for key, value in BUILD_OPTIONS.items()],
            ],
            "baseline-configure.log",
            seconds=120,
        )
        run(
            [
                "cmake",
                "--build",
                "build-benchmark",
                "--target",
                "generativeqc",
                "--parallel",
            ],
            "baseline-build.log",
            seconds=300,
        )
        run(["ccache", "--show-stats"], "ccache-after.log")
        base_build = verify_build(root)
        run(["readelf", "-d", "-n", base_build["library"]], "base-elf.log")
        if base_build["compiler"] != head_build["compiler"]:
            raise ValueError("baseline/head compiler configuration mismatch")
        report["builds"] = {"base": base_build, "head": head_build}
        shutil.copyfile(base_build["library"], libraries / "base.so")
        if sha256(libraries / "base.so") != base_build["library_sha256"]:
            raise ValueError("baseline library snapshot identity mismatch")
        load_directory = output / "load"
        load_directory.mkdir()
        library = load_directory / "libgenerativeqc.so"
        # Never reuse checkout-preserved .pyc files across source switches.
        # One fresh, fixed empty prefix plus no writes gives every arm the same
        # import/cache state without clearing a repository or compiler cache.
        pycache = output / "empty-pycache"
        pycache.mkdir()
        report["canonical_execution_paths"] = {
            "checkout": str(root),
            "pythonpath": str(root / "python"),
            "library": str(library),
            "pycache_prefix": str(pycache),
            "write_bytecode": False,
        }
        python = Path(sys.executable).absolute()
        # Do not resolve the .venv Python symlink: that would discard its venv.
        report["python_executable"] = str(python)
        observations = {}
        remaining = MAX_PROFILE_BYTES
        for arm in ARMS:
            role = arm.split("-", 1)[0]
            build = report["builds"][role]
            source_sha = actual_base if role == "base" else args.merge_sha
            run(["git", "checkout", "--detach", source_sha], f"{arm}-checkout.log")
            run(["git", "rev-parse", "HEAD"], f"{arm}-source-head.log")
            if (artifact / f"{arm}-source-head.log").read_text().strip() != source_sha:
                raise ValueError("paired source checkout identity mismatch")
            run(
                ["git", "diff", "--exit-code", "HEAD", "--"],
                f"{arm}-source-clean.log",
            )
            if sha256(root / BENCHMARK) != selection["source_sha256"]:
                raise ValueError("benchmark source changed between arms")
            if baseline.environment_fingerprint() != environment:
                raise ValueError(
                    "runtime/CPU fingerprint changed during paired diagnostic"
                )
            snapshot = libraries / (role + ".so")
            if sha256(snapshot) != build["library_sha256"]:
                raise ValueError("native library changed between paired arms")
            shutil.copyfile(snapshot, library)
            if sha256(library) != build["library_sha256"]:
                raise ValueError("canonical load-path library identity mismatch")
            profile = output / "profiles" / arm
            profile.mkdir(parents=True)
            if any(pycache.iterdir()):
                raise ValueError("paired Python bytecode prefix is not empty")
            env = {
                **os.environ,
                "PYTHONPYCACHEPREFIX": str(pycache),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(root / "python"),
                "GENERATIVEQC_LIBRARY": str(library),
                "CODSPEED_SKIP_UPLOAD": "true",
            }
            command = arm_command(python, root, profile)
            report["arms"][arm] = {
                "command": command,
                "library_sha256": build["library_sha256"],
                "environment": environment,
                "source_sha": source_sha,
            }
            started = time.monotonic()
            try:
                run(command, f"{arm}.log", env=env, seconds=ARM_SECONDS)
                if baseline.environment_fingerprint() != environment:
                    raise ValueError("runtime/CPU fingerprint changed during an arm")
                if sha256(library) != build["library_sha256"]:
                    raise ValueError("native library changed during an arm")
                if any(pycache.iterdir()):
                    raise ValueError(
                        "paired Python bytecode prefix changed during an arm"
                    )
                observations[arm] = parse_arm(profile)
                report["arms"][arm]["measurements"] = observations[arm]
            finally:
                files, remaining = preserve_profiles(profile, artifact / arm, remaining)
                report["arms"][arm]["raw_files"] = files
                report["arms"][arm]["execution_seconds_not_benchmark_metric"] = (
                    time.monotonic() - started
                )
                write_json(artifact / "report.json", report)
        report["comparisons"] = comparison(observations)
        report["status"] = "complete; manual performance review required"
        print(
            "Same-runner ABBA diagnostic complete; no CodSpeed score or performance clearance."
        )
        exit_code = 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"::error::Same-runner diagnostic failed: {report['error']}")
    finally:
        if restore_checkout:
            try:
                # Give restoration its own small budget, even after timeout.
                # Never use --force/reset/clean or discard unexpected changes.
                with (artifact / "restore-checkout.log").open("w") as handle:
                    run_bounded(
                        ["git", "checkout", "--detach", args.merge_sha],
                        cwd=root,
                        stdout=handle,
                        timeout=30,
                    )
                report["restored_tested_checkout"] = True
            except (OSError, subprocess.SubprocessError) as exc:
                report["restore_error"] = f"{type(exc).__name__}: {exc}"
                report["status"] = "failed to restore tested checkout"
                print(f"::error::{report['status']}: {report['restore_error']}")
                exit_code = 1
        write_json(artifact / "report.json", report)
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary).open("a") as handle:
                handle.write(
                    "\n### Same-runner CPU diagnostic\n\n"
                    + report["status"]
                    + ". Artifact counters are not a CodSpeed backend score; no performance clearance.\n"
                )
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

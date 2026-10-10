"""Qualify PR CodSpeed simulation against a published master environment.

GitHub-hosted runners can change CPU vendor/model and glibc dispatch between
runs. A PR sample is comparable only with a published, matching master baseline
at its exact base SHA. Missing proof is advisory, not a regression.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path

_SCHEMA = "generativeqc.codspeed-cpu-baseline.v2"
_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
_REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_PR_EXTRA_CASES = frozenset({"wb97mv"})


def _command(*args: str) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


def _cpu_identity(cpuinfo: str) -> dict[str, str]:
    """Record effective CPU dispatch features, not a variable core number."""
    first = cpuinfo.split("\n\n", 1)[0]
    items = dict(line.split(":", 1) for line in first.splitlines() if ":" in line)
    items = {key.strip(): value.strip() for key, value in items.items()}
    return {
        "vendor": items.get("vendor_id", ""),
        "model": items.get("model name", ""),
        "flags": " ".join(sorted(items.get("flags", "").split())),
    }


def environment_fingerprint() -> dict[str, object]:
    """Capture the environment that changes CPU simulation cost/dispatch."""
    cpu = _cpu_identity(Path("/proc/cpuinfo").read_text(encoding="utf-8"))
    packages = {}
    for package in ("numpy", "pytest", "pytest-codspeed"):
        packages[package] = importlib.metadata.version(package)
    return {
        "cpu": cpu,
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "python": platform.python_version(),
        "compiler": _command("g++", "--version").splitlines()[0],
        "ccache": os.environ.get("CCACHE_VERSION", ""),
        "packages": packages,
        "simulation": "CodSpeedHQ/action-v5.0.1/pytest-codspeed-5.0.3",
    }


def benchmark_selection() -> dict[str, object]:
    """Bind selector settings to the benchmark source that interprets them."""
    source = Path(__file__).resolve().parents[1] / "benchmarks/test_cpu_codspeed.py"
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tier": os.environ.get("GENERATIVEQC_CODSPEED_TIER", "full"),
        "extra_cases": sorted(
            {
                item.strip()
                for item in os.environ.get(
                    "GENERATIVEQC_CODSPEED_EXTRA_CASES", ""
                ).split(",")
                if item.strip()
            }
        ),
    }


def _selection_matches(recorded: object, requested: dict[str, object]) -> bool:
    if not isinstance(recorded, dict):
        return False
    digest = requested.get("source_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        return False
    if recorded.get("source_sha256") != digest:
        return False
    # Only PR-tier receipts are published. Fail closed if another tier is added
    # until its selection semantics are explicitly qualified here.
    if recorded.get("tier") != "pr" or requested.get("tier") != "pr":
        return False
    available = recorded.get("extra_cases")
    needed = requested.get("extra_cases")
    for extras in (available, needed):
        if not isinstance(extras, list):
            return False
        if not all(
            isinstance(item, str) and item in _PR_EXTRA_CASES for item in extras
        ):
            return False
        if extras != sorted(set(extras)):
            return False
    # A superset baseline changes the preceding in-process allocation history.
    # Instrumentation resets simulated caches, not allocator/application state.
    # The unchanged benchmark source fixes execution order for equal selectors.
    return available == needed


def _baseline_name(sha: str) -> str:
    if not _SHA_RE.fullmatch(sha):
        raise ValueError("invalid 40-character baseline commit SHA")
    return f"codspeed-cpu-baseline-{sha}"


def _lookup_baseline(repo: str, sha: str) -> dict[str, object]:
    if not _REPO_RE.fullmatch(repo):
        raise ValueError("invalid repository name")
    artifact_name = _baseline_name(sha)
    response = json.loads(
        _command(
            "gh",
            "api",
            "-X",
            "GET",
            f"repos/{repo}/actions/artifacts",
            "-f",
            f"name={artifact_name}",
            "-f",
            "per_page=100",
        )
    )
    artifacts = [
        artifact
        for artifact in response.get("artifacts", [])
        if artifact.get("name") == artifact_name
        and not artifact.get("expired", True)
        and artifact.get("workflow_run", {}).get("head_sha") == sha
        and artifact.get("workflow_run", {}).get("head_branch") == "master"
    ]
    if not artifacts:
        raise LookupError("no successful published baseline for the exact PR base")
    selected = max(artifacts, key=lambda artifact: artifact.get("created_at", ""))
    run_id = selected["workflow_run"]["id"]
    with tempfile.TemporaryDirectory(prefix="codspeed-baseline-") as directory:
        _command(
            "gh",
            "run",
            "download",
            str(run_id),
            "-n",
            artifact_name,
            "-D",
            directory,
            "-R",
            repo,
        )
        receipt = json.loads(
            (Path(directory) / "baseline.json").read_text(encoding="utf-8")
        )
        if not isinstance(receipt, dict):
            raise TypeError("baseline receipt must be a JSON object")
        return receipt


def qualify(
    baseline: dict[str, object],
    sha: str,
    current: dict[str, object],
    selection: dict[str, object],
) -> tuple[bool, str]:
    """Fail closed on an old schema, stale source, or different CPU/runtime."""
    if baseline.get("schema") != _SCHEMA or baseline.get("sha") != sha:
        return False, "baseline identity/schema mismatch"
    if not _selection_matches(baseline.get("benchmark_selection"), selection):
        return False, "baseline does not match the requested benchmark source/selection"
    recorded = baseline.get("environment")
    if not isinstance(recorded, dict):
        return False, "baseline environment missing"
    differences = sorted(
        key
        for key in recorded.keys() | current.keys()
        if recorded.get(key) != current.get(key)
    )
    if differences:
        return False, "different benchmark environments: " + ", ".join(differences)
    cpu = current.get("cpu", {})
    if not isinstance(cpu, dict) or not cpu.get("model") or not cpu.get("flags"):
        return False, "CPU model or instruction flags unavailable"
    return True, "exact-master-base and runtime environment match"


def _output(qualified: bool, reason: str) -> None:
    status = "qualified" if qualified else "unqualified"
    print(f"CodSpeed comparison {status}: {reason}")
    if not qualified:
        print(f"::warning::Skipping CodSpeed performance upload: {reason}")
    if destination := os.environ.get("GITHUB_OUTPUT"):
        with Path(destination).open("a", encoding="utf-8") as handle:
            handle.write(f"qualified={'true' if qualified else 'false'}\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(f"### CodSpeed CPU comparison: {status}\n\n{reason}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    record = subcommands.add_parser("record")
    record.add_argument("--sha", required=True)
    record.add_argument("--output", type=Path, required=True)
    check = subcommands.add_parser("qualify")
    check.add_argument("--base-sha", required=True)
    check.add_argument("--repo", required=True)
    args = parser.parse_args()
    if args.command == "record":
        _baseline_name(args.sha)
        artifact = {
            "schema": _SCHEMA,
            "sha": args.sha,
            "environment": environment_fingerprint(),
            "benchmark_selection": benchmark_selection(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    try:
        current = environment_fingerprint()
        baseline = _lookup_baseline(args.repo, args.base_sha)
        matched, reason = qualify(
            baseline, args.base_sha, current, benchmark_selection()
        )
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        LookupError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as exc:
        matched = False
        reason = f"baseline qualification unavailable: {type(exc).__name__}"
    _output(matched, reason)


if __name__ == "__main__":
    main()

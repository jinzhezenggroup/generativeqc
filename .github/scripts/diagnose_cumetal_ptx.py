"""Capture one exact PTX input and its typed-compiler error after a failed gate.

This is diagnostic evidence only. It neither retries the required gate nor
interprets the legacy invocation or offline compiler result as QC acceptance.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ENDPOINT = (
    "tests/python/test_cuda_runtime.py::test_cuda_minimal_rhf_matches_cpu_reference"
)
KERNEL = (
    "_ZN12generativeqc3scf14cuda_execution39build_shell_primitive_pair_cache_kernel"
    "ENS1_11DeviceBatchEPNS1_17PrimitivePairDataE"
)
SUITE_SECONDS = 180


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_command(
    command: list[str],
    name: str,
    output: Path,
    env: dict[str, str],
    timeout: float,
    deadline: float,
) -> dict[str, object]:
    started = time.monotonic()
    timeout = min(timeout, deadline - started)
    result: dict[str, object] = {
        "command": command,
        "timeout_seconds": max(0, timeout),
        "returncode": None,
        "timed_out": False,
    }
    if timeout <= 0:
        result["error"] = "diagnostic suite budget exhausted"
        return result
    stdout = output / f"{name}.stdout.txt"
    stderr = output / f"{name}.stderr.txt"
    result.update(stdout=stdout.name, stderr=stderr.name)
    with stdout.open("w") as out, stderr.open("w") as err:
        try:
            process = subprocess.Popen(
                command, stdout=out, stderr=err, env=env, start_new_session=True
            )
        except OSError as error:
            result["error"] = str(error)
            return result
        try:
            result["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            result["returncode"] = process.wait()
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def diagnose(output: Path, env: dict[str, str]) -> int:
    # Refuse stale evidence. No source, compiler cache, or earlier result is reset.
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + SUITE_SECONDS
    compiler = Path(env["CUMETAL_PREFIX"]) / "bin/cumetalc"
    report: dict[str, object] = {
        "purpose": "non-gating PTX compiler diagnosis; not QC acceptance",
        "suite_budget_seconds": SUITE_SECONDS,
        "endpoint": ENDPOINT,
        "entry": KERNEL,
        "python": sys.version,
        "platform": platform.platform(),
        "tested_commit": env.get("GITHUB_SHA"),
        "pr_head": env.get("CUMETAL_DIAGNOSTIC_PR_HEAD"),
        "provider_pin": env["CUMETAL_COMMIT"],
        "developer_dir": env.get("DEVELOPER_DIR"),
        "compiler_sha256": file_sha256(compiler) if compiler.is_file() else None,
        "precision": "fast48; inherited binary32 libdevice limitation unchanged",
        "commands": {},
    }

    def save() -> None:
        (output / "diagnostic.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    def run(
        name: str, command: list[str], timeout: int, command_env: dict[str, str]
    ) -> dict[str, object]:
        report["commands"][name] = {"command": command, "status": "starting"}
        save()
        result = run_command(command, name, output, command_env, timeout, deadline)
        report["commands"][name] = result
        save()
        return result

    save()
    for name, command in (
        ("provider-version", [str(compiler), "--version"]),
        ("xcode-version", ["xcodebuild", "-version"]),
        ("metal-version", ["xcrun", "metal", "--version"]),
        ("provider-head", ["git", "-C", env["CUMETAL_SOURCE"], "rev-parse", "HEAD"]),
        (
            "provider-status",
            [
                "git",
                "-C",
                env["CUMETAL_SOURCE"],
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
        ),
    ):
        if run(name, command, 5, env)["returncode"] != 0:
            report["status"] = f"unable to verify {name}"
            save()
            return 1
    provider_head = (output / "provider-head.stdout.txt").read_text().strip()
    if (
        provider_head != env["CUMETAL_COMMIT"]
        or (output / "provider-status.stdout.txt").read_text().strip()
    ):
        report["status"] = (
            "provider checkout is modified or does not match the selected pin"
        )
        save()
        return 1

    legacy_env = {
        **env,
        "CUMETAL_PTX_BACKEND": "legacy",
        "CUMETAL_FP64_MODE": "fast48",
        "CUMETAL_DEBUG_REGISTRATION": "1",
        "CUMETAL_TRACE_GPU": "1",
        "CUMETAL_DEBUG_DUMP_PTX_DIR": str(output / "ptx"),
        # Upstream returns cache hits before the PTX dump hook. Use a fresh,
        # diagnostic-only cache directory; existing caches remain untouched.
        "CUMETAL_CACHE_DIR": str(output / "diagnostic-jit-legacy-fast48"),
    }
    report["capture_backend"] = "legacy"
    legacy_result = run(
        "legacy-capture",
        [
            sys.executable,
            "-m",
            "pytest",
            ENDPOINT,
            "-vv",
            "-ra",
            "-s",
            f"--junitxml={output / 'legacy-capture.xml'}",
        ],
        90,
        legacy_env,
    )
    ptx = output / "ptx" / f"{KERNEL}.ptx"
    if not ptx.is_file() or ptx.stat().st_size == 0:
        report["status"] = "exact first-kernel PTX was not captured"
        save()
        return 1
    report["input"] = {
        "path": str(ptx.relative_to(output)),
        "bytes": ptx.stat().st_size,
        "sha256": file_sha256(ptx),
    }
    if legacy_result["timed_out"]:
        report["status"] = "legacy capture timed out; retained PTX, no compiler retry"
        save()
        return 1
    entries = re.findall(r"\.entry\s+([^\s(]+)\s*\(", ptx.read_text())
    if entries.count(KERNEL) != 1:
        report["status"] = "captured PTX does not contain exactly one selected entry"
        save()
        return 1
    typed_env = {
        **env,
        "CUMETAL_PTX_BACKEND": "cumetal-ir",
        "CUMETAL_FP64_MODE": "fast48",
    }
    result = run(
        "typed-compiler",
        [
            str(compiler),
            str(ptx),
            "--backend=cumetal-ir",
            "--fp64=fast48",
            "--ptx-strict",
            "--entry",
            KERNEL,
            "--emit=msl",
            "--no-link",
            "-o",
            str(output / "typed-output.metal"),
        ],
        60,
        typed_env,
    )
    report["status"] = (
        "typed compiler timed out"
        if result["timed_out"]
        else "typed compiler could not start"
        if result.get("error")
        else "diagnostic captured"
    )
    save()
    print(
        f"Typed compiler diagnostic returncode={result['returncode']}; not QC acceptance"
    )
    return 1 if result["timed_out"] or result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(
        diagnose(Path(os.environ["CUMETAL_DIAGNOSTIC_OUTPUT"]), dict(os.environ))
    )

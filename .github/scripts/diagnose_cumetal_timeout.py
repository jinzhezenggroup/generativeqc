"""Observe one partially warmed RHF retry without qualifying the failed gate."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ENDPOINT = (
    "tests/python/test_cuda_runtime.py::test_cuda_minimal_rhf_matches_cpu_reference"
)
SUITE_SECONDS = 180
SNAPSHOT_SECONDS = 15
SAMPLE_SECONDS = 60


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_inventory(root: Path) -> list[dict[str, object]]:
    """Identify existing/added artifacts without clearing or changing the cache."""
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def process_snapshot(pid: int, output: Path, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid,ppid,pgid,etime,time,state,command"],
            capture_output=True,
            text=True,
            timeout=min(2, remaining),
            check=False,
        )
        # Keep only the diagnostic's process group, not unrelated runner jobs.
        rows = result.stdout.splitlines()
        selected = [line for line in rows[1:] if line.split()[2:3] == [str(pid)]]
        output.write_text("\n".join(rows[:1] + selected) + "\n")
    except (OSError, subprocess.TimeoutExpired) as error:
        output.write_text(f"Process snapshot unavailable: {error}\n")


def sample_process(pid: int, output: Path, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0 or sys.platform != "darwin":
        return
    # macOS's built-in sampler can distinguish lowering/compiler waits from
    # a Metal execution wait. Failure is evidence absence, not endpoint failure.
    try:
        result = subprocess.run(
            ["sample", str(pid), "1", "-file", str(output)],
            capture_output=True,
            text=True,
            timeout=min(5, remaining),
            check=False,
        )
        output.with_suffix(".status.txt").write_text(
            f"exit={result.returncode}\n{result.stdout}{result.stderr}"
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        output.with_suffix(".status.txt").write_text(f"Sample unavailable: {error}\n")


def observe(
    command: list[str], output: Path, env: dict[str, str], deadline: float
) -> dict:
    started = time.monotonic()
    if started >= deadline:
        return {"error": "diagnostic budget exhausted", "timed_out": True}
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )
    assert process.stdout is not None

    def pump() -> None:
        with (output / "endpoint-live.txt").open("w") as stream:
            for line in process.stdout:
                stamped = f"t+{time.monotonic() - started:.3f}s {line}"
                stream.write(stamped)
                stream.flush()
                print(stamped, end="", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    snapshot = 0
    next_snapshot = started
    sampled = False
    timed_out = False
    while process.poll() is None:
        now = time.monotonic()
        if now >= deadline:
            timed_out = True
            break
        if now >= next_snapshot:
            process_snapshot(
                process.pid, output / f"process-{snapshot:02d}.txt", deadline
            )
            snapshot += 1
            next_snapshot = now + SNAPSHOT_SECONDS
        if not sampled and now - started >= SAMPLE_SECONDS:
            sampled = True
            sample_process(process.pid, output / "endpoint-sample.txt", deadline)
        try:
            process.wait(timeout=max(0.001, min(0.2, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass
    # Kill descendants too, including an offline compiler left behind at exit.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    returncode = process.wait()
    reader.join(timeout=5)
    return {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "live_log": "endpoint-live.txt",
        "note": "Diagnostic completion cannot qualify the failed 90-second gate",
    }


def diagnose(output: Path, env: dict[str, str]) -> int:
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + SUITE_SECONDS
    cache = Path(env["CUMETAL_CACHE_DIR"])
    compiler = Path(env["CUMETAL_PREFIX"]) / "bin/cumetalc"
    report = {
        "purpose": "partially warmed timeout diagnosis; not QC acceptance",
        "suite_budget_seconds": SUITE_SECONDS,
        "endpoint": ENDPOINT,
        "provider_pin": env["CUMETAL_COMMIT"],
        "tested_commit": env.get("GITHUB_SHA"),
        "pr_head": env.get("CUMETAL_DIAGNOSTIC_PR_HEAD"),
        "platform": platform.platform(),
        "compiler_sha256": sha256(compiler),
        "library_sha256": sha256(Path(env["GENERATIVEQC_LIBRARY"])),
        "backend": env["CUMETAL_PTX_BACKEND"],
        "fp64_mode": env["CUMETAL_FP64_MODE"],
        "precision": "fast48; inherited binary32 libdevice limitation unchanged",
        "cache_dir": str(cache),
        "cache_policy": "reuse failed gate cache as-is; no cold-runtime claim",
        "cache_before": cache_inventory(cache),
    }

    def save() -> None:
        (output / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    if report["backend"] != "cumetal-ir" or report["fp64_mode"] != "fast48":
        raise ValueError(
            "diagnostic must use the same strict cumetal-ir/fast48 contract"
        )
    for name, args in (
        ("head", ["rev-parse", "HEAD"]),
        ("status", ["status", "--porcelain", "--untracked-files=no"]),
    ):
        result = subprocess.run(
            ["git", "-C", env["CUMETAL_SOURCE"], *args],
            capture_output=True,
            text=True,
            timeout=max(0.001, min(5, deadline - time.monotonic())),
            check=True,
        )
        report[f"provider_{name}"] = result.stdout.strip()
    if report["provider_head"] != report["provider_pin"] or report["provider_status"]:
        report["error"] = "provider checkout is modified or does not match selected pin"
        save()
        return 1
    command = [
        sys.executable,
        "-u",
        "-m",
        "pytest",
        ENDPOINT,
        "-vv",
        "-ra",
        "-s",
        f"--junitxml={output / 'diagnostic-only.xml'}",
    ]
    report["command"] = command
    save()
    report["observation"] = observe(
        command,
        output,
        {
            **env,
            "CUMETAL_DEBUG_REGISTRATION": "1",
            "CUMETAL_TRACE_GPU": "1",
            "CUMETAL_DIAGNOSTIC_PHASES": "1",
            "PYTHONUNBUFFERED": "1",
        },
        deadline,
    )
    report["cache_after"] = cache_inventory(cache)
    save()
    print("CuMetal timeout evidence retained; required gate remains failed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        diagnose(Path(os.environ["CUMETAL_DIAGNOSTIC_OUTPUT"]), dict(os.environ))
    )

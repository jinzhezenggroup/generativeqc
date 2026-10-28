"""Finite compilation measurement with retained private cold/hit cache and aggregate RSS."""

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import psutil

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--seconds", type=float, default=7200)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)
a.cache.mkdir(parents=True, exist_ok=True)
cmd = a.command[1:] if a.command[0] == "--" else a.command
launcher = subprocess.check_output(["which", "sccache"], text=True).strip()
version = subprocess.check_output([launcher, "--version"], text=True).strip()
started = time.monotonic()
peak = 0
samples = 0
server = job = None
with (
    tempfile.TemporaryDirectory(prefix="gqc-build-", dir="/tmp") as tmp,
    a.output.with_suffix(".log").open("w") as log,
    a.output.with_suffix(".server.log").open("w") as slog,
):
    env = {
        **os.environ,
        "SCCACHE_SERVER_UDS": tmp + "/server.sock",
        "SCCACHE_DIR": str(a.cache.resolve() / "objects"),
        "SCCACHE_CACHED_CONF": str(a.cache.resolve() / "cached-config"),
        "SCCACHE_NO_DAEMON": "1",
        "SCCACHE_IDLE_TIMEOUT": str(int(a.seconds)),
    }
    env.pop("SCCACHE_STARTUP_NOTIFY", None)
    env.pop("SCCACHE_START_SERVER", None)
    try:
        server = subprocess.Popen(
            [launcher],
            env={**env, "SCCACHE_START_SERVER": "1"},
            stdout=slog,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while not Path(env["SCCACHE_SERVER_UDS"]).exists():
            if server.poll() is not None or time.monotonic() - started > 30:
                raise RuntimeError("private cache server failed")
            time.sleep(0.05)
        status = json.loads(
            subprocess.check_output(
                [launcher, "--dist-status"], env=env, text=True, timeout=30
            )
        )
        assert set(status) == {"Disabled"}, status
        before = subprocess.check_output(
            [launcher, "--show-stats", "--stats-format=json"],
            env=env,
            text=True,
            timeout=30,
        )
        start = time.monotonic()
        job = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        while job.poll() is None:
            rss = 0
            for proc in psutil.Process().children(recursive=True):
                try:
                    rss += proc.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, rss)
            samples += 1
            if time.monotonic() - start > a.seconds:
                raise TimeoutError("finite compilation deadline")
            time.sleep(0.1)
        elapsed = time.monotonic() - start
        after = subprocess.check_output(
            [launcher, "--show-stats", "--stats-format=json"],
            env=env,
            text=True,
            timeout=30,
        )
        a.output.write_text(
            json.dumps(
                dict(
                    command=cmd,
                    cwd=os.getcwd(),
                    launcher=launcher,
                    version=version,
                    cache=str(a.cache.resolve()),
                    seconds=elapsed,
                    returncode=job.returncode,
                    aggregate_peak_rss_bytes=peak,
                    sample_count=samples,
                    sample_period_seconds=0.1,
                    memory_definition="sum of RSS for live descendant build, generator, compiler, linker and private foreground sccache processes; shared pages counted per process",
                    cache_before=json.loads(before),
                    cache_after=json.loads(after),
                ),
                indent=2,
            )
            + "\n"
        )
        if job.returncode:
            raise SystemExit(job.returncode)
    finally:
        for proc in (job, server):
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()

"""Exercise timing semantics, redaction and optional workflow isolation."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tools import ci_host_resources as resources
from tools import report_ci_ninja_timing as timing

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "tools/report_ci_ninja_timing.py"
BOUNDARY = "Record fresh CLI Ninja timing boundary"
BUILD = "Build monolithic C++ CUDA CLI (release or fast PR)"
REPORT = "Report completed CLI Ninja command timings"


def log(*rows: str, version: int = 5) -> bytes:
    return (
        f"# ninja log v{version}\n" + "\n".join(rows) + ("\n" if rows else "")
    ).encode()


def row(start: int, end: int, output: str, key: str = "abcd") -> str:
    return f"{start}\t{end}\t1791886400000000000\t{output}\t{key}"


def test_multioutput_edges_and_parallel_time_are_not_wall_time() -> None:
    edges = timing.parse_log(
        log(
            row(10, 110, "generated/one.cu"),
            row(10, 110, "/private/build/generated/one.cu"),
            row(10, 110, "generated/two.cu"),
            row(20, 210, "CMakeFiles/generativeqc_core.dir/src/one.cu.o", "bcde"),
            row(110, 220, "generativeqc", "cdef"),
        )
    )
    assert len(edges) == 3
    summary = timing.render(edges)
    assert "Output records: 5" in summary
    assert "Redacted output names: 1" in summary
    assert "Cumulative command duration (parallel work): 0.400 s" in summary
    assert (
        "Recorded timestamp span (first start to last completion): 0.210 s" in summary
    )
    assert "not wall time" in summary
    assert "neither metric establishes the dependency critical path" in summary
    assert "/private" not in summary and "abcd" not in summary
    assert "Generated output | 1 | 0.100" in summary
    assert "CUDA object | 1 | 0.190" in summary


@pytest.mark.parametrize("version", [5, 6, 7])
def test_supported_versions_and_zero_completed_commands(version: int) -> None:
    assert timing.parse_log(log(version=version)) == []
    assert "No completed command records" in timing.render([])


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"# ninja log v8\n",
        b"# ninja log v5\n\n",
        log(row(0, 20, "generativeqc"))[:-1],
        log("0\t20\t0\tgenerativeqc"),
        log("nan\t20\t0\tgenerativeqc\tabcd"),
        log("-1\t20\t0\tgenerativeqc\tabcd"),
        log("0\t20\tnan\tgenerativeqc\tabcd"),
        log("0\t20\t0\tgenerativeqc\tnot-a-hash"),
        log(row(20, 10, "generativeqc")),
        log(row(0, timing.MAX_TIME_MS + 1, "generativeqc")),
        log(row(0, 10, "generated/\x1b[31msecret")),
        log(row(0, 10, "generated/\x00secret")),
        log(row(0, 10, "")),
    ],
)
def test_malformed_or_truncated_logs_are_not_partial_success(raw: bytes) -> None:
    with pytest.raises(ValueError):
        timing.parse_log(raw)


@pytest.mark.parametrize(
    "rows",
    [
        # A timestamp reset between invocations must not form one envelope.
        [row(0, 100, "generated/a"), row(0, 50, "generated/b", "bcde")],
        # Even an increasing second invocation is invalid when it rebuilds an output.
        [row(0, 100, "generativeqc"), row(0, 200, "generativeqc", "bcde")],
        # Compaction/reordering cannot be treated as adjacent multi-output edges.
        [
            row(0, 100, "generated/a"),
            row(0, 100, "generated/b", "bcde"),
            row(0, 100, "generated/c"),
        ],
    ],
)
def test_ambiguous_invocations_and_reordering_are_refused(rows: list[str]) -> None:
    with pytest.raises(ValueError):
        timing.parse_log(log(*rows))


@pytest.mark.parametrize(
    "output",
    [
        "/home/private/secret.cu",
        "../secret",
        "generated/../secret",
        "generated/./secret",
        "generated//secret",
        "C:\\private\\secret",
        "generated/secret|payload",
        "generated/secret`payload",
        "generated/<img>",
        "generated/[secret](url)",
        "generated/ümlaut",
        "private/secret",
        "generated/" + "a" * 181,
    ],
)
def test_unapproved_output_names_are_redacted(output: str) -> None:
    summary = timing.render(timing.parse_log(log(row(0, 1, output))))
    assert "[redacted]" in summary and output not in summary
    assert "Redacted output names: 1" in summary


def test_input_and_record_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timing, "MAX_BYTES", 30)
    with pytest.raises(ValueError):
        timing.parse_log(log(row(0, 1, "generativeqc")))
    monkeypatch.setattr(timing, "MAX_BYTES", 4096)
    monkeypatch.setattr(timing, "MAX_RECORDS", 1)
    with pytest.raises(ValueError):
        timing.parse_log(log(row(0, 1, "generated/a"), row(1, 2, "generated/b")))
    monkeypatch.setattr(timing, "MAX_LINE", 10)
    with pytest.raises(ValueError):
        timing.parse_log(log(row(0, 1, "generativeqc")))


def test_output_is_bounded_and_longest_groups_are_sorted() -> None:
    edges = timing.parse_log(
        log(*(row(0, n, f"generated/file{n}.cu", f"{n:x}") for n in range(1, 1001)))
    )
    summary = timing.render(edges)
    assert len(summary.encode()) < timing.MAX_SUMMARY_BYTES
    assert "generated/file1000.cu" in summary
    assert "generated/file981.cu" in summary
    assert "generated/file980.cu" not in summary
    assert summary.index("generated/file1000.cu") < summary.index(
        "generated/file999.cu"
    )


def invoke(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), "--build-dir", str(tmp_path / "build"), *args],
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


@pytest.mark.parametrize("existing", [False, True])
def test_freshness_requires_an_absent_log_even_if_historical_times_increase(
    tmp_path: Path, existing: bool
) -> None:
    build = tmp_path / "build"
    build.mkdir()
    path = build / ".ninja_log"
    contents = log(row(0, 10, "generated/a"), row(0, 20, "generated/b", "b"))
    if existing:
        path.write_bytes(contents)
    assert invoke(tmp_path, "--check-fresh").returncode == 0
    assert (tmp_path / "output").read_text() == f"fresh={str(not existing).lower()}\n"
    path.write_bytes(contents)
    result = invoke(tmp_path, "--fresh-log", str(not existing).lower())
    assert result.returncode == 0
    assert ("Timing unavailable" in result.stdout) == existing
    assert path.read_bytes() == contents  # Diagnostics never edit/clear build state.


@pytest.mark.parametrize(
    "kind", ["missing", "invalid", "symlink", "directory", "fifo", "parent-symlink"]
)
def test_io_failures_and_unsafe_files_are_non_gating_and_redacted(
    tmp_path: Path, kind: str
) -> None:
    build = tmp_path / "build"
    build.mkdir()
    path = build / ".ninja_log"
    if kind == "invalid":
        path.write_text("/private/secret=do-not-leak")
    elif kind == "symlink":
        path.symlink_to(tmp_path / "private-secret")
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "parent-symlink":
        build.rmdir()
        target = tmp_path / "private-secret"
        target.mkdir()
        (target / ".ninja_log").write_bytes(log(row(0, 1, "generativeqc")))
        build.symlink_to(target, target_is_directory=True)
    result = invoke(tmp_path, "--fresh-log", "true")
    assert result.returncode == 0
    assert "Timing unavailable" in result.stdout
    assert "private" not in result.stdout + result.stderr
    assert str(tmp_path) not in result.stdout + result.stderr


def test_summary_write_failure_keeps_stdout_and_does_not_fail(tmp_path: Path) -> None:
    (tmp_path / "build").mkdir()
    (tmp_path / "build/.ninja_log").write_bytes(log(row(0, 10, "generativeqc")))
    (tmp_path / "summary").mkdir()
    result = invoke(tmp_path, "--fresh-log", "true")
    assert result.returncode == 0
    assert "CLI executable | 1 | 0.010" in result.stdout
    assert "summary unavailable" in result.stdout
    assert str(tmp_path) not in result.stdout + result.stderr


def workflow_job(workflow: str, name: str) -> str:
    section = workflow.split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [A-Za-z0-9_-]+:\n", section, maxsplit=1)[0]


def workflow_steps() -> dict[str, str]:
    # Match the adjacent CI tests: these bounded step fixtures need only the
    # standard library, not a YAML package absent from hosted Python test jobs.
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    section = workflow_job(workflow, "native-cli-artifact")
    return {part.splitlines()[0]: part for part in section.split("      - name: ")[1:]}


def workflow_script(step: str) -> str:
    value = step.split("        run: ", 1)[1]
    if value.startswith("|\n"):
        return textwrap.dedent(value[2:])
    if value.startswith(">-\n"):
        return " ".join(textwrap.dedent(value[3:]).splitlines())
    return value.strip()


def test_workflow_diagnostics_are_bounded_non_gating_and_share_existing_build() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    steps = workflow_steps()
    names = list(steps)
    assert names.index(BOUNDARY) + 1 == names.index(BUILD)
    assert names.index(BUILD) + 1 == names.index(REPORT)
    for name in (BOUNDARY, REPORT):
        step = steps[name]
        assert "\n        continue-on-error: true\n" in step
        assert "\n        timeout-minutes: 1\n" in step
        condition = step.split("\n        if: ", 1)[1].splitlines()[0]
        assert "steps.cli_scope.outputs.tier != 'skip'" in condition
    reporter = steps[REPORT]
    condition = reporter.split("\n        if: ", 1)[1].splitlines()[0]
    assert "always()" in condition
    assert "steps.native_cli_build.outcome != 'skipped'" in condition
    assert "steps.native_cli_build.outcome != ''" in condition
    assert (
        "\n          FRESH_NINJA_LOG: ${{ steps.cli_ninja_boundary.outputs.fresh }}\n"
        in reporter
    )
    assert "continue-on-error:" not in steps[BUILD]
    producer = workflow_job(workflow, "native-cli-artifact")
    consumer = workflow_job(workflow, "native-cli-qualification")
    anchor = re.search(r"^    steps: &([A-Za-z0-9_]+)$", producer, re.MULTILINE)
    assert anchor is not None
    assert f"\n    steps: *{anchor[1]}\n" in consumer


@pytest.mark.parametrize("exit_code", [0, 23])
def test_actual_workflow_build_then_report_keeps_build_result(
    tmp_path: Path, exit_code: int
) -> None:
    steps = workflow_steps()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    cmake = bindir / "cmake"
    cmake.write_text(
        f"#!{sys.executable}\n"
        + "import pathlib, sys\n"
        + "if '--build' in sys.argv:\n"
        + " p = pathlib.Path('build/cuda-release-cli'); p.mkdir(parents=True, exist_ok=True)\n"
        + f" (p / '.ninja_log').write_bytes({log(row(10, 30, 'generativeqc'))!r})\n"
        + f" sys.exit({exit_code})\n"
    )
    cmake.chmod(0o755)
    ccache = bindir / "ccache"
    ccache.write_text("#!/bin/sh\nexit 0\n")
    ccache.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "CLI_FAST_COMPILE": "OFF",
    }
    results = {}
    for name in (BOUNDARY, BUILD, REPORT):
        if name == REPORT:
            env["FRESH_NINJA_LOG"] = next(
                line.split("=", 1)[1]
                for line in (tmp_path / "output").read_text().splitlines()
                if line.startswith("fresh=")
            )
        script = (
            workflow_script(steps[name])
            .replace(
                "python3 tools/report_ci_ninja_timing.py",
                f"{shlex.quote(sys.executable)} {shlex.quote(str(HELPER))}",
            )
            .replace(
                "python3 tools/ci_host_resources.py",
                f"{shlex.quote(sys.executable)} {shlex.quote(str(RESOURCE_HELPER))}",
            )
        )
        results[name] = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", script],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    assert results[BOUNDARY].returncode == results[REPORT].returncode == 0
    assert results[BUILD].returncode == exit_code
    assert "CLI executable | 1 | 0.020" in (tmp_path / "summary").read_text()


# Keep executable diagnostics coverage in the existing module so this repair
# does not redistribute the modulo-sharded core suite by adding a test file.
RESOURCE_HELPER = ROOT / "tools/ci_host_resources.py"


def resource_command(code: str) -> list[str]:
    return [
        sys.executable,
        str(RESOURCE_HELPER),
        "--interval-seconds",
        "0.05",
        "--",
        sys.executable,
        "-c",
        code,
    ]


@pytest.mark.parametrize("exit_code", [0, 23])
def test_live_samples_are_flushed_and_preserve_command_io_and_status(
    tmp_path: Path, exit_code: int
) -> None:
    output = tmp_path / "live.log"
    with output.open("w") as stream:
        child = subprocess.Popen(
            resource_command(
                "import os,sys,time; "
                "assert sys.argv[1] == 'argument with spaces'; "
                "assert os.environ['DIAGNOSTIC_TEST_VALUE'] == 'private-marker'; "
                "print('build stdout', flush=True); "
                "print('build stderr', file=sys.stderr, flush=True); "
                f"time.sleep(0.6); sys.exit({exit_code})"
            )
            + ["argument with spaces"],
            env={**os.environ, "DIAGNOSTIC_TEST_VALUE": "private-marker"},
            stdout=stream,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 5
            while "CI_HOST_RESOURCES" not in output.read_text():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert child.poll() is None  # Not buffered until command completion.
            _, stderr = child.communicate(timeout=5)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
    assert child.returncode == exit_code
    content = output.read_text()
    records = [
        json.loads(line.removeprefix("CI_HOST_RESOURCES "))
        for line in content.splitlines()
        if line.startswith("CI_HOST_RESOURCES ")
    ]
    assert len(records) >= 2
    assert "build stdout" in content and "build stderr" in stderr
    assert "private-marker" not in content + stderr
    assert "argument with spaces" not in content + stderr
    assert "host_compiler_rss_bytes" in records[0]


def test_resource_allowlist_and_missing_platform_metrics(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text(
        "MemTotal: 100 kB\nMemAvailable: 40 kB\n"
        "SwapTotal: 20 kB\nSwapFree: 10 kB\nPrivate: 123 kB\n"
    )
    pressure = proc / "pressure"
    pressure.mkdir()
    (pressure / "cpu").write_text(
        "some avg10=1.50 avg60=2.00 avg300=0.01 total=100 secret=private-marker\n"
    )
    for pid, name in (("123", "cicc"), ("456", "private-marker")):
        process = proc / pid
        process.mkdir()
        (process / "comm").write_text(name + "\n")
        (process / "statm").write_text("20 5 0 0 0 0 0\n")
        (process / "cmdline").write_text("private-marker")
        (process / "environ").write_text("private-marker")
    result = resources.snapshot(proc, tmp_path)
    assert result["memory_kib"] == {
        "MemTotal": 100,
        "MemAvailable": 40,
        "SwapTotal": 20,
        "SwapFree": 10,
    }
    assert result["cpu_pressure"]["some"]["avg10"] == 1.5
    assert result["memory_pressure"] is None
    rss = result["host_compiler_rss_bytes"]
    assert rss["count"] == 1
    assert rss["sum"] == rss["max"] == 5 * os.sysconf("SC_PAGE_SIZE")
    assert "private-marker" not in json.dumps(result)
    assert "Private" not in json.dumps(result)
    missing = resources.snapshot(tmp_path / "missing", tmp_path / "missing")
    assert missing and all(value is None for value in missing.values())


@pytest.mark.parametrize(
    "failure",
    [
        "sample",
        "write",
        "start",
        "init",
        "close",
        "missing",
        "blocked",
    ],
)
@pytest.mark.parametrize("exit_code", [0, 19])
def test_optional_sampler_failures_never_replace_build_result(
    tmp_path: Path, failure: str, exit_code: int
) -> None:
    setup = {
        "sample": "r.snapshot = lambda: 1 / 0",
        "write": (
            "def broken_write(*args): raise OSError('private-marker')\n"
            "r.os.write = broken_write"
        ),
        "start": "r.threading.Thread.start = lambda self: 1 / 0",
        "init": "r.Sampler.__init__ = lambda self, *args: 1 / 0",
        "close": "r.Sampler.close = lambda self: 1 / 0",
        "missing": f"r.snapshot = lambda: original(Path({str(tmp_path / 'absent')!r}), Path({str(tmp_path / 'absent')!r}))",
        "blocked": "r.snapshot = lambda: time.sleep(60)",
    }[failure]
    code = (
        "import sys,time; from pathlib import Path; "
        "from tools import ci_host_resources as r; original=r.snapshot\n"
        + setup
        + "\nstatus = r.run([sys.executable, '-c', "
        + repr(f"import time; time.sleep(0.2); raise SystemExit({exit_code})")
        + "], 0.05)\n"
        # Restore stdout before interpreter finalization so only sampler I/O fails.
        + "sys.stdout = sys.__stdout__; raise SystemExit(status)"
    )
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    assert time.monotonic() - started < 4
    assert "private-marker" not in result.stdout + result.stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX CI signal contract")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_forwarding_reaps_only_owned_child_and_stops_sampling(
    tmp_path: Path, signum: int
) -> None:
    ready = tmp_path / "ready"
    pending = tmp_path / "ready.pending"
    # A separate long-lived process proves forwarding never hits the caller's
    # process group or a shared compiler-cache server.
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    child = subprocess.Popen(
        resource_command(
            "import os,pathlib,time; "
            # Publish readiness only after the complete PID has been written.
            # File creation by write_text is visible before its contents are.
            f"pathlib.Path({str(pending)!r}).write_text(str(os.getpid())); "
            f"pathlib.Path({str(pending)!r}).replace({str(ready)!r}); "
            "time.sleep(30)"
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        build_pid = int(ready.read_text())
        child.send_signal(signum)
        stdout, _ = child.communicate(timeout=5)
        assert child.returncode == 128 + signum
        with pytest.raises(ProcessLookupError):
            os.kill(build_pid, 0)
        assert unrelated.poll() is None
        assert "CI_HOST_RESOURCES" in stdout
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_sampler_budget_and_normal_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(resources, "MAX_SAMPLES", 3)
    import threading

    log_path = tmp_path / "samples.log"
    before = set(threading.enumerate())
    with log_path.open("w") as stream:
        monkeypatch.setattr(resources.sys, "stdout", stream)
        assert (
            resources.run(
                [sys.executable, "-c", "import time; time.sleep(0.2)"], interval=0.01
            )
            == 0
        )
    lines = log_path.read_text().splitlines(keepends=True)
    assert len(lines) == 3
    assert all(len(line.encode()) <= resources.MAX_RECORD_BYTES for line in lines)
    assert set(threading.enumerate()) == before


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_build_terminating_itself_preserves_signal_status(signum: int) -> None:
    result = subprocess.run(
        resource_command(f"import os; os.kill(os.getpid(), {int(signum)})"),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 128 + signum


def test_process_scan_is_bounded_and_marks_partial_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(5):
        process = tmp_path / str(index)
        process.mkdir()
        (process / "comm").write_text("cicc\n")
        (process / "statm").write_text("20 5\n")
    monkeypatch.setattr(resources, "MAX_PROCESSES", 2)
    result = resources.snapshot(tmp_path, tmp_path)["host_compiler_rss_bytes"]
    assert result["scan_truncated"] is True
    assert result["count"] == 2


@pytest.mark.parametrize("exit_code", [0, 23])
def test_full_log_pipe_cannot_abort_interpreter_or_change_child_stdout(
    exit_code: int,
) -> None:
    """A real stalled pipe must not leave a buffered-I/O lock held at shutdown."""
    read_fd, write_fd = os.pipe()
    process = None
    try:
        os.set_blocking(write_fd, False)
        with pytest.raises(BlockingIOError):
            while True:
                os.write(write_fd, b"x" * 4096)
        os.set_blocking(write_fd, True)
        code = (
            "import os,time; assert os.get_blocking(1); time.sleep(0.3); "
            f"raise SystemExit({exit_code})"
        )
        started = time.monotonic()
        process = subprocess.Popen(
            resource_command(code),
            stdout=write_fd,
            stderr=subprocess.PIPE,
            text=True,
        )
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == exit_code, stderr
        assert time.monotonic() - started < 4
        assert os.get_blocking(write_fd)
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        os.close(read_fd)
        os.close(write_fd)


def test_cancellation_during_sampler_setup_never_starts_build(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-start"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).touch()",
    ]
    code = (
        "import signal,sys; from tools import ci_host_resources as r\n"
        "original = r.Sampler.__init__\n"
        "def cancel(self, *args):\n"
        " original(self, *args)\n"
        " signal.raise_signal(signal.SIGTERM)\n"
        "r.Sampler.__init__ = cancel\n"
        f"raise SystemExit(r.run({command!r}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 143, result.stderr
    assert not marker.exists()

"""Exercise bounded source attribution without publishing compiler arguments."""

from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tools import ci_compile_trace as trace
from tools import ci_host_resources as resources

if TYPE_CHECKING:
    from typing import TextIO


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "checkout"
    (root / "tools").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src/unit.cu").write_text("// public source\n")
    shutil.copy2(trace.__file__, root / "tools/ci_compile_trace.py")
    monkeypatch.setattr(trace, "ROOT", root)
    return root


def process(proc: Path, pid: int, parent: int, started: int, rss: int = 1) -> None:
    path = proc / str(pid)
    path.mkdir(parents=True, exist_ok=True)
    fields = ["S", str(parent), *(["0"] * 17), str(started)]
    (path / "stat").write_text(f"{pid} (cicc) {' '.join(fields)}\n")
    (path / "comm").write_text("cicc\n")
    (path / "statm").write_text(f"100 {rss} 0\n")
    (path / "cmdline").write_text("private-marker")
    (path / "environ").write_text("private-marker")


def marker(directory: Path, pid: int, started: int, source: str) -> None:
    directory.mkdir(exist_ok=True)
    (directory / f"{pid}.json").write_text(
        json.dumps({"start_ticks": started, "source": source})
    )


def test_numeric_ancestry_attributes_only_live_validated_source(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc, traces = tmp_path / "proc", tmp_path / "traces"
    process(proc, 123, 456, 10, 9)
    process(proc, 456, 1, 20, 2)
    marker(traces, 456, 20, "src/unit.cu")
    original = Path.open

    def no_private_reads(self: Path, *args: object, **kwargs: object) -> TextIO:
        assert self.name not in {"cmdline", "environ"}
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_private_reads)
    sample = resources.snapshot(proc, tmp_path, traces)
    largest = sample["largest_compiler_processes"]
    assert largest[0] == {
        "pid": 123,
        "compiler": "cicc",
        "rss_bytes": 9 * os.sysconf("SC_PAGE_SIZE"),
        "source": "src/unit.cu",
    }
    assert "private-marker" not in json.dumps(sample)
    assert str(checkout) not in json.dumps(sample)
    assert resources.compiler_source(123, proc, traces, 99) is None
    marker(traces, 456, 19, "src/unit.cu")
    assert resources.compiler_source(123, proc, traces, 10) is None


@pytest.mark.parametrize(
    "source",
    [
        "../private.cu",
        "/private.cu",
        "src/missing.cu",
        "src/unit.cu\nsecret",
        "src/link.cu",
    ],
)
def test_unknown_or_escaping_paths_never_leave_the_sampler(
    checkout: Path, tmp_path: Path, source: str
) -> None:
    outside = tmp_path / "private.cu"
    outside.write_text("private-marker")
    (checkout / "src/link.cu").symlink_to(outside)
    proc, traces = tmp_path / "proc", tmp_path / "traces"
    process(proc, 123, 1, 10)
    marker(traces, 123, 10, source)
    assert resources.compiler_source(123, proc, traces, 10) is None
    assert (
        resources.snapshot(proc, tmp_path, traces)["host_compiler_rss_bytes"]["count"]
        == 1
    )


def test_bounded_detail_missing_records_and_ancestry_cycles(
    checkout: Path, tmp_path: Path
) -> None:
    proc, traces = tmp_path / "proc", tmp_path / "absent"
    for pid in range(10, 30):
        process(proc, pid, pid, pid, pid)
    sample = resources.snapshot(proc, tmp_path, traces)
    details = sample["largest_compiler_processes"]
    assert len(details) == resources.MAX_COMPILER_DETAILS
    assert [entry["pid"] for entry in details] == [29, 28, 27, 26]
    assert all(entry["source"] is None for entry in details)
    assert len(json.dumps(sample).encode()) < resources.MAX_RECORD_BYTES - 100


def test_record_is_atomic_and_generated_paths_are_scoped(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = checkout / "build/cuda-release-cli/generated/unit.cu"
    source.parent.mkdir(parents=True)
    source.write_text("// generated public source\n")
    traces = tmp_path / "traces"
    monkeypatch.setenv(trace.TRACE_ENV, str(traces))
    original = os.replace

    def before_publish(source_file: Path, destination: Path) -> None:
        assert not destination.exists()
        assert not list(traces.glob("*.json"))
        data = json.loads(source_file.read_text())
        assert data["source"] == "build/cuda-release-cli/generated/unit.cu"
        original(source_file, destination)

    monkeypatch.setattr(os, "replace", before_publish)
    trace.record(["compiler", str(source)])
    assert len(list(traces.iterdir())) == 1


def test_live_descendant_remains_attributable_after_prefix_exec(
    checkout: Path, tmp_path: Path
) -> None:
    traces, proc = tmp_path / "traces", tmp_path / "proc"
    proc.mkdir()
    child_code = (
        "import ctypes,os,time; "
        "ctypes.CDLL(None).prctl(15,b'cicc',0,0,0); "
        "print(os.getpid(),flush=True); time.sleep(30)"
    )
    parent_code = (
        "import subprocess,sys; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        "raise SystemExit(child.wait())"
    )
    compiler = subprocess.Popen(
        [
            str(checkout / "tools/ci_compile_trace.py"),
            sys.executable,
            "-c",
            parent_code,
            str(checkout / "src/unit.cu"),
        ],
        env={**os.environ, trace.TRACE_ENV: str(traces)},
        stdout=subprocess.PIPE,
        text=True,
    )
    child_pid = None
    try:
        # One complete flushed line is the readiness handshake, not file existence.
        assert compiler.stdout is not None
        assert select.select([compiler.stdout], [], [], 5)[0]
        child_pid = int(compiler.stdout.readline())
        for pid in (compiler.pid, child_pid):
            (proc / str(pid)).symlink_to(Path("/proc") / str(pid))
        sample = resources.snapshot(proc, tmp_path, traces)
        assert sample["largest_compiler_processes"][0]["source"] == "src/unit.cu"
        assert sample["largest_compiler_processes"][0]["pid"] == child_pid
    finally:
        if child_pid is not None:
            os.kill(child_pid, signal.SIGTERM)
        compiler.wait(timeout=10)


@pytest.mark.parametrize("name", ["PIPE", "XFSZ"])
def test_prefix_restores_python_ignored_native_signals(
    checkout: Path, name: str
) -> None:
    signum = getattr(signal, f"SIG{name}", None)
    if signum is None:
        pytest.skip("signal unavailable")
    command = ["/bin/sh", "-c", f"kill -{name} $$; exit 23"]
    direct = subprocess.run(command, check=False, timeout=10)
    wrapped = subprocess.run(
        [str(checkout / "tools/ci_compile_trace.py"), *command],
        check=False,
        timeout=10,
    )
    assert direct.returncode == wrapped.returncode == -signum


@pytest.mark.parametrize("failure", ["blocked", "thread_start", "record"])
def test_optional_record_setup_cannot_stall_or_replace_compiler_status(
    failure: str,
) -> None:
    setup = {
        "blocked": "t.record = lambda command: time.sleep(60)",
        "thread_start": "t.threading.Thread.start = lambda self: 1 / 0",
        "record": "t.record = lambda command: 1 / 0",
    }[failure]
    code = (
        "import sys,time; from tools import ci_compile_trace as t; "
        + setup
        + "; sys.argv=['prefix','/bin/sh','-c','printf compiler; exit 23']; "
        "raise SystemExit(t.main())"
    )
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 23
    assert result.stdout == "compiler" and result.stderr == ""
    assert time.monotonic() - started < trace.MAX_SETUP_SECONDS + 2


@pytest.mark.parametrize("outcome", [0, 23, -signal.SIGTERM])
@pytest.mark.parametrize("recording", ["working", "unwritable", "ambiguous"])
def test_prefix_exec_preserves_pid_argv_environment_io_and_status(
    checkout: Path, tmp_path: Path, outcome: int, recording: str
) -> None:
    traces = tmp_path / "traces"
    if recording == "unwritable":
        traces.write_text("not a directory")
    arguments = [str(checkout / "src/unit.cu"), "argument with spaces"]
    if recording == "ambiguous":
        (checkout / "src/other.cpp").write_text("// another input\n")
        arguments.append(str(checkout / "src/other.cpp"))
    code = (
        "import json,os,signal,sys; "
        "print(json.dumps([os.getpid(),sys.argv[1:],os.environ['PRIVATE_MARKER']]),flush=True); "
        "print('compiler stderr',file=sys.stderr,flush=True); "
        + (
            f"os.kill(os.getpid(),{-outcome})"
            if outcome < 0
            else f"sys.exit({outcome})"
        )
    )
    child = subprocess.Popen(
        [
            str(checkout / "tools/ci_compile_trace.py"),
            sys.executable,
            "-c",
            code,
            *arguments,
        ],
        env={
            **os.environ,
            trace.TRACE_ENV: str(traces),
            "PRIVATE_MARKER": "private-marker",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output, error = child.communicate(timeout=10)
    assert child.returncode == outcome
    assert json.loads(output) == [child.pid, arguments, "private-marker"]
    assert error == "compiler stderr\n"
    if recording == "working":
        records = list(traces.iterdir())
        assert [p.name for p in records] == [f"{child.pid}.json"]
        data = records[0].read_text()
        assert json.loads(data)["source"] == "src/unit.cu"
        assert "private-marker" not in data and "argument with spaces" not in data
    elif recording == "ambiguous":
        assert not traces.exists()


def test_real_ccache_reuses_objects_across_the_optional_prefix(
    checkout: Path, tmp_path: Path
) -> None:
    cache, compiler = shutil.which("ccache"), shutil.which("c++")
    if cache is None or compiler is None:
        pytest.skip("verified compiler cache and host compiler required")
    subprocess.run([cache, "--version"], check=True, capture_output=True, timeout=10)
    source = checkout / "src/probe.cpp"
    # A source-specific symbol keeps this bounded probe distinct from other tests.
    name = "probe_" + str(os.getpid()) + "_" + str(tmp_path.stat().st_ino)
    source.write_text(f"int {name}() {{ return 17; }}\n")
    prefix = str(checkout / "tools/ci_compile_trace.py")
    traces = tmp_path / "traces"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"CCACHE_PREFIX", "CCACHE_PREFIX_CPP"}
    }
    env.update({trace.TRACE_ENV: str(traces), "CCACHE_BASEDIR": str(checkout)})
    command = [cache, compiler, "-c", str(source), "-o", str(tmp_path / "probe.o")]
    subprocess.run(command, env=env, check=True, capture_output=True, timeout=30)
    expected = (tmp_path / "probe.o").read_bytes()
    env["CCACHE_PREFIX"] = prefix
    env["CCACHE_PREFIX_CPP"] = prefix
    subprocess.run(command, env=env, check=True, capture_output=True, timeout=30)
    assert not traces.exists()  # Existing cache hit bypasses the prefix.
    assert (tmp_path / "probe.o").read_bytes() == expected
    source.write_text(f"int {name}() {{ return 23; }}\n")
    subprocess.run(command, env=env, check=True, capture_output=True, timeout=30)
    records = list(traces.glob("*.json"))
    assert records
    assert all(
        json.loads(path.read_text())["source"] == "src/probe.cpp" for path in records
    )
    expected = (tmp_path / "probe.o").read_bytes()
    env.pop("CCACHE_PREFIX")
    env.pop("CCACHE_PREFIX_CPP")
    subprocess.run(command, env=env, check=True, capture_output=True, timeout=30)
    assert (tmp_path / "probe.o").read_bytes() == expected

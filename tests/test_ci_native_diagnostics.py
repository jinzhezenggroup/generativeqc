"""Subprocess contracts; deliberately faulting children must remain failures."""

from __future__ import annotations

import ctypes
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux CI diagnostics")


@pytest.fixture(scope="module")
def helper(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("native-diagnostics-build")
    launcher = shutil.which("ccache")
    assert launcher, "A verified compiler cache is required"
    subprocess.run([launcher, "--version"], check=True, capture_output=True)
    compiler = shutil.which("cc")
    assert compiler, "The CI C compiler is required"
    source = ROOT / "tools/native_fault_record.c"
    obj, library = directory / "record.o", directory / "record.so"
    subprocess.run(
        [
            launcher,
            compiler,
            "-std=c11",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
    )
    subprocess.run(
        [launcher, compiler, "-shared", str(obj), "-o", str(library)], check=True
    )
    probe = directory / "probe.c"
    probe.write_text(r"""
#define _GNU_SOURCE
#include <signal.h>
#include <unistd.h>
static void prior(int sig) { (void)sig; (void)write(2, "PRIOR\n", 6); }
static void recursive(int sig) { prior(sig); raise(SIGSEGV); }
static void recursive_same(int sig) { prior(sig); raise(sig); }
void install_prior(int recurse) {
    struct sigaction a = {0};
    a.sa_handler = recurse == 2 ? recursive_same : (recurse ? recursive : prior);
    sigemptyset(&a.sa_mask);
    sigaction(SIGABRT, &a, 0);
}
int NPdgemm(void) { return 17; }
void native_fault(void) { volatile int *p = (int *)0; *p = 1; }
""")
    probe_obj, probe_lib = (
        directory / "probe.o",
        directory / "libopenblas_diagnostic_probe.so",
    )
    subprocess.run(
        [launcher, compiler, "-fPIC", "-O0", "-c", str(probe), "-o", str(probe_obj)],
        check=True,
    )
    subprocess.run(
        [launcher, compiler, "-shared", str(probe_obj), "-o", str(probe_lib)],
        check=True,
    )
    return library, probe_lib


def child(
    tmp_path: Path, helper: tuple[Path, Path], body: str, *, before: str = ""
) -> subprocess.CompletedProcess[str]:
    library, _probe = helper
    source = f"""
import ctypes, faulthandler, os, resource, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from tools.pytest_native_diagnostics import State
resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))
faulthandler.enable()
{before}
state = State(Path({str(tmp_path)!r}), Path({str(library)!r}))
{body}
"""
    return subprocess.run(
        [sys.executable, "-c", source],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


@pytest.mark.parametrize(
    ("body", "sig"),
    [("ctypes.string_at(0)", signal.SIGSEGV), ("os.abort()", signal.SIGABRT)],
)
def test_native_signal_and_python_traceback_survive(
    tmp_path: Path, helper: tuple[Path, Path], body: str, sig: signal.Signals
) -> None:
    before_limit = resource.getrlimit(resource.RLIMIT_CORE)
    result = child(tmp_path, helper, body)
    assert result.returncode == -sig, result.stderr
    assert "Fatal Python error" in result.stderr
    record = next(tmp_path.glob("*.native.log")).read_text()
    assert f"signal={hex(sig)}" in record
    assert "module=libc.so" in record and "file_offset=0x" in record
    assert len(record.encode()) <= 512
    assert " pc=" not in record and " address=" not in record
    assert resource.getrlimit(resource.RLIMIT_CORE) == before_limit
    assert not list(tmp_path.glob("core*"))


@pytest.mark.parametrize("recurse", [0, 1, 2])
def test_chained_handler_cannot_turn_fault_into_success(
    tmp_path: Path, helper: tuple[Path, Path], recurse: int
) -> None:
    probe = helper[1]
    result = child(
        tmp_path,
        helper,
        "os.abort()",
        before=f"ctypes.CDLL({str(probe)!r}).install_prior({int(recurse)})",
    )
    assert result.returncode == -signal.SIGABRT, result.stderr
    assert result.stderr.count("PRIOR") == 1
    assert len(next(tmp_path.glob("*.native.log")).read_text().splitlines()) == 1


def test_late_provider_mapping_and_unknown_module(
    tmp_path: Path, helper: tuple[Path, Path]
) -> None:
    _library, probe = helper
    result = child(
        tmp_path,
        helper,
        f"""
provider = ctypes.CDLL({str(probe)!r})
assert provider.NPdgemm() == 17
provider.native_fault()
""",
    )
    assert result.returncode == -signal.SIGSEGV
    record = next(tmp_path.glob("*.native.log")).read_text()
    assert "module=libopenblas_diagnostic_probe.so file_offset=" in record
    modules = json.loads(next(tmp_path.glob("*.modules.json")).read_text())
    assert all("/" not in m["name"] for m in modules)
    assert all(set(m) == {"name", "file_offset", "size"} for m in modules)
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    unlisted = unknown / "private_provider.so"
    shutil.copyfile(probe, unlisted)
    result = child(unknown, helper, f"ctypes.CDLL({str(unlisted)!r}).native_fault()")
    assert result.returncode == -signal.SIGSEGV
    assert "module=unresolved" in next(unknown.glob("*.native.log")).read_text()
    assert "private_provider" not in next(unknown.glob("*.native.log")).read_text()


def test_history_bound_and_process_local_cleanup(
    tmp_path: Path, helper: tuple[Path, Path]
) -> None:
    from tools.pytest_native_diagnostics import HISTORY_LIMIT, State

    previous = resource.getrlimit(resource.RLIMIT_CORE)
    state = State(tmp_path, helper[0])
    try:
        for _ in range(3000):
            state.write("start", nodeid="x" * 10000)
        assert next(tmp_path.glob("*.jsonl")).stat().st_size <= HISTORY_LIMIT
        assert '"history_limit"' in next(tmp_path.glob("*.jsonl")).read_text()
    finally:
        state.close()
    assert resource.getrlimit(resource.RLIMIT_CORE) == previous


def test_runtime_diagnostic_errors_do_not_change_lookup(
    tmp_path: Path, helper: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import pytest_native_diagnostics as diagnostics

    state = diagnostics.State(tmp_path, helper[0])
    try:

        def unavailable() -> None:
            raise OSError("synthetic diagnostic storage failure")

        monkeypatch.setattr(diagnostics, "allowed_modules", unavailable)
        provider = ctypes.CDLL(str(helper[1]))
        assert provider.NPdgemm() == 17
        state.refresh_modules()
        state.history.close()
        state.write("start", nodeid="ordinary-test")
    finally:
        state.close()


def test_xdist_crash_and_assertion_remain_failures(
    tmp_path: Path, helper: tuple[Path, Path]
) -> None:
    tests = tmp_path / "test_child.py"
    tests.write_text(
        "import ctypes\ndef test_pass(): pass\ndef test_assertion(): assert False\ndef test_crash(): ctypes.string_at(0)\n"
    )
    artifacts = tmp_path / ".artifacts/native-diagnostics"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tests),
            "-q",
            "-n",
            "1",
            "-p",
            "tools.pytest_native_diagnostics",
            f"--native-diagnostics-dir={artifacts}",
            f"--native-diagnostics-library={helper[0]}",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "2 failed, 1 passed" in result.stdout
    records = [
        json.loads(line)
        for p in artifacts.glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert sum(r["event"] == "start" for r in records) == 3
    assert any(r.get("outcome") == "failed" for r in records)
    assert any(r.get("nodeid", "").endswith("::test_crash") for r in records)
    assert any(p.stat().st_size for p in artifacts.glob("*.native.log"))
    assert all(r["worker"].startswith("gw") for r in records)
    assert not list(tmp_path.rglob("core.*"))


def test_workflow_keeps_diagnostics_in_failure_artifacts() -> None:
    source = (ROOT / ".github/workflows/ci.yml").read_text()
    assert '--native-diagnostics-dir="$PWD/.artifacts/native-diagnostics"' in source
    assert "path: .artifacts/" in source and "retention-days: 14" in source
    preparation = source.split("      - name: Prepare bounded native diagnostics\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "        if: matrix.shard == 'core-b'\n" in preparation
    assert "ccache --version" in preparation
    assert (
        ".venv/bin/python -m pytest tests/test_ci_native_diagnostics.py -q"
        in preparation
    )
    assert '"${diagnostic_args[@]}"' in source
    wheels = (ROOT / ".github/workflows/wheels.yml").read_text()
    assert "  push:\n    branches: [master]" in wheels


def test_startup_failure_restores_descriptors_and_core_limit(
    tmp_path: Path, helper: tuple[Path, Path]
) -> None:
    from tools.pytest_native_diagnostics import State

    previous = resource.getrlimit(resource.RLIMIT_CORE)
    descriptors = set(Path("/proc/self/fd").iterdir())
    with pytest.raises(OSError):
        State(tmp_path, tmp_path / "missing-helper.so")
    assert resource.getrlimit(resource.RLIMIT_CORE) == previous
    assert set(Path("/proc/self/fd").iterdir()) == descriptors


def test_diagnostics_do_not_intercept_external_termination(
    tmp_path: Path, helper: tuple[Path, Path]
) -> None:
    script = f"""
import sys, time
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from tools.pytest_native_diagnostics import State
state = State(Path({str(tmp_path)!r}), Path({str(helper[0])!r}))
print('ready', flush=True)
time.sleep(30)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.1)
        process.terminate()
        assert process.wait(timeout=5) == -signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert next(tmp_path.glob("*.native.log")).read_text() == ""


def test_cleanup_failure_does_not_skip_other_cleanup(
    tmp_path: Path, helper: tuple[Path, Path]
) -> None:
    from tools.pytest_native_diagnostics import State

    previous = resource.getrlimit(resource.RLIMIT_CORE)
    state = State(tmp_path, helper[0])
    state.history.close()
    with Path("/dev/full").open("w") as full:
        state.history = full
        state.write("report", nodeid="test-still-has-its-original-outcome")
        state.close()
    assert state.native_fd is None
    assert resource.getrlimit(resource.RLIMIT_CORE) == previous
    assert ("history", "OSError") in state.cleanup_errors


def test_concurrent_records_and_refresh_are_serialized(
    tmp_path: Path, helper: tuple[Path, Path]
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from tools.pytest_native_diagnostics import State

    state = State(tmp_path, helper[0])

    def record(index: int) -> None:
        for count in range(32):
            state.refresh_modules()
            state.write("start", nodeid=f"test-{index}-{count}")

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(record, i) for i in range(8)]
            for future in futures:
                future.result(timeout=10)
    finally:
        state.close()
    records = [
        json.loads(line)
        for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()
    ]
    assert sum(r["event"] == "start" for r in records) == 256


def test_repeated_install_keeps_original_faulthandler(
    tmp_path: Path, helper: tuple[Path, Path]
) -> None:
    result = child(
        tmp_path,
        helper,
        "assert state.lib.diagnostic_install(state.native_fd) == -2\nctypes.string_at(0)",
    )
    assert result.returncode == -signal.SIGSEGV
    assert "Fatal Python error" in result.stderr
    assert len(next(tmp_path.glob("*.native.log")).read_text().splitlines()) == 1


def test_cleanup_preserves_a_later_installed_handler(
    tmp_path: Path, helper: tuple[Path, Path]
) -> None:
    result = child(
        tmp_path,
        helper,
        f"ctypes.CDLL({str(helper[1])!r}).install_prior(0)\nstate.close()\nos.kill(os.getpid(), {int(signal.SIGABRT)})\nprint('returned')",
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("PRIOR") == 1
    assert "returned" in result.stdout


def test_interpolated_python_step_respects_github_expression_budget() -> None:
    # GitHub rewrites a mixed scalar as format('escaped literal', expressions).
    # Runner ExpressionConstants.MaxLength is 21000. Raw YAML length misses
    # quote/brace escaping and format-call/placeholder overhead.
    # https://github.com/actions/runner/blob/main/src/Sdk/DTObjectTemplating/ObjectTemplating/TemplateReader.cs
    source = (ROOT / ".github/workflows/ci.yml").read_text()
    step = source.split("      - name: Run Python tests with coverage\n", 1)[1].split(
        "\n      - name:", 1
    )[0]
    literal = step.split("        run: |\n", 1)[1]
    script = "\n".join(line[10:] for line in literal.splitlines()) + "\n"
    # Conservative upper bound: count every quote/brace, even inside expressions,
    # plus four characters per placeholder/argument and the format wrapper.
    upper_bound = (
        len(script.encode("utf-16-le")) // 2
        + script.count("'")
        + script.count("{")
        + script.count("}")
        + 4 * script.count("${{")
        + len("format('')")
    )
    assert upper_bound <= 21000

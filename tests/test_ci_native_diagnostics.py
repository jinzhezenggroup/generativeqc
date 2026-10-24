"""Subprocess contracts; deliberately faulting children must remain failures."""

from __future__ import annotations

import ctypes
import importlib.util
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
            "-U_FORTIFY_SOURCE",
            "-D_FORTIFY_SOURCE=3",
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


@pytest.mark.parametrize(
    ("fault", "sig"),
    [("ctypes.string_at(0)", signal.SIGSEGV), ("os.abort()", signal.SIGABRT)],
)
def test_xdist_crash_and_assertion_remain_failures(
    tmp_path: Path, helper: tuple[Path, Path], fault: str, sig: signal.Signals
) -> None:
    tests = tmp_path / "test_child.py"
    tests.write_text(
        "import ctypes, os\ndef test_pass(): pass\ndef test_assertion(): assert False\n"
        f"def test_crash(): {fault}\n"
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
            "-p",
            "tools.pytest_safe_diagnostics",
            "--diagnostic-timeout=0.001",
            f"--diagnostics-dir={tmp_path / 'safe-diagnostics'}",
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
    exits = [
        json.loads(line)
        for path in (tmp_path / "safe-diagnostics").glob("controller-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    crashed = [record for record in exits if record.get("returncode") == -sig]
    assert len(crashed) == 1
    assert crashed[0]["signal"] == sig
    assert crashed[0]["signal_name"] == sig.name
    assert crashed[0]["returncode_source"] == "local_popen"
    assert any(record.get("returncode") == 0 for record in exits)
    assert all("resources" in record for record in exits)


def test_workflow_keeps_diagnostics_in_failure_artifacts() -> None:
    source = (ROOT / ".github/workflows/ci.yml").read_text()
    assert '--native-diagnostics-dir="$PWD/.artifacts/native-diagnostics"' in source
    assert "path: .artifacts/" in source and "retention-days: 14" in source
    preparation = source.split("      - name: Prepare bounded native diagnostics\n", 1)[
        1
    ].split("      - name:", 1)[0]
    spec = importlib.util.spec_from_file_location(
        "ci_scope_workflow_contracts", ROOT / "tests/python/test_ci_prose_workflow.py"
    )
    assert spec is not None and spec.loader is not None
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    for shard in (
        "core-a",
        "core-b",
        "core-c",
        "compiler-heavy",
        "runtime-heavy",
        "uccsd-forces",
        "posthf",
        "ecp-forces",
    ):
        for outcome, scope in (
            ("success", "full"),
            ("success", "prose"),
            ("success", ""),
            ("failure", "prose"),
        ):
            expected = shard in {"core-a", "core-b", "core-c"} and not (
                outcome == "success" and scope == "prose"
            )
            assert (
                admission.selected(
                    preparation,
                    {
                        "matrix.shard": shard,
                        "steps.prose_scope.outcome": outcome,
                        "steps.prose_scope.outputs.scope": scope,
                    },
                )
                == expected
            )
    assert "ccache --version" in preparation
    assert (
        ".venv/bin/python -m pytest tests/test_ci_native_diagnostics.py -q"
        in preparation
    )
    assert '"${diagnostic_args[@]}"' in source
    wheels = (ROOT / ".github/workflows/wheels.yml").read_text()
    assert "  push:\n    branches: [master]" in wheels


@pytest.mark.parametrize(
    ("shard", "enabled"),
    [
        ("core-a", True),
        ("core-b", True),
        ("core-c", True),
        ("runtime-heavy", False),
        ("compiler-heavy", False),
        ("posthf", False),
        ("ecp-forces", False),
    ],
)
def test_workflow_diagnostic_arguments_follow_shard_scope(
    tmp_path: Path, shard: str, enabled: bool
) -> None:
    source = (ROOT / ".github/workflows/ci.yml").read_text()
    step = source.split("      - name: Run Python tests with coverage\n", 1)[1].split(
        "\n      - name:", 1
    )[0]
    selection = (
        "diagnostic_args=()"
        + step.split("diagnostic_args=()", 1)[1].split(".venv/bin/python -m pytest", 1)[
            0
        ]
    )
    selection = selection.replace("${{ matrix.shard }}", shard)
    result = subprocess.run(
        ["bash", "-e", "-c", selection + '\n printf "%s\\n" "${diagnostic_args[@]}"'],
        cwd=tmp_path,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    expected = (
        [
            "-p",
            "tools.pytest_native_diagnostics",
            f"--native-diagnostics-dir={tmp_path}/.artifacts/native-diagnostics",
            f"--native-diagnostics-library={tmp_path}/libnative-fault-record.so",
        ]
        if enabled
        else []
    )
    assert result.stdout.splitlines() == (expected or [""])


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


@pytest.mark.parametrize("compiler_name", ["gcc", "clang"])
@pytest.mark.parametrize("fortify", [2, 3])
def test_helper_compiles_with_fortified_libc(
    tmp_path: Path, compiler_name: str, fortify: int
) -> None:
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"Optional {compiler_name} compiler is not installed")
    launcher = shutil.which("ccache")
    assert launcher, "A verified compiler cache is required"
    subprocess.run([launcher, "--version"], check=True, capture_output=True)
    subprocess.run(
        [
            launcher,
            compiler,
            "-std=c11",
            "-fPIC",
            "-O2",
            "-U_FORTIFY_SOURCE",
            f"-D_FORTIFY_SOURCE={fortify}",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            str(ROOT / "tools/native_fault_record.c"),
            "-o",
            str(tmp_path / "record.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_safe_watchdog_one_shot_cancel_and_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import faulthandler
    import threading
    import time

    from tools.pytest_safe_diagnostics import Watchdog

    fired = threading.Event()
    calls = []

    def dump(**kwargs: object) -> None:
        calls.append(kwargs)
        fired.set()

    monkeypatch.setattr(faulthandler, "dump_traceback", dump)
    watchdog = Watchdog(tmp_path, 0.02)
    try:
        watchdog.arm("test_first")
        assert fired.wait(2)
        time.sleep(0.04)
        assert len(calls) == 1
        watchdog.arm("test_cancelled")
        watchdog.cancel()
        time.sleep(0.04)
        assert len(calls) == 1
        fired.clear()
        watchdog.arm("test_next")
        assert fired.wait(2)
        assert len(calls) == 2
        watchdog.arm("test_pending_on_shutdown")
    finally:
        watchdog.close()
    assert not watchdog.thread.is_alive()
    watchdog.close()
    watchdog.arm("test_after_shutdown")
    assert len(calls) == 2
    text = watchdog.path.read_text()
    assert '"test_first"' in text and '"test_next"' in text
    assert "test_cancelled" not in text and "test_after_shutdown" not in text
    assert "test_pending_on_shutdown" not in text


def test_safe_watchdog_dump_errors_do_not_stop_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import faulthandler
    import threading

    from tools.pytest_safe_diagnostics import Watchdog

    fired = threading.Event()

    def broken_dump(**kwargs: object) -> None:
        fired.set()
        raise OSError("synthetic diagnostic failure")

    monkeypatch.setattr(faulthandler, "dump_traceback", broken_dump)
    watchdog = Watchdog(tmp_path, 0.001)
    try:
        watchdog.arm("test_still_runs")
        assert fired.wait(2)
        watchdog.cancel()
        assert watchdog.thread.is_alive()
    finally:
        watchdog.close()


def test_safe_watchdog_stack_churn_and_fatal_recorder_coexist(
    tmp_path: Path, helper: tuple[Path, Path]
) -> None:
    result = child(
        tmp_path,
        helper,
        f"""
import time
from tools.pytest_safe_diagnostics import Watchdog

def forbidden(*args, **kwargs):
    raise AssertionError('must not use the GIL-free C watchdog')
faulthandler.dump_traceback_later = forbidden

def recurse(n):
    a, b, c = 1, 2, 3
    if n:
        return recurse(n - 1)

watchdog = Watchdog(Path({str(tmp_path / "timed")!r}), 0.001)
try:
    watchdog.arm('stack_churn')
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        recurse(123)
finally:
    watchdog.close()
assert not watchdog.thread.is_alive()
assert 'stack_churn' in watchdog.path.read_text()
ctypes.string_at(0)
""",
    )
    # A genuine native failure must still carry its original signal and records.
    assert result.returncode == -signal.SIGSEGV, result.stdout + result.stderr
    assert "Fatal Python error: Segmentation fault" in result.stderr
    assert "signal=0xb" in next(tmp_path.glob("*.native.log")).read_text()
    assert "CI timed traceback:" in result.stderr and "stack_churn" in result.stderr
    trace = next((tmp_path / "timed").glob("*.traceback.log")).read_text()
    assert "Thread 0x" in trace and '"threshold_seconds": 0.001' in trace


def test_safe_exit_recorder_retains_unknown_transport(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from tools.pytest_safe_diagnostics import ExitRecorder

    recorder = ExitRecorder(tmp_path)
    recorder.node_down(SimpleNamespace(gateway=SimpleNamespace(id="gw-remote")), "dead")
    recorder.close()
    record = json.loads(recorder.path.read_text().splitlines()[1])
    assert record["returncode"] is None and record["signal"] is None
    assert record["returncode_source"] == "unavailable_transport"
    assert record["worker"] == "gw-remote"


def test_safe_plugin_rejects_simultaneous_unsafe_timer(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "tools.pytest_safe_diagnostics",
            "--diagnostic-timeout=0.01",
            f"--diagnostics-dir={tmp_path / 'diagnostics'}",
            "-o",
            "faulthandler_timeout=0.01",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "requires faulthandler_timeout=0" in result.stderr


def test_safe_workflow_preserves_threshold_coverage_and_limits() -> None:
    source = (ROOT / ".github/workflows/ci.yml").read_text()
    step = source.split("      - name: Run Python tests with coverage\n", 1)[1].split(
        "\n      - name:", 1
    )[0]
    assert "-o faulthandler_timeout=0" in step
    assert "-p tools.pytest_safe_diagnostics --diagnostic-timeout=900" in step
    assert '--diagnostics-dir="$PWD/.artifacts/python-diagnostics"' in step
    assert '"${diagnostic_args[@]}"' in step
    assert '"${coverage_args[@]}"' in step
    assert '-n 4 --dist="$dist_mode"' in step
    assert "&& 90 || 30" in step
    assert "&& 100 || 40" in source
    assert "faulthandler_timeout=900" not in source
    preservation = source.split(
        "      - name: Preserve transient benchmark/debug artifacts on failure\n", 1
    )[1].split("\n      - name:", 1)[0]
    assert "if: failure()" in preservation
    assert "path: .artifacts/" in preservation and "retention-days: 14" in preservation
    assert ".venv/bin/python -m pytest tests/test_ci_native_diagnostics.py -q" in source


def test_safe_watchdog_reports_delay_for_gil_held_native_call(tmp_path: Path) -> None:
    import time

    from tools.pytest_safe_diagnostics import Watchdog

    watchdog = Watchdog(tmp_path, 0.001)
    libc = ctypes.PyDLL(None)  # Deliberately retain the GIL through the native sleep.
    libc.usleep.argtypes = [ctypes.c_uint]
    libc.usleep.restype = ctypes.c_int
    try:
        watchdog.arm("gil_held_native_call")
        assert libc.usleep(50000) == 0
        deadline = time.monotonic() + 2
        while not watchdog.path.stat().st_size and time.monotonic() < deadline:
            time.sleep(0.001)
    finally:
        watchdog.close()
    record = json.loads(watchdog.path.read_text().splitlines()[0])
    assert record["delay_seconds"] >= 0.03


def test_safe_plugin_storage_failure_preserves_original_outcome(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("keep")
    test = tmp_path / "test_original.py"
    test.write_text("def test_original(): assert False, 'original failure'\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test),
            "-q",
            "-p",
            "tools.pytest_safe_diagnostics",
            "--diagnostic-timeout=0.001",
            f"--diagnostics-dir={blocked}",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == pytest.ExitCode.TESTS_FAILED
    assert "original failure" in result.stdout and "1 failed" in result.stdout
    assert "CI diagnostics unavailable: FileExistsError" in result.stderr
    assert blocked.read_text() == "keep"


def test_safe_timer_never_calls_the_unlocked_c_watchdog() -> None:
    import ast

    from tools import pytest_safe_diagnostics

    tree = ast.parse(Path(pytest_safe_diagnostics.__file__).read_text())
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call, ast.Attribute) and call.attr == "dump_traceback_later"
        for call in calls
    )
    assert any(
        isinstance(call, ast.Attribute) and call.attr == "dump_traceback"
        for call in calls
    )

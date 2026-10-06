"""Private foreground cache workers must obey the compiler's finite lifetime."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from generativeqc_compiler.common import compiler_cache, compiler_process, cuda_adapter
from generativeqc_compiler.common.compiler_process import CompileResult, run_compiler
from generativeqc_compiler.common.cuda_target import cuda_target_info


@pytest.fixture
def cache_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Exercise real process groups without requiring sockets in unit tests."""
    launcher = tmp_path / "sccache"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, signal, subprocess, sys, time\n"
        f"root = pathlib.Path({str(tmp_path)!r})\n"
        "endpoint = pathlib.Path(os.environ['SCCACHE_SERVER_UDS'])\n"
        "if os.environ.get('SCCACHE_START_SERVER') == '1':\n"
        " assert os.environ['SCCACHE_NO_DAEMON'] == '1'\n"
        " assert 'SCCACHE_STARTUP_NOTIFY' not in os.environ\n"
        " child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
        " (root / (endpoint.parent.name + '.json')).write_text(json.dumps(\n"
        "  {'server':os.getpid(), 'child':child.pid, 'endpoint':str(endpoint), 'cache':os.environ['SCCACHE_DIR']}))\n"
        " if os.environ.get('QC_TEST_STARTUP') == 'fail':\n"
        "  print('server startup failed', flush=True); sys.exit(17)\n"
        " if os.environ.get('QC_TEST_STARTUP') != 'hang': endpoint.touch()\n"
        " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        " time.sleep(60)\n"
        "elif sys.argv[1:] == ['--dist-status']:\n"
        " print(os.environ.get('QC_TEST_DIST', '{\"Disabled\":\"disabled\"}'))\n"
        "else:\n"
        " assert sys.argv[1] != '--stop-server'\n"
        " if sys.argv[1] == 'timeout': time.sleep(60)\n"
        " if sys.argv[1] == 'slow': time.sleep(0.3)\n"
        " print('compiled', flush=True)\n"
        " sys.exit(17 if sys.argv[1] == 'failure' else 0)\n"
    )
    launcher.chmod(0o755)
    monkeypatch.setenv("GENERATIVEQC_JIT_SCCACHE_ROOT", str(tmp_path / "pool"))
    monkeypatch.setenv("SCCACHE_SERVER_UDS", str(tmp_path / "shared.sock"))
    monkeypatch.setenv("SCCACHE_STARTUP_NOTIFY", str(tmp_path / "shared-notify"))
    monkeypatch.setattr(
        compiler_cache,
        "resolve_compiler_cache",
        lambda _timeout: compiler_cache.CompilerCacheLauncher(
            launcher, "sccache", "sccache 0.16.0"
        ),
    )
    return tmp_path


def _running(pid: int) -> bool:
    # Linux can briefly retain killed grandchildren as zombies until init reaps.
    stat = Path(f"/proc/{pid}/stat")
    try:
        if stat.read_text().split()[2] == "Z":
            return False
    except (FileNotFoundError, ProcessLookupError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _assert_workers_stopped(root: Path, count: int) -> None:
    workers = [json.loads(path.read_text()) for path in root.glob("gqc-scc-*.json")]
    assert len(workers) == count
    deadline = time.monotonic() + 2
    for worker in workers:
        while any(_running(worker[key]) for key in ("server", "child")):
            assert time.monotonic() < deadline, worker
            time.sleep(0.01)
        assert not Path(worker["endpoint"]).exists()
        assert worker["endpoint"] != str(root / "shared.sock")


@pytest.mark.parametrize(
    "command,code", [("success", 0), ("failure", 17), ("timeout", 124)]
)
def test_owned_workers_stop_on_every_outcome(
    cache_workers: Path, command: str, code: int
) -> None:
    result = compiler_cache.run_cached_compiler([command], 0.5, label="probe")
    assert result.returncode == code
    assert result.timed_out == (command == "timeout")
    assert result.duration_seconds < 1.5
    _assert_workers_stopped(cache_workers, 1)


@pytest.mark.parametrize("startup,code", [("fail", 17), ("hang", 124)])
def test_startup_uses_the_same_budget_and_cleans_children(
    cache_workers: Path, monkeypatch: pytest.MonkeyPatch, startup: str, code: int
) -> None:
    monkeypatch.setenv("QC_TEST_STARTUP", startup)
    result = compiler_cache.run_cached_compiler(["success"], 0.2, label="probe")
    assert result.returncode == code
    assert result.duration_seconds < 1.2
    _assert_workers_stopped(cache_workers, 1)


@pytest.mark.parametrize(
    "distribution", ['{"NotConnected":[null,"remote"]}', "bad json"]
)
def test_remote_or_unknown_workers_fail_closed(
    cache_workers: Path, monkeypatch: pytest.MonkeyPatch, distribution: str
) -> None:
    monkeypatch.setenv("QC_TEST_DIST", distribution)
    result = compiler_cache.run_cached_compiler(["success"], 2, label="probe")
    assert result.returncode == 1
    assert "local sccache workers" in result.stderr
    _assert_workers_stopped(cache_workers, 1)


def test_concurrent_timeout_does_not_kill_another_invocation(
    cache_workers: Path,
) -> None:
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            timed = executor.submit(
                compiler_cache.run_cached_compiler, ["timeout"], 0.2, label="probe"
            )
            successful = executor.submit(
                compiler_cache.run_cached_compiler, ["slow"], 2, label="probe"
            )
            assert timed.result().timed_out
            assert successful.result().returncode == 0
        assert sentinel.poll() is None
        _assert_workers_stopped(cache_workers, 2)
    finally:
        os.killpg(sentinel.pid, signal.SIGKILL)
        sentinel.wait()


def test_cuda_link_preserves_timeout_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cuda_adapter,
        "run_cached_compiler",
        lambda *a, **k: CompileResult(124, True, 0.1, "out", "timeout"),
    )
    adapter = cuda_adapter.CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    with pytest.raises(subprocess.TimeoutExpired) as error:
        adapter.link(tmp_path / "driver.cu", [], tmp_path / "driver", timeout=0.1)
    assert error.value.stdout == "out"


@pytest.mark.parametrize("code", [0, 17])
def test_client_exit_reclaims_children_with_detached_output(code: int) -> None:
    script = (
        "import subprocess,sys; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print(child.pid, flush=True); "
        f"raise SystemExit({code})"
    )
    result = run_compiler([sys.executable, "-c", script], 2, label="probe")
    assert result.returncode == code and not result.timed_out
    pid = int(result.stdout)
    deadline = time.monotonic() + 2
    while _running(pid):
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_cache_slots_reuse_sequentially_and_separate_live_owners(
    cache_workers: Path,
) -> None:
    deadline = time.monotonic() + 3
    with (
        compiler_cache._sccache_slot(deadline) as (first, _),
        compiler_cache._sccache_slot(deadline) as (second, _),
    ):
        assert first != second
    with compiler_cache._sccache_slot(deadline) as (replay, _):
        assert replay == first


def test_server_inherits_lease_after_parent_closes_it(cache_workers: Path) -> None:
    deadline = time.monotonic() + 3
    with compiler_cache._sccache_slot(deadline) as (first, descriptor):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            pass_fds=(descriptor,),
        )
    try:
        with compiler_cache._sccache_slot(deadline) as (second, _):
            assert second != first
    finally:
        child.kill()
        child.wait()
    with compiler_cache._sccache_slot(deadline) as (replay, _):
        assert replay == first


def test_expired_slot_budget_does_not_acquire_a_cache(cache_workers: Path) -> None:
    with (
        pytest.raises(subprocess.TimeoutExpired),
        compiler_cache._sccache_slot(time.monotonic() - 1),
    ):
        pytest.fail("expired acquisition must not yield")


def test_jit_store_never_overlaps_shared_disk_cache(
    cache_workers: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCCACHE_DIR", str(cache_workers))
    with pytest.raises(ValueError, match="outside the shared"):
        compiler_cache.run_cached_compiler(["success"], 2, label="probe")


@pytest.mark.parametrize("format", ["toml", "json"])
def test_file_config_shared_cache_cannot_contain_jit_pool(
    cache_workers: Path, monkeypatch: pytest.MonkeyPatch, format: str
) -> None:
    monkeypatch.delenv("SCCACHE_DIR", raising=False)
    config = cache_workers / f"config.{format}"
    if format == "json":
        config.write_text(json.dumps({"cache": {"disk": {"dir": str(cache_workers)}}}))
    else:
        config.write_text(f'[cache.disk]\ndir = "{cache_workers}"\n')
    monkeypatch.setenv("SCCACHE_CONF", str(config))
    with pytest.raises(ValueError, match="outside the shared"):
        compiler_cache.run_cached_compiler(["success"], 2, label="probe")


def test_default_shared_cache_cannot_contain_jit_pool(
    cache_workers: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SCCACHE_DIR", raising=False)
    monkeypatch.setenv("SCCACHE_CONF", str(cache_workers / "missing-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_workers))
    monkeypatch.setenv(
        "GENERATIVEQC_JIT_SCCACHE_ROOT", str(cache_workers / "sccache/jit")
    )
    monkeypatch.setattr(compiler_cache.sys, "platform", "linux")
    with pytest.raises(ValueError, match="outside the shared"):
        compiler_cache.run_cached_compiler(["success"], 2, label="probe")


def test_process_launch_time_consumes_the_same_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = subprocess.Popen

    def delayed(command: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        time.sleep(0.25)
        return original(command, **kwargs)

    monkeypatch.setattr(compiler_process.subprocess, "Popen", delayed)
    result = run_compiler(
        [sys.executable, "-c", "import time; time.sleep(0.2)"], 0.3, label="probe"
    )
    assert result.timed_out
    assert result.duration_seconds < 0.6


@pytest.mark.parametrize(
    "setting,value",
    [
        ("SCCACHE_CACHE_SIZE", "1G"),
        ("SCCACHE_DIRECT", "true"),
        ("SCCACHE_LOCAL_RW_MODE", "READ_ONLY"),
    ],
)
def test_disk_policy_override_cannot_hide_default_shared_root(
    cache_workers: Path, monkeypatch: pytest.MonkeyPatch, setting: str, value: str
) -> None:
    monkeypatch.delenv("SCCACHE_DIR", raising=False)
    config = cache_workers / "config"
    config.write_text(f'[cache.disk]\ndir = "{cache_workers / "other"}"\n')
    monkeypatch.setenv("SCCACHE_CONF", str(config))
    monkeypatch.setenv(setting, value)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_workers))
    monkeypatch.setenv(
        "GENERATIVEQC_JIT_SCCACHE_ROOT", str(cache_workers / "sccache/jit")
    )
    monkeypatch.setattr(compiler_cache.sys, "platform", "linux")
    with pytest.raises(ValueError, match="outside the shared"):
        compiler_cache.run_cached_compiler(["success"], 2, label="probe")

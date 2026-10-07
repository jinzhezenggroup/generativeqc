"""Real sccache process/cache qualification, mandatory in its focused CI step."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from generativeqc_compiler.common import compiler_cache


@pytest.fixture
def real_sccache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    if os.environ.get("GENERATIVEQC_REQUIRE_SCCACHE_TEST") != "1":
        pytest.skip("real sccache lifecycle runs in the mandatory focused CI step")
    cache = shutil.which("sccache")
    compiler = shutil.which("g++")
    if cache is None or compiler is None:
        pytest.fail("required real sccache/g++ lifecycle prerequisites are missing")
    compiler_cache._resolve_compiler_cache.cache_clear()
    monkeypatch.setenv("SCCACHE_DIR", str(tmp_path / "shared-cache"))
    monkeypatch.setenv("GENERATIVEQC_JIT_SCCACHE_ROOT", str(tmp_path / "pool"))
    shared = tmp_path / "shared-cache"
    shared.mkdir()
    (shared / ".sccachetmp-live-peer").write_bytes(b"unrelated live cache write")
    # This is a fresh per-test configuration, never the user's shared daemon.
    config = tmp_path / "config"
    config.write_text("")
    monkeypatch.setenv("SCCACHE_CONF", str(config))
    monkeypatch.setenv("SCCACHE_CACHED_CONF", str(tmp_path / "cached-config"))
    monkeypatch.setenv("SCCACHE_SERVER_UDS", str(tmp_path / "unrelated.sock"))
    wrapper = tmp_path / "g++"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys, time\n"
        f"root = pathlib.Path({str(tmp_path)!r})\n"
        "if '-c' in sys.argv:\n"
        " tag = next(x.split('=', 1)[1] for x in sys.argv if x.startswith('-DGQC_TAG='))\n"
        " (root / (tag + '.pid')).write_text(str(os.getpid()))\n"
        " with (root / 'compiles').open('a') as log: log.write(tag + '\\n')\n"
        " if '-DGQC_HANG=1' in sys.argv: time.sleep(60)\n"
        " if '-DGQC_SLOW=1' in sys.argv: time.sleep(4)\n"
        " if '-DGQC_FAIL=1' in sys.argv: sys.exit(19)\n"
        f"os.execv({compiler!r}, [{compiler!r}, *sys.argv[1:]])\n"
    )
    wrapper.chmod(0o755)
    source = tmp_path / "probe.cpp"
    source.write_text("int answer() { return 2028; }\n")
    servers: list[tuple[subprocess.Popen[str], str, str]] = []
    original = subprocess.Popen

    def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        process = original(*args, **kwargs)
        env = kwargs.get("env") or {}
        if env.get("SCCACHE_START_SERVER") == "1":
            servers.append((process, env["SCCACHE_SERVER_UDS"], env["SCCACHE_DIR"]))
        return process

    monkeypatch.setattr(compiler_cache.subprocess, "Popen", popen)
    assert compiler_cache.resolve_compiler_cache().name == "sccache"
    return {"root": tmp_path, "compiler": wrapper, "source": source, "servers": servers}


def _command(probe: dict[str, Any], name: str, *options: str) -> list[str]:
    return [
        str(probe["compiler"]),
        "-DGQC_TAG=" + name.replace(".", "_"),
        *options,
        "-c",
        str(probe["source"]),
        "-o",
        str(probe["root"] / name),
    ]


def _assert_cleanup(probe: dict[str, Any]) -> None:
    assert (
        probe["root"] / "shared-cache" / ".sccachetmp-live-peer"
    ).read_bytes() == b"unrelated live cache write"
    assert probe["servers"]
    endpoints = []
    for process, endpoint, _cache in probe["servers"]:
        assert process.poll() is not None
        assert not Path(endpoint).exists()
        assert endpoint != str(probe["root"] / "unrelated.sock")
        endpoints.append(endpoint)
    assert len(set(endpoints)) == len(endpoints)
    for path in probe["root"].glob("*.pid"):
        pid = int(path.read_text())
        deadline = time.monotonic() + 2
        while True:
            stat = Path(f"/proc/{pid}/stat")
            try:
                if stat.read_text().split()[2] == "Z":
                    break
            except (FileNotFoundError, ProcessLookupError):
                break
            assert time.monotonic() < deadline, f"compiler {pid} survived cleanup"
            time.sleep(0.01)


def test_real_sccache_success_reuses_cache_across_private_servers(
    real_sccache: dict[str, Any],
) -> None:
    command = _command(real_sccache, "success.o")
    first = compiler_cache.run_cached_compiler(command, 30, label="probe")
    assert first.returncode == 0, first.stderr
    output = real_sccache["root"] / "success.o"
    original = output.read_bytes()
    output.unlink()
    second = compiler_cache.run_cached_compiler(command, 30, label="probe")
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == original
    assert (real_sccache["root"] / "compiles").read_text().splitlines() == ["success_o"]
    assert len(real_sccache["servers"]) == 2
    _assert_cleanup(real_sccache)


def test_real_sccache_compiler_failure_cleans_server(
    real_sccache: dict[str, Any],
) -> None:
    result = compiler_cache.run_cached_compiler(
        _command(real_sccache, "failure.o", "-DGQC_FAIL=1"), 30, label="probe"
    )
    assert result.returncode == 19 and not result.timed_out, result.stderr
    assert not (real_sccache["root"] / "failure.o").exists()
    _assert_cleanup(real_sccache)


def test_real_sccache_timeout_cleans_compiler_and_server(
    real_sccache: dict[str, Any],
) -> None:
    result = compiler_cache.run_cached_compiler(
        _command(real_sccache, "timeout.o", "-DGQC_HANG=1"), 3, label="probe"
    )
    assert result.timed_out and result.returncode == 124, result.stderr
    assert (real_sccache["root"] / "timeout_o.pid").exists(), (
        "probe must reach real compilation before timeout"
    )
    assert result.duration_seconds < 4
    _assert_cleanup(real_sccache)
    assert not (real_sccache["root"] / "timeout.o").exists()


def test_real_sccache_concurrent_calls_are_isolated(
    real_sccache: dict[str, Any],
) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        timed = executor.submit(
            compiler_cache.run_cached_compiler,
            _command(real_sccache, "timed.o", "-DGQC_HANG=1"),
            3,
            label="probe",
        )
        successful = executor.submit(
            compiler_cache.run_cached_compiler,
            _command(real_sccache, "other.o", "-DGQC_SLOW=1"),
            30,
            label="probe",
        )
        assert timed.result().timed_out
        result = successful.result()
        assert result.returncode == 0, result.stderr
    assert (real_sccache["root"] / "timed_o.pid").exists()
    assert (real_sccache["root"] / "other.o").exists()
    _assert_cleanup(real_sccache)


def test_real_sccache_overlapping_writes_have_exclusive_reusable_stores(
    real_sccache: dict[str, Any],
) -> None:
    left = _command(real_sccache, "left.o", "-DGQC_SLOW=1")
    right = _command(real_sccache, "right.o", "-DGQC_SLOW=1")
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            compiler_cache.run_cached_compiler, left, 30, label="probe"
        )
        deadline = time.monotonic() + 10
        while not real_sccache["servers"]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        second = executor.submit(
            compiler_cache.run_cached_compiler, right, 30, label="probe"
        )
        for future in (first, second):
            result = future.result()
            assert result.returncode == 0, result.stderr
    assert len(real_sccache["servers"]) == 2
    assert len({server[2] for server in real_sccache["servers"]}) == 2
    before = (real_sccache["root"] / "compiles").read_text().splitlines()
    assert sorted(before) == ["left_o", "right_o"]
    # Replay each exact command in the same stable slot. Holding the lower lease
    # models another active caller; it does not touch or clear either cache.
    replay = compiler_cache.run_cached_compiler(left, 30, label="probe")
    assert replay.returncode == 0, replay.stderr
    with compiler_cache._sccache_slot(time.monotonic() + 30):
        replay = compiler_cache.run_cached_compiler(right, 30, label="probe")
        assert replay.returncode == 0, replay.stderr
    assert (real_sccache["root"] / "compiles").read_text().splitlines() == before
    _assert_cleanup(real_sccache)

"""Exercise the finite benchmark wrapper without loading a CUDA runtime."""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from benchmarks import direct_force_reuse


def test_scheduler_visibility_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never create an output or start a GPU process outside Slurm."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit, match="2"):
        direct_force_reuse.main([str(tmp_path / "not-started")])
    assert not (tmp_path / "not-started").exists()


@pytest.mark.parametrize("timeout", [False, True])
@pytest.mark.parametrize("candidate", ["automatic", "relative-reuse"])
def test_population_budget_and_owned_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout: bool, candidate: str
) -> None:
    """Preserve device identity, five-call worker protocol and finite cleanup."""
    for name, value in (
        ("SLURM_JOB_ID", "fixture"),
        ("CUDA_VISIBLE_DEVICES", "GPU-scheduler-assigned"),
        ("GENERATIVEQC_LIBRARY", "/fixture/libgenerativeqc.so"),
        ("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS", "relative-reuse"),
    ):
        monkeypatch.setenv(name, value)
    launches, signals = [], []

    class Child:
        pid = 123456

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            launches.append((command, kwargs))
            self.waits = 0

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            if self.waits == 1:
                assert timeout == 1800
            if self.waits == 2:
                assert timeout == 10
            if timeout is not None and should_timeout:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return 0

    should_timeout = timeout
    monkeypatch.setattr(direct_force_reuse.subprocess, "Popen", Child)
    monkeypatch.setattr(
        direct_force_reuse.os, "killpg", lambda *args: signals.append(args)
    )
    output = tmp_path / "population"
    arguments = [str(output), "--candidate", candidate]
    if timeout:
        with pytest.raises(SystemExit, match="124"):
            direct_force_reuse.main(arguments)
        assert (output / "baseline/exit-status.txt").read_text() == "124\n"
        assert len(signals) == 2 and all(row[0] == Child.pid for row in signals)
        assert not (output / "candidate").exists()
    else:
        direct_force_reuse.main(arguments)
        assert len(launches) == 2
        assert all(row[0] == Child.pid for row in signals)
        assert (output / "candidate/exit-status.txt").read_text() == "0\n"
        selected = launches[1][1]["env"].get("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS")
        assert selected == (None if candidate == "automatic" else candidate)
    assert (
        launches[0][1]["env"]["GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS"]
        == "spatial-reuse"
    )
    for command, options in launches:
        assert "--worker" in command and options["start_new_session"]
        assert options["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-scheduler-assigned"


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "GENERATIVEQC_LIBRARY"):
        monkeypatch.setenv(name, "fixture")


@pytest.mark.parametrize("primary", ["timeout", "interrupt", "error", "nonzero"])
@pytest.mark.parametrize("cleanup", ["gone", "permission", "timeout"])
def test_primary_failure_and_receipt_survive_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheduler: None,
    primary: str,
    cleanup: str,
) -> None:
    """Secondary cleanup errors must not replace the original failure."""
    failure = {
        "timeout": subprocess.TimeoutExpired("fixture", 1800),
        "interrupt": KeyboardInterrupt("fixture"),
        "error": RuntimeError("fixture"),
    }.get(primary)

    class Child:
        pid = 123456

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calls = 0

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            self.calls += 1
            if self.calls == 1:
                if failure is not None:
                    raise failure
                return 7
            if cleanup == "timeout":
                raise subprocess.TimeoutExpired("cleanup", timeout)
            return 0

    def signal_group(pid: int, sig: int) -> None:
        assert pid == Child.pid
        if cleanup == "gone":
            raise ProcessLookupError("gone")
        if cleanup == "permission":
            raise PermissionError("controlled cleanup failure")

    monkeypatch.setattr(direct_force_reuse.subprocess, "Popen", Child)
    monkeypatch.setattr(direct_force_reuse.os, "killpg", signal_group)
    output = tmp_path / "receipt"
    expected = {"timeout": 124, "nonzero": 7, "interrupt": 130, "error": 1}[primary]
    with pytest.raises(BaseException) as caught:
        direct_force_reuse.main([str(output)])
    if primary in ("timeout", "nonzero"):
        assert isinstance(caught.value, SystemExit) and caught.value.code == expected
    else:
        assert caught.value is failure
    assert (output / "baseline/exit-status.txt").read_text() == f"{expected}\n"
    assert not (output / "candidate").exists()


@pytest.mark.parametrize(
    "existing", ["reference.json", "native.json", ".worker-started", "unrelated"]
)
def test_worker_reuse_preserves_all_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheduler: None,
    existing: str,
) -> None:
    """Refuse a reused arm before endpoint imports or any reference write."""
    output = tmp_path / "arm"
    output.mkdir()
    (output / existing).write_text("original evidence")
    with pytest.raises(FileExistsError):
        direct_force_reuse.main([str(output), "--worker"])
    assert {path.name: path.read_text() for path in output.iterdir()} == {
        existing: "original evidence"
    }


@pytest.mark.parametrize(
    "kind", ["output", "ancestor", "reference", "native", "dangling"]
)
def test_worker_symlinks_preserve_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheduler: None,
    kind: str,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "sentinel").write_text("original evidence")
    output = tmp_path / "arm"
    if kind == "output":
        output.symlink_to(target, target_is_directory=True)
    elif kind == "ancestor":
        output.symlink_to(target, target_is_directory=True)
        output = output / "child"
    else:
        output.mkdir()
        name = "native.json" if kind == "native" else "reference.json"
        (output / name).symlink_to(
            target / ("missing" if kind == "dangling" else "sentinel")
        )
    with pytest.raises((ValueError, FileExistsError)):
        direct_force_reuse.main([str(output), "--worker"])
    assert (target / "sentinel").read_text() == "original evidence"
    assert not (target / "child").exists() and not (target / "missing").exists()


@pytest.mark.parametrize("leader_exits", [False, True])
def test_owned_descendant_is_retired(tmp_path: Path, leader_exits: bool) -> None:
    """Use only the finite harmless process group started by this test."""
    import json
    import os
    import signal
    import sys
    import time

    if sys.platform != "linux":
        pytest.skip("Linux process-state witness")
    marker = tmp_path / "owned-descendant.json"
    descendant = (
        "import json,os,pathlib,signal,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"pathlib.Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid(),'pgid':os.getpgrp()}}));"
        "time.sleep(15)"
    )
    leader = (
        "import subprocess,sys,time,pathlib;"
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}]);"
        f"ready_path=pathlib.Path({str(marker)!r});\n"
        "while not ready_path.exists():time.sleep(.01)\n"
        + ("" if leader_exits else "time.sleep(15)")
    )
    child = subprocess.Popen([sys.executable, "-c", leader], start_new_session=True)
    owned = None

    def alive() -> bool:
        if owned is None:
            return False
        path = Path(f"/proc/{owned['pid']}/stat")
        return path.exists() and path.read_text().split()[2] not in ("Z", "X")

    try:
        deadline = time.monotonic() + 3
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        owned = json.loads(marker.read_text())
        assert owned["pgid"] == child.pid != os.getpgrp()
        if leader_exits:
            child.wait(timeout=3)
        direct_force_reuse._stop(child)
        deadline = time.monotonic() + 3
        while alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not alive()
    finally:
        # Only our start_new_session group; do this even when testing old code.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=3)
        deadline = time.monotonic() + 3
        while alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not alive()


@pytest.mark.parametrize("fail_setup", [False, True])
def test_worker_claim_and_endpoint_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheduler: None,
    fail_setup: bool,
) -> None:
    """Claim before setup, retain failures, and keep the endpoint arguments."""
    import json
    import sys
    import types

    from benchmarks import __name__ as package_name

    dft = types.ModuleType("benchmarks.readme_dft")
    omol = types.ModuleType("benchmarks.readme_omol25")
    oracle = types.ModuleType("tools.generativeqc_validation.xc_diagonal_evidence")
    monkeypatch.setattr(
        omol,
        "native_endpoint_record",
        lambda item, **kw: {"fock_builds": None},
        raising=False,
    )
    monkeypatch.setattr(
        oracle, "REFERENCE", {"load": lambda member: {"oracle": member}}, raising=False
    )
    calls = []
    failure = RuntimeError("controlled endpoint setup failure")

    def endpoint() -> None:
        calls.append(sys.argv[:])
        if fail_setup:
            raise failure
        item = types.SimpleNamespace(
            precision={
                "operator_work_counters_valid": True,
                "mixed_stage_fock_builds": 1,
                "strict_stage_fock_builds": 2,
                "post_scf_fock_builds": 3,
            }
        )
        assert omol.native_endpoint_record(item)["fock_builds"] == 6
        item.precision = None
        assert omol.native_endpoint_record(item)["fock_builds"] is None

    monkeypatch.setattr(dft, "main", endpoint, raising=False)
    for module in (dft, omol, oracle):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    for module in (dft, omol):
        monkeypatch.setattr(
            sys.modules[package_name],
            module.__name__.split(".")[-1],
            module,
            raising=False,
        )
    monkeypatch.setattr(sys, "argv", ["fixture"])
    output = tmp_path / "arm"
    output.mkdir()
    if fail_setup:
        with pytest.raises(RuntimeError) as caught:
            direct_force_reuse.main([str(output), "--worker", "--atoms", "48"])
        assert caught.value is failure
    else:
        direct_force_reuse.main([str(output), "--worker", "--atoms", "48"])
    assert json.loads((output / "reference.json").read_text()) == {
        "oracle": "raw/reference-r2scan-3c-48.json.xz"
    }
    assert calls == [
        [
            "readme_dft",
            "--method",
            "r2scan-3c",
            "native",
            "--atoms",
            "48",
            "--basis-file",
            "python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json",
            "--density-fitting",
            "none",
            "--repeats",
            "5",
            "--reference",
            str(output / "reference.json"),
            "--output",
            str(output / "native.json"),
        ]
    ]
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        direct_force_reuse.main([str(output), "--worker"])
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_parent_rejects_existing_output(tmp_path: Path, scheduler: None) -> None:
    output = tmp_path / "population"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    with pytest.raises(FileExistsError):
        direct_force_reuse.main([str(output)])
    assert {path.name: path.read_text() for path in output.iterdir()} == {
        "sentinel": "preserve"
    }


@pytest.mark.parametrize("primary", ["error", "timeout", "nonzero"])
def test_receipt_failure_preserves_primary_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheduler: None,
    primary: str,
) -> None:
    failure = (
        subprocess.TimeoutExpired("fixture", 1800)
        if primary == "timeout"
        else RuntimeError("fixture")
    )

    class Child:
        pid = 123456

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            if primary == "nonzero":
                return 7
            raise failure

    original_open = Path.open

    def open_path(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name == "exit-status.txt":
            raise PermissionError("controlled unwritable receipt")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(direct_force_reuse.subprocess, "Popen", Child)
    monkeypatch.setattr(direct_force_reuse, "_stop", lambda child: None)
    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(BaseException) as caught:
        direct_force_reuse.main([str(tmp_path / "population")])
    if primary == "timeout":
        assert isinstance(caught.value, SystemExit) and caught.value.code == 124
        assert caught.value.__cause__ is failure
    elif primary == "nonzero":
        assert isinstance(caught.value, SystemExit) and caught.value.code == 7
    else:
        assert caught.value is failure

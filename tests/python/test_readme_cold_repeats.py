"""Cold3 protocol, allocation metadata and archived-v1 compatibility."""

import json
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.readme_cold_repeats import SCHEMA, aggregate, population_order
from benchmarks.readme_defaults import allocation_metadata
from tests.python.test_readme_defaults import receipt


def matrix() -> list[dict]:
    records = []
    for repeat in range(3):
        for method, mode, atoms, arm in population_order(repeat):
            record = receipt(arm)
            record.update(
                method=method,
                mode=mode,
                atoms=atoms,
                process_repeat=repeat,
                accelerator={"nvidia_smi": {"uuid": "same-gpu"}},
                native_build={"library_sha256": "same-library"},
                runner_sha256="same-runner",
            )
            record["samples"][0]["seconds"] = [5, 90, 4][repeat]
            for sample in record["samples"]:
                sample["forces"] = [[0.0, 0.0, 0.0]] * atoms
            records.append(record)
    return records


def test_all_cold_outliers_are_retained_and_warm_grouping_is_explicit() -> None:
    rows = aggregate(matrix())
    assert len(rows) == 128
    assert rows[0]["cold"] == {"seconds": [5, 90, 4], "median": 5, "min": 4, "max": 90}
    assert rows[0]["warm"]["process_medians"] == [2, 2, 2]
    assert len(rows[0]["warm"]["all_seconds"]) == 9


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "failed", "gpu", "source", "warm", "nonfinite", "shape"],
)
def test_no_complete_summary_for_a_partial_or_mixed_campaign(change: str) -> None:
    records = matrix()
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records.append(deepcopy(records[0]))
    elif change == "failed":
        records[0]["status"] = "failed"
    elif change == "gpu":
        records[0]["accelerator"]["nvidia_smi"]["uuid"] = "other-gpu"
    elif change == "source":
        records[0]["source_revision"] = "b" * 40
    elif change == "warm":
        records[0]["samples"][2]["repeat"] = 0
    elif change == "nonfinite":
        records[0]["samples"][0]["seconds"] = float("nan")
    else:
        records[0]["samples"][0]["forces"] = [[0, 0, 0]]
    with pytest.raises(ValueError):
        aggregate(records)


def test_qz_never_fabricates_slurm_and_rejects_unbound_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("SLURM_JOB_ID", "LSB_JOBID", "GQC_QZ_JOB_NAME", "GQC_QZ_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="real GPU work"):
        allocation_metadata("test-job")
    monkeypatch.setenv("GQC_QZ_JOB_NAME", "test-job")
    monkeypatch.setenv("GQC_QZ_PROJECT", "project")
    monkeypatch.setattr("socket.gethostname", lambda: "other-job--pod")
    with pytest.raises(ValueError):
        allocation_metadata("test-job")
    monkeypatch.setattr(
        "socket.gethostname",
        lambda: "job-12345678-1234-1234-1234-123456789abc-worker-0",
    )
    assert allocation_metadata("test-job")["kind"] == "qz"
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    assert allocation_metadata() == {"kind": "slurm"}
    with pytest.raises(ValueError, match="combine"):
        allocation_metadata("test-job")


def test_each_round_balances_first_engine_order() -> None:
    for repeat in range(3):
        order = population_order(repeat)
        first = [order[i][3] for i in range(0, len(order), 2)]
        assert first.count("standard") == first.count("gpu4pyscf") == 32
        assert len(set(order)) == 128


def test_both_recorded_qz_host_conventions_are_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.setenv("GQC_QZ_JOB_NAME", "bound-job")
    monkeypatch.setenv("GQC_QZ_PROJECT", "project")
    for hostname in (
        "job-12345678-1234-1234-1234-123456789abc-worker-0",
        "bound-job--notebook",
    ):
        monkeypatch.setattr("socket.gethostname", lambda hostname=hostname: hostname)
        assert allocation_metadata("bound-job")["platform_receipt_required"] is True


@pytest.mark.parametrize("device", ["h100", "rtx5090"])
def test_original_v1_cohorts_keep_every_aggregate_value(device: str) -> None:
    from benchmarks.paper_hfdft_replay import DEFAULT, populations, read_xz

    payload = read_xz((DEFAULT / device / "samples.json.xz").read_bytes())
    assert SCHEMA == "generativeqc.readme-cold-repeats.v1"
    actual = {tuple(row["key"]): row for row in aggregate(payload["records"])}
    expected = {tuple(row["key"]): row for row in populations(payload["records"])}
    assert actual == expected


@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize(
    "relative_paths", [False, True], ids=["absolute-paths", "relative-paths"]
)
def test_acquisition_keeps_fixed_process_census_and_never_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_first: bool,
    relative_paths: bool,
) -> None:
    from benchmarks import readme_cold_repeats as collector

    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.chdir(tmp_path)
    if relative_paths:
        (tmp_path / "checkout").mkdir()
        (tmp_path / "inputs").mkdir()
        (tmp_path / "runtime").mkdir()
        if os.name == "posix":
            (tmp_path / "system-python").write_text("synthetic-only")
            (tmp_path / "runtime" / "fake-python").symlink_to(
                tmp_path / "system-python"
            )
    templates = {
        (r["arm"], r["method"], r["mode"], r["atoms"], r["process_repeat"]): r
        for r in matrix()
    }
    invoked = []

    class Child:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            invoked.append(argv)
            arm = argv[3]
            method, mode = (
                argv[argv.index("--method") + 1],
                argv[argv.index("--mode") + 1],
            )
            atoms = int(argv[argv.index("--atoms") + 1])
            target = Path(argv[argv.index("--output") + 1])
            if not target.is_absolute():
                target = Path(kwargs["cwd"]) / target
            target.parent.mkdir(parents=True, exist_ok=True)
            repeat = int(target.stem.rsplit("-r", 1)[1])
            record = deepcopy(templates[(arm, method, mode, atoms, max(0, repeat))])
            record.pop("process_repeat")
            record["samples"] = record["samples"][
                : int(argv[argv.index("--warm-repeats") + 1]) + 1
            ]
            target.write_text(json.dumps(record), encoding="utf-8")

        def wait(self, timeout: float) -> int:
            assert timeout > 0
            return 1 if fail_first else 0

    monkeypatch.setattr(collector.subprocess, "Popen", Child)
    config = {
        "source": "checkout" if relative_paths else str(tmp_path),
        "inputs": "inputs" if relative_paths else str(tmp_path),
        "source_revision": "a" * 40,
        "deadline_seconds": 1000,
        "process_timeout_seconds": 30,
        "interpreters": {
            a: "./runtime/fake-python" if relative_paths else "fake-python"
            for a in ("standard", "gpu4pyscf")
        },
        "environment": {},
        "arm_environment": {a: {} for a in ("standard", "gpu4pyscf")},
    }
    output = Path("acquisition") if relative_paths else tmp_path / "acquisition"
    if fail_first:
        with pytest.raises(RuntimeError, match="no retry"):
            collector.acquire(config, output)
    else:
        collector.acquire(config, output)
    campaign = json.loads((output / "campaign.json").read_bytes())
    for argv in invoked:
        assert Path(argv[argv.index("--output") + 1]).is_absolute()
        if relative_paths:
            assert Path(argv[0]).is_absolute()
            assert Path(argv[0]).name == "fake-python"
            assert Path(argv[0]).parent.name == "runtime"
        assert (
            Path(argv[argv.index("--inputs") + 1])
            == (tmp_path / "inputs" if relative_paths else tmp_path).resolve()
        )
    if fail_first:
        assert len(invoked) == 1
        assert campaign["status"] == "failed"
    else:
        assert len(invoked) == 512
        assert campaign["formal_processes"] == 384
        assert campaign["formal_endpoints"] == 1536
        assert len([r for r in campaign["commands"] if r["stage"] == "prewarm"]) == 128


def test_interruption_stops_descendant_after_launcher_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks import readme_cold_repeats as collector

    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")
    state = {"leader": True, "worker": True}
    tick = [0.0]

    def clock() -> float:
        tick[0] += 1.0
        return tick[0]

    monkeypatch.setattr(collector.time, "monotonic", clock)
    monkeypatch.setattr(collector.time, "sleep", lambda seconds: None)
    kill = getattr(signal, "SIGKILL", 9)
    monkeypatch.setattr(collector.signal, "SIGKILL", kill, raising=False)

    class Child:
        pid = 99000001

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def wait(self, timeout: float) -> int:
            if state["leader"]:
                raise subprocess.TimeoutExpired("synthetic-launcher", timeout)
            return 0

    def signal_group(pid: int, sig: int) -> None:
        assert pid == Child.pid
        if sig == signal.SIGTERM:
            state["leader"] = False
        elif sig == kill:
            state["leader"] = state["worker"] = False

    monkeypatch.setattr(collector.subprocess, "Popen", Child)
    monkeypatch.setattr(collector.os, "killpg", signal_group, raising=False)
    config = {
        "source": str(tmp_path),
        "inputs": str(tmp_path),
        "source_revision": "a" * 40,
        "deadline_seconds": 30,
        "process_timeout_seconds": 0.1,
        "interpreters": {a: "synthetic-launcher" for a in ("standard", "gpu4pyscf")},
        "environment": {},
        "arm_environment": {a: {} for a in ("standard", "gpu4pyscf")},
    }
    with pytest.raises(subprocess.TimeoutExpired):
        collector.acquire(config, tmp_path / "failed")
    assert not state["worker"]


@pytest.mark.skipif(
    sys.platform != "linux", reason="real Linux process-group resource cleanup"
)
def test_real_process_group_does_not_leave_term_ignoring_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks import readme_cold_repeats as collector

    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-process-test-only")
    marker = tmp_path / "worker.pid"
    worker = "import os,signal,sys,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)"
    launcher = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);time.sleep(60)"
    original = subprocess.Popen
    children = []

    def spawn(argv: list[str], **kwargs: object) -> subprocess.Popen:
        child = original(
            [sys.executable, "-c", launcher, worker, str(marker)], **kwargs
        )
        children.append(child)
        deadline = time.monotonic() + 10
        while not marker.exists():
            if child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("synthetic worker did not become ready")
            time.sleep(0.01)
        return child

    def live_worker() -> bool:
        path = Path("/proc") / marker.read_text() / "stat"
        try:
            return path.read_text().split()[2] != "Z"
        except FileNotFoundError:
            return False

    monkeypatch.setattr(collector.subprocess, "Popen", spawn)
    config = {
        "source": str(tmp_path),
        "inputs": str(tmp_path),
        "source_revision": "a" * 40,
        "deadline_seconds": 30,
        "process_timeout_seconds": 0.05,
        "interpreters": {
            a: "synthetic-process-only" for a in ("standard", "gpu4pyscf")
        },
        "environment": {},
        "arm_environment": {a: {} for a in ("standard", "gpu4pyscf")},
    }
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            collector.acquire(config, tmp_path / "failed")
        deadline = time.monotonic() + 3
        while live_worker() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not live_worker()
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)

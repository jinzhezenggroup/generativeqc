"""The retained D4 qualification driver fails closed without a GPU."""

from __future__ import annotations

import copy
import io
import json
import lzma
import runpy
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

BUNDLE = Path(__file__).resolve().parents[2] / "benchmarks/results/d4-capacity-20261029"


@pytest.fixture
def driver() -> SimpleNamespace:
    return SimpleNamespace(**runpy.run_path(str(BUNDLE / "qualify.py")))


def invoke(
    driver: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    samples: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    *,
    global_budget: bool = False,
) -> dict[str, Any]:
    """Replay the complete entry point with deterministic worker responses."""
    monkeypatch.setenv("D4_QUALIFY_GLOBAL_BUDGET", "1" if global_budget else "0")
    workers = []

    def start(*args: object, **kwargs: Any) -> SimpleNamespace:
        role = "AB"[len(workers)]
        rows = [record["rows"] for record in samples if record["engine"] == role]
        worker = SimpleNamespace(
            stdin=io.StringIO(),
            stdout=io.StringIO(
                json.dumps({"ready": (metadata or {}).get(role, {})})
                + "\n"
                + "\n".join(json.dumps({"rows": row}) for row in [rows[0], *rows])
                + "\n"
            ),
            stopped=False,
        )
        workers.append(worker)
        return worker

    def stop(worker: SimpleNamespace) -> None:
        worker.stopped = True

    output = tmp_path / "samples.json.xz"
    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setitem(driver.main.__globals__, "_stop_worker", stop)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify",
            "--baseline",
            "A",
            "--candidate",
            "B",
            "--baseline-library",
            "A.so",
            "--candidate-library",
            "B.so",
            "--backend",
            "cpu",
            "--groups",
            "8",
            "--output",
            str(output),
        ],
    )
    try:
        driver.main()
    finally:
        assert workers and all(worker.stopped for worker in workers)
    with lzma.open(output, "rt") as stream:
        return json.load(stream)


def population() -> list[dict[str, Any]]:
    item = {
        "energy": -1.0,
        "forces": [[0.0, 0.0, 0.0]],
        "iterations": 1,
        "fock_builds": 1,
        "grid_points": 1,
        "occupation": [1],
        "coulomb_exchange_work": {},
        "policy_work": {},
    }
    rows = [
        {"case": f"fixture-{i}", "phase": "cold", "seconds": 1.0, "items": [item]}
        for i in range(9)
    ]
    return [
        {"group": g, "engine": role, "rows": copy.deepcopy(rows)}
        for g in range(8)
        for role in ("ABBA" if g % 2 == 0 else "BAAB")
    ]


@pytest.mark.parametrize(
    "field,value,role",
    [
        ("energy", float("nan"), "B"),
        ("energy", float("inf"), "both"),
        ("energy", -float("inf"), "A"),
        ("forces", [[float("nan"), 0.0, 0.0]], "B"),
        ("forces", [[float("inf"), 0.0, 0.0]], "both"),
        ("seconds", float("inf"), "A"),
        ("seconds", float("nan"), "B"),
        ("seconds", 0.0, "both"),
        ("seconds", -1.0, "both"),
        ("seconds", sys.float_info.max, "both"),
    ],
)
def test_invalid_values_are_rejected(
    driver: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: float | list[list[float]],
    role: str,
) -> None:
    samples = population()
    for record in samples:
        if role in (record["engine"], "both"):
            row = record["rows"][0]
            (row if field == "seconds" else row["items"][0])[field] = value
    with pytest.raises(RuntimeError):
        invoke(driver, monkeypatch, tmp_path, samples)


@pytest.mark.parametrize(
    "field,value", [("energy", -0.5), ("forces", [[0.5, 0.0, 0.0]])]
)
def test_existing_scientific_limits_are_not_relaxed(
    driver: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: float | list[list[float]],
) -> None:
    samples = population()
    for record in samples:
        if record["engine"] == "B":
            record["rows"][0]["items"][0][field] = value
    with pytest.raises(RuntimeError, match="changed"):
        invoke(driver, monkeypatch, tmp_path, samples)


def test_valid_complete_entry_point(
    driver: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = invoke(driver, monkeypatch, tmp_path, population())
    assert report["all_within_1_05"] and report["maximum_ratio"] == 1.0


@pytest.mark.parametrize("role", ["A", "B", "both"])
def test_force_requested_rows_require_force_publication(
    driver: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
) -> None:
    samples = population()
    for record in samples:
        row = record["rows"][0]
        row["case"] += "/energy+forces"
        if role in (record["engine"], "both"):
            row["items"][0]["forces"] = None
    with pytest.raises(RuntimeError, match="missing requested forces"):
        invoke(driver, monkeypatch, tmp_path, samples)


def test_energy_only_rows_allow_no_force_publication(
    driver: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = population()
    for record in samples:
        for row in record["rows"]:
            row["case"] += "/energy"
            row["items"][0]["forces"] = None
    report = invoke(driver, monkeypatch, tmp_path, samples)
    assert report["all_within_1_05"] and report["max_force_difference"] == 0.0


@pytest.mark.parametrize(
    "candidate",
    [{}, {"global_budget": False}, {"global_budget": "true"}, {"global_budget": 1}],
)
def test_requested_budget_requires_explicit_candidate_admission(
    driver: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    candidate: dict[str, Any],
) -> None:
    with pytest.raises(RuntimeError, match="candidate did not enable"):
        invoke(
            driver,
            monkeypatch,
            tmp_path,
            population(),
            {"A": {"global_budget": False}, "B": candidate},
            global_budget=True,
        )


def test_budget_campaign_keeps_the_ordinary_baseline(
    driver: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = {"A": {"global_budget": False}, "B": {"global_budget": True}}
    report = invoke(
        driver, monkeypatch, tmp_path, population(), metadata, global_budget=True
    )
    assert report["all_within_1_05"] and report["metadata"] == metadata


def test_serialization_rejects_nonfinite_metadata(
    driver: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        invoke(
            driver,
            monkeypatch,
            tmp_path,
            population(),
            {"A": {"invalid": float("nan")}},
        )


@pytest.mark.parametrize("campaign", ["default", "budget"])
def test_actual_retained_population_replays_without_remeasurement(
    driver: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    campaign: str,
) -> None:
    with lzma.open(
        BUNDLE / f"integrated-cuda-{campaign}-samples.json.xz", "rt"
    ) as stream:
        retained = json.load(stream)
    report = invoke(
        driver,
        monkeypatch,
        tmp_path,
        retained["samples"],
        retained["metadata"],
        global_budget=campaign == "budget",
    )
    for key in (
        "groups",
        "endpoint_count",
        "ratios",
        "maximum_ratio",
        "max_energy_difference",
        "max_force_difference",
    ):
        assert report[key] == retained[key]
    assert report["samples"] == retained["samples"]


@pytest.mark.parametrize(
    "code,error",
    [
        ("import time;time.sleep(60)", TimeoutError),
        (
            "import sys,time;sys.stdout.write('{');sys.stdout.flush();time.sleep(60)",
            TimeoutError,
        ),
        ("pass", RuntimeError),
        ("print('invalid',flush=True)", ValueError),
    ],
)
def test_worker_response_deadline_and_failure_cleanup(
    driver: SimpleNamespace, code: str, error: type[Exception]
) -> None:
    worker = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    started = time.monotonic()
    try:
        with pytest.raises(error):
            driver._response(worker, 0.2)
    finally:
        driver._stop_worker(worker)
        worker.stdin.close()
        worker.stdout.close()
    assert worker.poll() is not None
    assert time.monotonic() - started < 12


def test_persistent_child_readiness_and_sample(driver: SimpleNamespace) -> None:
    code = "import sys;print('{\"ready\":{}}',flush=True);sys.stdin.readline();print('{\"rows\":[]}',flush=True);sys.stdin.readline()"
    worker = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert driver._response(worker, 2) == {"ready": {}}
        worker.stdin.write("{}\n")
        worker.stdin.flush()
        assert driver._response(worker, 2) == {"rows": []}
    finally:
        driver._stop_worker(worker)
        worker.stdin.close()
        worker.stdout.close()
    assert worker.poll() is not None


def test_startup_timeout_cleans_up_all_started_children(
    driver: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_start = subprocess.Popen
    workers = []

    def start(*args: object, **kwargs: Any) -> subprocess.Popen[str]:
        code = "import time;time.sleep(60)"
        if not workers:
            code = "import time;print('{\"ready\":{}}',flush=True);time.sleep(60)"
        worker = real_start([sys.executable, "-c", code], **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify",
            "--baseline",
            "A",
            "--candidate",
            "B",
            "--baseline-library",
            "A.so",
            "--candidate-library",
            "B.so",
            "--backend",
            "cpu",
            "--groups",
            "8",
            "--response-timeout",
            "0.5",
            "--output",
            str(tmp_path / "unused.xz"),
        ],
    )
    try:
        with pytest.raises(TimeoutError):
            driver.main()
        assert len(workers) == 2
        assert all(worker.poll() is not None for worker in workers)
    finally:
        for worker in workers:
            driver._stop_worker(worker)
            worker.stdin.close()
            worker.stdout.close()


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX inherited-pipe ownership")
def test_exited_parent_does_not_leave_a_descendant_pipe_owner(
    driver: SimpleNamespace,
) -> None:
    import select

    code = (
        "import os,signal,time;pid=os.fork();"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "print('{\"ready\":{}}',flush=True) if pid else None;"
        "os._exit(0) if pid else time.sleep(60)"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert driver._response(worker, 2) == {"ready": {}}
        worker.wait(timeout=2)
        with pytest.raises(TimeoutError):
            driver._response(worker, 0.2)
        assert select.select([worker.stdout], [], [], 2)[0]
        assert worker.stdout.readline() == ""
    finally:
        driver._stop_worker(worker)
        worker.stdin.close()
        worker.stdout.close()


def test_broadcastable_force_shape_mismatch_is_rejected(
    driver: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = population()
    for record in samples:
        if record["engine"] == "B":
            record["rows"][0]["items"][0]["forces"] = [0.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="force shape changed"):
        invoke(driver, monkeypatch, tmp_path, samples)

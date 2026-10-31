"""Qualification harnesses reject invalid science and bound worker lifetimes."""

from __future__ import annotations

import copy
import io
import json
import lzma
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2] / "benchmarks/results"
BUNDLES = ("precision-admission-20261029", "ks-d3-resources-20261028")


def population() -> list[dict[str, Any]]:
    item = {
        "energy": -1.0,
        "forces": [[0.0, 0.0, 0.0]],
        "correction_energy": -0.01,
        "correction_gradient": [[0.0, 0.0, 0.0]],
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
        {"group": group, "engine": role, "rows": copy.deepcopy(rows)}
        for group in range(8)
        for role in ("ABBA" if group % 2 == 0 else "BAAB")
    ]


def replay(
    bundle: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    samples: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    driver = runpy.run_path(str(ROOT / bundle / "qualify.py"))
    workers = []

    def start(*args: Any, **kwargs: Any) -> SimpleNamespace:
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

    output = tmp_path / "replay.json.xz"
    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setitem(driver["main"].__globals__, "_stop_worker", stop)
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
            "--output",
            str(output),
        ],
    )
    try:
        driver["main"]()
    finally:
        assert len(workers) == 2 and all(worker.stopped for worker in workers)
    with lzma.open(output, "rt") as stream:
        return json.load(stream)


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize(
    "field,value,role",
    [
        ("energy", float("nan"), "A"),
        ("energy", float("inf"), "B"),
        ("forces", [[float("nan"), 0.0, 0.0]], "A"),
        ("forces", [[float("inf"), 0.0, 0.0]], "B"),
        ("forces", [0.0, 0.0, 0.0], "B"),
        ("seconds", float("nan"), "A"),
        ("seconds", float("inf"), "B"),
        ("seconds", 0.0, "A"),
        ("seconds", -1.0, "B"),
        ("energy", -0.5, "B"),
        ("forces", [[0.5, 0.0, 0.0]], "B"),
    ],
)
def test_invalid_observations_fail_closed(
    bundle: str,
    field: str,
    value: Any,
    role: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = population()
    for record in samples:
        if record["engine"] == role:
            row = record["rows"][0]
            (row if field == "seconds" else row["items"][0])[field] = value
    with pytest.raises(RuntimeError):
        replay(bundle, monkeypatch, tmp_path, samples)


@pytest.mark.parametrize("field", ("correction_energy", "correction_gradient"))
def test_d3_correction_values_fail_closed(
    field: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = population()
    value = float("nan") if field == "correction_energy" else [[float("nan"), 0, 0]]
    samples[0]["rows"][0]["items"][0][field] = value
    with pytest.raises(RuntimeError, match="non-finite"):
        replay(BUNDLES[1], monkeypatch, tmp_path, samples)


@pytest.mark.parametrize("bundle", BUNDLES)
def test_valid_population_and_strict_json(
    bundle: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = replay(bundle, monkeypatch, tmp_path, population())
    assert report["all_within_1_05"] and report["maximum_ratio"] == 1.0
    with pytest.raises(ValueError):
        replay(
            bundle, monkeypatch, tmp_path, population(), {"A": {"bad": float("nan")}}
        )


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize(
    "code,error",
    [
        ("import time; time.sleep(60)", TimeoutError),
        (
            "import sys,time; sys.stdout.write('{'); sys.stdout.flush(); time.sleep(60)",
            TimeoutError,
        ),
        ("pass", RuntimeError),
    ],
)
def test_response_deadline_reaps_workers(
    bundle: str, code: str, error: type[Exception]
) -> None:
    driver = runpy.run_path(str(ROOT / bundle / "qualify.py"))
    worker = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        with pytest.raises(error):
            driver["_response"](worker, 0.2)
    finally:
        driver["_stop_worker"](worker)
        worker.stdin.close()
        worker.stdout.close()
    assert worker.poll() is not None

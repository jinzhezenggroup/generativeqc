"""Negative controls for native endpoint evidence; no CUDA/scientific doubles."""

from __future__ import annotations

import runpy
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

replay = runpy.run_path(
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/native-force-ao-20261117/replay.py"
)
validate = replay["validate"]
validate_endpoints = replay["validate_endpoints"]


def _payload() -> dict[str, Any]:
    """Supply transport-only fixtures, never a numerical qualification claim."""
    samples = []
    for mode in ("dense", "auto", "minimum"):
        for stage in ("cold", "warm", "moved", "moved-warm"):
            selected = mode == "auto"
            samples.append(
                {
                    "mode": mode,
                    "stage": stage,
                    "energy": -1.0,
                    "forces": [[0.0, 0.0, 0.0]],
                    "physical_residual_rms": 1e-10,
                    "endpoint_seconds": 0.8 if selected else 1.0,
                    "work": {
                        "indexed_grid": int(selected),
                        "grid_active_point_ao": 100 if selected else 200,
                        "grid_dense_point_ao": 200,
                        "grid_discovery_ao_jet_values": int(
                            selected and stage in ("cold", "moved")
                        )
                        * 800,
                        "grid_ao_jet_values": 400 if selected else 800,
                        "grid_projection_fma_pairs": 1000 if selected else 4000,
                    },
                }
            )
    return {
        "cases": [
            {
                "id": "synthetic-transport-only",
                "reference_metrics": [],
                "capacity": {
                    "automatic": [100, 0, 100, 200, 200, 300, 512, 1, 1],
                    "bounded": [100, 0, 100, 200, 200, 300, 256, 1, 1],
                },
                "references": [
                    {"energy": -1.0, "forces": [[0.0, 0.0, 0.0]]} for _ in range(2)
                ],
                "samples": samples,
            }
        ]
    }


def test_native_evidence_replay_recomputes_gates() -> None:
    summary = validate(_payload())
    assert summary["complete_native_endpoints"] == 12
    assert summary["maximum_force_error"] == 0


@pytest.mark.parametrize(
    "field,value",
    (
        ("energy", float("nan")),
        ("physical_residual_rms", -1e-10),
        ("physical_residual_rms", float("inf")),
        ("endpoint_seconds", 0),
        ("forces", [[0.0]]),
        ("forces", [[float("nan"), 0.0, 0.0]]),
        ("forces", [[1e-8, 0.0, 0.0]]),
        ("indexed_grid", 0),
        ("grid_active_point_ao", 200),
        ("grid_discovery_ao_jet_values", 0),
        ("grid_ao_jet_values", 800),
        ("grid_projection_fma_pairs", 4000),
    ),
)
def test_native_evidence_rejects_false_acceptance(field: str, value: Any) -> None:
    payload = _payload()
    row = payload["cases"][0]["samples"][4]
    target = row["work"] if field in row["work"] else row
    target[field] = value
    with pytest.raises(AssertionError):
        validate(payload)


def test_native_evidence_requires_every_stage_and_no_rediscovery() -> None:
    payload = _payload()
    missing = deepcopy(payload)
    missing["cases"][0]["samples"].pop()
    with pytest.raises(AssertionError):
        validate(missing)
    payload["cases"][0]["samples"][5]["work"]["grid_discovery_ao_jet_values"] = 800
    with pytest.raises(AssertionError):
        validate(payload)


def test_native_evidence_rejects_borrowed_nonlocal_grid() -> None:
    """A primary-grid map cannot authorize a distinct nonlocal quadrature."""
    payload = _payload()
    case = payload["cases"][0]
    case["input"] = {"grid": [32, 10, 20], "nonlocal_grid": [12, 4, 8]}
    for row in case["samples"]:
        row["work"]["shared_nonlocal_grid"] = 0
    assert validate(payload)["complete_native_endpoints"] == 12
    case["samples"][4]["work"]["shared_nonlocal_grid"] = 1
    with pytest.raises(AssertionError):
        validate(payload)


@pytest.mark.parametrize(
    "field,value",
    (
        ("minimum", float("nan")),
        ("minimum", 0),
        ("dimension", 0),
        ("relative_threshold", 1e-9),
    ),
)
def test_native_oracle_rejects_unmatched_metric_domain(field: str, value: Any) -> None:
    payload = _payload()
    case = payload["cases"][0]
    case["input"] = {"density_fitting_relative_threshold": 1e-10}
    metric = {
        "omega": 0.3,
        "dimension": 1,
        "minimum": 0.5,
        "maximum": 1.0,
        "relative_threshold": 1e-10,
    }
    case["reference_metrics"] = [metric]
    assert validate(payload)["complete_native_endpoints"] == 12
    metric[field] = value
    with pytest.raises(AssertionError):
        validate(payload)


def _endpoint_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct process/branch transport records, not performance evidence."""
    acceptance = _payload()
    acceptance["cases"][0]["input"] = {"method": "transport-rks", "fitted": True}
    runs = []
    for pair in range(4):
        for mode in ("dense", "auto"):
            samples = []
            for stage in ("cold", "warm", "warm", "moved", "moved-warm", "moved-warm"):
                sample = deepcopy(
                    next(
                        row
                        for row in acceptance["cases"][0]["samples"]
                        if row["mode"] == mode and row["stage"] == stage
                    )
                )
                sample["seconds"] = sample.pop("endpoint_seconds")
                sample["iterations"] = sample["fock_builds"] = int(
                    stage in ("cold", "moved")
                )
                samples.append(sample)
            runs.append(
                {
                    "pair": pair,
                    "device": "transport-only",
                    "case": acceptance["cases"][0]["id"],
                    "input": {"method": "transport-rks", "fitted": True},
                    "mode": mode,
                    "host": "synthetic",
                    "job": "synthetic",
                    "visible_devices": "synthetic",
                    "process": pair * 2 + int(mode == "auto"),
                    "samples": samples,
                }
            )
    return {"runs": runs}, acceptance


def test_native_endpoint_pairs_require_fresh_matched_branches() -> None:
    payload, acceptance = _endpoint_payload()
    summary = validate_endpoints(payload, acceptance)
    assert summary["complete_native_endpoints"] == 48
    assert summary["observations_not_global_promotion"][0]["pairs"] == 4


def test_native_endpoint_pairs_reject_borrowed_nonlocal_grid() -> None:
    """Matched processes still need independent grid/map ownership."""
    payload, acceptance = _endpoint_payload()
    inputs = acceptance["cases"][0]["input"]
    inputs.update(grid=[32, 10, 20], nonlocal_grid=[12, 4, 8])
    for run in payload["runs"]:
        run["input"] = deepcopy(inputs)
        for sample in run["samples"]:
            sample["work"]["shared_nonlocal_grid"] = 0
    assert validate_endpoints(payload, acceptance)["complete_native_endpoints"] == 48
    payload["runs"][1]["samples"][0]["work"]["shared_nonlocal_grid"] = 1
    with pytest.raises(AssertionError):
        validate_endpoints(payload, acceptance)


@pytest.mark.parametrize(
    "field,value",
    (
        ("seconds", float("nan")),
        ("seconds", 0),
        ("energy", float("inf")),
        ("forces", [[0.0]]),
        ("forces", [[1e-8, 0.0, 0.0]]),
        ("physical_residual_rms", -1e-10),
        ("iterations", 3),
        ("fock_builds", 0),
        ("stage", "warm"),
        ("indexed_grid", 0),
        ("grid_ao_jet_values", 800),
    ),
)
def test_native_endpoint_pairs_reject_false_acceptance(field: str, value: Any) -> None:
    payload, acceptance = _endpoint_payload()
    row = payload["runs"][1]["samples"][0]
    target = row["work"] if field in row["work"] else row
    target[field] = value
    with pytest.raises(AssertionError):
        validate_endpoints(payload, acceptance)


@pytest.mark.parametrize("identity", ("input", "job", "visible_devices", "process"))
def test_native_endpoint_pairs_reject_unmatched_identity(identity: str) -> None:
    payload, acceptance = _endpoint_payload()
    if identity == "process":
        payload["runs"][1][identity] = payload["runs"][0][identity]
    else:
        payload["runs"][1][identity] = "foreign"
    with pytest.raises((AssertionError, TypeError)):
        validate_endpoints(payload, acceptance)


@pytest.mark.parametrize("destination", ("output", "trace"))
def test_native_endpoint_rejects_retained_destinations_before_reading_receipt(
    destination: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from benchmarks import native_force_ao_endpoint as runner

    retained = runner.ROOT / "benchmarks/results"
    output = tmp_path / "native.json"
    if destination == "output":
        output = retained / "native.json"
    else:
        output.with_suffix(".trace.jsonl").symlink_to(retained / "native.trace.jsonl")

    def unexpected_receipt(*_args: object, **_kwargs: object) -> None:
        pytest.fail("raw output must be validated before reading the receipt")

    monkeypatch.setattr(runner.lzma, "open", unexpected_receipt)
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", "unopened")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_force_ao_endpoint",
            "--receipt",
            "unused",
            "--method",
            "pbe",
            "--mode",
            "auto",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 2
    message = capsys.readouterr().err
    if destination == "output":
        assert "--output" in message and "error:" in message
    else:
        assert "trace output:" in message and ".artifacts/benchmarks/" in message
    assert not output.exists()

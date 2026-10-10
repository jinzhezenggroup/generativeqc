"""Host-only regressions for the isolated LR acceptance population."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Self

import numpy as np
import pytest

from benchmarks import _support


@pytest.fixture
def endpoint() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "benchmarks/experiments/issue1855-lr-domain/endpoints.py"
    )
    spec = importlib.util.spec_from_file_location("lr_domain_endpoints", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def campaign(
    endpoint: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> SimpleNamespace:
    """Exercise the real runner and E/F gates with a GPU-free endpoint double."""
    scientific = {
        "grid": {},
        "repeats": 5,
        "geometries_bohr": [
            [[1, [float(i), 0.0, 0.0]] for i in range(12)],
            [[1, [float(i), 0.0, 0.001 if i == 1 else 0.0]] for i in range(12)],
        ],
    }
    records = []
    for geometry in range(2):
        for repeat in range(6):
            records.append(
                {
                    "geometry": geometry,
                    "phase": ("cold" if geometry == 0 else "moved")
                    if repeat == 0
                    else ("warm" if geometry == 0 else "moved-warm"),
                    "repeat": max(0, repeat - 1),
                    "energy": -76.0,
                    "forces": [[0.0, 0.0, 0.0] for _ in range(12)],
                    "converged": True,
                    "status": 0,
                }
            )
    reference = {"status": "measured", "protocol": scientific, "records": records}
    inputs = tmp_path / "identity"
    inputs.write_bytes(b"host test identity, no native library")
    reference_path = tmp_path / "reference.json"
    output = tmp_path / "result.json"
    calls = []
    updates = []
    seeds = []
    item_changes = {}

    class Batch:
        def __init__(self) -> None:
            self.updates = True
            self.seed = 0

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def set_warm_start_updates(self, enabled: bool) -> None:
            updates.append(enabled)
            self.updates = enabled

        def execute(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            seeds.append(self.seed)
            if self.updates:
                self.seed += 1
            if kwargs["strict"] and item_changes.get("status", 0) != 0:
                raise RuntimeError("strict mode rejected failed item")
            values = {
                "energy": -76.0,
                "forces": np.zeros((12, 3)),
                "converged": True,
                "status": 0,
                "status_message": "",
                "iterations": 1,
                "fock_builds": 1,
                "energy_change": 0.0,
                "density_rms": 0.0,
                "physical_residual_rms": 0.0,
                "warm_start_used": True,
                "warm_start_fallback": False,
                "ks_diagnostic": None,
            }
            values.update(item_changes)
            return SimpleNamespace(items=[SimpleNamespace(**values)])

    class Calculator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def prepare_batch(self, *args: Any, **kwargs: Any) -> Batch:
            return Batch()

    monkeypatch.setattr(endpoint, "Calculator", Calculator)
    monkeypatch.setattr(endpoint, "load_comparison_basis", lambda *a, **k: (None, None))
    monkeypatch.setattr(endpoint, "protocol", lambda *a, **k: scientific)
    monkeypatch.setattr(
        endpoint,
        "native_build_metadata",
        lambda calc: {
            "library_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest()
        },
    )
    for key, value in {
        "SLURM_JOB_ID": "host-test-only",
        "CUDA_VISIBLE_DEVICES": "host-test-only",
        "GENERATIVEQC_LIBRARY": str(inputs),
        "LD_PRELOAD": str(inputs),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "endpoints.py",
            "--reference",
            str(reference_path),
            "--basis-file",
            str(inputs),
            "--output",
            str(output),
            "--repeats",
            "1",
        ],
    )

    def run(value: dict[str, Any] | None = None) -> dict[str, Any]:
        reference_path.write_text(json.dumps(reference if value is None else value))
        endpoint.main()
        return json.loads(output.read_text())

    return SimpleNamespace(
        reference=reference,
        run=run,
        calls=calls,
        updates=updates,
        seeds=seeds,
        item_changes=item_changes,
        output=output,
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "empty",
        "missing-geometry",
        "missing-replay",
        "wrong-geometry",
        "duplicate",
        "missing-records",
    ],
)
def test_incomplete_reference_fails_before_native_calls(
    campaign: SimpleNamespace, invalid: str
) -> None:
    reference = copy.deepcopy(campaign.reference)
    rows = reference["records"]
    if invalid == "empty":
        rows.clear()
    elif invalid == "missing-geometry":
        reference["records"] = [row for row in rows if row["geometry"] == 0]
    elif invalid == "missing-replay":
        rows.pop()
    elif invalid == "wrong-geometry":
        rows[-1]["geometry"] = 2
    elif invalid == "duplicate":
        rows[-1] = copy.deepcopy(rows[-2])
    else:
        reference.pop("records")
    with pytest.raises((TypeError, ValueError), match="oracle"):
        campaign.run(reference)
    assert not campaign.calls


def test_complete_reference_keeps_all_independent_pairings(
    campaign: SimpleNamespace,
) -> None:
    payload = campaign.run()
    assert payload["status"] == "PASS"
    assert len(campaign.calls) == len(payload["records"]) == 24
    assert len(payload["independent_pairs"]) == 144
    assert all(pair["gate"]["gate"] for pair in payload["independent_pairs"])
    assert {pair["oracle_row"] for pair in payload["independent_pairs"]} == set(
        range(12)
    )


def test_replays_freeze_post_cold_and_post_move_native_density(
    campaign: SimpleNamespace,
) -> None:
    campaign.run()
    assert campaign.updates == [True, False] * 4
    assert campaign.seeds == [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2] * 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", 7),
        ("converged", False),
        ("energy", float("nan")),
        ("forces", [[float("inf"), 0.0, 0.0]] * 12),
        ("forces", [[0.0, 0.0, 0.0]]),
        ("forces", None),
    ],
)
def test_invalid_oracle_outputs_fail_before_native_calls(
    campaign: SimpleNamespace, field: str, value: Any
) -> None:
    reference = copy.deepcopy(campaign.reference)
    reference["records"][0][field] = value
    with pytest.raises(ValueError, match="oracle"):
        campaign.run(reference)
    assert not campaign.calls


def test_mismatched_protocol_fails_before_native_calls(
    campaign: SimpleNamespace,
) -> None:
    reference = copy.deepcopy(campaign.reference)
    reference["protocol"]["geometries_bohr"][1][1][1][2] += 0.001
    with pytest.raises(ValueError, match="protocol"):
        campaign.run(reference)
    assert not campaign.calls


def test_failed_native_item_is_journaled_before_rejection(
    campaign: SimpleNamespace,
) -> None:
    campaign.item_changes.update(
        status=7,
        converged=False,
        forces=None,
        energy=float("nan"),
        density_rms=float("inf"),
        status_message="SCF did not converge",
        iterations=100,
        ks_diagnostic=SimpleNamespace(
            to_payload=lambda: {
                "history": (
                    {
                        "density_change_max": float("nan"),
                        "physical_residual_max": float("inf"),
                    },
                ),
            }
        ),
    )
    with pytest.raises(RuntimeError, match="independent E/F gate failed"):
        campaign.run()
    payload = json.loads(campaign.output.read_text())
    assert payload["status"] == "failed"
    assert len(payload["records"]) == 1
    row = payload["records"][0]
    assert row["status"] == 7 and row["iterations"] == 100
    assert row["energy"] == "nan" and row["density_rms"] == "inf"
    assert row["native_ks_diagnostic"]["history"] == [
        {"density_change_max": "nan", "physical_residual_max": "inf"}
    ]
    assert row["seconds"] >= 0
    assert not payload["independent_pairs"][0]["gate"]["gate"]


@pytest.mark.parametrize(
    "change", [{"energy": -75.0}, {"forces": np.ones((12, 3))}, {"converged": False}]
)
def test_existing_numerical_and_convergence_gates_still_reject(
    campaign: SimpleNamespace, change: dict[str, Any]
) -> None:
    campaign.item_changes.update(change)
    with pytest.raises(RuntimeError, match="independent E/F gate failed"):
        campaign.run()
    payload = json.loads(campaign.output.read_text())
    assert payload["status"] == "failed"
    assert len(payload["records"]) == 1


@pytest.mark.parametrize("route", ["direct", "symlink", "traversal"])
def test_cli_rejects_retained_output_before_input_or_native_work(
    endpoint: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    route: str,
) -> None:
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    existing = retained / "existing.json"
    existing.write_text("original reviewed evidence")
    destination = existing
    if route == "symlink":
        alias = tmp_path / "retained-alias"
        alias.symlink_to(retained, target_is_directory=True)
        destination = alias / existing.name
    elif route == "traversal":
        destination = Path(".artifacts/../benchmarks/results/existing.json")
    for variable in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "endpoints.py",
            "--reference",
            "missing-reference.json",
            "--basis-file",
            "missing-basis.json",
            "--output",
            str(destination),
        ],
    )
    with pytest.raises(SystemExit) as error:
        endpoint.main()
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "argument --output:" in message and "raw_output_path" in message
    assert existing.read_text() == "original reviewed evidence"
    assert not (tmp_path / ".artifacts").exists()


@pytest.mark.parametrize(
    "output",
    [
        ".artifacts/benchmarks/run.json",
        "scratch/run.json",
        "benchmarks/results-copy/run.json",
    ],
)
def test_cli_preserves_scratch_output_and_complete_acceptance(
    endpoint: ModuleType,
    campaign: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    output: str,
) -> None:
    monkeypatch.setattr(_support, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    arguments = sys.argv.copy()
    arguments[arguments.index("--output") + 1] = output
    Path(arguments[arguments.index("--reference") + 1]).write_text(
        json.dumps(campaign.reference)
    )
    monkeypatch.setattr(sys, "argv", arguments)
    endpoint.main()
    payload = json.loads(Path(output).read_text())
    assert payload["status"] == "PASS"
    assert len(campaign.calls) == len(payload["records"]) == 24
    assert len(payload["independent_pairs"]) == 144

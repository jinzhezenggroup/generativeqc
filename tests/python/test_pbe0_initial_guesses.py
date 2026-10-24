"""Host-only guards for numerical-first, complete-endpoint guess screening."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks import pbe0_initial_guesses as guesses

if TYPE_CHECKING:
    from pathlib import Path


def write_record(path: Path, record: dict) -> None:
    """Keep malformed samples representable for fail-closed checker tests."""
    path.write_text(json.dumps(record))


@pytest.fixture
def campaign(tmp_path: Path) -> Path:
    """Build paired synthetic evidence; no native library or GPU is involved."""
    times = {"core": 100.0, "sap": 95.0, "mod-huckel": 80.0, "minao": 90.0}
    for atoms in (48, 96):
        protocol = {"atoms": atoms, "aos": 2}
        reference_path = tmp_path / f"reference-{atoms}.json"
        oracle = {
            "schema": guesses.SCHEMA,
            "engine": "reference",
            "status": "measured",
            "protocol": protocol,
            "energy": -10.0,
            "forces": [[0.0, 0.0, 0.0]] * atoms,
            "converged": True,
        }
        write_record(reference_path, oracle)
        for guess in guesses.GUESSES:
            for repeat in range(guesses.MINIMUM_REPEATS):
                density_path = tmp_path / f"{guess}-{atoms}-{repeat}.density.npy"
                np.save(density_path, np.asarray([2.0, 0.0, 0.0, 0.0]))
                record = {
                    **oracle,
                    "engine": "native",
                    "guess": guess,
                    "repeat": repeat,
                    "status_code": 0,
                    "physical_residual_rms": 1e-12,
                    "complete_seconds": times[guess],
                    "prepare_seconds": 1.0,
                    "seed_seconds": 2.0 if guess in ("sap", "mod-huckel") else 0.0,
                    "execute_seconds": times[guess]
                    - (3.0 if guess in ("sap", "mod-huckel") else 1.0),
                    "reference_sha256": guesses.digest(reference_path),
                    "native_build": {"library_sha256": "same-library"},
                    "source_file_sha256": {"source.cpp": "same-source"},
                    "harness_sha256": "same-harness",
                    "host": "same-host",
                    "gpu": {"pci_bus_id": "same-device"},
                    "warm_start_used": guess in ("sap", "mod-huckel"),
                    "warm_start_fallback": False,
                    "initial_guess": {
                        "kind": "minao",
                        "outcome": "used",
                        "target_attempts": 1,
                        "preliminary_fock_builds": 0,
                        "work_counters_complete": True,
                    }
                    if guess == "minao"
                    else None,
                    "seed": {"selected": True, "work_counters_complete": True},
                    "final_density": {
                        "path": density_path.name,
                        "sha256": guesses.digest(density_path),
                    },
                }
                write_record(tmp_path / f"{guess}-{atoms}-{repeat}.json", record)
    return tmp_path


def change_sample(
    directory: Path, *, guess: str = "sap", repeat: int = 0, **changes: object
) -> None:
    """Mutate one non-median sample so every observation must be checked."""
    path = directory / f"{guess}-48-{repeat}.json"
    record = json.loads(path.read_text())
    record.update(changes)
    write_record(path, record)


def test_strict_five_percent_and_existing_default_baseline(campaign: Path) -> None:
    case = guesses.summarize(campaign)["cases"]["48"]
    assert not case["sap"]["comparisons"]["core"]["performance_gate"]
    assert case["minao"]["native_admission_gate"]
    assert case["mod-huckel"]["comparisons"]["minao"]["screening_gate"]
    assert case["mod-huckel"]["oracle_backed_seed"]
    assert not case["mod-huckel"]["native_admission_gate"]
    assert all(not row["automatic_policy_changed"] for row in case.values())


@pytest.mark.parametrize(
    "builds,kind,qualified",
    [(6, "huckel", True), (0, "huckel", False), (6, "minao", False)],
)
def test_native_huckel_charges_atomic_work(
    campaign: Path, builds: int, kind: str, qualified: bool
) -> None:
    for atoms in (48, 96):
        for repeat in range(guesses.MINIMUM_REPEATS):
            for guess in ("sap", "mod-huckel"):
                (campaign / f"{guess}-{atoms}-{repeat}.json").unlink()
            path = campaign / f"minao-{atoms}-{repeat}.json"
            record = json.loads(path.read_text())
            record["guess"] = "huckel"
            record["complete_seconds"] = 80.0
            record["execute_seconds"] = 79.0
            record["initial_guess"] = {
                **record["initial_guess"],
                "kind": kind,
                "preliminary_fock_builds": builds,
            }
            write_record(campaign / f"huckel-{atoms}-{repeat}.json", record)
    report = guesses.summarize(campaign, arms=("core", "minao", "huckel"))
    assert report["cases"]["96"]["huckel"]["native_admission_gate"] is qualified


def test_preparation_cost_is_not_omitted(campaign: Path) -> None:
    for repeat in range(3):
        change_sample(
            campaign,
            repeat=repeat,
            complete_seconds=110.0,
            seed_seconds=50.0,
            execute_seconds=59.0,
        )
    row = guesses.summarize(campaign)["cases"]["48"]["sap"]
    assert row["median_complete_seconds"] == 110.0
    assert not row["comparisons"]["core"]["performance_gate"]


@pytest.mark.parametrize("repeat", range(3))
def test_numerical_failure_in_any_sample_blocks_screening(
    campaign: Path, repeat: int
) -> None:
    change_sample(campaign, guess="mod-huckel", repeat=repeat, energy=-10.000001)
    row = guesses.summarize(campaign)["cases"]["48"]["mod-huckel"]
    assert row["comparisons"]["core"]["performance_gate"]
    assert not row["numerical_gate"]
    assert not row["comparisons"]["core"]["screening_gate"]


def test_density_root_difference_blocks_screening(campaign: Path) -> None:
    density_path = campaign / "mod-huckel-48-2.density.npy"
    np.save(density_path, np.asarray([2.0, 0.0, 0.0, 1e-6]))
    change_sample(
        campaign,
        guess="mod-huckel",
        repeat=2,
        final_density={
            "path": density_path.name,
            "sha256": guesses.digest(density_path),
        },
    )
    row = guesses.summarize(campaign)["cases"]["48"]["mod-huckel"]
    assert not row["numerical_gate"]


@pytest.mark.parametrize(
    "field", ["native_build", "harness_sha256", "source_file_sha256", "host", "gpu"]
)
def test_mixed_provenance_fails_closed(campaign: Path, field: str) -> None:
    change_sample(campaign, **{field: "changed"})
    with pytest.raises(ValueError, match="mixed source/library/device"):
        guesses.summarize(campaign)


def test_missing_repeat_fails_closed(campaign: Path) -> None:
    (campaign / "sap-96-2.json").unlink()
    with pytest.raises(ValueError, match="incomplete paired repeats"):
        guesses.summarize(campaign)


def test_duplicate_repeat_fails_closed(campaign: Path) -> None:
    (campaign / "duplicate.json").write_bytes((campaign / "sap-48-0.json").read_bytes())
    with pytest.raises(ValueError, match="duplicate"):
        guesses.summarize(campaign)


@pytest.mark.parametrize(
    "field", ["complete_seconds", "prepare_seconds", "seed_seconds", "execute_seconds"]
)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_invalid_phase_times_fail_closed(
    campaign: Path, field: str, invalid: float
) -> None:
    change_sample(campaign, **{field: invalid})
    with pytest.raises(ValueError, match="invalid complete endpoint time"):
        guesses.summarize(campaign)


def test_complete_time_must_include_all_phases(campaign: Path) -> None:
    change_sample(campaign, seed_seconds=20.0)
    with pytest.raises(ValueError, match="invalid complete endpoint time"):
        guesses.summarize(campaign)


@pytest.mark.parametrize(
    "changes",
    [
        {"warm_start_fallback": True},
        {"seed": {"selected": False, "work_counters_complete": True}},
        {"seed": {"selected": True, "work_counters_complete": False}},
        {"warm_start_used": False},
    ],
)
def test_reference_seed_failure_or_retry_never_qualifies(
    campaign: Path, changes: dict
) -> None:
    change_sample(campaign, guess="mod-huckel", **changes)
    row = guesses.summarize(campaign)["cases"]["48"]["mod-huckel"]
    assert not row["comparisons"]["core"]["screening_gate"]


def test_density_hash_is_verified(campaign: Path) -> None:
    np.save(campaign / "sap-48-0.density.npy", np.zeros(4))
    with pytest.raises(ValueError, match="final-density evidence changed"):
        guesses.summarize(campaign)


@pytest.mark.parametrize("residual", [float("nan"), float("inf"), -1.0, 1e-9])
def test_physical_residual_is_an_independent_gate(
    campaign: Path, residual: float
) -> None:
    change_sample(campaign, guess="mod-huckel", physical_residual_rms=residual)
    row = guesses.summarize(campaign)["cases"]["48"]["mod-huckel"]
    assert not row["numerical_gate"]
    assert not row["comparisons"]["minao"]["screening_gate"]


def test_invalid_baseline_never_qualifies_a_candidate(campaign: Path) -> None:
    change_sample(campaign, guess="minao", energy=-10.000001)
    row = guesses.summarize(campaign)["cases"]["48"]["mod-huckel"]
    assert row["numerical_gate"]
    assert row["comparisons"]["minao"]["performance_gate"]
    assert not row["comparisons"]["minao"]["screening_gate"]


@pytest.mark.parametrize(
    "changes",
    [
        {"outcome": "preparation_failed"},
        {"work_counters_complete": False},
        {"preliminary_fock_builds": 1},
        {"target_attempts": 2},
    ],
)
def test_native_minao_fallback_or_incomplete_work_never_qualifies(
    campaign: Path, changes: dict
) -> None:
    record = json.loads((campaign / "minao-48-0.json").read_text())
    diagnostic = {**record["initial_guess"], **changes}
    change_sample(campaign, guess="minao", initial_guess=diagnostic)
    row = guesses.summarize(campaign)["cases"]["48"]["minao"]
    assert not row["native_admission_gate"]


def test_reference_hash_is_verified(campaign: Path) -> None:
    change_sample(campaign, reference_sha256="changed")
    with pytest.raises(ValueError, match="reference evidence changed"):
        guesses.summarize(campaign)


def test_seed_admission_uses_overlap_metric() -> None:
    overlap = np.diag([4.0, 1.0])
    density = np.diag([0.5, 0.0])
    report = guesses.validate_seed(density, overlap, 2)
    assert report["electron_error"] == 0.0
    assert report["maximum_occupation"] == 2.0


@pytest.mark.parametrize(
    "density",
    [np.diag([2.1, -0.1]), np.diag([1.0, 0.0]), np.asarray([[2.0, 0.1], [0.0, 0.0]])],
)
def test_invalid_seed_is_rejected_without_repair(density: np.ndarray) -> None:
    before = density.copy()
    with pytest.raises(ValueError):
        guesses.validate_seed(density, np.eye(2), 2)
    np.testing.assert_array_equal(density, before)


def test_gpu_measurement_requires_scheduler_before_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="finite srun"):
        guesses.measure(None)

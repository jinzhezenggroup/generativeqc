"""Host receipt tests: no diagnostic or mismatched work can promote residency."""

from __future__ import annotations

import copy

import pytest

from benchmarks.selective_eri_rhf_acceptance import PHASES, compare


def fixture() -> tuple[list[dict], list[dict], dict]:
    """Three complete fresh processes on one device with fixed semantic work."""
    receipt = {
        "schema": "selective-eri-rhf-endpoint-v1",
        "status": "complete",
        "input_sha256": "input",
        "geometry_hash": "geometry",
        "basis_hash": "basis",
        "nbf": 232,
        "budget_bytes": 2 << 30,
        "max_iterations": 80,
        "tolerance": 1e-9,
        "precision": "fp64",
        "screening_tolerance": 0.0,
        "method": "bounded-physical-reference-RHF",
        "moved_displacement_bohr": 0.05,
        "device": {"uuid": "gpu", "name": "5090", "driver_version": "580"},
        "slurm_job_id": "123",
        "timing_scope": "complete production E+F",
        "native_sha256": "master",
        "controls": {"GENERATIVEQC_RHF_RESIDENT_VALUES": "auto"},
        "samples": [
            {
                "phase": phase,
                "seconds": 10.0,
                "energy": -150.0,
                "forces": [0.1, -0.1, 0.0],
                "iterations": 12,
                "physical_fock_builds": 13,
                "execution_plan_reused": phase != "cold",
                "response_jk_actions": 3,
                "response_derivative_passes": 3,
                "response_orbital_actions": 0,
            }
            for phase in PHASES
        ],
    }
    baseline = [copy.deepcopy(receipt) for _ in range(3)]
    candidate = copy.deepcopy(baseline)
    for entry in candidate:
        entry["native_sha256"] = "candidate"
        for sample in entry["samples"]:
            sample["seconds"] = 8.0
    oracle = {
        "status": "complete",
        "input_sha256": "input",
        "moved_displacement_bohr": 0.05,
        "samples": {
            geometry: {"energy": -150.0, "forces": [0.1, -0.1, 0.0]}
            for geometry in ["base", "moved"]
        },
    }
    return baseline, candidate, oracle


def test_complete_same_work_improvement() -> None:
    report = compare(*fixture())
    assert report["qualified"]
    assert len(report["pairs"]) == 12
    assert report["phases"]["cold"]["gain_fraction"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "key", ["device", "budget_bytes", "input_sha256", "basis_hash", "slurm_job_id"]
)
def test_mismatched_identity_refuses(key: str) -> None:
    baseline, candidate, oracle = fixture()
    candidate[1][key] = "different"
    with pytest.raises(ValueError, match="identity mismatch"):
        compare(baseline, candidate, oracle)


@pytest.mark.parametrize(
    "key", ["iterations", "physical_fock_builds", "response_derivative_passes"]
)
def test_changed_work_cannot_qualify(key: str) -> None:
    baseline, candidate, oracle = fixture()
    candidate[0]["samples"][0][key] -= 1
    report = compare(baseline, candidate, oracle)
    assert not report["qualified"]
    assert any("changed semantic work" in failure for failure in report["failures"])


@pytest.mark.parametrize("phase", PHASES)
def test_local_gain_cannot_hide_endpoint_regression(phase: str) -> None:
    baseline, candidate, oracle = fixture()
    for receipt in candidate:
        receipt["samples"][PHASES.index(phase)]["seconds"] = 10.4
    assert not compare(baseline, candidate, oracle)["qualified"]


@pytest.mark.parametrize("bad", ["energy", "forces", "nonfinite", "timing"])
def test_numeric_refusal(bad: str) -> None:
    baseline, candidate, oracle = fixture()
    sample = candidate[0]["samples"][0]
    if bad == "energy":
        sample["energy"] += 1e-5
    elif bad == "forces":
        sample["forces"][0] += 1e-4
    elif bad == "nonfinite":
        sample["forces"][0] = float("nan")
    else:
        sample["seconds"] = -1.0
    if bad in ["nonfinite", "timing"]:
        with pytest.raises(ValueError):
            compare(baseline, candidate, oracle)
    else:
        assert not compare(baseline, candidate, oracle)["qualified"]


def test_independent_oracle_rejects_common_bias() -> None:
    baseline, candidate, oracle = fixture()
    for receipt in [*baseline, *candidate]:
        receipt["samples"][0]["energy"] += 1e-5
    report = compare(baseline, candidate, oracle)
    assert not report["qualified"]
    assert any("independent E+F" in failure for failure in report["failures"])


@pytest.mark.parametrize(
    "bad", ["incomplete", "ablation", "same-provider", "two-pairs"]
)
def test_nonproduction_evidence_refuses(bad: str) -> None:
    baseline, candidate, oracle = fixture()
    if bad == "incomplete":
        candidate[0]["status"] = "failed"
    elif bad == "ablation":
        baseline[0]["controls"]["GENERATIVEQC_RHF_RESIDENT_VALUES"] = "0"
    elif bad == "same-provider":
        candidate[0]["native_sha256"] = "master"
    else:
        baseline, candidate = baseline[:2], candidate[:2]
    with pytest.raises(ValueError):
        compare(baseline, candidate, oracle)


def test_changing_cache_between_pairs_refuses() -> None:
    baseline, candidate, oracle = fixture()
    candidate[1]["controls"]["GENERATIVEQC_RHF_RESIDENT_BUCKET_MASK"] = "1"
    with pytest.raises(ValueError, match="policy changed"):
        compare(baseline, candidate, oracle)


def test_cold_preflight_is_not_four_phase_qualification() -> None:
    baseline, candidate, oracle = fixture()
    candidate[0]["samples"] = candidate[0]["samples"][:1]
    with pytest.raises(ValueError, match="four-phase"):
        compare(baseline, candidate, oracle)

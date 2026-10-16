"""Protect the descriptive gates and strict default-route evidence boundary."""

import json
import lzma
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REDUCER = ROOT / "benchmarks/results/r2scan3c-final-validation-20261015/reduce.py"
SELECTOR = "GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION"


def population(seconds: float) -> dict:
    """Use complete phases with invariant SCF work for the reduction fixture."""
    return {
        "records": [
            {
                "phase": phase,
                "complete_seconds": seconds,
                "iterations": 1,
                "fock_builds": 1,
            }
            for phase, count in {
                "cold": 1,
                "warm": 5,
                "moved": 1,
                "moved-warm": 5,
            }.items()
            for _repeat in range(count)
        ]
    }


def test_complete_cohort_gates_preserve_work_and_reject_regressions() -> None:
    compare = runpy.run_path(str(REDUCER))["compare"]
    before = population(10.0)
    for result in compare(before, population(9.0)).values():
        assert result["relative_reduction"] == pytest.approx(0.1)
        assert result["robust_descriptive_gate"] == 0.02
        assert result["passes_benefit_gate"]
        assert result["same_scf_work"]
    for result in compare(before, population(11.0)).values():
        assert not result["passes_benefit_gate"]
        assert not result["passes_nonregression_gate"]
    for result in compare(before, population(9.81)).values():
        assert not result["passes_benefit_gate"]
        assert result["passes_nonregression_gate"]
        assert result["relative_reduction"] > 0
    changed_work = population(9.0)
    changed_work["records"][0]["fock_builds"] = 2
    assert not compare(before, changed_work)["cold"]["same_scf_work"]
    assert compare(before, changed_work)["cold"]["work_confounded"]


@pytest.mark.parametrize(
    "settings",
    [
        {SELECTOR: "1"},
        {SELECTOR: "0"},
        {"GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS": "spatial-reuse"},
    ],
)
def test_explicit_candidate_schedules_cannot_qualify_default_evidence(
    tmp_path: Path, settings: dict[str, str]
) -> None:
    """A successful explicit-selector pilot cannot masquerade as default proof."""
    reduce_campaign = runpy.run_path(str(REDUCER))["reduce_campaign"]
    frozen = {"source_identity": "frozen", "consumer_source_file_sha256": {}}
    (tmp_path / "source-freeze.json").write_text(json.dumps(frozen))
    directory = tmp_path / "remote/measurements/candidate/cuda/r2scan-3c/48"
    directory.mkdir(parents=True)
    document = {
        "native_build": {"probe": {"source_identity": "frozen"}},
        "source_file_sha256": {},
        "native_schedule_settings": settings,
    }
    (directory / "native.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="declared defaults"):
        reduce_campaign(tmp_path)


@pytest.mark.parametrize("phase", ["cold", "warm", "moved", "moved-warm"])
@pytest.mark.parametrize("changed_work", [False, True])
def test_endpoint_regression_cannot_be_waived_by_changed_work(
    phase: str, changed_work: bool
) -> None:
    """Changed SCF work limits attribution but cannot waive a slower endpoint."""
    compare = runpy.run_path(str(REDUCER))["compare"]
    verify = runpy.run_path(str(REDUCER.with_name("refresh.py")))["verify_comparison"]
    control = population(10.0)
    candidate = population(9.81)
    for row in candidate["records"]:
        if row["phase"] == phase:
            row["complete_seconds"] = 11.0
            if changed_work:
                row["iterations"] = row["fock_builds"] = 2
    comparison = {
        "density_fitting": "none",
        "atoms": 96,
        "arms": {
            "control": {"engines": {"native": control}},
            "candidate": {"engines": {"native": candidate}},
        },
        "phases": compare(control, candidate),
    }
    with pytest.raises(ValueError, match="complete-endpoint nonregression"):
        verify(comparison)


def test_small_positive_benefit_and_work_attribution_remain_separate() -> None:
    """The benefit label is not an enablement gate; retain unmatched cold work."""
    compare = runpy.run_path(str(REDUCER))["compare"]
    verify = runpy.run_path(str(REDUCER.with_name("refresh.py")))["verify_comparison"]
    control = population(10.0)
    candidate = population(9.81)
    candidate["records"][0]["iterations"] = 2
    comparison = {
        "density_fitting": "none",
        "atoms": 48,
        "arms": {
            "control": {"engines": {"native": control}},
            "candidate": {"engines": {"native": candidate}},
        },
        "phases": compare(control, candidate),
    }
    verify(comparison)
    comparison["phases"]["cold"]["passes_nonregression_gate"] = False
    with pytest.raises(ValueError, match="statistics differ"):
        verify(comparison)


def test_retained_direct_regression_stays_blocked_under_the_policy_hold() -> None:
    """Narrowing production admission must not turn failed historical evidence green."""
    verify = runpy.run_path(str(REDUCER.with_name("refresh.py")))["verify_comparison"]
    observations = json.loads(
        lzma.decompress(REDUCER.with_name("observations.json.xz").read_bytes())
    )
    for comparison in observations["comparisons"]:
        if comparison["density_fitting"] == "none" and comparison["atoms"] == 96:
            moved = comparison["phases"]["moved"]
            assert moved["work_confounded"]
            assert not moved["passes_nonregression_gate"]
            assert moved["relative_reduction"] == pytest.approx(-0.15985183252516988)
            with pytest.raises(ValueError, match="none/96/moved: complete-endpoint"):
                verify(comparison)
        else:
            verify(comparison)

"""Reject misleading timing/precision receipts without GPU or oracle imports."""

from copy import deepcopy

import pytest

from benchmarks.pbe0_mixed_gradient import qualify


@pytest.fixture
def report() -> dict:
    row = {
        "energy": -1.0,
        "forces": [[0.0, 0.0, 0.0]],
        "iterations": 8,
        "fock_builds": 9,
        "scf_precision": {"mixed_stage_fock_builds": 0},
        "mixed_weighted_derivatives": True,
        "instrumented": True,
        "excluded_warmup_pair": False,
        "seconds": 100.0,
        "gradient_seconds": 50.0,
        "mixed_tasks_primitives_retries": [10, 20, 0],
        "same_state": {
            "maximum_force_difference": 0.0,
            "rms_force_difference": 0.0,
            "strict_replay_difference": 0.0,
            "mixed_replay_difference": 0.0,
            "mixed_replay_work": [10, 20, 0],
        },
    }
    samples = [
        {
            **deepcopy(row),
            "mixed_weighted_derivatives": mixed,
            "instrumented": False,
            "excluded_warmup_pair": warmup,
            "seconds": 100.0 if warmup else 2.0 if mixed else 3.0,
            "gradient_seconds": 1.0 if mixed else 2.0,
        }
        for mixed in (False, True)
        for warmup in (True, False)
    ]
    return {
        "reference": {"energy": -1.0, "forces": [[0.0, 0.0, 0.0]]},
        "protocol": {"force_gate_Eh_per_bohr": 1e-6},
        "gates": {"energy": 1e-7, "maximum_force": 1e-5, "rms_force": 3e-6},
        "qualification": row,
        "samples": samples,
    }


def test_separate_endpoint_and_gradient_ratios(report: dict) -> None:
    summary = qualify(report)
    assert summary["speedup"] == {"seconds": 1.5, "gradient_seconds": 2.0}
    assert summary["timing"]["mixed"]["seconds"]["count"] == 1


@pytest.mark.parametrize("work", ([0, 0, 0], [10, 0, 0], [10, 20, 1]))
def test_no_speed_claim_for_empty_or_retrying_mixed_route(
    report: dict, work: list
) -> None:
    report["qualification"]["mixed_tasks_primitives_retries"] = work
    with pytest.raises(ValueError, match="mixed work|retry"):
        qualify(report)


@pytest.mark.parametrize("field", ("energy", "forces", "seconds"))
def test_nonfinite_outputs_or_timing_fail_closed(report: dict, field: str) -> None:
    report["samples"][1][field] = (
        [[float("nan"), 0.0, 0.0]] if field == "forces" else float("nan")
    )
    with pytest.raises(ValueError, match="nonfinite|timing"):
        qualify(report)


def test_strict_comparator_keeps_its_tighter_oracle_gate(report: dict) -> None:
    report["samples"][1]["forces"] = [[2e-6, 0.0, 0.0]]
    with pytest.raises(ValueError, match="independent"):
        qualify(report)
    report["samples"][1]["mixed_weighted_derivatives"] = True
    report["samples"][0]["excluded_warmup_pair"] = False
    qualify(report)


def test_changed_scf_work_or_precision_is_not_a_gradient_speedup(report: dict) -> None:
    report["samples"][1]["fock_builds"] += 1
    with pytest.raises(ValueError, match="SCF work"):
        qualify(report)
    report["samples"][1]["fock_builds"] -= 1
    report["samples"][1]["scf_precision"]["mixed_stage_fock_builds"] = 1
    with pytest.raises(ValueError, match="FP64 SCF"):
        qualify(report)


def test_strict_replay_must_not_be_contaminated(report: dict) -> None:
    report["qualification"]["same_state"]["strict_replay_difference"] = 1e-6
    with pytest.raises(ValueError, match="replay"):
        qualify(report)


@pytest.mark.parametrize("missing", ("geometry", "grid", "retry", "work"))
def test_broad_fp32_claim_needs_nonempty_equal_work(report: dict, missing: str) -> None:
    for row in [report["qualification"], *report["samples"]]:
        row["precision_mask"] = 7 if row["mixed_weighted_derivatives"] else 0
        row["semantic_work"] = {"points": 42, "projection_fma_pairs": 1024}
    receipt = [10, 20, 0, 42, 42, 0]
    report["qualification"]["mixed_tasks_primitives_retries"] = receipt
    report["qualification"]["same_state"]["mixed_replay_work"] = receipt
    qualify(report)
    if missing == "work":
        report["samples"][0]["semantic_work"]["points"] += 1
    elif missing == "geometry":
        receipt[3] = 0
    elif missing == "grid":
        receipt[4] = 0
    else:
        receipt[5] = 1
    with pytest.raises(ValueError, match="work|retry"):
        qualify(report)

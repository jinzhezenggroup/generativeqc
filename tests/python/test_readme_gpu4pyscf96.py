"""A reference timing supplement cannot replace unrelated or native evidence."""

from __future__ import annotations

import copy

import pytest

from tools.render_readme_gpu4pyscf96 import PHASE_COUNTS, overlay_reference


@pytest.fixture
def observations() -> tuple[dict, dict]:
    """Separate native resource hints from the shared scientific protocol."""
    protocol = {"atoms": 96, "grid": [48, 16, 32], "basis": "retained-basis"}
    point = {
        "atoms": 96,
        "protocol": {**protocol, "density_fitting_memory_budget_bytes": 12 << 30},
        "engines": {
            "native": {"status": "completed_unchecked", "records": ["historical"]},
            "reference": {"status": "not_run"},
        },
        "independent_references": ["historical native gate owner"],
    }
    summary = {
        "density_fitting": True,
        "methods": {
            "wb97m-v": [{"atoms": 48, "historical": True}, point],
            "pbe": [{"historical": True}],
        },
    }
    update = {
        "atoms": 96,
        "protocol": {**protocol, "density_fitting_memory_budget_bytes": 0},
        "engines": {
            "reference": {
                "status": "measured",
                "outcome": {"exit_code": 0, "time_limit_seconds": None},
                "records": [
                    {"phase": phase, "complete_seconds": repeat + 1.0}
                    for phase, count in PHASE_COUNTS.items()
                    for repeat in range(count)
                ],
            }
        },
    }
    return summary, update


def test_reference_overlay_preserves_native_and_other_populations(
    observations: tuple[dict, dict],
) -> None:
    """Updating the reference does not reinterpret a historically unchecked run."""
    summary, update = observations
    original = copy.deepcopy(summary)
    result = overlay_reference(summary, update)
    assert summary == original
    assert result["methods"]["pbe"] == original["methods"]["pbe"]
    assert result["methods"]["wb97m-v"][0] == original["methods"]["wb97m-v"][0]
    point = result["methods"]["wb97m-v"][1]
    assert point["engines"]["reference"] == update["engines"]["reference"]
    point["engines"]["reference"] = original["methods"]["wb97m-v"][1]["engines"][
        "reference"
    ]
    assert point == original["methods"]["wb97m-v"][1]


@pytest.mark.parametrize(
    "failure", ["protocol", "case", "status", "phase", "exit", "timeout"]
)
def test_reference_overlay_rejects_incomplete_or_different_runs(
    observations: tuple[dict, dict], failure: str
) -> None:
    """Partial timing populations and different inputs cannot fill the missing point."""
    summary, update = observations
    reference = update["engines"]["reference"]
    if failure == "protocol":
        update["protocol"]["grid"] = [24, 8, 16]
    elif failure == "case":
        update["atoms"] = 48
    elif failure == "status":
        reference["status"] = "failed"
    elif failure == "phase":
        reference["records"].pop()
    elif failure == "exit":
        reference["outcome"]["exit_code"] = 1
    else:
        reference["outcome"]["time_limit_seconds"] = 1800
    with pytest.raises(ValueError):
        overlay_reference(summary, update)

"""Protect complete plot populations and independent-error disclosure offline."""

from __future__ import annotations

import copy
import hashlib
import json
import lzma
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks import readme_defaults_summary
from benchmarks.readme_defaults import COMMON_GATES, METHODS, SIZES

if TYPE_CHECKING:
    from types import ModuleType

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks/results"


@pytest.fixture(scope="module")
def replay() -> ModuleType:
    """Load the publication's actual scalar-gate and timing reducer without plotting."""
    return readme_defaults_summary


@pytest.fixture
def populations() -> dict:
    """Use artificial values so tests constrain behavior, not historical winners."""
    return {
        "schema": readme_defaults_summary.SCHEMA,
        "sample_columns": list(readme_defaults_summary.SAMPLE_COLUMNS),
        "common_observable_gates": copy.deepcopy(COMMON_GATES),
        "populations": [
            {
                "key": [arm, method, mode, atoms],
                "allocation": ["123", "0", "GPU-synthetic"],
                "samples": [
                    [seconds, 1, 2, 0, 0.0, 0.0, 0.0]
                    for seconds in (
                        [4.0, 1.0, 9.0, 2.0]
                        if arm == "standard"
                        else [5.0, 3.0, 3.0, 3.0]
                    )
                ],
            }
            for arm in ("standard", "gpu4pyscf")
            for method in METHODS
            for mode in ("direct", "df")
            for atoms in SIZES
        ],
    }


def test_frozen_density_warm_uses_every_sample(
    replay: ModuleType, populations: dict
) -> None:
    """The median, not the first call or minimum, determines a warm timing win."""
    result = replay.audit(populations)
    assert result["complete_endpoints"] == 512
    assert all(row["wins"] == 32 for row in result["wins"])
    native = populations["populations"][0]
    native["samples"][3][0] = 10.0
    result = replay.audit(populations)
    assert (
        next(
            row
            for row in result["wins"]
            if row["mode"] == "direct" and row["phase"] == "warm"
        )["wins"]
        == 31
    )


def test_accuracy_failure_is_not_a_qualified_win(
    replay: ModuleType, populations: dict
) -> None:
    """An inaccurate completed call remains a timing point, not scientific acceptance."""
    populations["populations"][0]["samples"][1][5] = 2e-4
    result = replay.audit(populations)
    warm = next(
        row
        for row in result["wins"]
        if row["mode"] == "direct" and row["phase"] == "warm"
    )
    assert warm["wins"] == 32
    assert warm["qualified_wins"] == 31
    assert result["qualification"]["standard"]["failed"] == 1


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "partial", "nonfinite", "different_gpu", "gates"],
)
def test_reject_incomplete_or_mismatched_evidence(
    replay: ModuleType, populations: dict, mutation: str
) -> None:
    """Storage completeness, pairing and original tolerances are acceptance prerequisites."""
    first = populations["populations"][0]
    if mutation == "missing":
        populations["populations"].pop()
    elif mutation == "duplicate":
        populations["populations"][-1] = copy.deepcopy(first)
    elif mutation == "partial":
        first["samples"].pop()
    elif mutation == "nonfinite":
        first["samples"][1][0] = float("nan")
    elif mutation == "different_gpu":
        first["allocation"][2] = "GPU-another"
    else:
        populations["common_observable_gates"]["force_max_abs_hartree_per_bohr"] *= 2
    with pytest.raises(ValueError):
        replay.audit(populations)


@pytest.mark.parametrize(
    "campaign",
    [
        "readme-rtx5090-20261111",
        "readme-rtx5090-composed-20261111",
        "readme-h100-20261111",
    ],
)
def test_retained_samples_recompute_the_published_summary(
    replay: ModuleType, campaign: str
) -> None:
    """Exercise every real timing/error sample without GPU execution or winner pinning."""
    directory = RESULTS / campaign
    manifest = json.loads((directory / "manifest.json").read_text())
    payload = (directory / "samples.json.xz").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["samples_sha256"]
    data = json.loads(lzma.decompress(payload))
    assert replay.audit(data) == manifest["summary"]
    assert all(len(row["raw_receipt_sha256"]) == 64 for row in data["populations"])
    for name, checksum in data["references"].items():
        assert (
            hashlib.sha256((RESULTS.parents[1] / name).read_bytes()).hexdigest()
            == checksum
        )


def test_superseded_attempts_are_retained_but_not_selected_for_speed(
    replay: ModuleType, populations: dict
) -> None:
    """A disclosed failed pair's partial observations cannot replace a complete cohort."""
    previous = copy.deepcopy(populations["populations"][0])
    previous["samples"] = previous["samples"][:2]
    previous["samples"][0][0] = 1000.0
    populations["superseded_populations"] = [previous]
    result = replay.audit(populations)
    assert result["complete_endpoints"] == 512
    assert result["superseded_complete_endpoints"] == 2
    assert all(row["wins"] == 32 for row in result["wins"])
    previous["samples"][0][0] = float("nan")
    with pytest.raises(ValueError):
        replay.audit(populations)

"""Offline rejection, provenance and scientific-contract gates for Becke evidence."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from benchmarks._exact_fp64_evidence import load
from benchmarks.dft_becke_qualification import qualify_becke
from benchmarks.dft_streaming_qualification import _quadrature_definition

FAMILY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/dft-adjusted-becke-20261109"
)


@pytest.fixture(scope="module")
def retained() -> tuple[dict, dict]:
    """Load authenticated originals once; tests edit only temporary copies."""
    _, files = load(FAMILY)
    return json.loads((FAMILY / "cohort.json").read_text()), files


def test_rejected_candidate_stays_rejected_and_keeps_incomplete_diagnostics() -> None:
    report = runpy.run_path(str(FAMILY / "replay.py"))["replay"]()
    assert not report["default_promotion_qualified"]
    assert not report["all_readme_leadership_established"]
    a100 = report["cards"]["a100"]
    assert a100["allocation_exit"] == 1
    assert len(a100["missing_diagnostics"]) == 4
    assert not a100["profiles"]
    assert len(a100["performance_failures"]) == 3
    assert report["cards"]["5090"]["single_pairs"]["pbe0-48"]["cold_ratio"] > 1.03
    for method in a100["methods"].values():
        assert method["cold_means"]["control"] > method["reference_cold_mean"]
        assert max(method["paired_warm_ratios"]) <= 1.03


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("forces", "Independent numerical gate"),
        ("image", "Wrong normal ELF"),
        ("work", "initial-guess"),
        ("control", "Candidate is not automatic"),
        ("missing-warm", "Incomplete endpoint"),
        ("missing-primary", "Missing endpoint"),
        ("missing-profile", "Missing endpoint"),
        ("collector", "Incomplete cold collector"),
    ),
)
def test_corrupt_science_work_image_controls_or_population_cannot_replay(
    tmp_path: Path, retained: tuple[dict, dict], change: str, message: str
) -> None:
    cohort, files = retained
    for name, text in files.items():
        if name.startswith("native-5090/"):
            path = tmp_path / name.removeprefix("native-5090/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if change.startswith("missing-") and change != "missing-warm":
        name = (
            "pbe0-2-candidate.json"
            if change == "missing-primary"
            else "pbe0-candidate-profile.json"
        )
        (tmp_path / name).unlink()
    elif change == "control":
        (tmp_path / "pbe0-2-candidate.env").write_text(
            "GENERATIVEQC_STATIONARY_BECKE_SCALED_PRODUCTS=1\n"
        )
    elif change == "collector":
        path = tmp_path / "pbe0-candidate-cold-kernels.tsv"
        path.write_text(path.read_text().replace("dropped=0", "dropped=1", 1))
    else:
        name = (
            "identity-pbe0-2-candidate.json"
            if change == "image"
            else "pbe0-2-candidate.json"
        )
        path = tmp_path / name
        record = json.loads(path.read_text())
        if change == "image":
            record["library_sha256"] = "0" * 64
        elif change == "forces":
            record["samples"][0]["forces"][0][0] += 0.1
        elif change == "work":
            record["samples"][0]["initial_guess"]["preliminary_iterations"] += 1
        else:
            record["samples"].pop()
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        qualify_becke(tmp_path, "5090", image=cohort["candidate_image"])


def test_quadrature_matching_ignores_only_implementation_point_count() -> None:
    cpu = {"level": 6, "nonlocal_grid": {"level": 5, "points": 90064}}
    gpu = {"level": 6, "nonlocal_grid": {"level": 5, "points": 90112}}
    assert _quadrature_definition(cpu) == _quadrature_definition(gpu)


@pytest.mark.parametrize("field", ("level", "radial", "pruning", "configuration"))
def test_quadrature_matching_preserves_all_nonlocal_configuration(field: str) -> None:
    baseline = {"level": 6, "nonlocal_grid": {"level": 5, "points": 90064}}
    changed = {"level": 6, "nonlocal_grid": dict(baseline["nonlocal_grid"])}
    changed["nonlocal_grid"][field] = "different"
    assert _quadrature_definition(baseline) != _quadrature_definition(changed)

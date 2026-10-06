"""Offline scientific consumers of the complete normalization comparison."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.common.evidence import block_error
from generativeqc_compiler.common.performance import assess_comparison

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_json, load_publication_record

BUNDLE = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/becke-normalize-20261006"
)


def verify_pairs(record: dict[str, Any], reference: dict[str, Any]) -> None:
    """Recompute vector gates, solver work and schedule-only A/B equivalence.

    Normalization itself is bitwise checked in the CUDA gate. Complete forces
    also include unchanged native reductions, so their stricter A/B gate uses
    1e-10 absolute tolerance rather than promising endpoint bitwise identity.
    """
    assert record["stage"] == reference["stage"] == "complete"
    assert record["protocol"] == reference["protocol"]
    assert not record["feasibility"] and not record["profile_intrusive"]
    assert len(record["samples"]) == len(record["priming"]) == 20
    assert len(record["setup"]) == 2
    atoms = record["protocol"]["atoms"]
    points = atoms * 48 * 16 * 32
    for phase, geometry in (("warm", 0), ("moved-warm", 1)):
        oracle = next(
            row
            for row in reference["records"]
            if row["geometry"] == geometry and row["phase"] in ("cold", "moved")
        )
        samples = [row for row in record["samples"] if row["phase"] == phase]
        assessment = assess_comparison(samples)
        assert assessment == record["assessments"][phase]
        comparison = assessment["workloads"]["energy-plus-force"]
        if atoms == 96:
            assert assessment["status"] == "pass"
        else:
            assert comparison["relative_improvement"] > -comparison["noise_floor"]
        counters = []
        vectors = []
        for sample in samples:
            diagnostic = sample["diagnostics"]
            assert diagnostic["iterations"] == diagnostic["fock_builds"] == 1
            assert (
                diagnostic["warm_start_used"] and not diagnostic["warm_start_fallback"]
            )
            assert len(diagnostic["native_ks_diagnostic"]["history"]) == 1
            assert (
                block_error(
                    [diagnostic["energy"]], [oracle["energy"]], atol=1e-8, rtol=0
                )
                == diagnostic["errors"]["energy"]
            )
            assert (
                block_error(diagnostic["forces"], oracle["forces"], atol=1e-7, rtol=0)
                == diagnostic["errors"]["forces"]
            )
            assert all(error["passed"] for error in diagnostic["errors"].values())
            expected = [
                1,
                int(sample["selection"] == "candidate"),
                points // 256,
                points,
            ]
            assert diagnostic["normalize_metrics"] == [expected]
            work = diagnostic["native_force_components"]["becke_owners"]["stationary"]
            assert work["profile_enabled"] is False
            assert (
                work["work_counters"]["becke_normalization_atom_entries"]
                == points * atoms
            )
            counters.append(work["work_counters"])
            vectors.append(diagnostic["forces"])
        assert all(counter == counters[0] for counter in counters)
        for vector in vectors[1:]:
            np.testing.assert_allclose(vector, vectors[0], atol=1e-10, rtol=0)


def test_publication_and_complete_pairs() -> None:
    """Selected bytes authenticate the complete raw vectors, not only medians."""
    manifest = load_json(BUNDLE / "publication.json")
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    for atoms in (48, 96):
        record = load_publication_record(
            BUNDLE, role="samples", name=f"pairs-{atoms}.json"
        )
        reference = load_publication_record(
            BUNDLE, role="input", name=f"reference-{atoms}.json"
        )
        verify_pairs(record, reference)


@pytest.mark.parametrize("field", ["forces", "iterations", "normalize_metrics"])
def test_changed_results_cannot_reuse_passing_summary(field: str) -> None:
    """Reject forged force gates, solver work or cumulative replay counters."""
    record = deepcopy(
        load_publication_record(BUNDLE, role="samples", name="pairs-96.json")
    )
    reference = load_publication_record(BUNDLE, role="input", name="reference-96.json")
    diagnostic = record["samples"][0]["diagnostics"]
    if field == "forces":
        diagnostic[field][0][0] += 1e-4
    elif field == "iterations":
        diagnostic[field] += 1
    else:
        diagnostic[field][0][2] *= 2
    with pytest.raises(AssertionError):
        verify_pairs(record, reference)

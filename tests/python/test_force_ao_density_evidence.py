"""Scientific consumers of the live, explicitly source-scoped force experiment.

Recompute all-repeat numerical gates and producer conservation, rather than
retaining a historical report solely for a checksum or sample-count assertion.
"""

from pathlib import Path

import numpy as np
import pytest

from tools.generativeqc_validation.record import load_publication_record

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/pbe0-ao-density-force-20261004"
)
FOLLOWUP_DIRECTORY = DIRECTORY.with_name("pbe0-ao-density-integration-20261004")
FOLLOWUP_SOURCES = {
    "pr-head": "b319ecfb7c5437621d1b9fc3bb13f51dc99c5411ccc38dc1a6803b3dac7aea45",
    "integrated": "1db3843899d84a5c6c0529d74dd132d981a656242d1ec213a56273bd5138a621",
}


@pytest.mark.parametrize(
    ("campaign", "atoms"),
    [
        (campaign, atoms)
        for campaign in ("master", "prototype", "pr-head", "integrated")
        for atoms in (3, 6, 12, 24, 48, 96)
    ]
    + [(campaign, 4) for campaign in ("prototype", "pr-head", "integrated")],
)
def test_frozen_force_screening_all_repeat_scientific_gates(
    campaign: str, atoms: int
) -> None:
    """No timing/iteration filter may remove an inaccurate energy or force row."""
    entries = load_publication_record(
        FOLLOWUP_DIRECTORY if campaign in FOLLOWUP_SOURCES else DIRECTORY,
        role="samples",
        name=f"{campaign}-{atoms}.json.gz",
    )
    reference = entries["reference"]
    phases = [
        ("cold", 0, 0),
        *[("warm", 0, repeat) for repeat in range(5)],
        ("moved", 1, 0),
        *[("moved-warm", 1, repeat) for repeat in range(5)],
    ]
    for entry in entries.values():
        if campaign in FOLLOWUP_SOURCES:
            assert entry["receipt"]["source_identity"] == FOLLOWUP_SOURCES[campaign]
        assert [
            (row["phase"], row["geometry"], row["repeat"])
            for row in entry["result"]["records"]
        ] == phases
    xc_objects = reference["receipt"]["reference_xc_objects"]
    assert xc_objects and all(item["on_gpu"] for item in xc_objects)
    for name, entry in entries.items():
        if name == "reference":
            continue
        native = entry["result"]
        assert native["protocol"] == reference["result"]["protocol"]
        assert entry["outcome"]["exit_code"] == 0
        for row in native["records"]:
            assert row["status"] == 0 and row["converged"]
            if campaign in FOLLOWUP_SOURCES:
                assert row["fock_builds_source"] == "ks_diagnostic.fock_builds"
                assert isinstance(row["fock_builds"], int)
                assert row["fock_builds"] >= row["iterations"]
            assert np.isfinite(row["complete_seconds"]) and row["complete_seconds"] > 0
            force = np.asarray(row["forces"])
            assert force.shape == (atoms, 3) and np.isfinite(force).all()
            for oracle in reference["result"]["records"]:
                if oracle["geometry"] != row["geometry"]:
                    continue
                assert abs(row["energy"] - oracle["energy"]) <= 1e-8
                np.testing.assert_allclose(force, oracle["forces"], atol=1e-7, rtol=0)


@pytest.mark.parametrize("campaign", ("original", "pr-head", "integrated"))
def test_observed_generic_force_work_and_independent_source_arrays(
    campaign: str,
) -> None:
    """The producer gates conserve work; both source channels remain accurate."""
    if campaign == "original":
        record = load_publication_record(
            DIRECTORY, role="samples", name="producer-sources.json.gz"
        )
        rows = record["rows"]
    else:
        record = load_publication_record(
            FOLLOWUP_DIRECTORY, role="samples", name=f"{campaign}-work.json.gz"
        )
        rows = record["producer_rows"]
        assert rows[0]["source_identity"] == FOLLOWUP_SOURCES[campaign]
    expected = np.asarray(record["accepted_snapshot"]["derivative_sources"]).reshape(-1)
    if campaign != "original":
        clean_rows = record["clean_rows"]
        assert clean_rows[0]["source_identity"] == FOLLOWUP_SOURCES[campaign]
        assert len(clean_rows) == 7
        for row in clean_rows[1:]:
            assert not row["instrumented"]
            assert np.isfinite(row["gpu_seconds"]) and row["gpu_seconds"] > 0
            np.testing.assert_allclose(row["derivatives"], expected, atol=1e-9, rtol=0)
    assert [row["label"] for row in rows if row["kind"] == "replay"] == [
        "work-0",
        "work-1",
    ]
    for row in rows:
        if row["kind"] == "replay":
            assert row["instrumented"]
            np.testing.assert_allclose(row["derivatives"], expected, atol=1e-9, rtol=0)
        elif row["kind"] == "actual-generic-force-work":
            assert row["angular_order"] >= 4
            assert row["decoded"] == sum(
                row[key]
                for key in (
                    "schwarz_rejected",
                    "density_rejected",
                    "zero_weight_rejected",
                    "admitted",
                )
            )
            if row["angular_order"] <= 6:
                assert row["explicit_gradient_evaluations"] == row["admitted"]
                assert row["dual3_gradient_evaluations"] == 0
            else:
                assert row["explicit_gradient_evaluations"] == 0
                assert row["dual3_gradient_evaluations"] <= 3 * row["admitted"]

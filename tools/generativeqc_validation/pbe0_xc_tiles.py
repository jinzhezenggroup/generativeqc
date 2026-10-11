"""Independent consumers of complete SCF-tile comparison evidence."""

from typing import Any

import numpy as np
from generativeqc_compiler.common.evidence import block_error, canonical_hash
from generativeqc_compiler.common.performance import assess_comparison


def verify_pairs(record: dict[str, Any], reference: dict[str, Any]) -> None:
    """Recompute vector gates, replay semantics, map work and force invariance.

    AO discovery fields are retained geometry receipts, not repeated warm work.
    Map visit counts describe one traversal; ``xc_evaluations`` is solve-local.
    Independently converged arm snapshots must not be called identical density
    bytes, and cold/moved setup histories are not normalized into speed claims.
    """
    _verify_pairs(record, reference, point_family=False)


def verify_point_family_pairs(
    record: dict[str, Any],
    reference: dict[str, Any],
    *,
    require_improvement: bool = False,
) -> None:
    """Audit fixed-work E+F evidence, retaining valid negative timing results.

    Promotion additionally requires ``require_improvement=True``. A valid
    numerical/work receipt alone does not authorize changing the default.
    """
    _verify_pairs(
        record,
        reference,
        point_family=True,
        require_improvement=require_improvement,
    )


def _verify_pairs(
    record: dict[str, Any],
    reference: dict[str, Any],
    *,
    point_family: bool,
    require_improvement: bool = True,
) -> None:
    assert record["schema"] == (
        "generativeqc.pbe0-xc-point-family-pairs.v1"
        if point_family
        else "generativeqc.pbe0-xc-tile-pairs.v1"
    )
    assert record["stage"] == reference["stage"] == "complete"
    assert record["protocol"] == reference["protocol"]
    assert record["scf_tiles"] == {
        "baseline": 256,
        "candidate": 256 if point_family else 512,
    }
    if point_family:
        assert record["point_specialization"] is True
        assert record["point_consumer"] == {
            "baseline": "generic",
            "candidate": "pbe-specialized",
        }
        request = record["point_batch_request"]
        assert request is None or (
            request["tiles"] >= 2 and request["device_bytes"] >= 0
        )
    assert record["force_tile_points"] == 256 and not record["feasibility"]
    assert record["density_scope"] == (
        "separate independently converged publicly frozen warm snapshots"
    )
    assert len(record["samples"]) == 20
    assert len(record["setup"]) == len(record["priming"]) == 4
    atoms, aos = record["protocol"]["atoms"], record["protocol"]["aos"]
    assert atoms in ((12, 48, 96) if point_family else (48, 96))
    points = atoms * 48 * 16 * 32
    pair_visits = points * atoms * (atoms - 1) // 2
    for group in ("setup", "priming", "samples"):
        for row in record[group]:
            geometry = row.get("geometry", int(row.get("phase") == "moved-warm"))
            oracle = next(
                item
                for item in reference["records"]
                if item["geometry"] == geometry and item["phase"] in ("cold", "moved")
            )
            diagnostic = row["diagnostics"]
            expected_errors = {
                "energy": block_error(
                    [diagnostic["energy"]], [oracle["energy"]], atol=1e-8, rtol=0
                ),
                "forces": block_error(
                    diagnostic["forces"], oracle["forces"], atol=1e-7, rtol=0
                ),
            }
            assert diagnostic["errors"] == expected_errors
            assert all(error["passed"] for error in expected_errors.values())
            tile = record["scf_tiles"][row["selection"]]
            ks = diagnostic["native_ks_diagnostic"]
            assert ks["tile_points"] == tile and ks["grid_points"] == points
            assert len(ks["history"]) == diagnostic["iterations"]
            ao = diagnostic["native_scf_ao_work"]
            assert ao["selected"] == 1 and ao["cutoff"] == 1e-16
            assert ao["tiles"] == points // tile
            assert ao["dense_point_ao_square_sum"] == points * aos * aos
            raw = diagnostic["native_force_work_raw"]
            assert raw["grid_tile_points_requested"] == 256
            work = diagnostic["native_force_components"]["becke_owners"]["stationary"][
                "work_counters"
            ]
            assert work["becke_phase_points"] == points
            assert work["phased_becke_batches"] == points // 256
            assert work["becke_pair_primal_visits"] == pair_visits
            assert work["becke_reverse_pair_visits"] == pair_visits
            assert work["becke_normalization_atom_entries"] == points * atoms
            assert work["becke_phase_launches"] == 7 * points // 256
            assert work["becke_profile_batches"] == 0

    for phase, geometry in (("warm", 0), ("moved-warm", 1)):
        samples = [row for row in record["samples"] if row["phase"] == phase]
        assessment = assess_comparison(samples)
        assert assessment == record["assessments"][phase]
        if require_improvement:
            assert assessment["status"] == "pass"
        else:
            # Full sample counts and recomputed timing summaries distinguish
            # an executed negative result from missing or invalid measurements.
            assert assessment["status"] in ("pass", "not-run")
            assert assessment.get("workloads")
        expected_inputs = canonical_hash(
            {
                "protocol": record["protocol"],
                "geometry": geometry,
                "density_scope": record["density_scope"],
            }
        )
        force_counts = []
        for row in samples:
            assert row["inputs_hash"] == expected_inputs
            diagnostic = row["diagnostics"]
            assert diagnostic["iterations"] == diagnostic["fock_builds"] == 1
            assert (
                diagnostic["warm_start_used"] and not diagnostic["warm_start_fallback"]
            )
            assert diagnostic["native_scf_ao_work"]["xc_evaluations"] == 1
            force_counts.append(
                diagnostic["native_force_components"]["becke_owners"]["stationary"][
                    "work_counters"
                ]
            )
            np.testing.assert_allclose(
                diagnostic["forces"],
                samples[0]["diagnostics"]["forces"],
                atol=1e-10,
                rtol=0,
            )
        assert all(count == force_counts[0] for count in force_counts)
        if point_family:
            # Each arm independently discovers its AO map during preparation.
            # Retain that measured duration, but compare only semantic work;
            # all other (including future) fields must still agree exactly.
            ao_work = [
                {
                    name: value
                    for name, value in row["diagnostics"]["native_scf_ao_work"].items()
                    if name != "discovery_seconds"
                }
                for row in samples
            ]
            assert all(work == ao_work[0] for work in ao_work)

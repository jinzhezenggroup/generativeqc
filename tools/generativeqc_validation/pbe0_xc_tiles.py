"""Independent consumers of complete SCF-tile comparison evidence."""

import math
from itertools import pairwise
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
    Legacy receipts remain auditable. Current promotion requires explicit point
    policy plus v3 constructor/prepare/first-E+F segment accounting; those segments
    do not establish a continuous cold-wallclock comparison.
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
    schemas = (
        (
            "generativeqc.pbe0-xc-point-family-pairs.v1",
            "generativeqc.pbe0-xc-point-family-pairs.v2",
            "generativeqc.pbe0-xc-point-family-pairs.v3",
        )
        if point_family
        else ("generativeqc.pbe0-xc-tile-pairs.v1",)
    )
    assert record["schema"] in schemas
    explicit_policy = point_family and record["schema"].endswith((".v2", ".v3"))
    current_lifecycle = point_family and record["schema"].endswith(".v3")
    if point_family and require_improvement:
        assert current_lifecycle, "current promotion requires v3 lifecycle accounting"
    assert record["stage"] == reference["stage"] == "complete"
    assert record["protocol"] == reference["protocol"]
    if current_lifecycle or "lifecycle_timing" in record:
        _verify_lifecycle_timing(record)
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
        if explicit_policy:
            assert record["point_spin_channels"] == 1
            policies = record["point_equal_spin_policy"]
            assert policies["baseline"] == policies["candidate"]
            assert policies["baseline"] in ("0", "1")
            assert record["point_family_effect"] == (
                "ordinary-physical-family-binding"
                if policies["baseline"] == "0"
                else "shared-method-bound-equal-spin-physical"
            )
            # The shared equal-spin physical consumer cannot establish a gain
            # attributable to physical family specialization.
            if require_improvement:
                assert policies["baseline"] == "0"
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
            if explicit_policy:
                assert (
                    diagnostic["requested_equal_spin_policy"]
                    == policies[row["selection"]]
                )
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


def _verify_lifecycle_timing(record: dict[str, Any]) -> None:
    """Recompute measured own-work costs without hiding interleaved gaps."""
    timing = record["lifecycle_timing"]
    assert type(timing["version"]) is int and timing["version"] == 1
    assert timing["continuous_cold_wallclock"] is False
    prepared = {row["selection"]: row for row in record["preparation"]}
    assert len(record["preparation"]) == 2 and set(prepared) == {
        "baseline",
        "candidate",
    }
    assert len(record["setup"]) == 4
    cold = {
        row["selection"]: row["execution_interval"]
        for row in record["setup"]
        if type(row["geometry"]) is int and row["geometry"] == 0
    }
    assert set(cold) == set(prepared)
    observed = set()
    for row in record["setup"]:
        key = row["selection"], row["geometry"]
        assert type(key[1]) is int
        assert key not in observed and key[0] in prepared and key[1] in (0, 1)
        observed.add(key)
        owner = prepared[key[0]]
        constructor = owner["calculator_interval"]
        preparation = owner["preparation_interval"]
        execution = row["execution_interval"]
        for interval in (constructor, preparation, execution):
            assert len(interval) == 2
            assert all(
                type(value) in (int, float) and math.isfinite(value)
                for value in interval
            )
            assert interval[1] >= interval[0]
        assert owner["calculator_seconds"] == constructor[1] - constructor[0]
        assert owner["seconds"] == preparation[1] - preparation[0]
        assert row["seconds"] == execution[1] - execution[0]
        segments = [constructor, preparation, execution] if key[1] == 0 else [execution]
        if key[1] == 1:
            assert execution[0] >= cold[key[0]][1]
        assert all(right[0] >= left[1] for left, right in pairwise(segments))
        lifecycle = row["lifecycle"]
        assert lifecycle["segments"] == segments
        for interval in lifecycle["segments"]:
            assert all(
                type(value) in (int, float) and math.isfinite(value)
                for value in interval
            )
        for value in (
            owner["calculator_seconds"],
            owner["seconds"],
            row["seconds"],
            lifecycle["own_work_seconds"],
            lifecycle["observed_span_seconds"],
            lifecycle["excluded_gap_seconds"],
        ):
            assert type(value) in (int, float) and math.isfinite(value) and value >= 0
        assert lifecycle["own_work_seconds"] == sum(
            end - begin for begin, end in segments
        )
        assert lifecycle["observed_span_seconds"] == segments[-1][1] - segments[0][0]
        assert lifecycle["excluded_gap_seconds"] == sum(
            right[0] - left[1] for left, right in pairwise(segments)
        )
        assert lifecycle["continuous"] is (key[1] == 1)

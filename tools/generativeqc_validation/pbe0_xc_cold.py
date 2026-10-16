"""Independent checks of process-cold PBE-family endpoint observations."""

import math
from typing import Any

from generativeqc_compiler.common.evidence import block_error, canonical_hash
from generativeqc_compiler.common.performance import assess_comparison


def finite_number(value: Any) -> float:
    assert type(value) in (int, float) and math.isfinite(value)
    return float(value)


def verify_observation(
    row: dict[str, Any], reference: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Check actual process/result boundaries, independent vectors and work."""
    assert row["exit_code"] == 0 and row["timeout"] is False
    assert row["synchronized"] is True
    assert (
        not row.get("reader_error")
        and not row.get("parse_error")
        and not row.get("collector_error")
    )
    assert not row.get("duplicate_endpoint") and not row.get("duplicate_worker")
    endpoint, worker = row["endpoint"], row["worker"]
    assert worker["status"] == "complete"
    assert type(worker["item_status"]) is int and worker["item_status"] == 0
    assert worker["converged"] is True
    assert worker["worker_sha256"] == manifest["worker_sha256"]
    assert worker["sdk_source_head"] == manifest["sdk_source_head"]
    assert endpoint == worker["endpoint"]
    assert endpoint["nonce"] == row["nonce"]
    assert endpoint["pid"] == row["child_pid"] != row["parent_pid"]
    if row["process_lineage"] == "direct":
        assert endpoint["pid"] == row["launcher_pid"]
        assert endpoint["ppid"] == row["parent_pid"]
    else:
        assert row["process_lineage"] == "windows-launcher"
        assert endpoint["ppid"] == row["launcher_pid"] != row["child_pid"]
    assert endpoint["selection"] == row["selection"]
    assert endpoint["geometry"] == row["geometry"]
    assert endpoint["synchronized"] is True
    start = finite_number(row["launch_started"])
    delivered = finite_number(row["endpoint_received"])
    completed = finite_number(row["process_completed"])
    assert start < delivered <= completed
    assert finite_number(row["seconds"]) == completed - start
    assert finite_number(row["delivery_seconds"]) == delivered - start
    assert finite_number(row["process_seconds"]) == completed - start
    assert worker["protocol"] == reference["protocol"]
    assert reference["stage"] == "complete"
    atoms, aos = reference["protocol"]["atoms"], reference["protocol"]["aos"]
    assert atoms in (48, 96) and type(aos) is int and aos > 0
    points = atoms * 48 * 16 * 32
    oracle = next(
        x
        for x in reference["records"]
        if x["geometry"] == row["geometry"] and x["phase"] in ("cold", "moved")
    )
    finite_number(endpoint["energy"])
    assert len(endpoint["forces"]) == atoms
    for xyz in endpoint["forces"]:
        assert len(xyz) == 3
        for value in xyz:
            finite_number(value)
    assert block_error([endpoint["energy"]], [oracle["energy"]], atol=1e-8, rtol=0)[
        "passed"
    ]
    assert block_error(endpoint["forces"], oracle["forces"], atol=1e-7, rtol=0)[
        "passed"
    ]
    assert worker["warm_start_requested"] is False
    assert worker["warm_start_used"] is False
    assert worker["warm_start_fallback"] is False
    for key in ("iterations", "fock_builds"):
        assert type(worker[key]) is int and worker[key] > 0
    ks = worker["native_ks_diagnostic"]
    assert ks["tile_points"] == 256 and ks["grid_points"] == points
    assert len(ks["history"]) == worker["iterations"]
    for index, step in enumerate(ks["history"]):
        assert type(step["iteration"]) is int and step["iteration"] == index + 1
        for key in ("nuclear", "one_electron", "hartree", "xc"):
            finite_number(step["components"][key])
        for key in ("density_change_max", "physical_residual_max"):
            assert finite_number(step[key]) >= 0
        if step["energy_change"] is None:
            assert index == 0
        else:
            assert finite_number(step["energy_change"]) >= 0
        assert step["electrons"]
        assert all(finite_number(x) >= 0 for x in step["electrons"])
        assert type(step["occupation_stabilized"]) is bool
    jk = ks["coulomb_exchange_work"]
    assert jk["complete"] is True
    for key in ("coulomb_applications", "exchange_applications"):
        assert type(jk[key]) is int and jk[key] > 0
    raw = worker["native_force_work_raw"]
    assert type(raw["grid_tile_points_requested"]) is int
    assert raw["grid_tile_points_requested"] == 256
    assert raw["grid_tile_schedule"] == "explicit"
    work = worker["native_force_components"]["becke_owners"]["stationary"][
        "work_counters"
    ]
    pairs = points * atoms * (atoms - 1) // 2
    expected = {
        "becke_phase_points": points,
        "phased_becke_batches": points // 256,
        "becke_pair_primal_visits": pairs,
        "becke_reverse_pair_visits": pairs,
        "becke_normalization_atom_entries": points * atoms,
        "becke_phase_launches": 7 * points // 256,
        "becke_profile_batches": 0,
    }
    assert all(type(work[k]) is int and work[k] == v for k, v in expected.items())
    ao = worker["native_scf_ao_work"]
    assert ao["selected"] == 1 and ao["cutoff"] == 1e-16
    assert ao["tiles"] == points // 256
    assert ao["dense_point_ao_square_sum"] == points * aos * aos
    assert type(ao["xc_evaluations"]) is int and ao["xc_evaluations"] > 0
    build = worker["native_build"]
    assert build["probe"]["source_identity"] == manifest["native_identity"]
    assert build["library_path"] == manifest["library_path"]
    assert build["library_sha256"] == manifest["library_sha256"]
    device = build["probe"]["device"]
    assert type(device["release_build"]) in (bool, int)
    assert type(device["fast_compile"]) in (bool, int)
    assert device["release_build"] == 1 and device["fast_compile"] == 0
    assert type(device["major"]) is int and type(device["minor"]) is int
    assert (device["major"], device["minor"]) == (12, 0)
    allocation = worker["allocation"]
    assert allocation["slurm_job_id"] or allocation["inspire_job_name"]
    assert allocation["cuda_visible_devices"] or allocation["nvidia_visible_devices"]


def verify_pairs(
    record: dict[str, Any],
    reference: dict[str, Any],
    *,
    allow_feasibility: bool = False,
    require_improvement: bool = False,
) -> None:
    """Require all requested pairs and equal cold semantic work; no speed waiver."""
    assert record["schema"] == "generativeqc.pbe0-xc-point-family-cold.v1"
    assert record["stage"] == "complete"
    assert record["protocol"] == reference["protocol"]
    repeats = record["repeats"]
    assert type(repeats) is int
    assert type(record["feasibility"]) is bool
    assert not record["feasibility"] or allow_feasibility
    assert repeats >= 5 or (
        allow_feasibility and record["feasibility"] is True and repeats == 1
    )
    if require_improvement:
        assert record["feasibility"] is False and repeats >= 5
    rows = record["observations"]
    assert len(rows) == 4 * repeats
    assert len({row["nonce"] for row in rows}) == len(rows)
    frozen_build = rows[0]["worker"]["native_build"]
    frozen_allocation = rows[0]["worker"]["allocation"]
    for row in rows:
        verify_observation(row, reference, record["manifest"])
        assert row["worker"]["native_build"] == frozen_build
        assert row["worker"]["allocation"] == frozen_allocation
        assert row["inputs_hash"] == canonical_hash(
            {
                "protocol": record["protocol"],
                "geometry": row["geometry"],
                "density_scope": "fresh process, owner and density",
            }
        )
    for geometry in (0, 1):
        samples = [r for r in rows if r["geometry"] == geometry]
        summary = assess_comparison(samples)
        assert summary == record["assessments"][str(geometry)]
        if require_improvement:
            assert summary["status"] == "pass"
        for pair in range(repeats):
            selected = [
                r for r in rows if r["geometry"] == geometry and r["pair"] == pair
            ]
            assert len(selected) == 2 and {r["selection"] for r in selected} == {
                "baseline",
                "candidate",
            }
            left, right = (r["worker"] for r in selected)
            assert left["native_build"] == right["native_build"]
            for key in ("iterations", "fock_builds"):
                assert left[key] == right[key]
            assert (
                left["native_ks_diagnostic"]["coulomb_exchange_work"]
                == right["native_ks_diagnostic"]["coulomb_exchange_work"]
            )
            assert (
                left["native_force_components"]["work_counts"]
                == right["native_force_components"]["work_counts"]
            )
            assert (
                left["native_force_work_raw"]["tensor_work"]
                == right["native_force_work_raw"]["tensor_work"]
            )
            assert (
                left["native_force_components"]["stationary_integral_derivative_route"]
                == right["native_force_components"][
                    "stationary_integral_derivative_route"
                ]
            )
            assert (
                left["native_force_components"]["becke_owners"]["stationary"][
                    "work_counters"
                ]
                == right["native_force_components"]["becke_owners"]["stationary"][
                    "work_counters"
                ]
            )

            def ao_work(worker: dict[str, Any]) -> dict[str, Any]:
                return {
                    k: v
                    for k, v in worker["native_scf_ao_work"].items()
                    if k != "discovery_seconds"
                }

            assert ao_work(left) == ao_work(right)

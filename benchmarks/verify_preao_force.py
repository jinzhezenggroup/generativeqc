"""Verify complete current-source A/B/C, independent gates and executed work."""

import argparse
import json
import typing
from pathlib import Path
from statistics import median

from generativeqc_compiler.common.performance import assess_comparison
from generativeqc_compiler.common.provenance import canonical_hash

from benchmarks._retention import raw_output_path
from benchmarks.readme_omol25 import check_record
from tools.generativeqc_validation.record import load_record

PRODUCERS = ("dense", "sampled-jets", "pre-ao-envelope-native-csr")
PHASES = ("cold", "warm", "moved", "moved-warm")


def load_report(path: Path) -> dict[str, typing.Any]:
    """Read raw JSON or its byte-preserving published gzip storage member."""
    return load_record(path if path.exists() else path.with_suffix(path.suffix + ".gz"))


def verify_census(
    record: dict[str, typing.Any],
    work: dict[str, typing.Any],
    producer: str,
) -> dict[str, typing.Any]:
    """Cross-check the native launch census with independent lease span totals."""
    grid = work["grid_metrics"]["ao_grid_work"]
    selection = work["resident_ao_selection"]
    assert selection["derivative_order"] == 2
    maps = selection["work"]
    points = work["grid_work_plan"]["grid_points"]
    nao = selection["full_ao_capacity"]
    tiles = work["grid_work_plan"]["tile_count"]
    if producer == "dense":
        visits, squares, nonempty = points * nao, points * nao**2, tiles
        assert maps is None
    else:
        assert maps["dense_budget_tiles"] == maps["dense_capability_tiles"] == 0
        assert maps.get("dense_allocation_tiles", 0) == 0
        assert maps["tile_count"] == tiles
        visits = maps["point_ao_visits"]
        squares = maps["point_ao_square_sum"]
        nonempty = tiles - maps["empty_tile_count"]
        assert maps["dense_point_ao_square_sum"] == points * nao**2
        if producer == "pre-ao-envelope-native-csr":
            if maps.get("occupancy_declined", False):
                assert not maps["native_csr_ready"]
                assert maps["dense_occupancy_tiles"] == tiles
            else:
                assert maps["native_csr_ready"]
            assert maps["host_ao_label_lookups"] == maps["ao_map_h2d_bytes"] == 0
            assert grid["ao_map_h2d_bytes"] == 0
            assert grid["discovery_point_ao"] == grid["discovery_ao_jet_values"] == 0
            assert maps["numeric_peak_bound_bytes"] <= maps["budget_bytes"]
            if record["phase"] in ("warm", "moved-warm"):
                assert maps["discovery_seconds"] == maps["discovery_d2h_bytes"] == 0
                assert maps["discovery_offsets_h2d_bytes"] == maps["discoveries"] == 0
    assert grid["evaluation_tiles"] == grid["feature_passes"] == tiles
    assert grid["evaluation_points"] == points
    assert grid["active_point_ao"] == grid["deriv2_point_ao"] == visits
    assert grid["ao_jet_values"] == 10 * visits
    assert grid["evaluation_passes"] == grid["deriv2_passes"] == nonempty
    assert grid["deriv0_passes"] == grid["deriv1_passes"] == grid["deriv3_passes"] == 0
    assert grid["dense_point_ao"] == points * nao
    projection_jets = grid["projection_matrices"] // nonempty
    assert projection_jets in (1, 4)
    assert grid["projection_passes"] == nonempty
    assert grid["projection_fma_pairs"] == projection_jets * squares
    assert grid["projection_output_values"] == projection_jets * visits
    assert grid["identical_spin_copy_bytes"] == 8 * projection_jets * visits
    # This matched PBE0/RKS protocol has uniform tiles and one physical density.
    # Full identity tiles alias the density; only selected tiles gather D[I,I].
    tile_points = work["grid_work_plan"]["tile_points"]
    assert points % tile_points == 0
    full_tiles = nonempty - grid["density_gather_passes"]
    assert 0 <= full_tiles <= nonempty
    assert squares % tile_points == 0
    assert (
        grid["density_gather_elements"] == squares // tile_points - full_tiles * nao**2
    )
    assert not work.get("geometry_point_preparation_enabled", False)
    return grid


def verify_public_default(report: dict[str, typing.Any]) -> None:
    """Require the candidate to leave real public policy dispatch unmodified."""
    assert report["p0c_execution_mode"] == "auto"
    for work in report["p0c_force_work"]:
        policy = work["force_active_ao_policy"]
        assert policy["decision"] == "selected"
        assert policy["producer"] == "pre-ao-envelope-native-csr"
        assert policy["actual_mode"] in ("selected", "dense-identity")


def main() -> None:
    """Validate five complete interleaved campaigns and separate phase profiles."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument(
        "--atoms", type=int, nargs="+", choices=(3, 24, 48, 96), default=(48, 96)
    )
    parser.add_argument("--without-profiles", action="store_true")
    parser.add_argument("--require-public-default", action="store_true")
    parser.add_argument("--warm-default-authorized", action="store_true")
    arguments = parser.parse_args()
    reports = {}
    raw = []
    for atoms in arguments.atoms:
        oracle = load_report(arguments.directory / f"reference-{atoms}.json")
        assert oracle["status"] == "measured"
        rows = {producer: {phase: [] for phase in PHASES} for producer in PRODUCERS}
        work_rows = {
            producer: {phase: [] for phase in PHASES} for producer in PRODUCERS
        }
        chronological = []
        source_identity = None
        campaign_scheduler = None
        numerical_gates = []
        for round_index in range(5):
            order = PRODUCERS if round_index % 2 == 0 else tuple(reversed(PRODUCERS))
            for producer in order:
                path = arguments.directory / f"{atoms}-{round_index}-{producer}.json"
                report = load_report(path)
                assert report["status"] == "measured" and report["stage"] == "complete"
                assert not report["p0c_intrusive_stage_profile"]
                assert report["p0c_domain_producer"] == producer
                if arguments.require_public_default and producer == PRODUCERS[-1]:
                    verify_public_default(report)
                assert report["protocol"] == oracle["protocol"]
                identity = (report["source_file_sha256"], report["native_build"])
                if source_identity is None:
                    source_identity = identity
                    campaign_scheduler = report["scheduler"]
                assert identity == source_identity
                assert report["scheduler"] == campaign_scheduler
                assert campaign_scheduler["SLURM_JOB_PARTITION"] == "main"
                assert len(report["records"]) == len(report["p0c_force_work"]) == 4
                for record, work in zip(
                    report["records"], report["p0c_force_work"], strict=True
                ):
                    assert record["converged"] and record["status"] == 0
                    assert record["native_ks_diagnostic"]["history"]
                    assert (
                        record["native_ks_diagnostic"]["fock_builds"]
                        <= record["fock_builds"]
                    )
                    gates = [
                        check_record(record, reference)
                        for reference in oracle["records"]
                        if reference["geometry"] == record["geometry"]
                    ]
                    assert gates and all(gate["gate"] is True for gate in gates), gates
                    numerical_gates.extend(gates)
                    grid = verify_census(record, work, producer)
                    phase = record["phase"]
                    rows[producer][phase].append(record)
                    work_rows[producer][phase].append(work)
                    chronological.append((producer, record))
                    raw.append(
                        {
                            "atoms": atoms,
                            "round": round_index,
                            "producer": producer,
                            "phase": phase,
                            "seconds": record["complete_seconds"],
                            "force_grid_work": grid,
                            "scf_ao_work": record["native_scf_ao_work"],
                            "map_work": work["resident_ao_selection"]["work"],
                            "force_endpoint_seconds": work["endpoint_seconds"],
                            "timeline": work.get("timeline"),
                            "fock_builds": record["fock_builds"],
                            "fock_history": record["native_ks_diagnostic"]["history"],
                            "scheduler": report["scheduler"],
                            "accelerator": report["environment"]["accelerator"],
                            "gates": gates,
                        }
                    )
        comparisons = {}
        for baseline, candidate in (
            ("dense", "sampled-jets"),
            ("dense", "pre-ao-envelope-native-csr"),
            ("sampled-jets", "pre-ao-envelope-native-csr"),
        ):
            samples = []
            for producer, record in chronological:
                if producer not in (baseline, candidate):
                    continue
                samples.append(
                    {
                        "selection": "baseline"
                        if producer == baseline
                        else "candidate",
                        "workload": f"{atoms}/{record['phase']}",
                        "seconds": record["complete_seconds"],
                        "synchronized": True,
                        "inputs_hash": canonical_hash(oracle["protocol"]),
                    }
                )
            comparisons[f"{baseline} -> {candidate}"] = assess_comparison(samples)
        reductions = {}
        for phase in PHASES:
            dense = work_rows["dense"][phase][0]["grid_metrics"]["ao_grid_work"]
            selected = work_rows["pre-ao-envelope-native-csr"][phase][0][
                "grid_metrics"
            ]["ao_grid_work"]
            reduced = {
                name: 1 - selected[name] / dense[name]
                for name in ("active_point_ao", "ao_jet_values", "projection_fma_pairs")
            }
            if work_rows["pre-ao-envelope-native-csr"][phase][0][
                "resident_ao_selection"
            ]["work"].get("occupancy_declined", False):
                assert atoms < 48 and all(value == 0 for value in reduced.values()), (
                    reduced
                )
            else:
                assert all(value > 0.02 for value in reduced.values()), reduced
            reductions[phase] = reduced
        profiles = {}
        if not arguments.without_profiles:
            for producer in PRODUCERS:
                profile = load_report(
                    arguments.directory / f"profile-{atoms}-{producer}.json"
                )
                assert (
                    profile["status"] == "measured" and profile["stage"] == "complete"
                )
                assert profile["p0c_intrusive_stage_profile"] is True
                assert profile["p0c_domain_producer"] == producer
                if arguments.require_public_default and producer == PRODUCERS[-1]:
                    verify_public_default(profile)
                assert profile["protocol"] == oracle["protocol"]
                assert (
                    profile["source_file_sha256"],
                    profile["native_build"],
                ) == source_identity
                assert profile["scheduler"] == campaign_scheduler
                assert len(profile["records"]) == len(profile["p0c_force_work"]) == 4
                profiles[producer] = []
                for record, work in zip(
                    profile["records"], profile["p0c_force_work"], strict=True
                ):
                    gates = [
                        check_record(record, reference)
                        for reference in oracle["records"]
                        if reference["geometry"] == record["geometry"]
                    ]
                    assert gates and all(gate["gate"] is True for gate in gates), gates
                    verify_census(record, work, producer)
                    assert work["device_profile_enabled"] is True
                    profiles[producer].append(
                        {
                            "phase": record["phase"],
                            "force_work": work,
                            "gates": gates,
                            "scheduler": profile["scheduler"],
                        }
                    )
        reports[str(atoms)] = {
            "scheduler": campaign_scheduler,
            "source_identity": {
                "source_file_sha256": source_identity[0],
                "native_build": source_identity[1],
            },
            "comparisons": comparisons,
            "work_reductions": reductions,
            "numerical_gates": {
                "passed": True,
                "max_energy_error": max(
                    gate["energy_error"] for gate in numerical_gates
                ),
                "max_force_error": max(gate["force_error"] for gate in numerical_gates),
            },
            "separate_intrusive_profiles": profiles,
            "medians": {
                producer: {
                    phase: median(
                        row["complete_seconds"] for row in rows[producer][phase]
                    )
                    for phase in PHASES
                }
                for producer in PRODUCERS
            },
            "fock_histories": {
                producer: {
                    phase: [row["fock_builds"] for row in rows[producer][phase]]
                    for phase in PHASES
                }
                for producer in PRODUCERS
            },
        }
        candidate = comparisons["dense -> pre-ao-envelope-native-csr"]["workloads"]
        reports[str(atoms)]["warm_qualification"] = {
            "objective": "user-authorized stable warm/moved-warm improvement; all phases retained",
            "warm_passed": all(
                candidate[f"{atoms}/{phase}"]["significant"]
                for phase in ("warm", "moved-warm")
            ),
            "cold_moved_no_material_regression": all(
                candidate[f"{atoms}/{phase}"]["relative_improvement"]
                > -candidate[f"{atoms}/{phase}"]["noise_floor"]
                for phase in ("cold", "moved")
            ),
        }
        # A dense occupancy fallback is useful only if its bookkeeping does
        # not materially regress any phase. Sparse domains must improve both
        # warm phases; neither branch hides the cold/moved measurements.
        dense_fallback = all(
            work_rows[PRODUCERS[-1]][phase][0]["resident_ao_selection"]["work"].get(
                "occupancy_declined", False
            )
            for phase in PHASES
        )
        warm = reports[str(atoms)]["warm_qualification"]
        reports[str(atoms)]["guarded_default_qualified"] = (
            all(
                candidate[f"{atoms}/{phase}"]["relative_improvement"]
                > -candidate[f"{atoms}/{phase}"]["noise_floor"]
                for phase in PHASES
            )
            if dense_fallback
            else warm["warm_passed"] and warm["cold_moved_no_material_regression"]
        )
    result = {
        "schema": "generativeqc.preao-abc-qualification.v1",
        "atoms": list(arguments.atoms),
        "profiles_complete": not arguments.without_profiles,
        "public_default_verified": arguments.require_public_default,
        "default_promotion_authorized": arguments.warm_default_authorized,
        "default_promotion_eligible": (
            arguments.warm_default_authorized
            and arguments.require_public_default
            and not arguments.without_profiles
            and all(report["guarded_default_qualified"] for report in reports.values())
        ),
        "scf_census_scope": (
            "point_ao_visits is one-traversal inventory; xc_evaluations is solve-local; "
            "all actual Fock histories are retained separately"
        ),
        "reports": reports,
        "raw": raw,
    }
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                atoms: {
                    key: value
                    for key, value in report.items()
                    if key != "separate_intrusive_profiles"
                }
                for atoms, report in reports.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

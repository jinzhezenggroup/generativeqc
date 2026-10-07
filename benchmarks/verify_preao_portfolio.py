"""Qualify public pre-AO dispatch against the already enabled sampled default."""

from __future__ import annotations

import argparse
import json
import typing
from pathlib import Path
from statistics import median

from generativeqc_compiler.common.performance import assess_comparison
from generativeqc_compiler.common.provenance import canonical_hash

from benchmarks._retention import raw_output_path
from benchmarks.readme_omol25 import check_record
from benchmarks.verify_preao_force import PHASES, load_report, verify_census

MODES = ("current-default", "auto")
NATIVE = "pre-ao-envelope-native-csr"
WORK_COUNTERS = ("active_point_ao", "ao_jet_values", "projection_fma_pairs")


def accelerator_identity(accelerator: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Compare device identity, not post-run power, temperature or clock telemetry."""
    return {key: value for key, value in accelerator.items() if key != "nvidia_smi"}


def verify_dispatch(report: dict[str, typing.Any], mode: str) -> str:
    """Reject producer overrides and retain the incumbent sampled policy verbatim."""
    assert mode in MODES and report["p0c_execution_mode"] == mode
    producer = report["p0c_domain_producer"]
    assert producer in (
        ("dense", "sampled-jets") if mode == MODES[0] else (NATIVE, "sampled-jets")
    )
    for work in report["p0c_force_work"]:
        policy = work["force_active_ao_policy"]
        assert (policy["producer"] or "dense") == producer
        if producer == "dense":
            assert policy["decision"] == policy["actual_mode"] == "dense"
            assert policy["profile_id"] is None
        else:
            assert policy["decision"] == "selected"
            assert policy["actual_mode"] in ("selected", "dense-identity")
            assert policy["profile_id"] == (
                "cuda-resident-preao-native-csr-v1"
                if producer == NATIVE
                else "ordinary-direct-active-ao-cost-v3"
            )
            assert policy["cutoff"] == 1e-16
            assert policy["cache_bytes"] == (
                64 << 20 if producer == NATIVE else 16 << 20
            )
    return producer


def warm_eligibility(
    comparison: dict[str, typing.Any], atoms: int, *, sparse: bool
) -> bool:
    """Sparse promotion needs both warm wins; identity paths need no regression."""
    workloads = comparison["workloads"]
    no_regression = all(
        workloads[f"{atoms}/{phase}"]["relative_improvement"]
        > -workloads[f"{atoms}/{phase}"]["noise_floor"]
        for phase in PHASES
    )
    return no_regression and (
        not sparse
        or all(
            workloads[f"{atoms}/{phase}"]["significant"]
            for phase in ("warm", "moved-warm")
        )
    )


def qualify(directory: Path, atoms_list: list[int]) -> dict[str, typing.Any]:
    """Verify five AB/BA campaigns, four E+F phases and separate intrusive runs."""
    reports, raw = {}, []
    for atoms in atoms_list:
        oracle = load_report(directory / f"reference-{atoms}.json")
        assert oracle["status"] == "measured"
        rows = {mode: {phase: [] for phase in PHASES} for mode in MODES}
        work_rows = {mode: {phase: [] for phase in PHASES} for mode in MODES}
        samples, gates, profiles = [], [], {}
        identity = scheduler = accelerator = None
        producers = {}

        def validate(
            report: dict[str, typing.Any],
            mode: str,
            *,
            intrusive: bool,
            protocol: dict[str, typing.Any] = oracle["protocol"],
            campaign_producers: dict[str, str] = producers,
        ) -> str:
            nonlocal identity, scheduler, accelerator
            assert report["status"] == "measured" and report["stage"] == "complete"
            assert report["p0c_intrusive_stage_profile"] is intrusive
            assert report["protocol"] == protocol
            current = (report["source_file_sha256"], report["native_build"])
            if identity is None:
                identity, scheduler, accelerator = (
                    current,
                    report["scheduler"],
                    accelerator_identity(report["environment"]["accelerator"]),
                )
            assert current == identity and report["scheduler"] == scheduler
            assert (
                accelerator_identity(report["environment"]["accelerator"])
                == accelerator
            )
            assert scheduler["SLURM_JOB_PARTITION"] == "main"
            assert scheduler["SLURM_JOB_ID"] and scheduler["SLURM_GPUS_ON_NODE"] == "1"
            assert len(report["records"]) == len(report["p0c_force_work"]) == 4
            assert tuple(record["phase"] for record in report["records"]) == PHASES
            producer = verify_dispatch(report, mode)
            if mode in campaign_producers:
                assert producer == campaign_producers[mode]
            campaign_producers[mode] = producer
            return producer

        def numerical(
            record: dict[str, typing.Any],
            references: list[dict[str, typing.Any]] = oracle["records"],
            campaign_gates: list[dict[str, typing.Any]] = gates,
        ) -> list[dict[str, typing.Any]]:
            assert record["converged"] and record["status"] == 0
            assert record["native_ks_diagnostic"]["history"]
            assert (
                record["native_ks_diagnostic"]["fock_builds"] <= record["fock_builds"]
            )
            checked = [
                check_record(record, reference)
                for reference in references
                if reference["geometry"] == record["geometry"]
            ]
            assert checked and all(gate["gate"] is True for gate in checked), checked
            campaign_gates.extend(checked)
            return checked

        for round_index in range(5):
            for mode in MODES if round_index % 2 == 0 else tuple(reversed(MODES)):
                report = load_report(directory / f"{atoms}-{round_index}-{mode}.json")
                producer = validate(report, mode, intrusive=False)
                for record, work in zip(
                    report["records"], report["p0c_force_work"], strict=True
                ):
                    checked = numerical(record)
                    grid = verify_census(record, work, producer)
                    assert not work["device_profile_enabled"]
                    phase = record["phase"]
                    rows[mode][phase].append(record)
                    work_rows[mode][phase].append(work)
                    samples.append(
                        {
                            "selection": "baseline"
                            if mode == MODES[0]
                            else "candidate",
                            "workload": f"{atoms}/{phase}",
                            "seconds": record["complete_seconds"],
                            "synchronized": True,
                            "inputs_hash": canonical_hash(oracle["protocol"]),
                        }
                    )
                    raw.append(
                        {
                            "atoms": atoms,
                            "round": round_index,
                            "mode": mode,
                            "producer": producer,
                            "phase": phase,
                            "seconds": record["complete_seconds"],
                            "force_grid_work": grid,
                            "scf_ao_work": record["native_scf_ao_work"],
                            "map_work": work["resident_ao_selection"]["work"],
                            "force_endpoint_seconds": work["endpoint_seconds"],
                            "fock_builds": record["fock_builds"],
                            "fock_history": record["native_ks_diagnostic"]["history"],
                            "gates": checked,
                        }
                    )

        reductions = {}
        sparse = False
        for phase in PHASES:
            phase_reductions = []
            for baseline, candidate in zip(
                work_rows[MODES[0]][phase], work_rows[MODES[1]][phase], strict=True
            ):
                before = baseline["grid_metrics"]["ao_grid_work"]
                after = candidate["grid_metrics"]["ao_grid_work"]
                reduced = {
                    name: 1 - after[name] / before[name] for name in WORK_COUNTERS
                }
                maps = candidate["resident_ao_selection"]["work"]
                if producers[MODES[1]] == "sampled-jets":
                    assert (
                        baseline["force_active_ao_policy"]
                        == candidate["force_active_ao_policy"]
                    )
                    assert all(value == 0 for value in reduced.values())
                elif maps.get("occupancy_declined", False):
                    assert all(value == 0 for value in reduced.values())
                else:
                    assert producers[MODES[0]] == "dense"
                    assert all(value > 0.02 for value in reduced.values()), reduced
                    sparse = True
                phase_reductions.append(reduced)
            reductions[phase] = phase_reductions

        for mode in MODES:
            profile = load_report(directory / f"profile-{atoms}-{mode}.json")
            producer = validate(profile, mode, intrusive=True)
            profiles[mode] = []
            for record, work in zip(
                profile["records"], profile["p0c_force_work"], strict=True
            ):
                checked = numerical(record)
                verify_census(record, work, producer)
                assert work["device_profile_enabled"] is True
                profiles[mode].append(
                    {"phase": record["phase"], "force_work": work, "gates": checked}
                )

        comparison = assess_comparison(samples)
        assert "workloads" in comparison, comparison
        reports[str(atoms)] = {
            "scheduler": scheduler,
            "accelerator": accelerator,
            "source_identity": {
                "source_file_sha256": identity[0],
                "native_build": identity[1],
            },
            "producers": producers,
            "comparison": comparison,
            "work_reductions": reductions,
            "sparse_extension": sparse,
            "guarded_default_qualified": warm_eligibility(
                comparison, atoms, sparse=sparse
            ),
            "independent_numerical_gates": {
                "count": len(gates),
                "max_energy_error": max(gate["energy_error"] for gate in gates),
                "max_force_error": max(gate["force_error"] for gate in gates),
            },
            "separate_intrusive_profiles": profiles,
            "medians": {
                mode: {
                    phase: median(
                        record["complete_seconds"] for record in rows[mode][phase]
                    )
                    for phase in PHASES
                }
                for mode in MODES
            },
            "fock_histories": {
                mode: {
                    phase: [
                        record["native_ks_diagnostic"]["history"]
                        for record in rows[mode][phase]
                    ]
                    for phase in PHASES
                }
                for mode in MODES
            },
        }
    return {
        "schema": "generativeqc.preao-portfolio-qualification.v1",
        "atoms": atoms_list,
        "objective": "user-authorized stable warm/moved-warm improvement with incumbent sampled preservation",
        "scf_census_scope": "point_ao_visits is one-traversal inventory; xc_evaluations is solve-local; actual Fock histories are retained",
        "default_promotion_eligible": all(
            report["guarded_default_qualified"] for report in reports.values()
        )
        and any(report["sparse_extension"] for report in reports.values()),
        "reports": reports,
        "raw": raw,
    }


def main() -> None:
    """Keep the reviewed results tree publisher-owned; write raw qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--atoms", type=int, nargs="+", default=(3, 24, 48))
    parser.add_argument("--output", type=raw_output_path, required=True)
    arguments = parser.parse_args()
    report = qualify(arguments.directory, list(arguments.atoms))
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "default_promotion_eligible": report["default_promotion_eligible"],
                "reports": {
                    atoms: {
                        key: value
                        for key, value in row.items()
                        if key
                        in (
                            "producers",
                            "comparison",
                            "guarded_default_qualified",
                            "medians",
                            "independent_numerical_gates",
                        )
                    }
                    for atoms, row in report["reports"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

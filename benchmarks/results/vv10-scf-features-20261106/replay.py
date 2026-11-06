"""Replay source-matched endpoint, semantic-work and independent E/F gates."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path
from typing import Any

REFERENCE_GATES = (1e-5, 1e-4, 3e-5)
PAIRED_GATES = (1e-9, 1e-7, 3e-8)


def errors(sample: dict[str, Any], target: dict[str, Any]) -> tuple[float, ...]:
    """Compare all exported observables, not a scalar force summary."""
    differences = [
        actual - expected
        for row, other in zip(sample["forces"], target["forces"], strict=True)
        for actual, expected in zip(row, other, strict=True)
    ]
    assert len(differences) == 288
    return (
        abs(sample["energy"] - target["energy"]) / 96,
        max(map(abs, differences)),
        math.sqrt(math.fsum(value * value for value in differences) / 288),
    )


def replay() -> dict[str, Any]:
    """Historical reference latency is deliberately excluded from all rankings."""
    directory = Path(__file__).resolve().parent
    evidence = json.loads(
        lzma.decompress((directory / "evidence.json.xz").read_bytes())
    )
    assert evidence["schema"] == "generativeqc.vv10-scf-features.v1"
    reference_path = directory.parents[2] / evidence["reference"]["path"]
    reference_bytes = reference_path.read_bytes()
    assert (
        hashlib.sha256(reference_bytes).hexdigest() == evidence["reference"]["sha256"]
    )
    reference = json.loads(lzma.decompress(reference_bytes))
    reference_record = reference["records"]["A100/reference"]
    assert reference_record["status"] == "complete"
    assert reference_record["arm"] == "gpu4pyscf"
    targets = reference_record["samples"]
    assert len(targets) == 2, "expected cold and warm independent reference samples"
    for index, target in enumerate(targets):
        assert target["converged"]
        assert target["phase"] == ("cold" if index == 0 else "warm")
        assert math.isfinite(target["energy"])
        assert len(target["forces"]) == 96
        assert all(
            len(row) == 3 and all(math.isfinite(value) for value in row)
            for row in target["forces"]
        )
    summary: dict[str, Any] = {
        "decision": evidence["decision"],
        "source_identity": evidence["image"]["source_identity"],
        "fresh_gpu4pyscf_ranking": False,
        "independent_pairs": 0,
        "worst_reference_errors": [0.0, 0.0, 0.0],
        "worst_paired_errors": [0.0, 0.0, 0.0],
        "devices": {},
    }
    for device in ("A100", "RTX5090"):
        records = {
            arm: evidence["endpoints"][f"{device}/{arm}"]
            for arm in ("baseline", "default")
        }
        baseline = records["baseline"]["record"]
        for arm, population in records.items():
            record, identity = population["record"], population["identity"]
            assert population["exit_code"] == 0 and record["status"] == "complete"
            assert record["geometry_bohr"] == reference["cohort"]["geometry_bohr"]
            assert record["method"] == "wb97m-v" and record["mode"] == "direct"
            assert record["atoms"] == 96 and record["orbital_basis"] == "def2-SVP"
            assert record["basis_representation"] == "spherical"
            assert record["charge"] == reference["cohort"]["charge"] == 0
            assert record["spin"] == reference["cohort"]["spin"] == 0
            assert record["source_revision"] == evidence["image"]["source_identity"]
            for field in ("source_identity", "library_sha256"):
                assert (
                    record["native_build"][field]
                    == identity[field]
                    == evidence["image"][field]
                )
            assert record["scheduler"] == identity["scheduler"]
            assert record["scheduler"]["CUDA_VISIBLE_DEVICES"]
            assert ("A100" if device == "A100" else "5090") in record["accelerator"][
                "name"
            ]
            assert (
                record["protocol"]["cold"]
                == "constructor+prepare+SCF+forces+host-export"
            )
            assert not record["protocol"]["grid_overrides"]
            assert record["scf_feature_experiment"]["clean"]
            assert record["scf_feature_experiment"]["schedule"] == (
                "parallel-packet" if arm == "baseline" else None
            )
            assert len(record["samples"]) == 4
            for index, sample in enumerate(record["samples"]):
                assert sample["converged"] and not sample["warm_start_fallback"]
                assert sample["phase"] == ("cold" if index == 0 else "warm")
                assert sample["warm_start_used"] == (index != 0)
                assert sample["settings"] == baseline["samples"][index]["settings"]
                assert sample["settings"]["energy_tolerance"] == 1e-9
                assert sample["settings"]["density_tolerance"] == 1e-5
                assert sample["settings"]["screening_tolerance"] == 1e-12
                assert sample["settings"]["force_grid_response"]
                assert sample["precision"]["effective_bits"] == 64
                assert sample["precision"]["mixed_stage_fock_builds"] == 0
                assert sample["precision"]["final_residual_audits"] == 1
                assert (
                    sum(
                        solve["force_evaluations"]
                        for solve in sample["precision_validation"]["solves"]
                    )
                    == 1
                )
                assert sample["iterations"] == sample["fock_builds"]
                diagnostic = sample["ks_diagnostic"]
                work = diagnostic["coulomb_exchange_work"]
                assert work["complete"]
                assert work["coulomb_applications"] == sample["fock_builds"]
                assert work["exchange_applications"] == 2 * sample["fock_builds"]
                assert (
                    sample["fock_builds"] == baseline["samples"][index]["fock_builds"]
                )
                assert (
                    diagnostic["nonlocal_work"]
                    == baseline["samples"][index]["ks_diagnostic"]["nonlocal_work"]
                )
                if index:
                    assert sample["fock_builds"] == 1
                    assert diagnostic["nonlocal_work"]["pair_sweeps"] == 0
                for target in targets:
                    observed = errors(sample, target)
                    for position, (value, gate) in enumerate(
                        zip(observed, REFERENCE_GATES, strict=True)
                    ):
                        assert math.isfinite(value) and value < gate, (
                            device,
                            arm,
                            observed,
                        )
                        summary["worst_reference_errors"][position] = max(
                            summary["worst_reference_errors"][position], value
                        )
                    summary["independent_pairs"] += 1
                if arm == "default":
                    observed = errors(sample, baseline["samples"][index])
                    for position, (value, gate) in enumerate(
                        zip(observed, PAIRED_GATES, strict=True)
                    ):
                        assert math.isfinite(value) and value < gate, (device, observed)
                        summary["worst_paired_errors"][position] = max(
                            summary["worst_paired_errors"][position], value
                        )
        samples = {arm: value["record"]["samples"] for arm, value in records.items()}
        cold = {arm: rows[0]["seconds"] for arm, rows in samples.items()}
        warm = {
            arm: statistics.median(row["seconds"] for row in rows[1:])
            for arm, rows in samples.items()
        }
        ratios = [
            candidate["seconds"] / incumbent["seconds"]
            for incumbent, candidate in zip(
                samples["baseline"][1:], samples["default"][1:], strict=True
            )
        ]
        if evidence["decision"] == "promote":
            assert cold["default"] < cold["baseline"]
            assert warm["default"] / warm["baseline"] < 1.02
            assert max(ratios) < 1.05
        summary["devices"][device] = {
            "cold_seconds": cold,
            "warm_median_seconds": warm,
            "cold_change_percent": 100 * (cold["default"] / cold["baseline"] - 1),
            "warm_change_percent": 100 * (warm["default"] / warm["baseline"] - 1),
            "warm_per_call_ratios": ratios,
            "cold_focks": samples["default"][0]["fock_builds"],
            "cold_nonlocal_work": samples["default"][0]["ks_diagnostic"][
                "nonlocal_work"
            ],
        }
    assert summary["independent_pairs"] == 32
    integration = evidence["latest_integration"]
    assert integration is not None, "latest integration qualification is required"
    assert set(integration["endpoints"]) == {"A100", "RTX5090"}
    summary["latest_integration"] = {
        "base_commit": integration["base_commit"],
        "image": integration["image"],
        "independent_pairs": 0,
        "worst_cross_image_errors": [0.0, 0.0, 0.0],
        "devices": {},
    }
    for device, population in integration["endpoints"].items():
        record, identity = population["record"], population["identity"]
        assert record["status"] == "complete" and population["exit_code"] == 0
        assert record["geometry_bohr"] == reference["cohort"]["geometry_bohr"]
        assert record["method"] == "wb97m-v" and record["mode"] == "direct"
        assert (
            record["orbital_basis"] == "def2-SVP"
            and record["basis_representation"] == "spherical"
        )
        assert record["atoms"] == 96 and record["charge"] == record["spin"] == 0
        assert record["scheduler"] == identity["scheduler"]
        assert record["scheduler"]["CUDA_VISIBLE_DEVICES"]
        for field in ("source_identity", "library_sha256"):
            assert (
                record["native_build"][field]
                == identity[field]
                == integration["image"][field]
            )
        assert record["source_revision"] == integration["image"]["source_identity"]
        assert (
            record["protocol"]["cold"] == "constructor+prepare+SCF+forces+host-export"
        )
        assert not record["protocol"]["grid_overrides"]
        assert record["scf_feature_experiment"]["schedule"] is None
        assert len(record["samples"]) == 2
        baseline = evidence["endpoints"][f"{device}/default"]["record"]
        for index, sample in enumerate(record["samples"]):
            assert sample["converged"] and not sample["warm_start_fallback"]
            assert sample["settings"] == baseline["samples"][index]["settings"]
            assert sample["phase"] == ("cold" if index == 0 else "warm")
            assert sample["warm_start_used"] == (index != 0)
            assert sample["precision"]["effective_bits"] == 64
            assert sample["precision"]["mixed_stage_fock_builds"] == 0
            assert sample["precision"]["final_residual_audits"] == 1
            assert (
                sum(
                    solve["force_evaluations"]
                    for solve in sample["precision_validation"]["solves"]
                )
                == 1
            )
            assert (
                sample["iterations"]
                == sample["fock_builds"]
                == (16 if index == 0 else 1)
            )
            work = sample["ks_diagnostic"]["coulomb_exchange_work"]
            assert work["complete"]
            assert work["coulomb_applications"] == sample["fock_builds"]
            assert work["exchange_applications"] == 2 * sample["fock_builds"]
            assert (
                sample["ks_diagnostic"]["nonlocal_work"]
                == baseline["samples"][index]["ks_diagnostic"]["nonlocal_work"]
            )
            observed = errors(sample, baseline["samples"][index])
            for position, (value, gate) in enumerate(
                zip(observed, PAIRED_GATES, strict=True)
            ):
                assert math.isfinite(value) and value < gate, (device, observed)
                summary["latest_integration"]["worst_cross_image_errors"][position] = (
                    max(
                        summary["latest_integration"]["worst_cross_image_errors"][
                            position
                        ],
                        value,
                    )
                )
            for target in targets:
                observed = errors(sample, target)
                assert all(
                    math.isfinite(value) and value < gate
                    for value, gate in zip(observed, REFERENCE_GATES, strict=True)
                ), (device, observed)
                summary["latest_integration"]["independent_pairs"] += 1
        summary["latest_integration"]["devices"][device] = {
            "cold_seconds": record["samples"][0]["seconds"],
            "warm_seconds": record["samples"][1]["seconds"],
            "focks": [sample["fock_builds"] for sample in record["samples"]],
            "scope": "default-only latest-master integration; not a new paired performance ranking",
        }
    assert summary["latest_integration"]["independent_pairs"] == 8
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True))

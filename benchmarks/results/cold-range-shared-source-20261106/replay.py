"""Replay complete cold endpoints, source provenance and scientific gates offline."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
from pathlib import Path
from typing import Any


def errors(
    sample: dict[str, Any], target: dict[str, Any]
) -> tuple[float, float, float]:
    """Check all 288 exported force components, not a scalar force summary."""
    differences = [
        abs(actual - expected)
        for row, other in zip(sample["forces"], target["forces"], strict=True)
        for actual, expected in zip(row, other, strict=True)
    ]
    assert len(differences) == 288
    assert len(sample["forces"]) == 96
    assert all(len(row) == 3 for row in sample["forces"])
    return (
        abs(sample["energy"] - target["energy"]),
        max(differences),
        math.sqrt(math.fsum(value * value for value in differences) / len(differences)),
    )


def replay() -> dict[str, Any]:
    """Keep superseded measurements distinct from the final production source."""
    directory = Path(__file__).resolve().parent
    evidence = json.loads(
        lzma.decompress((directory / "evidence.json.xz").read_bytes())
    )
    assert evidence["schema"] == "generativeqc.cold-range-shared-source.v1"
    reference_bytes = (
        directory.parents[2] / evidence["reference"]["path"]
    ).read_bytes()
    assert (
        hashlib.sha256(reference_bytes).hexdigest() == evidence["reference"]["sha256"]
    )
    reference = json.loads(lzma.decompress(reference_bytes))
    targets = reference["records"]["A100/reference"]["samples"]
    assert len(targets) == 2 and all(target["converged"] for target in targets)
    summary: dict[str, Any] = {
        "decision": evidence["decision"],
        "fresh_gpu4pyscf_ranking": False,
        "cohorts": {},
    }
    for cohort_name, cohort in evidence["cohorts"].items():
        image = cohort["image"]
        assert cohort["build"]["exit_code"] == 0
        assert cohort["build"]["identity"] == image
        assert cohort["source_receipt"]["source_identity"] == image["source_identity"]
        assert cohort["operations"]["lsf_status"] == "DONE"
        assert cohort["operations"]["owned_transfer_stop"] == "OWNED_TRANSFER_STOPPED"
        assert (
            hashlib.sha256(cohort["patch"].encode()).hexdigest()
            == cohort["patch_sha256"]
        )
        devices = {}
        worst_reference_errors = [0.0, 0.0, 0.0]
        independent_pairs = 0
        for device in ("A100", "RTX5090"):
            population = cohort["devices"][device]
            assert population["exit_code"] == 0
            assert population["raw_oracle"]["passed"]
            assert population["raw_oracle"]["shared_source_values"] > 0
            for result in population["raw_oracle"]["results"]:
                assert "range oracle checks 16" in result
                maximum = float(result.split("maximum_error ", 1)[1].split()[0])
                assert math.isfinite(maximum) and maximum < 2e-9
            if device == "RTX5090":
                assert population["raw_oracle"]["tests"] == (
                    3 if cohort_name == "final" else 2
                )
                assert population["raw_oracle"]["skipped"] == 0
                assert "ERROR SUMMARY: 0 errors" in population["sanitizer"]
            records = population["records"]
            baseline = records["component"]
            for arm in ("component", "default"):
                record = records[arm]
                identity = population["identities"][arm]
                assert record["status"] == "complete"
                assert record["source_revision"] == image["source_identity"]
                for key in ("source_identity", "library_sha256"):
                    assert record["native_build"][key] == identity[key] == image[key]
                assert record["scheduler"] == identity["scheduler"]
                assert record["scheduler"]["CUDA_VISIBLE_DEVICES"]
                assert ("A100" if device == "A100" else "5090") in record[
                    "accelerator"
                ]["name"]
                assert record["method"] == "wb97m-v" and record["mode"] == "direct"
                assert record["atoms"] == 96 and record["orbital_basis"] == "def2-SVP"
                assert record["basis_representation"] == "spherical"
                assert record["geometry_bohr"] == reference["cohort"]["geometry_bohr"]
                assert record["charge"] == reference["cohort"]["charge"] == 0
                assert record["spin"] == reference["cohort"]["spin"] == 0
                assert record["protocol"]["cold"] == (
                    "constructor+prepare+SCF+forces+host-export"
                )
                assert not record["protocol"]["grid_overrides"]
                experiment = record["source_schedule_experiment"]
                assert experiment["clean"] and experiment["arm"] == arm
                assert experiment["component_source"] == (
                    "1" if arm == "component" else None
                )
                assert len(record["samples"]) == 2
                for index, sample in enumerate(record["samples"]):
                    incumbent = baseline["samples"][index]
                    assert sample["converged"] and not sample["warm_start_fallback"]
                    assert sample["phase"] == ("cold" if index == 0 else "warm")
                    assert sample["warm_start_used"] == bool(index)
                    assert sample["iterations"] == sample["fock_builds"]
                    assert sample["fock_builds"] == incumbent["fock_builds"]
                    assert sample["settings"] == incumbent["settings"]
                    settings = sample["settings"]
                    assert settings["energy_tolerance"] == 1e-9
                    assert settings["density_tolerance"] == 1e-5
                    assert settings["screening_tolerance"] == 1e-12
                    assert settings["force_grid_response"]
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
                    diagnostic = sample["ks_diagnostic"]
                    assert (
                        diagnostic["nonlocal_work"]
                        == incumbent["ks_diagnostic"]["nonlocal_work"]
                    )
                    work = diagnostic["coulomb_exchange_work"]
                    assert work["complete"]
                    assert work["coulomb_applications"] == sample["fock_builds"]
                    assert work["exchange_applications"] == 2 * sample["fock_builds"]
                    if index:
                        assert sample["fock_builds"] == 1
                        assert diagnostic["nonlocal_work"]["pair_sweeps"] == 0
                    energy_error, force_error, _ = errors(sample, incumbent)
                    assert energy_error <= 2e-9 and force_error <= 2e-8
                    for target in targets:
                        energy_error, force_error, force_rms = errors(sample, target)
                        assert energy_error / 96 < 1e-5 and force_error < 1e-4
                        assert force_rms < 3e-5
                        for position, value in enumerate(
                            (energy_error / 96, force_error, force_rms)
                        ):
                            worst_reference_errors[position] = max(
                                worst_reference_errors[position], value
                            )
                        independent_pairs += 1
            phases = {}
            for index, phase in enumerate(("cold", "warm")):
                previous = baseline["samples"][index]
                current = records["default"]["samples"][index]
                ratio = current["seconds"] / previous["seconds"]
                energy_error, force_error, force_rms = errors(current, previous)
                phases[phase] = {
                    "component_seconds": previous["seconds"],
                    "default_seconds": current["seconds"],
                    "seconds_saved": previous["seconds"] - current["seconds"],
                    "change_percent": 100 * (ratio - 1),
                    "energy_abs_error": energy_error,
                    "force_max_abs_error": force_error,
                    "force_rms_error": force_rms,
                    "focks": current["fock_builds"],
                    "nonlocal_work": current["ks_diagnostic"]["nonlocal_work"],
                }
                if evidence["decision"] == "promote" and cohort_name == "final":
                    assert ratio < (0.98 if phase == "cold" else 1.03)
            devices[device] = phases
        summary["cohorts"][cohort_name] = {
            "base_commit": cohort["base_commit"],
            "image": image,
            "devices": devices,
            "independent_pairs": independent_pairs,
            "worst_reference_errors": worst_reference_errors,
        }
        assert independent_pairs == 16
    assert "final" in summary["cohorts"]
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True))

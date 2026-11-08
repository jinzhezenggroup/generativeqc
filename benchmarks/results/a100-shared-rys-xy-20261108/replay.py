"""Replay immutable observations without turning a retained failed gate green."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import runpy
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
load_observations = runpy.run_path(
    str(ROOT.parent / "hf-screened-handoff-rejections-20261108" / "replay_support.py")
)["load_observations"]


def require(condition: bool, message: str) -> None:
    """Fail closed even when the interpreter runs with optimization enabled."""
    if not condition:
        raise ValueError(message)


def load(directory: Path = ROOT) -> tuple[dict, dict]:
    """Authenticate exact retained UTF-8 bytes, including unsuccessful jobs."""
    cohort = json.loads((directory / "cohort.json").read_text())
    return cohort, load_observations(directory, cohort)


def observable_error(actual: dict, target: dict) -> dict:
    """Use unchanged absolute E/F gates, never native-to-native parity alone."""
    actual_forces, expected_forces = actual["forces"], target["forces"]
    require(len(actual_forces) == len(expected_forces) > 0, "wrong force shape")
    differences = []
    for actual_row, expected_row in zip(actual_forces, expected_forces, strict=True):
        require(len(actual_row) == len(expected_row) == 3, "wrong Cartesian shape")
        differences.extend(
            abs(left - right)
            for left, right in zip(actual_row, expected_row, strict=True)
        )
    energy = abs(actual["energy"] - target["energy"])
    require(
        math.isfinite(energy) and all(math.isfinite(value) for value in differences),
        "nonfinite independent observable",
    )
    force = max(differences)
    return {
        "energy": energy,
        "force": force,
        "passed": bool(
            actual.get("converged") is True
            and target.get("converged") is True
            and energy < 1e-8
            and force < 1e-7
        ),
    }


def replay(directory: Path = ROOT) -> dict:
    """Report performance, parity, strict requests and default failures separately."""
    cohort, capsule = load(directory)

    def record(name: str) -> dict:
        return json.loads(capsule[name])

    def require_image(identity: dict) -> None:
        require(
            all(
                identity.get(key) == cohort[key]
                for key in ("source_identity", "library_sha256")
            ),
            "wrong auxiliary measured image",
        )

    def require_retained(name: str, digest: str) -> None:
        require(
            hashlib.sha256(capsule[name].encode()).hexdigest() == digest,
            "unqualified historical auxiliary provenance: " + name,
        )

    # These are immutable evidence inputs, not fingerprints of current production
    # code. The old strict producer omitted basis/method, and the extra producer
    # omitted per-record image metadata. Only this audited historical recipe and
    # same-job chain can supply that context; never infer it for arbitrary records.
    require_retained(
        "drivers/strict-native.py",
        "80973857ec1d7d34645de1cc58d80927fb3ebeb141e7cf97eb74240872a444ab",
    )
    require_retained(
        "drivers/qualify-extra.py",
        "015109fc788b3f46a319812b6d64c7a062640a58c4e7b6bc97e82b744523b475",
    )
    historical_extra = {
        "a100": (
            "0128c8193843f0eaf7a675805c4dc211b763a70c21052335c126300c71e8e51c",
            "ccc4318fea4be87b969b9631def87cae0946ee80260bcb049f427cfa848bdb51",
            "LSB_JOBID",
            "229003",
        ),
        "5090": (
            "ef33f1aaf13183e5eeac0a87cf6a6f110ee5c46871b881fe1f6874c42681938c",
            "912732b9086e1fd5d75f2553d8ea3579a3fd6842ab525b872326500128646344",
            "SLURM_JOB_ID",
            "16818",
        ),
    }
    require_retained(
        "final2/image-identity.json",
        "fb526af14e6a76fef340aed2e5df3e7f950bf2eba4a6825debcef309b3bbbd52",
    )
    require_image(record("final2/image-identity.json"))

    references = {
        size: record(name) for size, name in cohort["reference_files"].items()
    }
    if "converged" not in references["96"]:
        # This historical schema omitted convergence. Its audited producer
        # rejects unconverged SCF before exporting either independent target.
        # Bind the selected observables to that exact retained producer, rather
        # than treating every reference without a convergence field as valid.
        producer_text = capsule["oracle-hf96-producer-228320.json"]
        producer = json.loads(producer_text)
        for card in ("a100", "5090"):
            provenance = record(f"receipts/native-{card}-frozen-hf96-gates.json")
            require(
                provenance["producer_job"] == 228320
                and provenance["producer_identity"]["scheduler"]["LSB_JOBID"]
                == "228320"
                and provenance["oracle_text_sha256"]
                == hashlib.sha256(producer_text.encode()).hexdigest()
                and provenance["scientific_recipe_sha256"][
                    "tests/python/test_hf_cold_screening_cuda.py"
                ]
                == "80a5c3b806654d50ab33245beb096e9147f7e8af67cdbfd4de9fe853d84615ff"
                and len(producer) == 2
                and references["96"] == producer[0],
                "Unqualified historical HF96 reference provenance",
            )
        references["96"] = {**references["96"], "converged": True}
    require(
        all(reference.get("converged") is True for reference in references.values()),
        "Unconverged independent reference",
    )
    summary = {
        "source_identity": cohort["source_identity"],
        "cards": {},
        "independent_gates": {"absolute_energy": 1e-8, "maximum_force": 1e-7},
        "paired_warm_ratio_ceiling": 1.03,
    }
    promotion = True
    for card in ("a100", "5090"):
        worker_digest, extra_digest, scheduler_key, job = historical_extra[card]
        require_retained(f"drivers/worker-final2-{card}.sh", worker_digest)
        require_retained(f"final-{card}/extra-native.json", extra_digest)
        identity = record(f"final-{card}/identity-48-4-control.json")
        require_image(identity)
        require(
            identity["scheduler"].get(scheduler_key) == job
            and capsule[f"final-{card}/exit-code.txt"].strip() == "0",
            "incomplete historical auxiliary job provenance",
        )
        cases = {}
        for atoms in (96, 48):
            populations = []
            for ordinal, arm in enumerate(
                ("control", "candidate", "candidate", "control"), 1
            ):
                name = f"final-{card}/hf-{atoms}-{ordinal}-{arm}.json"
                data = record(name)
                require(
                    data["status"] == "complete" and len(data["samples"]) == 4,
                    "incomplete population",
                )
                require(
                    data["method"] == "hf"
                    and data["mode"] == "direct"
                    and data["arm"] == "standard"
                    and data["atoms"] == atoms,
                    "changed scientific endpoint",
                )
                require(
                    data["native_build"]["source_identity"] == cohort["source_identity"]
                    and data["native_build"]["library_sha256"]
                    == cohort["library_sha256"],
                    "wrong measured image",
                )
                for sample in data["samples"]:
                    require(
                        sample["converged"]
                        and math.isfinite(sample["seconds"])
                        and sample["seconds"] > 0,
                        "incomplete timed result",
                    )
                    require(
                        (
                            sample["settings"]["energy_tolerance"],
                            sample["settings"]["density_tolerance"],
                            sample["settings"]["screening_tolerance"],
                        )
                        == (1e-9, 1e-5, 1e-12),
                        "strict-request solve relabeled as default",
                    )
                populations.append(data)
            original = populations[0]
            if "geometry_bohr" in references[str(atoms)]:
                require(
                    original["geometry_bohr"]
                    == references[str(atoms)]["geometry_bohr"],
                    "changed independent geometry",
                )
            for data in populations:
                for key in (
                    "geometry_bohr",
                    "orbital_basis",
                    "basis_representation",
                    "charge",
                    "spin",
                    "protocol",
                ):
                    require(data[key] == original[key], "changed paired input: " + key)
                for sample in data["samples"]:
                    require(
                        sample["settings"] == original["samples"][0]["settings"],
                        "changed paired controls",
                    )
            samples = [sample for data in populations for sample in data["samples"]]
            parity = [
                observable_error(left, right)
                for left, right in itertools.combinations(samples, 2)
            ]
            require(len(parity) == 120, "missing native sample pairs")
            paired = []
            for control_index, candidate_index in ((0, 1), (3, 2)):
                ratios = [
                    candidate["seconds"] / control["seconds"]
                    for control, candidate in zip(
                        populations[control_index]["samples"],
                        populations[candidate_index]["samples"],
                        strict=True,
                    )
                ]
                paired.append(
                    {
                        "candidate_over_control": ratios,
                        "each_warm_within_3pct": all(
                            value <= 1.03 for value in ratios[1:]
                        ),
                    }
                )
            independent = [
                observable_error(sample, references[str(atoms)]) for sample in samples
            ]
            case = {
                "cold_median_seconds": {
                    "control": statistics.median(
                        populations[index]["samples"][0]["seconds"] for index in (0, 3)
                    ),
                    "candidate": statistics.median(
                        populations[index]["samples"][0]["seconds"] for index in (1, 2)
                    ),
                },
                "paired_timing": paired,
                "all_120_native_pairs_pass": all(item["passed"] for item in parity),
                "independent_default_observations": independent,
                "target_iterations": [
                    data["samples"][0]["iterations"] for data in populations
                ],
                "target_fock_builds": [
                    data["samples"][0]["fock_builds"] for data in populations
                ],
                "initial_guess_fock_builds": [
                    data["samples"][0]["initial_guess"]["preliminary_fock_builds"]
                    for data in populations
                ],
                "quartet_work_counters_valid": [
                    data["samples"][0]["incremental_direct_jk"][
                        "quartet_work_counters_valid"
                    ]
                    for data in populations
                ],
            }
            promotion &= case["all_120_native_pairs_pass"] and all(
                item["passed"] for item in independent
            )
            promotion &= all(item["each_warm_within_3pct"] for item in paired)
            cases[str(atoms)] = case
        strict = {}
        for arm in ("control", "candidate"):
            data = record(f"final-{card}/hf48-tight-{arm}.json")
            require_image(data.get("native_build", {}))
            require(
                data.get("schema") == "hf48-tight-native.v1"
                and data.get("geometry_bohr") == references["48"]["geometry_bohr"]
                and data.get("xy_control") == ("0" if arm == "control" else None)
                and data.get("scheduler", {}).get(scheduler_key) == job,
                "changed strict auxiliary scientific input or job",
            )
            require(
                len(data["samples"]) == 4
                and (
                    data["energy_tolerance"],
                    data["density_tolerance"],
                    data["screening_tolerance"],
                )
                == (1e-12, 1e-10, 1e-14),
                "changed strict diagnostic",
            )
            strict[arm] = [
                observable_error(sample, references["48"]) for sample in data["samples"]
            ]
            promotion &= all(item["passed"] for item in strict[arm])
        moved_target = record("history-qualification-a100/extra-reference.json")[
            "observations"
        ][1]
        extra_record = record(f"final-{card}/extra-native.json")
        require(
            extra_record.get("engine") == "native"
            and extra_record.get("schema") == "cooperative-roots-extra.v1",
            "changed auxiliary endpoint",
        )
        extra = extra_record["observations"]
        original96 = record(f"final-{card}/hf-96-1-control.json")
        extra_settings = dict(original96["samples"][0]["settings"])
        # The historical extra producer predates this null HF-only field.
        require(
            extra_settings.pop("resolved_force_tile_points") is None,
            "changed auxiliary HF force settings",
        )
        moved_geometry = json.loads(json.dumps(original96["geometry_bohr"]))
        moved_geometry[1][1][0] += 0.01
        require(
            moved_target.get("name") == "hf96-moved"
            and moved_target.get("geometry_bohr") == moved_geometry,
            "changed auxiliary independent geometry",
        )
        require(
            [item["name"] for item in extra] == ["hf96-moved", "hf96-restored"],
            "missing displaced/restored endpoint",
        )
        restored = []
        for item, target in zip(extra, (moved_target, references["96"]), strict=True):
            expected_geometry = (
                moved_geometry
                if item["name"] == "hf96-moved"
                else original96["geometry_bohr"]
            )
            require(
                item.get("geometry_bohr") == expected_geometry
                and item.get("settings") == extra_settings
                and item.get("warm_start_used") is True,
                "changed auxiliary scientific input",
            )
            error = observable_error(item, target)
            restored.append(
                {
                    "name": item["name"],
                    **error,
                    "iterations": item["iterations"],
                    "fock_builds": item["fock_builds"],
                }
            )
            promotion &= error["passed"]
        summary["cards"][card] = {
            "default_cases": cases,
            "strict_request_hf48": strict,
            "moved_and_restored_hf96": restored,
            "auxiliary_provenance": {
                "strict_hf48": "recorded native image; retained producer scientific recipe",
                "moved_and_restored_hf96": (
                    "retained observation and producer bytes; authenticated same-job image; "
                    "no per-record native image measurement in historical schema"
                ),
            },
        }
    summary["independent_default_promotion_qualified"] = bool(promotion)
    return summary


def main() -> None:
    """A successful replay may truthfully report a failed promotion gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-promotion", action="store_true")
    arguments = parser.parse_args()
    summary = replay()
    print(json.dumps(summary, indent=2, allow_nan=False))
    if (
        arguments.require_promotion
        and not summary["independent_default_promotion_qualified"]
    ):
        raise SystemExit(
            "Independent default promotion remains blocked; retained failures are not passes"
        )


if __name__ == "__main__":
    main()

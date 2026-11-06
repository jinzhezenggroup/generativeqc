"""Independently replay complete DF acceptance and the separately bounded WB study."""

import argparse
import hashlib
import json
import lzma
import statistics
import sys
from pathlib import Path


def check(value: object, message: str) -> None:
    if not value:
        raise ValueError(message)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(not sys.flags.optimize and not args.output.exists(), "Preserve output")
    root = args.task
    base = root / "base"
    folder = root / "qualification-v3"
    campaign = read(folder / "campaign.json")
    decl = read(root / "qualification-preparation-v3.json")
    build = read(base / "build-receipts-v3/build.json")
    check(
        sha(root / "qualification-preparation-v3.json")
        == campaign["declaration_sha256"],
        "Declaration differs",
    )
    check(
        all(sha(root / n) == h for n, h in decl["scripts"].items()), "Producer differs"
    )
    check(
        all(sha(base / n) == h for n, h in decl["base_files"].items()),
        "Base input differs",
    )
    check(
        campaign["source"] == build["source"] == read(base / "source-v3.json")
        and campaign["build_artifacts"] == build["artifacts"],
        "Source/build differs",
    )
    check(
        campaign["native_source_identity"]
        == build["native_source_identity"]
        == decl["native_source_identity"],
        "Native identity differs",
    )
    check(
        campaign["job"] == "gqc-h100-cold-qualified-1105-v3"
        and sha(base / "build-receipts-v3/build.json")
        == campaign["build_receipt_sha256"],
        "Build/job binding differs",
    )
    check(
        sha(base / "artifact-cache-snapshot-v1/manifest.json")
        == campaign["cache_manifest_sha256"],
        "Artifact cache identity differs",
    )
    prep = read(base / "measurement-preparation.json")
    check(
        all(sha(base / "inputs" / n) == h for n, h in prep["input_sha256"].items()),
        "Scientific inputs differ",
    )
    sys.path[:0] = [str(args.repository), str(args.repository / "python")]
    from benchmarks.readme_defaults_accuracy import errors
    from benchmarks.readme_defaults_qualification import qualify
    from benchmarks.readme_defaults_report import reduce_record

    oracles = json.loads(lzma.decompress((base / "oracles.json.xz").read_bytes()))
    reuse = read(base / "reference-reuse.json")
    studies = {s["id"]: s for s in decl["studies"]}
    index = {}
    failures = []
    calls = 0
    for row in campaign["runs"]:
        plan_path = folder / (row["name"] + ".plan.json")
        raw_path = folder / (row["name"] + ".json")
        plan = read(plan_path)
        study = studies[row["study"]]
        key = (row["study"], row["pair_index"], row["variant"])
        check(
            key not in index and key[1] in range(study["rounds"]),
            "Duplicate/unexpected population",
        )
        variant = next(v for v in study["variants"] if v["name"] == key[2])
        check(
            row["plan_sha256"] == sha(plan_path)
            and plan["execution_controls"]
            == row["effective_controls"]
            == variant["controls"],
            "Plan/control binding differs",
        )
        check(
            all(plan[k] == study[k] for k in ["method", "mode", "atoms"])
            and plan["pair_index"] == key[1]
            and plan["variant"] == key[2],
            "Scientific identity differs",
        )
        check(
            plan["source_revision"] == build["source"]["revision"]
            and plan["native_source_identity"] == build["native_source_identity"]
            and plan["build_artifacts"] == build["artifacts"],
            "Dispatched source differs",
        )
        check(
            plan["endpoint_sha256"] == decl["scripts"]["endpoint-v3.py"]
            and plan["input_sha256"] == prep["input_sha256"],
            "Dispatched producer/input differs",
        )
        if not raw_path.exists():
            failures.append(
                {"population": key, "reason": "No raw endpoint", "parent": row}
            )
            continue
        raw = read(raw_path)
        check(
            row.get("raw_sha256") == sha(raw_path)
            and raw["plan_sha256"] == sha(plan_path),
            "Raw provenance differs",
        )
        check(
            raw["runner_sha256"] == plan["endpoint_sha256"]
            and raw["driver_sha256"] == plan["driver_sha256"],
            "Executed producer differs",
        )
        check(
            all(
                raw[k] == plan[k]
                for k in [
                    "method",
                    "mode",
                    "atoms",
                    "arm",
                    "geometry_bohr",
                    "source_revision",
                    "qz_allocation",
                ]
            ),
            "Returned inputs differ",
        )
        check(
            raw["accelerator"]["nvidia_smi"]["uuid"] == campaign["device_uuid"]
            and raw["accelerator"]["compute_capability"] == [9, 0],
            "GPU differs",
        )
        if key[2] != "gpu4pyscf" and raw["status"] == "complete":
            check(
                raw["native_source_identity"] == build["native_source_identity"]
                and raw["native_build"]["library_sha256"]
                == build["artifacts"]["libgenerativeqc.so"],
                "Executed native image differs",
            )
        qualification = qualify([raw], oracles, reference_reuse=reuse)[0]
        accepted = (
            row.get("returncode") == 0
            and raw["status"] == "complete"
            and len(raw["samples"]) == 4
            and qualification["status"] == "qualified"
        )
        index[key] = {
            "raw": raw,
            "qualification": qualification,
            "accepted": accepted,
            "reduced": reduce_record(raw),
            "raw_sha256": sha(raw_path),
        }
        calls += len(raw["samples"])
        if not accepted:
            failures.append(
                {"population": key, "qualification": qualification, "parent": row}
            )
    check(calls == campaign["returned_EF"], "E/F count differs")
    results = []
    for study_id, study in studies.items():
        result = {
            "id": study_id,
            "pairs": [],
            "population_failures": [
                r for r in failures if r["population"][0] == study_id
            ],
        }
        observed = next(
            (s for s in campaign["studies"] if s["id"] == study_id),
            {"status": "NOT_STARTED", "pairs": []},
        )
        for i in range(study["rounds"]):
            values = {
                v["name"]: index.get((study_id, i, v["name"]))
                for v in study["variants"]
            }
            if not all(v and v["accepted"] for v in values.values()):
                continue
            off = values["off"]
            candidate = values[study["candidate"]]
            reference = values["gpu4pyscf"]
            matched = []
            for left, right in zip(
                off["raw"]["samples"], candidate["raw"]["samples"], strict=True
            ):
                comparison = errors(
                    right["energy"], right["forces"], left, study["atoms"]
                )
                comparison["matched_pass"] = all(
                    comparison[k] <= v for k, v in study["matched_gates"].items()
                )
                matched.append(comparison)
            check(
                matched == read(folder / f"{study_id}-pair{i}-matched.json"),
                "Producer matched errors differ",
            )
            work = candidate["raw"]["samples"][0].get("incremental_direct_jk")
            work_pass = not study.get("require_incremental", False) or bool(
                work and work.get("active") is True and work.get("delta_builds", 0) > 0
            )
            pair = {
                "pair_index": i,
                "matched_pass": all(r["matched_pass"] for r in matched) and work_pass,
                "matched_errors": matched,
                "incremental_work": work,
                "cold_candidate_over_off": candidate["reduced"]["cold"]["median"]
                / off["reduced"]["cold"]["median"],
                "warm_candidate_over_off": candidate["reduced"]["warm"]["median"]
                / off["reduced"]["warm"]["median"],
                "cold_reference_over_candidate": reference["reduced"]["cold"]["median"]
                / candidate["reduced"]["cold"]["median"],
                "warm_reference_over_candidate": reference["reduced"]["warm"]["median"]
                / candidate["reduced"]["warm"]["median"],
                "arms": {
                    name: {
                        "raw_sha256": value["raw_sha256"],
                        "qualification": value["qualification"],
                        "cold": value["reduced"]["cold"],
                        "warm": value["reduced"]["warm"],
                        "cold_initial_guess": value["raw"]["samples"][0].get(
                            "initial_guess"
                        ),
                        "cold_policy": value["raw"]["samples"][0].get(
                            "precision_validation"
                        ),
                        "cold_incremental": value["raw"]["samples"][0].get(
                            "incremental_direct_jk"
                        ),
                    }
                    for name, value in values.items()
                },
            }
            producer = next(p for p in observed["pairs"] if p["pair_index"] == i)
            check(
                all(
                    pair[k] == producer[k]
                    for k in [
                        "matched_pass",
                        "cold_candidate_over_off",
                        "warm_candidate_over_off",
                        "cold_reference_over_candidate",
                    ]
                ),
                "Producer statistics differ",
            )
            result["pairs"].append(pair)
        result["all_declared_pairs_qualified"] = (
            len(result["pairs"]) == study["rounds"]
            and all(p["matched_pass"] for p in result["pairs"])
            and not result["population_failures"]
        )
        result["producer_status"] = observed["status"]
        if result["pairs"]:
            cold = [p["cold_candidate_over_off"] for p in result["pairs"]]
            warm = [p["warm_candidate_over_off"] for p in result["pairs"]]
            ref = [p["cold_reference_over_candidate"] for p in result["pairs"]]
            result["summary"] = {
                "cold_candidate_over_off": cold,
                "median_cold_reduction": 1 - statistics.median(cold),
                "warm_candidate_over_off": warm,
                "median_warm_change": statistics.median(warm) - 1,
                "warm_guard_pass": all(
                    r <= decl["warm_candidate_over_off_limit"] for r in warm
                ),
                "cold_reference_over_candidate": ref,
                "median_cold_reference_over_candidate": statistics.median(ref),
                "reference_cold_win": len(ref) == 3
                and statistics.median(ref) > 1.05
                and all(r > 1 for r in ref),
                "cold_process_medians_seconds": {
                    v["name"]: statistics.median(
                        p["arms"][v["name"]]["cold"]["median"] for p in result["pairs"]
                    )
                    for v in study["variants"]
                },
            }
        results.append(result)
    hf96, hf48 = results[:2]
    science_pass = (
        hf96["all_declared_pairs_qualified"] and hf48["all_declared_pairs_qualified"]
    )
    performance_pass = (
        science_pass
        and hf96["summary"]["median_cold_reduction"] >= 0.02
        and all(r < 1 for r in hf96["summary"]["cold_candidate_over_off"])
        and hf96["summary"]["warm_guard_pass"]
    )
    terminal = (
        campaign["status"] == "COMPLETED_DECLARED_DF_QUALIFICATION_AND_BOUNDED_WB_STUDY"
        and campaign.get("final_artifacts_verified") is True
    )
    outcome = {
        "status": "PASS_CHANGED_DF_DEFAULT_SCIENCE_AND_COLD_PERFORMANCE"
        if terminal and performance_pass
        else "INCOMPLETE_OR_NOT_ACCEPTED_CHANGED_DEFAULT",
        "scope": "HF96 DF three-pair default acceptance plus HF48 numerical size guard; WB study reported separately; no fixed30 aggregate",
        "source_revision": build["source"]["revision"],
        "native_source_identity": build["native_source_identity"],
        "core_sha256": build["artifacts"]["libgenerativeqc.so"],
        "device_uuid": campaign["device_uuid"],
        "campaign_sha256": sha(folder / "campaign.json"),
        "qualified_populations": sum(v["accepted"] for v in index.values()),
        "returned_EF": calls,
        "changed_default_science_pass": science_pass,
        "changed_default_performance_pass": performance_pass,
        "studies": results,
        "failures": failures,
    }
    with args.output.open("x") as stream:
        json.dump(outcome, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {k: v for k, v in outcome.items() if k not in {"studies", "failures"}}
        )
    )


if __name__ == "__main__":
    main()

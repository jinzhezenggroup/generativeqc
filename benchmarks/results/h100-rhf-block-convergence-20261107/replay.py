"""Replay retained full E/F arrays and provenance; never execute CUDA or an oracle."""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONTROL = "GENERATIVEQC_HF_BLOCK_CONVERGENCE"
DENSE = {
    "energy_abs_per_atom_hartree": 1e-5,
    "force_max_abs_hartree_per_bohr": 1e-4,
    "force_rms_hartree_per_bohr": 3e-5,
}
STRICT = {
    "energy_abs_per_atom_hartree": 1e-9,
    "force_max_abs_hartree_per_bohr": 1e-7,
    "force_rms_hartree_per_bohr": 3e-8,
}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def errors(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    """Same intensive/full-array arithmetic as the retained HF opt-in readers."""
    for sample in (first, second):
        require(
            math.isfinite(sample["energy"])
            and len(sample["forces"]) == 96
            and all(
                len(row) == 3 and all(math.isfinite(x) for x in row)
                for row in sample["forces"]
            ),
            "Incomplete/nonfinite E/F",
        )
    delta = [
        x - y
        for row, other in zip(first["forces"], second["forces"], strict=True)
        for x, y in zip(row, other, strict=True)
    ]
    return {
        "energy_abs_per_atom_hartree": abs(first["energy"] - second["energy"]) / 96,
        "force_max_abs_hartree_per_bohr": max(map(abs, delta)),
        "force_rms_hartree_per_bohr": math.sqrt(math.fsum(x * x for x in delta) / 288),
    }


def within(values: dict[str, float], limits: dict[str, float]) -> None:
    require(
        all(math.isfinite(v) and v <= limits[k] for k, v in values.items()),
        "Observable gate missed",
    )


def same(actual: dict[str, float], reported: dict[str, float]) -> None:
    require(
        all(
            math.isclose(v, reported[k], rel_tol=2e-12, abs_tol=1e-18)
            for k, v in actual.items()
        ),
        "Reported arithmetic differs",
    )


def audit() -> dict[str, Any]:
    cohort = json.loads((HERE / "cohort.json").read_text())
    data = (HERE / cohort["capsule"]["path"]).read_bytes()
    require(
        len(data) == cohort["capsule"]["bytes"]
        and sha(data) == cohort["capsule"]["sha256"],
        "Capsule changed",
    )
    stream = lzma.LZMADecompressor()
    decoded = stream.decompress(data, max_length=8 << 20)
    require(
        stream.eof
        and not stream.unused_data
        and len(decoded) == cohort["capsule"]["decoded_bytes"],
        "Capsule extent differs",
    )
    capsule = json.loads(decoded)
    files = {}
    for name, row in capsule["files"].items():
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Unsafe member",
        )
        blob = base64.b64decode(row["base64"], validate=True)
        require(
            len(blob) == row["bytes"] and sha(blob) == row["sha256"],
            "Member changed: " + name,
        )
        files[name] = blob
    require(
        len(files) == cohort["capsule"]["original_files"]
        and sum(map(len, files.values()))
        == cohort["capsule"]["original_payload_bytes"],
        "Inventory omitted",
    )
    get = lambda name: json.loads(files[name])
    source = cohort["measured_source"]
    contract, image = (
        get("support/source-contract.json"),
        get("support/image-audit.json"),
    )
    require(
        contract["revision"] == source["revision"]
        and contract["tree"] == source["tree"]
        and contract["native_source_identity"]
        == image["native_source_identity"]
        == source["native_identity"]
        and image["artifacts"]["libgenerativeqc.so"] == source["core_sha256"],
        "Source/core differs",
    )
    require(
        cohort["scope"]["default"] == "off"
        and cohort["scope"]["all30_complete"] is False
        and cohort["scope"]["GPU4PySCF_timing_comparison"] is False,
        "Scope promoted",
    )
    require(
        sha(files["support/oracles.json.xz"]) == cohort["oracle_reuse"]["sha256"],
        "Original oracle changed",
    )
    oracle = [
        r
        for r in json.loads(lzma.decompress(files["support/oracles.json.xz"]))
        if r["method"] == "hf" and r["atoms"] == 96
    ]
    require(
        len(oracle) == 1
        and len(oracle[0]["samples"]) == 1
        and oracle[0]["common_observable_gates"] == DENSE,
        "Wrong dense oracle",
    )
    target = oracle[0]["samples"][0]
    settings = get("support/original-settings.json")["owners"]["native"]["settings"]
    review = get("support/endpoint-review.json")
    require(
        review["status"]
        == "THREE_PAIRED_COMPONENT_CONTRIBUTION_PASS_DEFAULT_OFF_OPT_IN_PR_SUPPORTED"
        and review["source_revision"] == source["revision"]
        and review["native_source_identity"] == source["native_identity"]
        and review["core_sha256"] == source["core_sha256"]
        and review["gpu_uuid"] == source["gpu_uuid"]
        and [r["result_sha256"] for r in review["rounds"]]
        == [r["result_sha256"] for r in cohort["rounds"]],
        "Independent completed review differs",
    )
    rows, dense, strict = [], [], []
    for round_spec in cohort["rounds"]:
        prefix = round_spec["prefix"]
        result = get(prefix + "/result.json")
        config_name = "support/config-round" + str(round_spec["round"]) + ".json"
        config = get(config_name)
        require(
            sha(files[prefix + "/result.json"]) == round_spec["result_sha256"]
            and result["status"] == "EXPLORATORY_SINGLE_PAIR_POSITIVE"
            and result["order"] == round_spec["order"]
            and result["complete_EF"] == 8
            and result["whole_goal_complete"] is False
            and result["default_promotion"] is False,
            "Round incomplete/promoted",
        )
        require(
            result["build_receipt_sha256"] == sha(files["support/image-audit.json"])
            and result["source_contract_sha256"]
            == sha(files["support/source-contract.json"])
            and result["fixture_review"]["sha256"]
            == sha(files["support/fixture-review.json"])
            and result["runner_sha256"]
            == sha(
                files["support/purpose-adapter.py"]
                if round_spec["round"] == 1
                else files["support/counterbalance-adapter.py"]
            ),
            "Execution provenance chain differs",
        )
        require(
            sha(files[config_name]) == result["config_sha256"]
            and config["owner_order"] == round_spec["order"]
            and config["new_oracle_runs"] == config["new_reference_runs"] == 0
            and config["numerical_gates"] == DENSE
            and config["strict_native_gates"] == STRICT
            and config["artifact_cache_manifest_sha256"]
            == get("support/config-round1.json")["artifact_cache_manifest_sha256"],
            "Controls/gates/cache-initialization configuration differs",
        )
        require(
            len(result["stages"]) == 8
            and all(x["returncode"] == 0 for x in result["stages"]),
            "Round failure omitted",
        )
        for stage in result["stages"]:
            for kind in ("stdout", "stderr"):
                require(
                    sha(files[prefix + "/" + stage["name"] + "." + kind])
                    == stage[kind + "_sha256"],
                    "Stage stream changed",
                )
        records, work = {}, {}
        for variant, value in (("off", "0"), ("on", "1")):
            raw, plan, q = (
                get(prefix + "/" + variant + "/" + name + ".json")
                for name in ("raw", "plan", "qualification")
            )
            owner = next(x for x in result["owners"] if x["variant"] == variant)
            require(
                all(
                    sha(files[prefix + "/" + variant + "/" + name + ".json"])
                    == owner[key]
                    for name, key in (
                        ("raw", "raw_sha256"),
                        ("plan", "plan_sha256"),
                        ("qualification", "qualification_sha256"),
                    )
                ),
                "Owner hash differs",
            )
            require(
                raw["status"] == "complete"
                and q["status"] == "qualified"
                and not q["reasons"]
                and q["cpu_gpu_spot"]["passed"]
                and q["reference_reuse"],
                "Independent qualification incomplete",
            )
            require(
                raw["plan_sha256"] == owner["plan_sha256"]
                and raw["source_revision"]
                == plan["source_revision"]
                == source["revision"]
                and raw["native_source_identity"]
                == plan["native_source_identity"]
                == source["native_identity"]
                and raw["native_build"]["library_sha256"] == source["core_sha256"]
                and raw["accelerator"]["nvidia_smi"]["uuid"] == source["gpu_uuid"]
                and raw["accelerator"]["runtime_version"] == 12090,
                "Executed epoch differs",
            )
            require(
                all(
                    raw[k] == plan[k] == v
                    for k, v in {
                        "method": "hf",
                        "mode": "direct",
                        "arm": "standard",
                        "atoms": 96,
                    }.items()
                )
                and raw["protocol"]["common_observable_gates"] == DENSE
                and plan["matched_comparison_policy"]["strict_native_hard_thresholds"]
                == STRICT,
                "Original physical recipe/gates differ",
            )
            controls = {
                "GENERATIVEQC_SCF_ORTHONORMAL_DIIS": "0",
                "GENERATIVEQC_CUDA_HF_SHARED_RYS_XY": "1",
                "GENERATIVEQC_CUDA_HF_SHARED_RYS_WEIGHTED_Z": "1",
                CONTROL: value,
            }
            require(
                raw["observed_execution_controls"]
                == plan["execution_controls"]
                == controls
                and raw["geometry_bohr"]
                == plan["geometry_bohr"]
                == oracle[0]["geometry_bohr"]
                and plan["pair_order"] == round_spec["order"]
                and plan["ordinal"] == round_spec["order"].index(variant)
                and plan["purpose"] == "FOCUSED_DIAGNOSTIC_NOT_ALL30_ACCEPTANCE"
                and raw["runner_sha256"]
                == plan["endpoint_sha256"]
                == sha(files["support/endpoint.py"]),
                "Recipe/order/endpoint differs",
            )
            require(
                raw["protocol"]["analytic_forces"]
                and not raw["protocol"]["trace_enabled"]
                and not raw["protocol"]["oracle_seed_used"],
                "Timing/physics boundary differs",
            )
            require(
                [(s["phase"], s["repeat"]) for s in raw["samples"]]
                == [("cold", 0), ("warm", 0), ("warm", 1), ("warm", 2)],
                "Population incomplete",
            )
            reports, work[variant] = [], []
            for index, sample in enumerate(raw["samples"]):
                p, inc = sample["precision"], sample["incremental_direct_jk"]
                require(
                    sample["converged"]
                    and sample["settings"] == settings
                    and math.isfinite(sample["seconds"])
                    and sample["seconds"] > 0
                    and p["requested_mode"] == "fp64"
                    and p["effective_bits"] == 64
                    and p["operator_work_counters_valid"]
                    and p["mixed_stage_fock_builds"] == 0
                    and p["mixed_precision_fock_threshold"] == 0
                    and sample["semantic_work"]["fock_builds"]
                    == sample["fock_builds"]
                    >= 1
                    and not inc["requested"]
                    and not inc["active"],
                    "Scientific/work policy differs",
                )
                require(
                    sample["warm_start_used"] == (index != 0)
                    and not sample["warm_start_fallback"],
                    "Warm ownership differs",
                )
                target_iters = (
                    p["refinement_iterations"] if index == 0 else sample["iterations"]
                )
                proposal = sample["iterations"] - target_iters if index == 0 else 0
                require(
                    (
                        1 <= proposal <= 8
                        and target_iters >= 1
                        and p["strict_refinement_applied"]
                    )
                    if index == 0
                    else (
                        p["refinement_iterations"] == 0
                        and not p["strict_refinement_applied"]
                    ),
                    "Phase work incomplete",
                )
                work[variant].append(
                    {
                        "phase": sample["phase"],
                        "iterations": sample["iterations"],
                        "focks": sample["fock_builds"],
                        "proposal": proposal,
                        "target": target_iters,
                    }
                )
                error = errors(sample, target)
                within(error, DENSE)
                reports.append(error)
            same(reports[0], q["cold"]["errors"][0])
            for actual, reported in zip(reports[1:], q["warm"]["errors"], strict=True):
                same(actual, reported)
            dense.extend(reports)
            records[variant] = raw
        for index, (off, on) in enumerate(
            zip(records["off"]["samples"], records["on"]["samples"], strict=True)
        ):
            value = errors(off, on)
            within(value, STRICT)
            same(value, result["strict_native_comparisons"][index]["errors"])
            strict.append(value)
        off, on = ([s["seconds"] for s in records[v]["samples"]] for v in ("off", "on"))
        warm = [b / a for a, b in zip(off[1:], on[1:], strict=True)]
        ratio = off[0] / on[0]
        require(
            ratio > 1 and all(x <= 1.05 for x in warm),
            "Contribution direction/warm guard failed",
        )
        require(
            math.isclose(
                ratio, result["timing_decision"]["cold_off_over_on"], rel_tol=1e-14
            ),
            "Reported timing differs",
        )
        rows.append(
            {
                "round": round_spec["round"],
                "order": round_spec["order"],
                "cold_off_seconds": off[0],
                "cold_on_seconds": on[0],
                "off_over_on": ratio,
                "reduction_percent": 100 * (1 - on[0] / off[0]),
                "saved_seconds": off[0] - on[0],
                "warm_on_over_off": warm,
                "work": work,
            }
        )
    failure = get("failed-v1/result.json")
    require(
        sha(files["failed-v1/result.json"]) == cohort["failure"]["sha256"]
        and failure["status"] == "FAILED_RETAINED"
        and not failure["owners"]
        and not any(s["name"] == "on" for s in failure["stages"])
        and b"ValueError: Wrong scope" in files["failed-v1/off.stderr"],
        "Original failure relabelled",
    )
    for stage in failure["stages"]:
        for kind in ("stdout", "stderr"):
            require(
                sha(files["failed-v1/" + stage["name"] + "." + kind])
                == stage[kind + "_sha256"],
                "Failure stream changed",
            )
    san = get("support/sanitizer-result.json")
    fixtures = get("support/fixture-review.json")
    require(
        fixtures["image_audit_sha256"] == sha(files["support/image-audit.json"])
        and fixtures["core_sha256"] == source["core_sha256"]
        and fixtures["sanitizer_result_sha256"]
        == sha(files["support/sanitizer-result.json"]),
        "Linked fixture qualification differs",
    )
    for name, digest in fixtures["ordinary_files"].items():
        if name.endswith((".stdout", ".stderr")):
            require(
                sha(files["qualification/actual-fixtures-v1/" + name]) == digest,
                "Ordinary linked-fixture stream differs",
            )
    for stage in san["stages"]:
        for kind in ("stdout", "stderr"):
            require(
                stage["returncode"] == 0
                and sha(
                    files[
                        "qualification/actual-sanitizers-v1/"
                        + stage["name"]
                        + "."
                        + kind
                    ]
                )
                == stage[kind + "_sha256"],
                "Sanitizer stream/status differs",
            )
    for tool, marker in (
        ("racecheck", "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"),
        ("synccheck", "ERROR SUMMARY: 0 errors"),
    ):
        lines = (
            files["qualification/actual-sanitizers-v1/" + tool + ".stdout"]
            .decode()
            .splitlines()
        )
        require(
            san["sanitizer_summaries"][tool]["passed"]
            and san["sanitizer_summaries"][tool]["matched_line_count"] == 1
            and lines[-1].endswith(marker),
            "Selected sanitizer gate differs",
        )
    return {
        "status": "THREE_NATIVE_PAIR_ARRAY_AND_PROVENANCE_REPLAY_PASS",
        "measured_source": source,
        "rounds": rows,
        "median_paired_off_over_on": statistics.median(r["off_over_on"] for r in rows),
        "median_paired_reduction_percent": statistics.median(
            r["reduction_percent"] for r in rows
        ),
        "median_saved_seconds": statistics.median(r["saved_seconds"] for r in rows),
        "maximum_warm_on_over_off": max(x for r in rows for x in r["warm_on_over_off"]),
        "independent_EF": len(dense),
        "strict_full_force_pairs": len(strict),
        "warm_guards": 9,
        "maximum_dense_errors": {k: max(r[k] for r in dense) for k in DENSE},
        "maximum_strict_errors": {k: max(r[k] for r in strict) for k in STRICT},
        "default": "off",
        "all30_complete": False,
        "GPU4PySCF_comparison": False,
        "new_native_or_oracle_execution": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))

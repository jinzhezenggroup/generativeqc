"""Replay complete baseline observations, including every force coordinate."""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import math
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The prepared private copy is deliberately not runnable as a repository reader.
# Install this directory under benchmarks/results before using this entry point.
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks.readme_defaults_qualification import identity, qualify

SOURCE = "000517f82dc62130459a6fd92a27518a158c49cc"
COHORT = "integrated-coverage-000517-v1"
BUILD = "integrated-000517-build-originals-v1/integrated-000517-build-v1"
AOT = "integrated-000517-build-originals-v1/integrated-000517-build-manifests-v1"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    check(
        manifest["schema"] == "generativeqc.h100-integrated-baseline-evidence.v1"
        and manifest["measured_source"] == SOURCE,
        "unexpected evidence identity",
    )
    for name, binding in manifest["files"].items():
        check(Path(name).name == name, "unsafe stored member")
        data = (HERE / name).read_bytes()
        check(
            {"bytes": len(data), "sha256": digest(data)} == binding,
            "stored payload differs",
        )
    documents = json.loads(lzma.decompress((HERE / "documents.json.xz").read_bytes()))
    index = json.loads(lzma.decompress((HERE / "bindings.json.xz").read_bytes()))
    bindings = index["documents"]
    values = np.load(
        io.BytesIO(lzma.decompress((HERE / "forces.npy.xz").read_bytes())),
        allow_pickle=False,
    )
    check(
        values.dtype.str == "<f8"
        and list(values.shape) == manifest["force_storage"]["shape"] == [83052]
        and np.isfinite(values).all()
        and len(index["force_slots"]) == 396,
        "complete FP64 force population differs",
    )
    end = 0
    for slot in index["force_slots"]:
        check(slot["begin"] == end, "overlapping or missing force coordinates")
        parent = documents[slot["document"]]
        for key in slot["path"][:-1]:
            parent = parent[key]
        descriptor = parent[slot["path"][-1]]
        check(
            descriptor == {"force_storage_begin": end, "atoms": slot["atoms"]},
            "force descriptor differs",
        )
        end += 3 * slot["atoms"]
        parent[slot["path"][-1]] = values[slot["begin"] : end].reshape(-1, 3).tolist()
    check(end == values.size, "unused force coordinates")
    check(
        set(documents) | set(index["byte_segments"]) == set(bindings),
        "missing original member binding",
    )
    for name, value in documents.items():
        check(
            digest(canonical(value)) == bindings[name]["normalized_sha256"],
            "normalized original JSON differs after full-force reconstruction",
        )
    text = lzma.decompress((HERE / "text-streams.bin.xz").read_bytes())
    cursor = 0
    for name, segment in index["byte_segments"].items():
        check(segment["begin"] == cursor, "text stream overlap or gap")
        cursor = segment["end"]
        data = text[segment["begin"] : cursor]
        check(
            len(data) == bindings[name]["normalized_bytes"]
            and digest(data) == bindings[name]["normalized_sha256"],
            "normalized producer/log/telemetry stream differs",
        )
    check(cursor == len(text), "unbound text bytes")
    check(
        sum(name.startswith(COHORT + "/") for name in bindings)
        + sum(name.startswith(COHORT + "-executed-producers/") for name in bindings)
        + int(COHORT + "-telemetry.csv" in bindings)
        == manifest["original_archive"]["members"]
        == 509,
        "terminal original archive member population differs",
    )

    def original_sha(name: str) -> str:
        """Read retained private-byte provenance, not a public-byte digest."""
        if name in bindings:
            return bindings[name]["sha256"]
        matches = [
            row["sha256"]
            for path, row in bindings.items()
            if path == COHORT + "-executed-producers/" + name
        ]
        check(len(matches) == 1, "missing executed producer binding: " + name)
        return matches[0]

    def load(name: str) -> dict:
        return documents[name]

    campaign = load(COHORT + "/campaign.json")
    prereg_name = "integrated-coverage-000517-preregistered-v1.json"
    prereg = load(prereg_name)
    build = load(BUILD + "/build-receipt.json")
    check(
        campaign["status"] == "COMPLETE_COVERAGE_ALL_QUALIFIED"
        and campaign["job"] == manifest["job"] == "gqc-h100-integrated-000517-1030-v1"
        and campaign["expanded_build_receipt_sha256"]
        == original_sha(BUILD + "/build-receipt.json")
        and build["status"]
        == "PASS_INTEGRATED_000517_SM90_PACKAGE_ONLY_MOLECULAR_NOT_RUN"
        and build["source_revision"] == SOURCE
        and len(build["artifacts"]) == 33,
        "source/core/AOT or complete campaign binding differs",
    )
    check(
        campaign["preregistration_sha256"] == original_sha(prereg_name)
        and prereg["expected_source_revision"] == SOURCE
        and original_sha(prereg["source_stage"])
        == prereg["source_stage_sha256"]
        == build["source_stage_sha256"]
        and original_sha("protocol-v1.json")
        == prereg["final_protocol_sha256"]
        == "2391b71fb49660f682f1b60056b2892d5516f01871d7479319f0be2cf925fdaa",
        "source stage or frozen original protocol differs",
    )
    check(
        all(original_sha(name) == value for name, value in campaign["scripts"].items()),
        "executed campaign producer differs",
    )
    final = load("protocol-v1.json")
    primary = {
        (r["method"], r["atoms"], r["route"]) for r in final["primary_physical_cases"]
    }
    check(
        len(primary) == 30
        and set(prereg["policies"]) == {"tile256", "tile1024"}
        and len(prereg["order"]) == len(campaign["runs"]) == 99,
        "fixed physical denominator or whole policies differ",
    )
    allocation = {"cpu.max": "2000000 100000", "memory.max": "214748364800"}
    check(
        campaign["allocation_cgroups"] == allocation
        and campaign["shared_memory_bytes"] == 16 * (1 << 30),
        "actual allocation differs",
    )
    fields = [v.strip() for v in campaign["device"]["snapshot"].split(",")]
    check(
        fields[:4]
        == ["NVIDIA H100 80GB HBM3", campaign["device"]["uuid"], "9.0", "Disabled"],
        "actual H100 architecture or MIG state differs",
    )
    runtime_name = COHORT + "/reference-runtime.json"
    runtime = load(runtime_name)
    check(
        original_sha(runtime_name) == campaign["reference_runtime_sha256"]
        and runtime["status"] == "PASS_REFERENCE_RUNTIME_AND_CUTENSOR_ONLY"
        and campaign["device"]["uuid"] in runtime["device"],
        "stock same-allocation reference runtime differs",
    )
    audit = load(BUILD + "/aot-audit.json")
    check(
        original_sha(BUILD + "/aot-audit.json") == build["aot_audit_sha256"]
        and audit["primary"]["architecture"] == "sm_90"
        and len(audit["primary"]["artifacts"]) == 16,
        "SM90 AOT metadata audit differs",
    )
    for row in audit["primary"]["artifacts"]:
        name = row["name"]
        metadata = load(AOT + "/generativeqc_stationary_" + name + ".json")
        check(
            metadata["binary_sha256"]
            == row["binary_sha256"]
            == build["artifacts"]["libgenerativeqc_stationary_" + name + ".so"]
            and metadata["architectures"] == ["sm_90"]
            and metadata["compile_contract"]["fp64"] is True
            and metadata["compile_contract"]["fmad"] is False
            and "cubin" in row["code_kinds"]
            and row["driver_ptx_jit_required"] is False,
            "native AOT architecture/precision/artifact binding differs",
        )
    for key in ["qualifier", "oracle"]:
        check(
            digest((ROOT / manifest[key]["path"]).read_bytes())
            == manifest[key]["sha256"],
            "independent qualifier/oracle source changed",
        )
    oracles = json.loads(
        lzma.decompress((ROOT / manifest["oracle"]["path"]).read_bytes())
    )
    reuse = load(COHORT + "/reference-reuse-v2.json")
    input_sets = {
        kind: {
            name: row["sha256"] for name, row in load(kind + ".json")["files"].items()
        }
        for kind in ["inputs-default", "inputs-hf-ccpvtz"]
    }
    base = load("build-pair-completed.json")["sources"]["base"]
    indexed = {}
    counts = {
        "qualified_populations": 0,
        "qualified_EF": 0,
        "fresh_reference_crosschecks": 0,
    }
    for declared, stage in zip(prereg["order"], campaign["runs"], strict=True):
        method, atoms, mode, policy_name = (
            declared[k] for k in ["method", "atoms", "mode", "policy"]
        )
        key = (method, atoms, mode, policy_name)
        stem = "-".join(map(str, key))
        reference = policy_name == "reference"
        policy = None if reference else prereg["policies"][policy_name]
        arm = "gpu4pyscf" if reference else "standard"
        prefix = COHORT + "/" + stem
        plan, raw = load(prefix + ".plan.json"), load(prefix + ".json")
        check(
            key not in indexed
            and stage["case"] == stem
            and stage["status"] == "COMPLETE"
            and stage["returncode"] == 0
            and stage["receipt_sha256"] == original_sha(prefix + ".json")
            and stage["plan_sha256"] == original_sha(prefix + ".plan.json")
            and raw["plan_sha256"] == stage["plan_sha256"],
            "missing/reordered/incomplete or incorrectly bound attempt",
        )
        expected_source = base["revision"] if reference else SOURCE
        check(
            (raw["method"], raw["atoms"], raw["mode"], raw["arm"])
            == (method, atoms, mode, arm)
            == (plan["method"], plan["atoms"], plan["mode"], plan["arm"])
            and raw["source_revision"] == plan["source_revision"] == expected_source
            and raw["geometry_bohr"] == plan["geometry_bohr"]
            and raw["protocol"]["common_observable_gates"] == prereg["numerical_gates"]
            and plan.get("execution_policy") == policy,
            "scientific model, source, geometry or policy differs",
        )
        fitted = method == "hf" and mode == "df"
        script = (
            ("hf-fit-reference.py" if fitted else "endpoint.py")
            if reference
            else "policy-endpoint-v5.py"
        )
        check(
            raw["runner_sha256"] == plan["endpoint_sha256"] == original_sha(script)
            and plan["driver_sha256"]
            == raw.get("driver_sha256", raw.get("driver_context_sha256"))
            == "f83e5c75204b3e2b8c1f8f6d5dd9c3fe5e6691b3462768b1778b97e47659b42a"
            and plan["input_sha256"]
            == input_sets["inputs-hf-ccpvtz" if fitted else "inputs-default"]
            and plan["build_artifacts"] == ({} if reference else build["artifacts"]),
            "executed endpoint/driver/input/core/AOT binding differs",
        )
        if fitted:
            check(
                plan["fit_policy"]
                == "EXPLICIT_CC_PVTZ_JKFIT_ACCURACY_CANDIDATE_NOT_FINAL",
                "HF fit contract differs",
            )
        eligible = (method, atoms, mode) in primary
        check(
            stage["primary_ratio_eligible"]
            == plan["primary_ratio_eligible"]
            == declared["primary_ratio_eligible"]
            == eligible
            and raw["accelerator"]["nvidia_smi"]["uuid"] == campaign["device"]["uuid"]
            and raw["accelerator"]["name"] == "NVIDIA H100 80GB HBM3"
            and raw["accelerator"]["compute_capability"] == [9, 0]
            and raw["qz_allocation"] == plan["qz_allocation"]
            and raw["qz_allocation"]["job_name"] == campaign["job"]
            and raw["scheduler"]["kind"] == "qz_job"
            and raw["slurm_job_id"] is None,
            "physical GPU, allocation or primary exclusions differ",
        )
        exact = qualify([raw], oracles, reference_reuse=reuse)[0]
        retained = dict(load(prefix + ".qualification.json"))
        retained["receipt_sha256"] = identity(raw)
        check(
            exact == retained
            and exact["status"] == stage["qualification"] == "qualified",
            "full scientific qualifier replay differs",
        )
        check(
            raw["status"] == "complete"
            and len(raw["samples"]) == 4
            and [s["phase"] for s in raw["samples"]]
            == ["cold", "warm", "warm", "warm"],
            "complete cold/warm E+F population differs",
        )
        if not reference:
            caps = raw.get("native_prepared_force_capabilities", [])
            check(
                raw["execution_policy"] == policy
                and raw["resolved_execution_policy"]
                == {
                    "ks_tile_points": None
                    if method == "hf"
                    else policy["ks_tile_points"],
                    "df_environment": policy["df_environment"],
                }
                and (method == "hf" or caps == [True] * 4)
                and stage["native_dft_force_capabilities"] == caps
                and stage["native_dft_force_accepted"] is True
                and raw["native_build"]["library_sha256"]
                == build["artifacts"]["libgenerativeqc.so"],
                "actual native force route/core or global policy differs",
            )
        median = statistics.median(s["seconds"] for s in raw["samples"][1:])
        check(math.isfinite(median) and median > 0, "invalid endpoint duration")
        indexed[key] = {"raw": raw, "warm_median": median, "eligible": eligible}
        counts["qualified_populations"] += 1
        counts["qualified_EF"] += 4
    comparisons, summaries = [], []
    for policy in ["tile256", "tile1024"]:
        ratios = []
        for (method, atoms, mode, name), row in indexed.items():
            if name != policy:
                continue
            reference = indexed[(method, atoms, mode, "reference")]
            for sample in row["raw"]["samples"]:
                a = np.asarray(sample["forces"], dtype=np.float64)
                for target in reference["raw"]["samples"]:
                    b = np.asarray(target["forces"], dtype=np.float64)
                    check(
                        a.shape == b.shape == (atoms, 3),
                        "fresh comparison force shape differs",
                    )
                    delta = a - b
                    errors = {
                        "energy_abs_per_atom_hartree": abs(
                            sample["energy"] - target["energy"]
                        )
                        / atoms,
                        "force_max_abs_hartree_per_bohr": float(np.max(np.abs(delta))),
                        "force_rms_hartree_per_bohr": float(
                            np.sqrt(np.mean(delta * delta))
                        ),
                    }
                    check(
                        all(
                            math.isfinite(v) and v <= prereg["numerical_gates"][k]
                            for k, v in errors.items()
                        ),
                        "fresh same-GPU E+F crosscomparison failed",
                    )
                    counts["fresh_reference_crosschecks"] += 1
            ratio = reference["warm_median"] / row["warm_median"]
            if row["eligible"]:
                ratios.append(ratio)
            comparisons.append(
                {
                    "method": method,
                    "atoms": atoms,
                    "mode": mode,
                    "policy": policy,
                    "native_warm_median": row["warm_median"],
                    "reference_warm_median": reference["warm_median"],
                    "reference_over_native": ratio,
                    "primary_input_eligible": row["eligible"],
                    "final_majority_vote": False,
                }
            )
        check(len(ratios) == 30, "fixed complete primary denominator differs")
        summaries.append(
            {
                "policy": policy,
                "fixed_primary_inputs": 30,
                "single_process_ratios_above_1_05": sum(r > 1.05 for r in ratios),
                "complete_primary_geomean": math.exp(
                    math.fsum(math.log(r) for r in ratios) / 30
                ),
            }
        )
    check(
        counts
        == {
            "qualified_populations": 99,
            "qualified_EF": 396,
            "fresh_reference_crosschecks": 1056,
        },
        "complete replay counts differ",
    )
    selected = load("integrated-global-policy-selected-v1.json")
    winner = max(
        summaries,
        key=lambda r: (r["complete_primary_geomean"], r["policy"] == "tile256"),
    )
    check(
        selected["selected_policy_name"] == winner["policy"] == "tile1024",
        "previous whole-policy selection differs",
    )
    print(
        json.dumps(
            {
                "status": "PASS_PUBLIC_ALL99_396_EF_1056_FRESH_CROSSCHECKS_AND_GLOBAL_POLICIES",
                "source_revision": SOURCE,
                **counts,
                "device": campaign["device"],
                "global_policy_summaries": summaries,
                "comparisons": comparisons,
                "scope": manifest["scope"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

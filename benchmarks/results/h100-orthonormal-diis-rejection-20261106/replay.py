"""Bounded standard-library offline replay of the rejected H100 DIIS screen.

No measured producer is executed. No GPU, compiler, solver, network, or Git
subprocess is used; old reference bytes come from the existing public capsule.
"""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATES = {
    "energy_abs_per_atom_hartree": 1e-5,
    "force_max_abs_hartree_per_bohr": 1e-4,
    "force_rms_hartree_per_bohr": 3e-5,
}
DIAGNOSTIC = dict(zip(GATES, (1e-9, 1e-7, 3e-8), strict=True))


def check(value: object, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def original_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def decode(data: bytes, size: int) -> dict | list:
    check(0 < size <= 64 << 20, "Unbounded evidence decode")
    decoder = lzma.LZMADecompressor(memlimit=192 << 20)
    raw = decoder.decompress(data, max_length=size + 1)
    check(
        decoder.eof and not decoder.unused_data and len(raw) == size,
        "Evidence decode differs",
    )
    return json.loads(raw)


def references(binding: dict) -> tuple[list, dict]:
    root = HERE.parent / binding["publication"]
    cohort = json.loads((root / "cohort.json").read_text())
    chunks = []
    for item in cohort["records"]["parts"]:
        data = (root / item["path"]).read_bytes()
        check(
            len(data) == item["bytes"] and sha(data) == item["sha256"],
            "Reference capsule part differs",
        )
        chunks.append(data)
    compressed = b"".join(chunks)
    check(sha(compressed) == cohort["records"]["sha256"], "Reference container differs")
    capsule = decode(compressed, cohort["records"]["decoded_bytes"])
    values = []
    for name, digest in (
        (binding["member"], binding["oracle_sha256"]),
        (binding["reuse_member"], binding["reuse_sha256"]),
    ):
        entry = next(row for row in capsule["files"] if row["path"] == name)
        check(entry["sha256"] == digest, "Original reference binding differs")
        blob = capsule["blobs"][digest]
        data = (
            blob["data"].encode()
            if blob["encoding"] == "utf8"
            else base64.b64decode(blob["data"], validate=True)
        )
        check(
            sha(data) == digest and len(data) == entry["bytes"] == blob["bytes"],
            "Original reference bytes differ",
        )
        values.append(
            json.loads(lzma.decompress(data))
            if name.endswith(".xz")
            else json.loads(data)
        )
    return values[0], values[1]


def vector(sample: dict, atoms: int) -> None:
    check(
        math.isfinite(sample["energy"])
        and len(sample["forces"]) == atoms
        and all(
            len(row) == 3 and all(math.isfinite(v) for v in row)
            for row in sample["forces"]
        ),
        "Incomplete/nonfinite full E/F vector",
    )


def errors(left: dict, right: dict, atoms: int) -> dict:
    vector(left, atoms)
    vector(right, atoms)
    delta = [
        a - b
        for row, other in zip(left["forces"], right["forces"], strict=True)
        for a, b in zip(row, other, strict=True)
    ]
    return dict(
        zip(
            GATES,
            (
                abs(left["energy"] - right["energy"]) / atoms,
                max(map(abs, delta)),
                math.sqrt(math.fsum(v * v for v in delta) / (3 * atoms)),
            ),
            strict=True,
        )
    )


def passes(values: dict, gates: dict = GATES, fraction: float = 1) -> bool:
    return all(
        math.isfinite(values[key]) and values[key] <= fraction * limit
        for key, limit in gates.items()
    )


def independent(record: dict, oracles: list, reuse: dict) -> list[dict]:
    indexed = {(r["method"], r["atoms"], r["backend"]): r for r in oracles}
    target = indexed[(record["method"], 96, "cuda")]
    cpu, gpu = (indexed[(record["method"], 3, backend)] for backend in ("cpu", "cuda"))
    check(
        target["geometry_bohr"] == record["geometry_bohr"]
        and cpu["geometry_bohr"] == gpu["geometry_bohr"],
        "Independent geometry differs",
    )
    for oracle in (target, cpu, gpu):
        check(
            oracle["status"] == "complete"
            and oracle["common_observable_gates"] == GATES
            and oracle["grid_stability_gate_fraction"] == 0.25,
            "Independent reference contract differs",
        )
        identity = sha(
            json.dumps(
                oracle, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        contract = reuse["entries"][identity]
        check(
            contract["source_revision"] == oracle["source_revision"]
            and contract["driver_sha256"] == oracle["driver_sha256"]
            and contract["reference_packages"] == oracle["environment"]["packages"]
            and record["environment"]["packages"] in contract["measurement_packages"]
            and contract["review"].strip()
            and contract["scientific_inputs"]
            == {
                k: record[k]
                for k in (
                    "method",
                    "orbital_basis",
                    "basis_representation",
                    "charge",
                    "spin",
                )
            },
            "Original audited reference reuse contract differs",
        )
        for sample in oracle["samples"]:
            check(
                sample["controls"]
                == {
                    "energy": 1e-11,
                    "orbital_gradient": 1e-7,
                    "screening": 1e-14,
                    "full_density_fock": True,
                    "force_grid_response": None if record["method"] == "hf" else True,
                },
                "Independent solver controls differ",
            )
    for oracle in (target, gpu):
        if len(oracle["samples"]) == 2:
            check(
                passes(errors(*oracle["samples"], oracle["atoms"]), fraction=0.25),
                "Dense grid stability failed",
            )
    cpu_sample = cpu["samples"][-1]
    same = [
        s
        for s in gpu["samples"]
        if s["level"] == cpu_sample["level"]
        and (s.get("nonlocal_grid") or {}).get("level")
        == (cpu_sample.get("nonlocal_grid") or {}).get("level")
    ]
    check(
        len(same) == 1 and passes(errors(cpu_sample, same[0], 3), fraction=0.25),
        "CPU/GPU spot check failed",
    )
    checks = [errors(sample, target["samples"][-1], 96) for sample in record["samples"]]
    check(
        all(passes(value) for value in checks), "Original independent E/F gate failed"
    )
    return checks


def admission_history(evidence: dict) -> dict:
    old, resumed = evidence["original_failed"], evidence["targeted_continuation"]
    check(
        sha(original_json(old["campaign"])) == old["campaign_sha256"]
        and old["campaign"]["status"] == "FAILED_OR_UNVERIFIED_ADMISSION"
        and len(old["campaign"]["runs"]) == 5,
        "Original failed attempt was relabeled",
    )
    check(
        sha(original_json(resumed["campaign"])) == resumed["campaign_sha256"]
        and resumed["campaign"]["status"]
        == "PASS_TWO_PBE0_CONTINUATION_WITH_PRIOR_HF_EVIDENCE_NOT_ACCURACY_OR_TIMING"
        and len(resumed["campaign"]["runs"]) == 2
        and len(resumed["campaign"]["reused_prior_HF_evidence"]["reused_HF"]) == 4,
        "Targeted continuation population differs",
    )
    for group in (old, resumed):
        for index, row in enumerate(group["campaign"]["runs"]):
            name = row["name"]
            raw, plan = group["raw"][name], group["plans"][name]
            check(
                sha(original_json(raw)) == row["raw_sha256"]
                and sha(original_json(plan))
                == row["plan_sha256"]
                == raw["plan_sha256"],
                "Admission receipt differs",
            )
            if group is old and index == 4:
                check(
                    row["returncode"] != 0
                    and "upstream/libxc/7.0.0" in raw["error"]
                    and "energy" not in raw
                    and "forces" not in raw,
                    "Pre-compute PBE0 failure differs",
                )
                continue
            check(
                row["returncode"] == 0
                and raw["status"].startswith("PASS_ADMISSION_ONLY")
                and raw["converged"]
                and raw["executed_backend"] == "cuda"
                and not raw["accuracy_qualified"]
                and not raw["timing_qualified"],
                "Admission was overqualified",
            )
            vector(raw, 6)
            counters = raw["trace"]["orthonormal_diis_residuals"]
            check(
                sum(counters.values()) > 0
                if raw["observed_selector"] == "1"
                else sum(counters.values()) == 0,
                "Admission selector evidence differs",
            )
            if group is old:
                text = evidence["original_HF_traces"][name + ".trace.jsonl"]
                check(
                    sha(text.encode()) == raw["trace"]["trace_sha256"],
                    "Original HF trace differs",
                )
                rows = [json.loads(line) for line in text.splitlines()]
                counts = {
                    kind: sum(
                        r["counters"].get("orthonormal_diis_residuals", 0)
                        for r in rows
                        if r["execution"] == kind
                    )
                    for kind in ("stream", "graph_capture")
                }
                check(
                    all(r["valid"] for r in rows) and counts == counters,
                    "HF capture/stream counters differ",
                )
            else:
                check(
                    plan["declaration"]["method"] == "pbe0", "Continuation repeated HF"
                )
    return {
        "original_attempt": "FAILED_PBE0_ASSET_INPUT_BEFORE_EF",
        "original_HF_reused": 4,
        "fresh_continuation_PBE0": 2,
        "one_original_six_success_attempt": False,
        "independent_accuracy_or_timing_qualification": False,
    }


def replay() -> dict:
    cohort = json.loads((HERE / "cohort.json").read_text())
    item = cohort["evidence"]
    compressed = (HERE / item["path"]).read_bytes()
    check(
        len(compressed) == item["bytes"] and sha(compressed) == item["sha256"],
        "Evidence bytes differ",
    )
    evidence = decode(compressed, item["decoded_bytes"])
    patch_binding = cohort["source_patch"]
    stored_patch = (HERE / patch_binding["stored_path"]).read_bytes()
    check(
        len(stored_patch) == patch_binding["stored_bytes"]
        and sha(stored_patch) == patch_binding["stored_sha256"],
        "Stored candidate patch differs",
    )
    decoder = lzma.LZMADecompressor(memlimit=192 << 20)
    patch = decoder.decompress(stored_patch, max_length=patch_binding["bytes"] + 1)
    check(decoder.eof and not decoder.unused_data, "Candidate patch decode differs")
    check(
        len(patch) == cohort["source_patch"]["bytes"]
        and sha(patch) == cohort["source_patch"]["sha256"],
        "Candidate patch differs",
    )
    check(
        evidence["decision"] == "REJECT_NOT_PROMOTABLE"
        and not evidence["production_changed"],
        "Rejection differs",
    )
    pilot = evidence["pilot"]
    campaign = pilot["campaign"]
    check(
        sha(original_json(campaign)) == cohort["campaign_sha256"]
        and campaign["returned_EF"] == 24
        and campaign["status"] == "COMPLETE_RAW_COHORT_OFFLINE_REDUCTION_PENDING"
        and campaign["config"]["rounds"] == 1,
        "Pilot cohort differs",
    )
    check(
        campaign["config"]["baseline"]["execution_controls"]
        == {"GENERATIVEQC_SCF_ORTHONORMAL_DIIS": "0"}
        and campaign["config"]["candidate"]["execution_controls"]
        == {"GENERATIVEQC_SCF_ORTHONORMAL_DIIS": "1"},
        "This is an explicit selector experiment, not public defaults",
    )
    for name, digest in campaign["producers"].items():
        check(
            sha(evidence["producer_sources"][name].encode()) == digest,
            "Executed producer differs",
        )
    check(
        campaign["matched_comparison_policy"]["threshold_role"] == "diagnostic_only"
        and not campaign["matched_comparison_policy"]["is_hard_gate"],
        "Pre-data DIIS amendment differs",
    )
    oracles, reuse = references(evidence["reference_reuse"])
    index, checks = {}, []
    expected = [
        (method, variant)
        for method in ("wb97m-v", "hf")
        for variant in ("baseline", "candidate", "gpu4pyscf")
    ]
    check(
        [
            (r["declaration"]["method"], r["declaration"]["variant"])
            for r in campaign["runs"]
        ]
        == expected,
        "One-round process ordering/population differs",
    )
    for row in campaign["runs"]:
        name = row["name"]
        raw, plan = pilot["records"][name], pilot["plans"][name]
        check(
            row["returncode"] == 0 and row["qualification"] == "qualified",
            "Pilot process did not qualify",
        )
        check(
            sha(original_json(raw)) == row["raw_sha256"]
            and sha(original_json(plan)) == row["plan_sha256"] == raw["plan_sha256"],
            "Original pilot receipt bytes differ",
        )
        check(
            raw["source_revision"] == cohort["source_revision"]
            and raw["mode"] == "direct"
            and raw["atoms"] == 96
            and raw["status"] == "complete"
            and raw["orbital_basis"] == "def2-SVP"
            and raw["basis_representation"] == "spherical"
            and raw["spin"] == raw["charge"] == 0
            and raw["warm_repeats"] == 3
            and len(raw["samples"]) == 4,
            "Scientific endpoint identity differs",
        )
        check(
            raw["accelerator"]["nvidia_smi"]["uuid"] == cohort["gpu_uuid"]
            and raw["accelerator"]["compute_capability"] == [9, 0]
            and raw["runner_sha256"]
            == campaign["producers"]["measurement-endpoint.py"],
            "GPU/producer differs",
        )
        check(
            raw["protocol"]["common_observable_gates"] == GATES
            and not raw["protocol"]["trace_enabled"]
            and not raw["protocol"]["oracle_seed_used"]
            and raw["protocol"]["analytic_forces"],
            "Timing/oracle policy differs",
        )
        check(
            raw["runtime_assets"] == plan["runtime_assets"]
            and raw["runtime_assets"]["manifest_sha256"]
            == evidence["runtime_source"]["manifest_sha256"]
            and raw["compiler_asset_resolution"]["upstream/libxc/7.0.0"]
            == plan["source_root"] + "/upstream/libxc/7.0.0",
            "Bound runtime assets/import root differ",
        )
        native = raw["arm"] == "standard"
        if native:
            check(
                raw["native_source_identity"] == cohort["native_source_identity"]
                and raw["native_build"]["library_sha256"] == cohort["core_sha256"],
                "Native source/core differs",
            )
            core = raw["native_build"]["library_path"]
            images = raw["runtime_images_after_endpoint"]
            check(
                core in images and images[core]["sha256"] == cohort["core_sha256"],
                "Mapped core differs",
            )
            expected_images = {
                str(Path(plan["library_root"]) / b["resolved_relative_path"]): {
                    "bytes": b["bytes"],
                    "sha256": b["sha256"],
                }
                for n, b in plan["build_artifact_bindings"].items()
                if n.endswith(".so")
            }
            check(
                all(
                    expected_images.get(name) == binding
                    for name, binding in images.items()
                    if Path(name).name.startswith("libgenerativeqc")
                ),
                "Mapped native/AOT image differs",
            )
        for i, sample in enumerate(raw["samples"]):
            vector(sample, 96)
            check(
                (sample["phase"], sample["repeat"])
                == (("cold", 0) if i == 0 else ("warm", i - 1))
                and sample["converged"]
                and math.isfinite(sample["seconds"])
                and sample["seconds"] > 0
                and sample["warm_start_used"] == bool(i),
                "Endpoint phases/own-cold replay differ",
            )
            key = "fock_builds" if native else "effective_potential_builds"
            check(
                type(sample[key]) is int
                and sample[key] >= 1
                and sample["semantic_work"][key] == sample[key],
                "SCF work missing",
            )
            if native:
                precision = sample["precision_validation"]
                check(
                    precision["profile"] == "standard"
                    and precision["solves"]
                    and precision["solve_count"] == len(precision["solves"])
                    and not sample["warm_start_fallback"],
                    "Executed Standard solve/warm ownership differs",
                )
        checks.extend(independent(raw, oracles, reuse))
        index[(raw["method"], plan["variant"])] = raw
    cases = []
    for method in ("wb97m-v", "hf"):
        off, on, reference = (
            index[(method, name)] for name in ("baseline", "candidate", "gpu4pyscf")
        )
        check(
            all(
                a["settings"] == b["settings"]
                for a, b in zip(off["samples"], on["samples"], strict=True)
            ),
            "Native settings differ",
        )
        matched = [
            errors(a, b, 96) for a, b in zip(on["samples"], off["samples"], strict=True)
        ]
        flags = [passes(value, DIAGNOSTIC) for value in matched]
        original = next(
            r
            for r in pilot["original_reduction"]["results"]
            if r["case_id"].startswith(method + "-water")
        )["pairs"][0]
        check(
            flags == [r["passed"] for r in original["matched_errors"]],
            "Original diagnostic misses were relabeled",
        )
        warm = [
            a["seconds"] / b["seconds"]
            for a, b in zip(on["samples"][1:], off["samples"][1:], strict=True)
        ]
        warm_median = statistics.median(
            s["seconds"] for s in on["samples"][1:]
        ) / statistics.median(s["seconds"] for s in off["samples"][1:])
        check(all(v <= 1.05 for v in warm) and warm_median <= 1.05, "Warm guard failed")
        cases.append(
            {
                "method": method,
                "cold_seconds": {
                    name: index[(method, name)]["samples"][0]["seconds"]
                    for name in ("baseline", "candidate", "gpu4pyscf")
                },
                "cold_candidate_over_baseline": on["samples"][0]["seconds"]
                / off["samples"][0]["seconds"],
                "cold_reference_over_candidate": reference["samples"][0]["seconds"]
                / on["samples"][0]["seconds"],
                "native_cold_iterations": [
                    r["samples"][0]["iterations"] for r in (off, on)
                ],
                "native_cold_fock_builds": [
                    r["samples"][0]["fock_builds"] for r in (off, on)
                ],
                "warm_candidate_over_baseline": warm_median,
                "warm_each_call_ratios": warm,
                "warm_guard_pass": True,
                "native_native_diagnostic_pass_flags": flags,
                "original_native_native_errors": original["matched_errors"],
            }
        )
    check(
        len(checks) == 24
        and cases[0]["native_cold_fock_builds"] == [18, 18]
        and cases[1]["native_cold_fock_builds"] == [11, 11],
        "Pilot semantic work differs",
    )
    return {
        "decision": "REJECT_NOT_PROMOTABLE",
        "production_changed": False,
        "processes": 6,
        "complete_EF": 24,
        "independent_accuracy_pass": True,
        "warm_guards_pass": True,
        "rounds": 1,
        "three_rounds_qualified": False,
        "all30_goal_achieved": False,
        "selected_policy_is_public_default": False,
        "source_revision": cohort["source_revision"],
        "native_source_identity": cohort["native_source_identity"],
        "core_sha256": cohort["core_sha256"],
        "gpu_uuid": cohort["gpu_uuid"],
        "worst_independent_errors": {key: max(r[key] for r in checks) for key in GATES},
        "cases": cases,
        "admission_history": admission_history(evidence["admission"]),
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))

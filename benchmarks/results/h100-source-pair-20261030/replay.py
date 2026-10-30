"""Offline full-vector replay of clean source pairs and separate diagnostics."""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks.df_component_ledger import validate_record
from benchmarks.readme_defaults import COMMON_GATES
from benchmarks.readme_defaults_qualification import identity, qualify


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
        manifest["schema"] == "generativeqc.h100-source-pair-evidence.v1",
        "unexpected manifest",
    )
    check(
        manifest["sources"]
        == {
            "base": "15ab66a68a061355fda833a93e88484dbfee6360",
            "candidate": "2241fa6a76698c09573c540b2822316ef7cdb908",
        },
        "measured source pair differs",
    )
    for name, binding in manifest["files"].items():
        data = (HERE / name).read_bytes()
        check(
            {"bytes": len(data), "sha256": digest(data)} == binding,
            "stored bytes differ",
        )
    payload = lzma.decompress((HERE / "observations.json.xz").read_bytes())
    check(b"/global_user/" not in payload, "private storage path in public JSON")
    bundle = json.loads(payload)
    manifest["original_json_bindings"] = json.loads(
        lzma.decompress((HERE / manifest["original_json_bindings_path"]).read_bytes())
    )
    force_storage = manifest["force_storage"]
    force_values = np.load(
        io.BytesIO(lzma.decompress((HERE / force_storage["path"]).read_bytes())),
        allow_pickle=False,
    )
    check(
        force_values.dtype.str == force_storage["dtype"] == "<f8"
        and list(force_values.shape) == force_storage["shape"]
        and np.isfinite(force_values).all(),
        "invalid full-precision force storage",
    )
    next_force = 0
    for cohort in bundle["cohorts"].values():
        for population in cohort["populations"]:
            for sample in population["record"]["samples"]:
                descriptor = sample["forces"]
                begin, atoms = descriptor["array_begin"], descriptor["atoms"]
                check(
                    type(begin) is int
                    and type(atoms) is int
                    and atoms == population["record"]["atoms"]
                    and begin == next_force
                    and begin + 3 * atoms <= force_values.size,
                    "missing/overlapping force coordinates",
                )
                next_force = begin + 3 * atoms
                sample["forces"] = (
                    force_values[begin:next_force].reshape(atoms, 3).tolist()
                )
    check(next_force == force_values.size, "unused force coordinates")
    checked_bindings = set()

    def bind(path: str, value: object) -> None:
        check(path not in checked_bindings, "duplicate original JSON binding")
        check(
            digest(canonical(value))
            == manifest["original_json_bindings"][path]["normalized_sha256"],
            "normalized JSON bytes differ",
        )
        checked_bindings.add(path)

    for key, path in [
        ("build", "build-pair-completed.json"),
        ("inputs", "inputs-default.json"),
        ("preregistration", "pbe0-pairs-preregistered-v2.json"),
        ("reference_reuse_audit", "reference-reuse-scientific-audit.json"),
    ]:
        bind(path, bundle[key])
    built = bundle["build"]
    check(
        built["status"] == "PASS_SOURCE_PAIR_BUILD_AND_SM90_AOT_PACKAGE_AUDIT",
        "unqualified build",
    )
    check(
        {k: v["revision"] for k, v in built["sources"].items()} == manifest["sources"],
        "build/source pair differs",
    )
    for name in ["oracle", "auxiliary_inputs"]:
        check(
            digest((ROOT / manifest[name]["path"]).read_bytes())
            == manifest[name]["sha256"],
            "independent fixture bytes differ",
        )
    oracles = json.loads(
        lzma.decompress((ROOT / manifest["oracle"]["path"]).read_bytes())
    )
    inputs = json.loads(
        lzma.decompress((ROOT / manifest["auxiliary_inputs"]["path"]).read_bytes())
    )
    input_hashes = {name: digest(canonical(value)) for name, value in inputs.items()}
    check(
        input_hashes == {n: v["sha256"] for n, v in bundle["inputs"]["files"].items()},
        "exact auxiliary exports differ",
    )
    producers = {}
    for name, binding in manifest["producers"].items():
        data = lzma.decompress((HERE / binding["path"]).read_bytes())
        check(
            len(data) == binding["bytes"] and digest(data) == binding["sha256"],
            "executed producer bytes differ",
        )
        producers[name] = data
    trace_stream = lzma.decompress((HERE / "traces.jsonl.xz").read_bytes())
    traces = {}
    previous_end = 0
    check(
        set(manifest["trace_segments"]) == set(manifest["traces"]),
        "missing trace segments",
    )
    for name, segment in manifest["trace_segments"].items():
        begin, end = segment["begin"], segment["end"]
        check(
            type(begin) is int
            and type(end) is int
            and begin == previous_end <= end <= len(trace_stream),
            "missing/overlapping trace bytes",
        )
        data = trace_stream[begin:end]
        check(
            {"bytes": len(data), "sha256": digest(data)} == manifest["traces"][name],
            "original trace bytes differ",
        )
        traces[name] = data
        previous_end = end
    check(previous_end == len(trace_stream), "unused trace bytes")
    paired = {}
    comparisons = []
    diagnostics = []
    cohort_counts = {}
    pilot_cases = {
        ("hf", 3, "direct"),
        ("pbe0", 3, "df"),
        ("pbe0", 48, "df"),
        ("pbe0", 96, "df"),
        ("pbe0", 96, "direct"),
    }
    diagnostic_cases = [
        (48, "df", "base", "standard", "default"),
        (48, "df", "candidate", "standard", "default"),
        (96, "df", "base", "standard", "default"),
        (96, "df", "candidate", "standard", "default"),
        (96, "df", "candidate", "fast", "default"),
        (96, "df", "candidate", "standard", "projection-off"),
        (96, "direct", "base", "standard", "default"),
        (96, "direct", "candidate", "standard", "default"),
    ]
    for folder, population_count, status in [
        ("pilot-v1", 15, "COMPLETE_PILOT"),
        ("pbe0-pairs-v1", 60, "COMPLETE_SOURCE_PAIR_MATRIX"),
        ("df-trace-v1", 8, "COMPLETE_INTRUSIVE_NUMERICAL_DIAGNOSTICS"),
    ]:
        cohort = bundle["cohorts"][folder]
        campaign = cohort["campaign"]
        bind(folder + "/campaign.json", campaign)
        bind(folder + "/reference-reuse-v2.json", cohort["reference_reuse"])
        check(
            campaign["status"] == status
            and len(campaign["runs"]) == len(cohort["populations"]) == population_count,
            "incomplete population",
        )
        check(
            campaign["build_receipt_sha256"]
            == manifest["original_json_bindings"]["build-pair-completed.json"][
                "sha256"
            ],
            "original build binding differs",
        )
        if folder != "df-trace-v1":
            runtime_path = (
                "reference-runtime-pilot.json"
                if folder == "pilot-v1"
                else folder + "/reference-runtime.json"
            )
            bind(runtime_path, cohort["reference_runtime"])
            runtime = cohort["reference_runtime"]
            check(
                runtime["status"] == "PASS_REFERENCE_RUNTIME_AND_CUTENSOR_ONLY"
                and campaign["device"]["uuid"] in runtime["device"],
                "actual reference runtime/device differs",
            )
            check(
                campaign["reference_runtime_sha256"]
                == manifest["original_json_bindings"][runtime_path]["sha256"],
                "runtime receipt binding differs",
            )
        if folder == "pbe0-pairs-v1":
            check(
                campaign["preregistration_sha256"]
                == manifest["original_json_bindings"][
                    "pbe0-pairs-preregistered-v2.json"
                ]["sha256"],
                "preregistered order binding differs",
            )
            check(
                campaign["allocation_cgroups"]
                == {"cpu.max": "2000000 100000", "memory.max": "214748364800"}
                and campaign["python_optimization"] == 0
                and campaign["shared_memory_bytes"] == 16 * (1 << 30),
                "allocation/producer controls differ",
            )
        seen = set()
        calls = 0
        for index, (stage, pop) in enumerate(
            zip(campaign["runs"], cohort["populations"], strict=True)
        ):
            stem = pop["id"]
            raw, plan = pop["record"], pop["plan"]
            for suffix, value in [
                (".json", raw),
                (".plan.json", plan),
                (".qualification.json", pop["qualification"]),
            ]:
                bind(folder + "/" + stem + suffix, value)
            check(
                stage["case"] == stem
                and stage["status"] == "COMPLETE"
                and stage.get("returncode", stage.get("exit_code")) == 0,
                "failed/incomplete process",
            )
            original_raw = manifest["original_json_bindings"][
                folder + "/" + stem + ".json"
            ]
            original_plan = manifest["original_json_bindings"][
                folder + "/" + stem + ".plan.json"
            ]
            check(
                stage["receipt_sha256"] == original_raw["sha256"]
                and stage["plan_sha256"]
                == raw["plan_sha256"]
                == original_plan["sha256"],
                "original raw/plan binding differs",
            )
            if folder == "pilot-v1":
                label = stem.rsplit("-", 1)[-1]
                key = (raw["method"], raw["atoms"], raw["mode"], label)
                check(
                    key[:3] in pilot_cases
                    and label in ["base", "candidate", "gpu4pyscf"],
                    "unexpected pilot case",
                )
                check(
                    stem == f"{key[0]}-{key[1]}-{key[2]}-{label}", "pilot label differs"
                )
            elif folder == "pbe0-pairs-v1":
                label = stage["source_label"]
                key = (raw["atoms"], raw["mode"], label, raw["arm"], stage["process"])
                declared = bundle["preregistration"]["order"][index]
                check(
                    key
                    == tuple(
                        declared[k]
                        for k in ["atoms", "mode", "source_label", "arm", "process"]
                    ),
                    "process execution order differs",
                )
                check(
                    raw["method"] == "pbe0"
                    and stem == f"pbe0-{key[0]}-{key[1]}-{key[2]}-{key[3]}-p{key[4]}",
                    "paired label differs",
                )
            else:
                atoms, mode, label, arm, variant = diagnostic_cases[index]
                key = diagnostic_cases[index]
                check(
                    stem == f"pbe0-{atoms}-{mode}-{label}-{arm}-{variant}"
                    and (raw["method"], raw["atoms"], raw["mode"], raw["arm"])
                    == ("pbe0", atoms, mode, arm),
                    "diagnostic label differs",
                )
                check(
                    plan["diagnostic_overrides"]
                    == (
                        {"GENERATIVEQC_DF_FINAL_PROJECTION": "off"}
                        if variant == "projection-off"
                        else {}
                    ),
                    "diagnostic control differs",
                )
                check(
                    raw["native_prepared_force_capabilities"]
                    == stage["native_prepared_force_capabilities"]
                    == [True] * 4,
                    "native prepared capability missing",
                )
            check(key not in seen, "duplicate scientific population")
            seen.add(key)
            which = "base" if label == "gpu4pyscf" else label
            check(
                raw["source_revision"] == built["sources"][which]["revision"]
                and raw["driver_sha256"] == built["sources"][which]["driver_sha256"],
                "source/driver identity differs",
            )
            adapter = "trace-endpoint.py" if folder == "df-trace-v1" else "endpoint.py"
            check(
                raw["runner_sha256"]
                == plan["endpoint_sha256"]
                == digest(producers[adapter]),
                "executed adapter differs",
            )
            check(
                raw["geometry_bohr"] == plan["geometry_bohr"]
                and plan["input_sha256"] == input_hashes,
                "scientific inputs differ",
            )
            check(
                raw["accelerator"]["nvidia_smi"]["uuid"] == campaign["device"]["uuid"]
                and raw["slurm_job_id"] is None
                and raw["scheduler"]["kind"] == "qz_job",
                "actual device/scheduler differs",
            )
            check(
                all(value == "1" for value in raw["thread_controls"].values()),
                "thread policy differs",
            )
            if label != "gpu4pyscf":
                build = next(s for s in built["stages"] if s["name"] == label)
                check(
                    raw["native_build"]["library_sha256"]
                    == build["artifacts"]["libgenerativeqc.so"]
                    and plan["build_artifacts"] == build["artifacts"],
                    "core/AOT bytes differ",
                )
            qualification = qualify(
                [raw], oracles, reference_reuse=cohort["reference_reuse"]
            )[0]
            # The declared private-prefix replacement changes the canonical
            # receipt identity. Every other qualifier field must replay exactly.
            expected_qualification = dict(pop["qualification"])
            expected_qualification["receipt_sha256"] = identity(raw)
            check(
                qualification == expected_qualification
                and qualification["status"] == "qualified",
                "qualifier differs or fails: " + stem,
            )
            oracle = next(
                o
                for o in oracles
                if (o["method"], o["atoms"], o["backend"])
                == (raw["method"], raw["atoms"], "cuda")
            )
            check(
                oracle["geometry_bohr"] == raw["geometry_bohr"]
                and raw["protocol"]["common_observable_gates"] == COMMON_GATES,
                "oracle geometry/gates differ",
            )
            check(
                len(raw["samples"]) == 4
                and [s["phase"] for s in raw["samples"]]
                == ["cold", "warm", "warm", "warm"],
                "missing/reordered E/F calls",
            )
            target = oracle["samples"][-1]
            trace_ids = set()
            previous = 0
            observed = []
            for sample in raw["samples"]:
                differences = [
                    x - y
                    for a, b in zip(sample["forces"], target["forces"], strict=True)
                    for x, y in zip(a, b, strict=True)
                ]
                check(
                    len(differences) == 3 * raw["atoms"]
                    and all(math.isfinite(v) for v in differences),
                    "invalid force vector",
                )
                errors = {
                    "energy_abs_per_atom_hartree": abs(
                        sample["energy"] - target["energy"]
                    )
                    / raw["atoms"],
                    "force_max_abs_hartree_per_bohr": max(map(abs, differences)),
                    "force_rms_hartree_per_bohr": math.sqrt(
                        math.fsum(v * v for v in differences) / len(differences)
                    ),
                }
                check(
                    all(
                        math.isfinite(errors[k]) and errors[k] <= limit
                        for k, limit in COMMON_GATES.items()
                    ),
                    "independent E/F gate failed",
                )
                check(
                    sample["phase"] == "cold" or sample["warm_start_used"],
                    "warm density-seed protocol differs",
                )
                calls += 1
                if folder == "df-trace-v1":
                    data = traces[stem + ".cuda.jsonl"]
                    bounds = sample["diagnostic_trace_byte_ranges"][
                        "GENERATIVEQC_DF_TRACE"
                    ]
                    check(
                        type(bounds["begin"]) is int
                        and type(bounds["end"]) is int
                        and previous == bounds["begin"] <= bounds["end"] <= len(data),
                        "invalid/noncontiguous component boundary",
                    )
                    previous = bounds["end"]
                    segment = data[bounds["begin"] : bounds["end"]]
                    check(segment.endswith(b"\n"), "partial component record")
                    components = [json.loads(line) for line in segment.splitlines()]
                    for component in components:
                        validate_record(component)
                        check(
                            component["id"] not in trace_ids,
                            "duplicate component operation",
                        )
                        trace_ids.add(component["id"])
                    responses = [
                        r
                        for r in components
                        if r["operation"] == "force_response"
                        and r["execution"] == "stream"
                        and r["counters"].get("atom_coordinates") == 3 * raw["atoms"]
                    ]
                    reused = sum(
                        r["counters"].get("response_final_fitted_projection_reused", 0)
                        for r in responses
                    )
                    check(
                        len(responses) == (2 if mode == "df" else 0)
                        and reused
                        == int(
                            mode == "df"
                            and label == "candidate"
                            and variant == "default"
                        ),
                        "completed response/reuse differs",
                    )
                    grouped = []
                    for response in responses:
                        times = defaultdict(float)
                        for region in response["regions"]:
                            if not region["name"].startswith("shell_"):
                                times[region["name"]] += region["gpu_ms"]
                        grouped.append(
                            {
                                "inclusive_gpu_ms_by_name": dict(times),
                                "counters": response["counters"],
                            }
                        )
                    observed.append(
                        {
                            "phase": sample["phase"],
                            "repeat": sample["repeat"],
                            "completed_projection_reuse": reused,
                            "responses": grouped,
                        }
                    )
            if folder == "pbe0-pairs-v1":
                paired[key] = statistics.median(
                    s["seconds"] for s in raw["samples"][1:]
                )
            elif folder == "df-trace-v1":
                check(
                    previous == len(traces[stem + ".cuda.jsonl"]),
                    "unattributed CUDA operations",
                )
                host_data = traces[stem + ".host.jsonl"]
                check(host_data.endswith(b"\n"), "partial host trace")
                host_ids = set()
                for record in (json.loads(line) for line in host_data.splitlines()):
                    check(
                        record["schema"] == "generativeqc.df_host_trace"
                        and record["version"] == 1
                        and record["valid"] is True
                        and record["id"] not in host_ids,
                        "invalid host record",
                    )
                    host_ids.add(record["id"])
                    for region in record["regions"]:
                        check(
                            region["finished"] is True
                            and region["failed"] is False
                            and all(
                                math.isfinite(region[k]) and region[k] >= 0
                                for k in ["wall_ms", "cpu_ms"]
                            ),
                            "unfinished/invalid host region",
                        )
                check(
                    stage["component_summary_sha256"]
                    == pop["original_component_summary_sha256"],
                    "original component summary binding differs",
                )
                for env, suffix in [
                    ("GENERATIVEQC_DF_TRACE", ".cuda.jsonl"),
                    ("GENERATIVEQC_DF_HOST_TRACE", ".host.jsonl"),
                ]:
                    check(
                        pop["trace_bindings"][env] == manifest["traces"][stem + suffix],
                        "trace original binding differs",
                    )
                check(
                    pop["trace_bindings"]["GENERATIVEQC_DF_PROGRESS_TRACE"]
                    == manifest["omitted_transient_progress"][stem + ".progress.jsonl"],
                    "private progress binding differs",
                )
                diagnostics.append({"case": stem, "phases": observed})
        if folder == "pilot-v1":
            check(
                seen
                == {
                    (*case, label)
                    for case in pilot_cases
                    for label in ["base", "candidate", "gpu4pyscf"]
                },
                "missing pilot case",
            )
        if folder == "pbe0-pairs-v1":
            check(
                seen
                == {
                    (atoms, mode, label, arm, process)
                    for atoms, mode in [
                        (48, "df"),
                        (96, "df"),
                        (48, "direct"),
                        (96, "direct"),
                    ]
                    for process in range(3)
                    for label, arm in [
                        ("base", "standard"),
                        ("candidate", "standard"),
                        ("gpu4pyscf", "gpu4pyscf"),
                        ("base", "fast"),
                        ("candidate", "fast"),
                    ]
                },
                "missing paired case",
            )
        cohort_counts[folder] = {
            "populations": len(seen),
            "qualified_EF_calls": calls,
            "uuid": campaign["device"]["uuid"],
        }
    check(
        checked_bindings == set(manifest["original_json_bindings"]),
        "unused/missing original JSON binding",
    )
    for atoms, mode in [(48, "df"), (96, "df"), (48, "direct"), (96, "direct")]:
        for arm in ["standard", "fast"]:
            base = [paired[(atoms, mode, "base", arm, p)] for p in range(3)]
            candidate = [paired[(atoms, mode, "candidate", arm, p)] for p in range(3)]
            reference = [
                paired[(atoms, mode, "gpu4pyscf", "gpu4pyscf", p)] for p in range(3)
            ]
            ratios = [r / c for r, c in zip(reference, candidate, strict=True)]
            comparisons.append(
                {
                    "atoms": atoms,
                    "mode": mode,
                    "arm": arm,
                    "base_process_medians": base,
                    "candidate_process_medians": candidate,
                    "reference_process_medians": reference,
                    "median_base_over_candidate": statistics.median(
                        b / c for b, c in zip(base, candidate, strict=True)
                    ),
                    "reference_over_candidate_pairs": ratios,
                    "median_reference_over_candidate": statistics.median(ratios),
                    "passes_reference_win_rule": statistics.median(ratios) > 1.05
                    and min(ratios) > 1,
                }
            )
    print(
        json.dumps(
            {
                "status": "PASS_83_POPULATIONS_332_EF_AND_COMPLETED_COMPONENT_REPLAY",
                "scope": "75 clean populations/300 E+F plus 8 separate intrusive populations/32 E+F. Four affected physical PBE0 cases only; profiles/repeats add no votes. Not full30-case acceptance. Transient progress is not published or replayed by this reader. Inclusive GPU-event and host intervals overlap; do not add them or infer hardware occupancy.",
                "cohorts": cohort_counts,
                "comparisons": comparisons,
                "diagnostics": diagnostics,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

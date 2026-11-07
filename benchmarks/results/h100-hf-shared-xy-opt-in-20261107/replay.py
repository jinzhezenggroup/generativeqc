"""Audit retained HF opt-in arrays/provenance; never execute CUDA or an oracle."""

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
REVISION = "a694beec91645e0c7002bcb84976e2041ae02fa3"
IDENTITY = "fea322c023b121519d3b35b50ed92fa022f6096ccdeed3b5b1fcf20f9598bc18"
CORE = "136cc845ed20041978dd6f98bafdd2610b84c93a40724298a8573f8493545443"
GPU = "GPU-2b5c4936-338e-88e5-49e3-e4d3cceccf52"
ORACLE = "d61a3e8ba0a8f8a8c08ee4c4b30c8e3e397e67863b8d75e7ba974e90cf0f85a4"
CONTROL = "GENERATIVEQC_CUDA_HF_SHARED_RYS_XY"
STRICT = {
    "energy_abs_per_atom_hartree": 1e-9,
    "force_max_abs_hartree_per_bohr": 1e-7,
    "force_rms_hartree_per_bohr": 3e-8,
}
DENSE = {
    "energy_abs_per_atom_hartree": 1e-5,
    "force_max_abs_hartree_per_bohr": 1e-4,
    "force_rms_hartree_per_bohr": 3e-5,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode(cohort: dict[str, Any], compressed: bytes) -> dict[str, bytes]:
    require(
        len(compressed) == cohort["capsule"]["bytes"]
        and sha(compressed) == cohort["capsule"]["sha256"],
        "capsule bytes differ",
    )
    require(len(compressed) < 1 << 20, "oversized capsule")
    stream = lzma.LZMADecompressor()
    decoded = stream.decompress(compressed, max_length=2 << 20)
    require(
        stream.eof
        and not stream.unused_data
        and len(decoded) == cohort["capsule"]["decoded_json_bytes"],
        "capsule extent differs",
    )
    capsule = json.loads(decoded)
    require(
        capsule["schema"] == "generativeqc.hf-xy-pilot-capsule.v1"
        and capsule["pilot_original_file_count"] == 23
        and capsule["pilot_original_bytes"] == 286045,
        "pilot inventory differs",
    )
    records = {}
    for name, row in capsule["files"].items():
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            "unsafe member",
        )
        data = base64.b64decode(row["base64"], validate=True)
        require(
            len(data) == row["bytes"] and sha(data) == row["sha256"],
            "member hash differs",
        )
        records[name] = data
    pilot = [v for k, v in records.items() if k.startswith("pilot/")]
    require(
        len(pilot) == 23 and sum(map(len, pilot)) == 286045,
        "original pilot bytes omitted",
    )
    require(
        len(records) == cohort["capsule"]["original_payload_count"]
        and sum(map(len, records.values()))
        == cohort["capsule"]["original_payload_bytes"],
        "support payload inventory differs",
    )
    return records


def errors(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    require(
        len(first["forces"]) == len(second["forces"]) == 96
        and all(len(row) == 3 for row in first["forces"] + second["forces"]),
        "full96x3 force shape differs",
    )
    values = [first["energy"], second["energy"]]
    values += [v for sample in (first, second) for row in sample["forces"] for v in row]
    require(all(math.isfinite(v) for v in values), "nonfinite E/F")
    delta = [
        x - y
        for a, b in zip(first["forces"], second["forces"], strict=True)
        for x, y in zip(a, b, strict=True)
    ]
    return {
        "energy_abs_per_atom_hartree": abs(first["energy"] - second["energy"]) / 96,
        "force_max_abs_hartree_per_bohr": max(map(abs, delta)),
        "force_rms_hartree_per_bohr": math.sqrt(math.fsum(v * v for v in delta) / 288),
    }


def within(value: dict[str, float], gates: dict[str, float]) -> None:
    require(
        all(math.isfinite(v) and v <= gates[k] for k, v in value.items()),
        "numerical gate failed",
    )


def same_errors(actual: dict[str, float], reported: dict[str, float]) -> None:
    require(
        all(
            math.isclose(v, reported[k], rel_tol=2e-12, abs_tol=1e-18)
            for k, v in actual.items()
        ),
        "reported numerical reduction differs",
    )


def audit(cohort: dict[str, Any], records: dict[str, bytes]) -> dict[str, Any]:
    source = cohort["measured_source"]
    require(
        source["revision"] == REVISION
        and source["native_identity"] == IDENTITY
        and source["core_sha256"] == CORE
        and source["gpu_uuid"] == GPU,
        "measured identity differs",
    )
    require(
        cohort["scope"]
        == {
            "default": "off",
            "publication_tree_device_measured": False,
            "all30_complete": False,
            "timing_population": "one fixed-order off/on cold pair plus three warm calls per owner",
            "reader": "retained full E/F array and provenance replay; no CUDA/SCF/oracle execution",
        },
        "scope promotion refused",
    )
    get = lambda name: json.loads(records[name])
    receipt = get("pilot/result.json")
    build = get("support/build-result.json")
    contract = get("support/source-contract.json")
    authority = get("support/endpoint-review.json")
    require(
        receipt["status"]
        == cohort["preserved_pilot_status"]
        == "INTEGRATION_PILOT_CORRECTNESS_PASS_PENDING_TIMING_DECISION"
        and authority["status"] == "PASS_FOR_DEFAULT_OFF_OPT_IN_PUBLICATION",
        "execution/review status differs",
    )
    require(
        sha(records["pilot/result.json"]) == authority["pilot_receipt_sha256"]
        and sha(records["support/build-result.json"]) == receipt["build_receipt_sha256"]
        and sha(records["support/source-contract.json"])
        == receipt["source_contract_sha256"]
        and sha(records["support/config.json"]) == receipt["config_sha256"]
        and sha(records["support/dispatch-admission.json"])
        == receipt["dispatch_admission_sha256"],
        "execution provenance chain differs",
    )
    require(
        contract["revision"] == REVISION
        and contract["tree"] == source["tree"]
        and contract["native_source_identity"]
        == build["native_source_identity"]
        == IDENTITY
        and build["artifacts"]["libgenerativeqc.so"] == CORE
        and authority["core_sha256"] == CORE
        and authority["gpu"]["uuid"] == GPU,
        "source/core/device binding differs",
    )
    require(
        receipt["order"] == ["xy_off", "xy_on"]
        and receipt["native_selector"] == 0
        and receipt["complete_EF"] == 8
        and receipt["whole_goal_complete"] is False
        and receipt["reference_processes"]
        == receipt["new_reference_or_EF_oracle_runs"]
        == 0
        and receipt["strict_native_gates"] == STRICT,
        "pilot protocol differs",
    )
    require(
        sha(records["support/pilot-producer.py"]) == receipt["runner_sha256"],
        "pilot producer differs",
    )
    require(
        GPU.encode() in records["pilot/gpu-before.stdout"], "actual GPU UUID differs"
    )
    for stage in receipt["stages"]:
        require(stage["status"] == "PASS" and stage["returncode"] == 0, "failed stage")
        for kind in ("stdout", "stderr"):
            data = records["pilot/" + stage["name"] + "." + kind]
            require(
                sha(data) == stage[kind + "_sha256"], "stage diagnostic omitted/changed"
            )
    require(len(receipt["stages"]) == 8, "stage population differs")
    require(
        sha(records["support/oracles.json.xz"])
        == receipt["original_oracle_sha256"]
        == ORACLE,
        "original independent oracle bytes differ",
    )
    oracles = json.loads(lzma.decompress(records["support/oracles.json.xz"]))
    selected = [r for r in oracles if r["method"] == "hf" and r["atoms"] == 96]
    require(
        len(selected) == 1 and len(selected[0]["samples"]) == 1,
        "dense reference selection differs",
    )
    reference = selected[0]["samples"][0]
    require(selected[0]["common_observable_gates"] == DENSE, "original gates differ")
    dense_errors, raw_records, plans = [], [], []
    for arm, value in [("xy_off", "0"), ("xy_on", "1")]:
        raw = get("pilot/" + arm + "/raw.json")
        plan = get("pilot/" + arm + "/plan.json")
        qualification = get("pilot/" + arm + "/qualification.json")
        row = next(
            r for r in authority["raw_and_qualifications"] if r["variant"] == arm
        )
        require(
            sha(records["pilot/" + arm + "/raw.json"]) == row["raw_sha256"]
            and sha(records["pilot/" + arm + "/qualification.json"])
            == row["qualification_sha256"],
            "reviewed raw/qualification bytes differ",
        )
        require(
            raw["status"] == "complete"
            and qualification["status"] == "qualified"
            and qualification["reasons"] == [],
            "unqualified population",
        )
        require(
            raw["plan_sha256"] == sha(records["pilot/" + arm + "/plan.json"])
            and raw["source_revision"] == plan["source_revision"] == REVISION
            and raw["native_source_identity"]
            == plan["native_source_identity"]
            == IDENTITY
            and raw["native_build"]["library_sha256"] == CORE,
            "raw plan/core binding differs",
        )
        require(
            raw["observed_execution_controls"]
            == plan["execution_controls"]
            == {"GENERATIVEQC_SCF_ORTHONORMAL_DIIS": "0", CONTROL: value},
            "control differs",
        )
        require(
            raw["geometry_bohr"]
            == plan["geometry_bohr"]
            == selected[0]["geometry_bohr"]
            and len(raw["geometry_bohr"]) == 96,
            "geometry differs",
        )
        require(
            raw["method"] == "hf"
            and raw["mode"] == "direct"
            and raw["arm"] == "standard"
            and raw["atoms"] == 96
            and raw["warm_repeats"] == 3
            and raw["orbital_basis"] == "def2-SVP"
            and raw["basis_representation"] == "spherical"
            and raw["charge"] == raw["spin"] == 0,
            "scientific inputs differ",
        )
        protocol = raw["protocol"]
        require(
            protocol["cold"] == "constructor+prepare+SCF+forces+host-export"
            and protocol["analytic_forces"]
            and not protocol["trace_enabled"]
            and not protocol["oracle_seed_used"]
            and not protocol["grid_overrides"]
            and not protocol["moved_geometry"],
            "endpoint boundary differs",
        )
        require(
            plan["expected_gpu_uuid"] == GPU
            and plan["build_artifacts"] == build["artifacts"]
            and plan["build_artifact_bindings"] == build["artifact_bindings"],
            "artifact/device plan differs",
        )
        images = raw["native_images_preflight"]
        require(
            len(images) == 1
            and next(iter(images.values()))["sha256"] == CORE
            and Path(next(iter(images))).name == "libgenerativeqc.so.0.1.0",
            "unexpected native image",
        )
        require(len(raw["samples"]) == 4, "cold/warm population incomplete")
        reports = []
        for index, sample in enumerate(raw["samples"]):
            require(
                sample["phase"] == ("cold" if index == 0 else "warm")
                and sample["repeat"] == max(index - 1, 0)
                and sample["converged"]
                and sample["iterations"]
                == sample["fock_builds"]
                == (11 if index == 0 else 1)
                and sample["warm_start_used"] == (index != 0)
                and not sample["warm_start_fallback"],
                "physical work/warm behavior differs",
            )
            require(
                math.isfinite(sample["seconds"]) and sample["seconds"] > 0,
                "invalid timing",
            )
            require(
                sample["precision"]["requested_mode"] == "fp64"
                and sample["precision"]["effective_bits"] == 64,
                "precision differs",
            )
            error = errors(sample, reference)
            within(error, DENSE)
            reports.append(error)
        same_errors(reports[0], qualification["cold"]["errors"][0])
        require(len(qualification["warm"]["errors"]) == 3, "warm qualification omitted")
        for error, reported in zip(
            reports[1:], qualification["warm"]["errors"], strict=True
        ):
            same_errors(error, reported)
        dense_errors.extend(reports)
        raw_records.append(raw)
        plans.append(plan)
    off_plan, on_plan = plans
    differences = {
        k for k in set(off_plan) | set(on_plan) if off_plan.get(k) != on_plan.get(k)
    }
    require(
        differences == {"variant", "execution_controls"},
        "plans differ beyond sole control",
    )
    strict_errors = []
    for off, on in zip(
        raw_records[0]["samples"], raw_records[1]["samples"], strict=True
    ):
        for key in (
            "phase",
            "repeat",
            "iterations",
            "fock_builds",
            "settings",
            "precision",
            "incremental_direct_jk",
            "warm_start_used",
            "warm_start_fallback",
        ):
            require(
                off[key] == on[key], "work/settings/precision equality failed: " + key
            )
        guesses = [
            {k: v for k, v in r["initial_guess"].items() if k != "preparation_seconds"}
            for r in (off, on)
        ]
        require(guesses[0] == guesses[1], "initial guess work differs")
        error = errors(off, on)
        within(error, STRICT)
        strict_errors.append(error)
    require(len(receipt["strict_native_errors"]) == 4, "strict comparison omitted")
    for error, reported in zip(
        strict_errors, receipt["strict_native_errors"], strict=True
    ):
        same_errors(error, reported)
    off, on = [[r["seconds"] for r in raw["samples"]] for raw in raw_records]
    cold_ratio = off[0] / on[0]
    warm = [b / a for a, b in zip(off[1:], on[1:], strict=True)]
    require(
        cold_ratio > 1 and all(v <= 1.05 for v in warm), "cold/warm timing gate failed"
    )
    require(
        math.isclose(cold_ratio, receipt["cold_off_over_on"], rel_tol=1e-14)
        and math.isclose(
            statistics.median(on[1:]) / statistics.median(off[1:]),
            receipt["warm_on_over_off"],
            rel_tol=1e-14,
        ),
        "timing reduction differs",
    )
    return {
        "status": "RETAINED_ARRAY_PROVENANCE_REPLAY_PASS",
        "source": REVISION,
        "full_EF_arrays": 8,
        "strict_pairs": 4,
        "cold_seconds": {"off": off[0], "on": on[0]},
        "cold_reduction_percent": 100 * (1 - on[0] / off[0]),
        "warm_seconds": {"off": off[1:], "on": on[1:]},
        "paired_warm_on_over_off": warm,
        "maximum_dense_errors": {k: max(r[k] for r in dense_errors) for k in DENSE},
        "maximum_strict_errors": {k: max(r[k] for r in strict_errors) for k in STRICT},
        "default": "off",
        "new_CUDA_SCF_or_oracle_execution": False,
        "all30_complete": False,
    }


def main() -> None:
    cohort = json.loads((HERE / "cohort.json").read_text())
    records = decode(cohort, (HERE / cohort["capsule"]["path"]).read_bytes())
    print(json.dumps(audit(cohort, records), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

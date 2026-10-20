"""Bounded derived evidence; original raw qualification remains a separate replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

from benchmarks._retention import raw_output_path
from benchmarks.cc_cupti_qualification import (
    EXTRA_WORK,
    audit_campaign,
    closed_bytes,
    json_record,
    validate_frozen_child,
)
from benchmarks.cc_cupti_source_recovery import (
    SourceUnavailable,
    bounded_gzip,
    recover_tooling_sources,
)
from benchmarks.cc_tile_environment import (
    CHILD_SCHEMA,
    atomic_json,
    canonical_hash,
    profile_environment,
)
from benchmarks.cc_tile_pipeline_campaign import CONTROLS, WORK, check_science
from benchmarks.cc_trace_reader import require

SCHEMA = "generativeqc.cc-cupti-derived-capsule.v1"
SPEC_SHA = "c6099b3f895b65a6aff8f36f3005ff2f9213bcfa0cd0b6b4ffd19e4bcf3af2b6"
WORKER_SHA = "50c82d378872dd930352389866c2b74a641fb74f9b7e7e9d853c778910d216c1"
ORACLE_SHA = "ba40ed0b759eb27a9f3909377a6da0795fd292019a9b079e58c82929941a059b"
ROLES = ("clean-1", "clean-2", "diagnostic")
METRICS = ("total_energy", "triples_energy", "ccsd_replay_r1_max", "ccsd_replay_r2_max")
ENDPOINT_KEYS = (
    "nbf",
    "naux",
    "forces_requested",
    "ccsd_mixed_evaluations",
    "ccsd_fp32_gemm_calls",
    "triples_fp32_gemms",
    *METRICS,
    *WORK,
    *EXTRA_WORK,
)
DOCUMENTS = (
    "spec.json",
    "worker.py",
    "input.input",
    "oracle.json",
    "campaign-result.json",
    "outer-guard-receipt.json",
    *(
        f"{role}/{name}.json"
        for role in ROLES
        for name in ("receipt", "endpoint", "execution_plan", "child_environment")
    ),
)


def compressed_json(path: Path) -> tuple[dict, bytes]:
    """Reject links/FIFOs, truncated streams, trailing members and expansion bombs."""
    decoded = bounded_gzip(closed_bytes(path))
    value = json.loads(decoded)
    require(isinstance(value, dict), "compressed evidence must be an object")
    json.dumps(value, allow_nan=False)  # Reject nonfinite nested values.
    return value, decoded


def normalized_argv(role: str) -> list[str]:
    """Only paths are replaced; scientific positional controls remain exact."""
    argv = ["@executable", "@input", f"@{role}/endpoint.json", *CONTROLS]
    if role == "diagnostic":
        argv += ["--cc-cupti-v1", "@backend", f"@{role}/capture"]
    return argv


def derive_capsule(root: Path, audit: dict) -> dict:
    """Whitelist scientific fields; never serialize SDK/allocation/access objects.

    Call the existing full auditor before this projection. Original byte
    identities and path-normalized derived documents are explicitly distinct.
    """
    spec, _ = json_record(root / "spec.json", SPEC_SHA)
    documents = {}
    for name in DOCUMENTS:
        raw = closed_bytes(root / name)
        documents[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    children = []
    for role in ROLES:
        receipt, _ = json_record(root / role / "receipt.json")
        endpoint, _ = json_record(root / role / "endpoint.json")
        plan, _ = json_record(root / role / "execution_plan.json")
        witness, _ = json_record(root / role / "child_environment.json")
        validate_frozen_child(spec, role, receipt, plan, witness)
        children.append(
            {
                "role": role,
                "receipt": {
                    key: receipt[key]
                    for key in (
                        "diagnostic",
                        "native_exit",
                        "supervised_child_wall_ns",
                        "frozen_inputs_unchanged",
                        "cleanup_errors",
                    )
                }
                | {
                    "native_process": {
                        key: receipt["native_process"][key]
                        for key in ("pid", "starttime")
                    }
                },
                "endpoint": {key: endpoint[key] for key in ENDPOINT_KEYS},
                "plan": {
                    "normalized_argv": normalized_argv(role),
                    "original_argv_sha256": canonical_hash(plan["argv"]),
                    "original_environment_sha256": canonical_hash(plan["environment"]),
                    "profile": plan["environment"]["profile"],
                    "profile_sha256": plan["profile_sha256"],
                    "locale": {
                        key: plan["environment"]["observation"][key]
                        for key in ("LANG", "LC_ALL", "LC_CTYPE")
                    },
                },
                "witness": {key: witness[key] for key in ("schema", "status", "pid")}
                | {
                    "process": {
                        key: witness["process"][key] for key in ("pid", "starttime")
                    },
                    "original_argv_sha256": canonical_hash(witness["argv"]),
                    "original_environment_sha256": witness["environment_sha256"],
                },
            }
        )
    return {
        "schema": SCHEMA,
        "projection_notice": "Derived whitelist/path-normalized documents, not original receipt bytes or a replacement raw capture.",
        "original_documents": documents,
        "identities": {
            "production_source": spec["production_source"],
            "artifact_sha256": {
                key: spec["run"]["sha256"][key]
                for key in (
                    "executable",
                    "library",
                    "backend",
                    "cupti",
                    "input",
                    "oracle",
                )
            },
            "gpu_model": "NVIDIA H200",
            "driver_version": spec["driver_version"],
        },
        "children": children,
        "reported_audit": {
            key: audit[key]
            for key in (
                "science",
                "selected_phase_qualification",
                "frozen_packet",
                "campaign_result_sha256",
                "raw_files",
                "loader_attestation_scope",
                "execution_supervision",
                "parent_issue",
            )
        },
    }


def verify_summary(capsule: dict, oracle: dict) -> dict[str, Any]:
    """Recompute scalar/reference, bits/work and summary identities, never raw joins."""
    require(capsule["schema"] == SCHEMA, "unsupported derived capsule")
    documents = capsule["original_documents"]
    require(
        set(documents) == set(DOCUMENTS), "missing/extra original document identities"
    )
    for identity in documents.values():
        require(
            re.fullmatch(r"[0-9a-f]{64}", identity["sha256"])
            and type(identity["bytes"]) is int
            and identity["bytes"] > 0,
            "invalid original document identity",
        )
    for name, expected in (
        ("spec.json", SPEC_SHA),
        ("worker.py", WORKER_SHA),
        ("oracle.json", ORACLE_SHA),
    ):
        require(documents[name]["sha256"] == expected, "frozen external anchor changed")
    require(
        [row["role"] for row in capsule["children"]] == list(ROLES),
        "missing/extra/reordered children",
    )
    profiles, processes, endpoints, numerical = set(), set(), [], []
    expected_profile = profile_environment(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":16:8",
        }
    )
    for child in capsule["children"]:
        role, receipt, plan, witness = (
            child[key] for key in ("role", "receipt", "plan", "witness")
        )
        require(
            plan["normalized_argv"] == normalized_argv(role)
            and plan["profile"] == expected_profile,
            "derived argv/profile changed",
        )
        require(
            canonical_hash(plan["profile"]) == plan["profile_sha256"],
            "derived profile hash changed",
        )
        require(
            witness["schema"] == CHILD_SCHEMA
            and witness["status"] == "PREPARED_FOR_EXEC"
            and plan["locale"]
            == {
                key: {"present": True, "value": "C.UTF-8"}
                for key in ("LANG", "LC_ALL", "LC_CTYPE")
            },
            "projected witness/locale contract differs",
        )
        require(
            all(
                re.fullmatch(r"[0-9a-f]{64}", plan[key])
                for key in ("original_argv_sha256", "original_environment_sha256")
            ),
            "projected original binding is not a digest",
        )
        require(
            plan["original_argv_sha256"] == witness["original_argv_sha256"]
            and plan["original_environment_sha256"]
            == witness["original_environment_sha256"],
            "projected plan/witness binding differs",
        )
        process = receipt["native_process"]
        require(
            process == witness["process"] and process["pid"] == witness["pid"],
            "projected child PID/start differs",
        )
        require(
            type(process["pid"]) is int
            and process["pid"] > 0
            and isinstance(process["starttime"], str)
            and process["starttime"].isdigit()
            and int(process["starttime"]) > 0,
            "invalid projected native process",
        )
        require(
            receipt["native_exit"] == 0
            and receipt["diagnostic"] is (role == "diagnostic")
            and receipt["cleanup_errors"] == []
            and receipt["frozen_inputs_unchanged"] is True,
            "reported child failed",
        )
        require(
            type(receipt["supervised_child_wall_ns"]) is int
            and receipt["supervised_child_wall_ns"] > 0,
            "reported child wall unavailable",
        )
        processes.add((process["pid"], process["starttime"]))
        profiles.add(plan["profile_sha256"])
        endpoint = child["endpoint"]
        require(set(endpoint) == set(ENDPOINT_KEYS), "derived endpoint fields differ")
        gates = check_science(endpoint, oracle)
        require(gates["status"] == "PASS", "derived independent scalar gates failed")
        numerical.append({"role": role, "gates": gates})
        endpoints.append(endpoint)
    require(
        len(processes) == 3 and len(profiles) == 1,
        "fresh process/profile identities differ",
    )
    for key in (*WORK, *EXTRA_WORK):
        require(
            all(type(row[key]) is int and row[key] >= 0 for row in endpoints)
            and len({row[key] for row in endpoints}) == 1,
            "derived work differs: " + key,
        )
    for key in METRICS:
        require(
            len({struct.pack("!d", row[key]) for row in endpoints}) == 1,
            "derived numerical bits differ",
        )
    reported = capsule["reported_audit"]
    identities = capsule["identities"]
    require(
        identities["production_source"]
        == reported["frozen_packet"]["production_source"]
        and identities["artifact_sha256"]["library"]
        == reported["frozen_packet"]["library_sha256"]
        and identities["artifact_sha256"]["input"] == documents["input.input"]["sha256"]
        and identities["artifact_sha256"]["oracle"] == ORACLE_SHA,
        "reported scientific artifact identity differs",
    )
    require(
        documents["campaign-result.json"]["sha256"]
        == reported["campaign_result_sha256"],
        "reported campaign identity differs",
    )
    require(
        reported["frozen_packet"]["spec_sha256"] == SPEC_SHA
        and reported["frozen_packet"]["worker_sha256"] == WORKER_SHA,
        "reported packet anchors differ",
    )
    ledger = reported["selected_phase_qualification"]["exclusive_ledger_ns"]
    window = reported["selected_phase_qualification"]["full_caller_cupti_window_ns"]
    require(
        all(type(value) is int and value >= 0 for value in ledger.values())
        and sum(ledger.values())
        == window["duration"]
        == window["end"] - window["start"],
        "reported exclusive ledger/window inconsistent",
    )
    return {
        "schema": "generativeqc.cc-cupti-capsule-verification.v1",
        "derived_scalar_bits_work_summary": "PASS_RECOMPUTED_AGAINST_PINNED_INDEPENDENT_ORACLE",
        "numerical": numerical,
        "full_raw_qualification": "UNAVAILABLE_ORIGINAL_RAW_NOT_SUPPLIED",
        "tooling_source_recovery": {"status": "UNAVAILABLE_NOT_REQUESTED"},
        "full_evidence_reproduction": "UNAVAILABLE_RAW_AND_TOOLING_REQUIRED",
        "full_evidence_reproduction_scope": "Measured tooling byte reconstruction and original recorded raw replay only; no new native build, scientific processes, current-upstream scientific qualification or performance requalification.",
        "reported_capture_scope": "Producer/Linux-auditor summary retained; no raw activity/source/ABI/lifetime joins recomputed from this capsule.",
        "reported_exclusive_ledger_ns": ledger,
        "execution_supervision_limitation": reported["execution_supervision"][
            "limitation"
        ],
        "whole_program_fine_attribution": "INCOMPLETE",
        "implicit_wait_causality_profitability": "UNAVAILABLE",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--tooling-output", type=raw_output_path)
    parser.add_argument(
        "--source-repository", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--source-revision", default="HEAD")
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    capsule, _ = compressed_json(args.capsule)
    oracle, oracle_bytes = compressed_json(args.oracle)
    require(
        hashlib.sha256(oracle_bytes).hexdigest() == ORACLE_SHA,
        "independent original oracle bytes changed",
    )
    result = verify_summary(capsule, oracle)
    if args.tooling_output is not None:
        try:
            result["tooling_source_recovery"] = recover_tooling_sources(
                args.source_repository,
                args.source_revision,
                args.tooling_output,
                capsule,
            )
        except SourceUnavailable as error:
            result["tooling_source_recovery"] = {
                "status": "UNAVAILABLE",
                "reason": str(error),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output, result)
            return 1
    if args.campaign is not None:
        actual = audit_campaign(args.campaign, SPEC_SHA, WORKER_SHA)
        require(
            capsule == derive_capsule(args.campaign, actual),
            "capsule projection differs from actual original evidence",
        )
        result["full_raw_qualification"] = actual["selected_phase_qualification"][
            "selected_phase_coverage"
        ]
        if result["tooling_source_recovery"]["status"].startswith("PASS"):
            result["full_evidence_reproduction"] = (
                "PASS_TOOLING_RECONSTRUCTION_AND_ORIGINAL_RAW_REPLAY"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

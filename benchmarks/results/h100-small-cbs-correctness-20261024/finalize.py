"""Root-only offline finalizer for actual qz CLI originals and runtime preflight.

This performs no network, scheduling, GPU or solver call. Root runs it only after
reviewing the frozen finite plan and the one allocation's real observations.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from native_worker import helper
from transport import ROOT_CHAT, checked, creation_identity, read, sha, write_new


def finalize(args: Any) -> None:
    plan = read(args.plan)
    preflight = read(args.preflight)
    if (
        sha(args.plan) != args.plan_sha256
        or sha(args.preflight) != args.preflight_sha256
        or preflight.get("prepared_plan_sha256") != args.plan_sha256
        or preflight.get("job_name") != plan["expected_job_name"]
        or preflight.get("endpoints_started") != 0
    ):
        raise ValueError("plan/preflight identity mismatch")
    observed = datetime.fromisoformat(preflight["observed_utc"])
    if (
        observed.tzinfo is None
        or not 0 <= (datetime.now(timezone.utc) - observed).total_seconds() < 300
    ):
        raise ValueError("root release window expired")
    if (args.preflight.parent / "gate-outcome.json").exists() or args.release.exists():
        raise ValueError("gate terminal or already released")
    identity = preflight["static_identity"]
    snapshot = read(checked(Path(identity["path"]), identity["sha256"]))
    if snapshot["whole_original_inputs"]["status"] != "PASS":
        raise ValueError("original source/environment identity did not pass")
    status_original, instances_original = read(args.status), read(args.instances)
    if (
        status_original.get("success") is not True
        or instances_original.get("success") is not True
    ):
        raise ValueError("actual CLI envelopes must both report success")
    status, instances = status_original["data"], instances_original["data"]
    if (
        status.get("name") != plan["expected_job_name"]
        or instances.get("name") != plan["expected_job_name"]
    ):
        raise ValueError("actual Job differs from fixed plan")
    canonical = creation_identity(status.get("created_at"))
    expected = creation_identity(getattr(args, "expected_created_at", None))
    low = creation_identity(getattr(args, "submission_not_before_ms", None))
    high = creation_identity(getattr(args, "submission_not_after_ms", None))
    if canonical != expected or not int(low) <= int(canonical) <= int(high):
        raise ValueError("actual creation differs from root snapshot/submission window")
    rows = instances.get("items", [])
    if len(rows) != 1:
        raise ValueError("expected one actual rank-0 instance")
    proof = {
        "schema": "generativeqc.h100-environment-allocation.v2",
        "provider": "qz",
        "name": plan["expected_job_name"],
        "status": status.get("status"),
        "gpu_count": 1,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "resources": status.get("resource"),
        "platform_status": status,
        "platform_instances": instances,
        "raw_status_response": status_original,
        "raw_instances_response": instances_original,
        "query_context": {
            "account_alias": "qz",
            "workspace": "分布式训练空间",
            "job_name": plan["expected_job_name"],
            "scope_source": "root_cli_query_arguments",
            "status_sha256": sha(args.status),
            "instances_sha256": sha(args.instances),
        },
        "original_status_sha256": sha(args.status),
        "original_instances_sha256": sha(args.instances),
        "platform_job_uid": None,
        "platform_job_uid_status": "UNAVAILABLE",
        "instances": [
            {
                "rank": rows[0].get("rank"),
                "node": rows[0].get("node"),
                "id": None,
                "platform_instance_id": None,
                "platform_instance_id_status": "UNAVAILABLE",
                "hostname": preflight["device"]["hostname"],
                "hostname_source": "runtime_preflight",
            }
        ],
        "runtime_preflight": {
            "path": str(args.preflight.resolve()),
            "sha256": sha(args.preflight),
        },
        "coordinator_chat": ROOT_CHAT,
        "opaque_id_policy": "UNAVAILABLE identifiers are not fabricated",
        "creation_binding": {
            "raw": status["created_at"],
            "canonical": canonical,
            "expected_canonical": expected,
            "submission_window_ms": {"not_before": low, "not_after": high},
        },
    }
    helper(plan).bind_allocated_instance(
        proof, plan["expected_job_name"], preflight["device"]
    )
    args.proof.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.proof, proof)
    # Refresh the finite window immediately before publishing; no late marker.
    if (datetime.now(timezone.utc) - observed).total_seconds() >= 300 or (
        args.preflight.parent / "gate-outcome.json"
    ).exists():
        raise ValueError("gate closed during finalization; proof retained, no release")
    write_new(
        args.release,
        {
            "schema": "generativeqc.h100-small-cbs-root-release.v2",
            "authority": "ONE_FINITE_BATCH",
            "root_coordinator_chat": ROOT_CHAT,
            "prepared_plan_sha256": sha(args.plan),
            "expected_job_name": plan["expected_job_name"],
            "allocation_created_at": canonical,
            "allocation_created_at_raw": status["created_at"],
            "expected_created_at_canonical": expected,
            "submission_window_ms": {"not_before": low, "not_after": high},
            "allocation_proof": str(args.proof.resolve()),
            "allocation_proof_sha256": sha(args.proof),
            "authorized_calls": 32,
            "authorized_science_processes": 4,
            "released_unix_seconds": time.time(),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "preflight", "status", "instances", "proof", "release"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "preflight-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in (
        "expected-created-at",
        "submission-not-before-ms",
        "submission-not-after-ms",
    ):
        parser.add_argument("--" + name, required=True)
    finalize(parser.parse_args())

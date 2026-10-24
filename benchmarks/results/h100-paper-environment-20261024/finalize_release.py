"""Root-only fast finalizer for observed status/instances and unique preflight.

No API or resource action occurs here. Root supplies scrubbed JSON originals,
reviews them, and invokes this once to publish a bound release marker.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gate_wait import ROOT_CHAT, final_plan, load_adapter, read_json, sha, write_new


def data(path: Path) -> dict:
    raw = read_json(path)
    if not isinstance(raw, dict) or raw.get("success") is not True:
        raise ValueError("original platform response envelope must report success=true")
    value = raw.get("data")
    if not isinstance(value, dict):
        raise TypeError("platform JSON must contain an object data envelope")
    return value


def gate_open(args: Any, preflight: dict) -> None:
    if (
        args.release_marker.parent / "job-outcome.json"
    ).exists() or args.release_marker.exists():
        raise ValueError(
            "gate already terminal or marker already exists; no late release"
        )
    observed = datetime.fromisoformat(preflight["observed_utc"])
    if (
        observed.tzinfo is None
        or (datetime.now(timezone.utc) - observed).total_seconds()
        >= preflight["wait_cap_seconds"]
    ):
        raise ValueError(
            "root release window expired; preserve attempt and do not publish"
        )


def finalize(args: Any, adapter: Any) -> dict:
    if sha(args.prepared_plan) != args.prepared_plan_sha256:
        raise ValueError("prepared plan differs from frozen snapshot")
    if sha(args.preflight) != args.preflight_sha256:
        raise ValueError("actual preflight file hash differs")
    prepared = read_json(args.prepared_plan)
    preflight = read_json(args.preflight)
    gate_open(args, preflight)
    status, instances = data(args.status), data(args.instances)
    query_context = read_json(args.query_context)
    name = status.get("name")
    expected_name = prepared.get("expected_job_name")
    if (
        not isinstance(expected_name, str)
        or not expected_name
        or name != expected_name
        or preflight.get("job_name") != expected_name
    ):
        raise ValueError("observed Job must match immutable prepared expected_job_name")
    if (
        not isinstance(name, str)
        or not name
        or status.get("status") != "RUNNING"
        or instances.get("name") != name
    ):
        raise ValueError("actual status/instances must identify one RUNNING Job")
    rows = instances.get("items", [])
    if len(rows) != 1:
        raise ValueError("one observed platform worker is required")
    row = rows[0]
    if not row.get("node") or row["node"] == args.excluded_node:
        raise ValueError("actual node missing or excluded")
    device = preflight.get("device", {})
    if (
        device.get("visible_device_count") != 1
        or device.get("compute_capability") != [9, 0]
        or "H100" not in device.get("name", "")
    ):
        raise ValueError("actual preflight does not prove one H100/SM90")
    if (
        preflight.get("adapter_sha256") != prepared["adapter_sha256"]
        or preflight.get("prepared_plan_sha256") != args.prepared_plan_sha256
    ):
        raise ValueError("actual preflight used another frozen caller/plan")
    if preflight.get("job_name") != name:
        raise ValueError("preflight names another Job")
    hostname = device.get("hostname")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("actual preflight hostname missing")
    args.output.mkdir(parents=True, exist_ok=False)
    # Preserve original response envelopes separately from normalized association.
    write_new(args.output / "status-original.json", read_json(args.status))
    write_new(args.output / "instances-original.json", read_json(args.instances))
    proof = {
        "schema": "generativeqc.h100-environment-allocation.v2",
        "provider": "qz",
        "name": name,
        "status": "RUNNING",
        "observed_utc": args.observed_utc,
        "gpu_count": 1,
        "resources": status.get("resource"),
        "platform_status": status,
        "platform_instances": instances,
        "raw_status_response": read_json(args.status),
        "raw_instances_response": read_json(args.instances),
        "query_context": query_context,
        "platform_job_uid": None,
        "platform_job_uid_status": "UNAVAILABLE",
        "instances": [
            {
                "rank": row.get("rank"),
                "node": row["node"],
                "platform_instance_id": None,
                "platform_instance_id_status": "UNAVAILABLE",
                "hostname": hostname,
                "hostname_source": "runtime_preflight",
            }
        ],
        "runtime_preflight": {
            "path": str(args.preflight),
            "sha256": args.preflight_sha256,
        },
        "original_status_sha256": sha(args.status),
        "original_instances_sha256": sha(args.instances),
        "coordinator_chat": ROOT_CHAT,
        "opaque_id_policy": "not exposed by supported CLI; never derived from pick/rank/pod hostname",
    }
    association = adapter.bind_allocated_instance(proof, name, device)
    proof_path = args.output / "allocation-proof.json"
    write_new(proof_path, proof)
    plan = copy.deepcopy(prepared)
    plan.update(
        job_name=name,
        allocation_proof=str(proof_path),
        allocation_proof_sha256=sha(proof_path),
        execution_authority="explicit root release for this actual Job/preflight only",
    )
    plan_path = args.output / "final-plan.json"
    write_new(plan_path, plan)
    marker = {
        "schema": "generativeqc.h100-environment-release.v1",
        "coordinator_chat": ROOT_CHAT,
        "preflight_sha256": args.preflight_sha256,
        "plan_path": str(plan_path),
        "plan_sha256": sha(plan_path),
    }
    # Apply the exact gate validator before publication, not after execution.
    final_plan(marker, prepared, args.preflight_sha256, args.excluded_node)
    write_new(
        args.output / "finalizer-validation.json",
        {
            "association": association,
            "marker": marker,
            "scope": "root release control, not scientific acceptance",
        },
    )
    gate_open(args, preflight)
    write_new(args.release_marker, marker)
    return {
        "job_name": name,
        "proof_path": str(proof_path),
        "proof_sha256": sha(proof_path),
        "plan_path": str(plan_path),
        "plan_sha256": sha(plan_path),
        "marker_path": str(args.release_marker),
        "marker_sha256": sha(args.release_marker),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-plan",
        "preflight",
        "status",
        "instances",
        "output",
        "release-marker",
        "adapter",
        "query-context",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "prepared-plan-sha256",
        "preflight-sha256",
        "adapter-sha256",
        "observed-utc",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--excluded-node", default="qb-prod-gpu095")
    args = parser.parse_args()
    print(
        json.dumps(
            finalize(args, load_adapter(args.adapter, args.adapter_sha256)),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

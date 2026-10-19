"""Retain a live qz single-GPU Job observation before native qualification.

Run on the operator host with the optional Inspire SDK, never in production.
This reads platform identity and a bounded container probe. It does not submit,
stop, or change a workload, access counters, or claim scientific acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks._retention import raw_output_path
from tools.audit_replay_allocations import _require
from tools.capture_prepared_allocations import write_new
from tools.residency_allocation import SCHEMA, validate_qz

PROBE = """import json,os,socket,subprocess
uuid=os.environ.get('NVIDIA_VISIBLE_DEVICES','')
result=subprocess.check_output(['nvidia-smi','--id='+uuid,
 '--query-gpu=uuid,mig.mode.current','--format=csv,noheader'],text=True,timeout=20)
print(json.dumps(dict(hostname=socket.gethostname(),environment={
 k:os.environ[k] for k in ('NVIDIA_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','SLURM_JOB_ID')
 if k in os.environ},nvidia_smi=result.strip())))
"""


def snapshot(client: Any, name: str, workspace: str, python: str) -> dict[str, Any]:
    """Bind API IDs and assigned instance to an independently observed container."""
    before = client.jobs.get(name, workspace=workspace)
    _require(before.status == "RUNNING", "qz job must be RUNNING")
    instances = client.jobs.instances(before.ref)
    _require(len(instances) == 1, "requires one qz instance")
    instance = instances[0]
    probe = client.jobs.exec(
        before.ref,
        instance=instance.label,
        cwd="/",
        command=shlex.quote(python) + " -c " + shlex.quote(PROBE),
        timeout=60,
        max_output_bytes=16384,
    )
    _require(
        probe.completed and probe.returncode == 0 and not probe.truncated,
        "qz container probe failed",
    )
    observation = json.loads(probe.stdout or probe.output)
    after = client.jobs.get(before.ref)
    rows = client.jobs.instances(before.ref)
    _require(
        after.status == "RUNNING"
        and len(rows) == 1
        and all(
            rows[0].raw.get(key) == instance.raw.get(key)
            for key in ("name", "node", "instance_status")
        ),
        "qz assignment changed during observation",
    )
    raw = after.raw
    job = {
        key: raw.get(key)
        for key in (
            "job_id",
            "workspace_id",
            "project_id",
            "name",
            "status",
            "gpu_count",
            "node_count",
            "logic_compute_group_name",
        )
    }
    selected = {
        key: instance.raw.get(key) for key in ("name", "node", "instance_status")
    }
    fields = [s.strip() for s in observation["nvidia_smi"].split(",")]
    _require(len(fields) == 2 and fields[1] in ("Disabled", "[N/A]"), "MIG unsupported")
    allocation = {
        "project_id": job["project_id"],
        "workspace_id": job["workspace_id"],
        "workload_id": job["job_id"],
        "workload_name": job["name"],
        "instance_id": selected["name"],
        "pod": selected["name"],
        "node": selected["node"],
        "compute_group": job["logic_compute_group_name"],
        "kind": "job",
        "status": "RUNNING",
        "gpu_count": job["gpu_count"],
        "gpu_uuid": fields[0],
    }
    receipt = {
        "schema": SCHEMA,
        "observed_unix_seconds": time.time(),
        "allocation": allocation,
        "api": {"job": job, "instance": selected},
    }
    validate_qz(
        receipt,
        uuid=fields[0],
        hostname=observation["hostname"],
        environment=observation["environment"],
        now=time.time(),
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True, choices=("qz",))
    parser.add_argument("--no-env-file", required=True, action="store_true")
    parser.add_argument("--job", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--python", required=True, help="absolute interpreter in the Job"
    )
    parser.add_argument("--output", required=True, type=raw_output_path)
    args = parser.parse_args()
    _require(args.python.startswith("/"), "remote interpreter must be absolute")
    # SDK config loading does not load a dotenv file; never bootstrap the CLI.
    from inspire.sdk import InspireClient

    with InspireClient(account=args.account) as client:
        receipt = snapshot(client, args.job, args.workspace, args.python)
    write_new(args.output, receipt)
    print(
        json.dumps(
            {
                "receipt_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "status": "OBSERVED",
                "scientific_acceptance": None,
            }
        )
    )


if __name__ == "__main__":
    main()

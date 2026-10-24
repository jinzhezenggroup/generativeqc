"""Root-controlled bounded Job gate; no endpoint starts before release.json.

The launcher performs only device/host preflight before the release marker.
It never creates, starts, stops or deletes a platform resource.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_support import stop_owned_process_group

ROOT_CHAT = "01a10e8f-7b0a-7b52-a330-b2de34309210"
ARMS = ("native-hf", "gpu4pyscf-hf", "cbs-c2")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def write_new(path: Path, data: dict) -> None:
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # The hard link atomically publishes complete bytes while refusing clobber.
    os.link(pending, path)
    pending.unlink()


def redact(text: str) -> str:
    text = re.sub(
        r"(?:https?|wss?)://[^\s\"'<>]+", "[REDACTED_URL]", text, flags=re.IGNORECASE
    )
    return re.sub(
        r"""(?i)("?(?:access_token|refresh_token|token|password|secret|authorization|credential)"?\s*[:=]\s*)("[^"]*"|'[^']*'|(?:Bearer\s+)?[^\s,;]+)""",
        r"\1[REDACTED]",
        text,
    )


def load_adapter(path: Path, expected: str) -> Any:
    if sha(path) != expected:
        raise ValueError("frozen adapter hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_setup_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final_plan(
    marker: dict, prepared: dict, preflight_sha: str, excluded_node: str
) -> tuple[Path, str]:
    if (
        marker.get("schema") != "generativeqc.h100-environment-release.v1"
        or marker.get("coordinator_chat") != ROOT_CHAT
    ):
        raise ValueError("release marker is not the root-controlled contract")
    if marker.get("preflight_sha256") != preflight_sha:
        raise ValueError("release marker does not bind this Job preflight")
    path = Path(marker["plan_path"]).resolve(strict=True)
    checksum = marker["plan_sha256"]
    if sha(path) != checksum:
        raise ValueError("final plan hash mismatch")
    plan = read_json(path)
    if (
        not prepared.get("expected_job_name")
        or plan.get("job_name") != prepared["expected_job_name"]
    ):
        raise ValueError("release differs from frozen expected Job name")
    if not isinstance(plan.get("job_name"), str) or not plan["job_name"]:
        raise ValueError("release Job name is missing")
    for key in (
        "schema",
        "sources",
        "expected_packages",
        "hf_inputs",
        "qm7",
        "qm7_sha256",
        "protocol_manifest",
        "protocol_manifest_sha256",
        "adapter",
        "adapter_sha256",
        "expected_job_name",
    ):
        if plan.get(key) != prepared.get(key):
            raise ValueError(f"release altered a frozen input: {key}")
    if plan.get("setup_calls") != list(ARMS):
        raise ValueError("release must contain exactly the three setup arms")
    caps = plan["caps"]
    if (
        caps.get("calls") != 3
        or caps.get("fresh_processes") != 3
        or caps.get("automatic_retries") != 0
    ):
        raise ValueError("release changed setup cardinality or retry policy")
    for key, bound in (
        ("per_process_seconds", 3600),
        ("job_seconds", 14400),
        ("termination_grace_seconds", 30),
    ):
        value = caps.get(key)
        if type(value) is not int or not 0 < value <= bound:
            raise ValueError(f"release exceeds finite cap: {key}")
    proof_path = Path(plan["allocation_proof"]).resolve(strict=True)
    if sha(proof_path) != plan["allocation_proof_sha256"]:
        raise ValueError("release allocation proof hash mismatch")
    proof = read_json(proof_path)
    observation = proof.get("runtime_preflight", {})
    if (
        observation.get("sha256") != preflight_sha
        or sha(Path(observation["path"])) != preflight_sha
    ):
        raise ValueError("allocation proof does not bind this exact runtime preflight")
    if proof.get("status") != "RUNNING" or proof.get("name") != plan.get("job_name"):
        raise ValueError("release lacks the real RUNNING Job identity")
    instances = proof.get("instances", [])
    if not instances or any(
        not x.get("node") or x["node"] == excluded_node for x in instances
    ):
        raise ValueError("release node is absent or shares the excluded #2677 node")
    return path, checksum


def run_child(command: list[str], folder: Path, cap: float) -> dict:
    def drain(pipe: Any, target: Path) -> None:
        with pipe, target.open("x", encoding="utf-8") as stream:
            for line in pipe:
                stream.write(redact(line))
                stream.flush()

    started = time.monotonic()
    child = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="backslashreplace",
        start_new_session=True,
    )
    threads = [
        threading.Thread(target=drain, args=(pipe, folder / name))
        for pipe, name in ((child.stdout, "stdout.log"), (child.stderr, "stderr.log"))
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        child.wait(timeout=cap)
    except subprocess.TimeoutExpired:
        timed_out = True
        stop_owned_process_group(child)
    finally:
        for thread in threads:
            thread.join(timeout=5)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("owned log drain did not finish within cleanup bound")
    return {
        "pid": child.pid,
        "returncode": child.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
    }


def gate(args: Any, adapter: Any) -> dict:
    if (
        not 0 < args.wait_seconds <= 300
        or not 0 < args.process_cap <= 3600
        or not 0 < args.job_cap <= 14400
    ):
        raise ValueError("gate/Job/process cap exceeds the approved bound")
    if sha(args.prepared_plan) != args.prepared_plan_sha256:
        raise ValueError("prepared plan hash mismatch")
    prepared = read_json(args.prepared_plan)
    if (
        not prepared.get("expected_job_name")
        or args.job_name != prepared["expected_job_name"]
    ):
        raise ValueError(
            "gate command Job name differs from frozen prepared expectation"
        )
    if (
        str(args.adapter) != prepared["adapter"]
        or args.adapter_sha256 != prepared["adapter_sha256"]
    ):
        raise ValueError("launcher adapter differs from frozen prepared plan")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    outcome = {
        "schema": "generativeqc.h100-environment-job.v1",
        "status": "preflight",
        "scientific_acceptance": "INCOMPLETE",
        "endpoints_started": 0,
        "attempts": [],
    }
    try:
        if args.release_marker.exists():
            raise ValueError("release marker predates this Job preflight")
        preflight = {
            "schema": "generativeqc.h100-environment-gate-preflight.v1",
            "job_name": args.job_name,
            "scope": "device/host only; no endpoint before root release",
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "device": adapter.device_observation(),
            "environment": adapter.environment_observation(),
            "adapter_sha256": args.adapter_sha256,
            "prepared_plan_sha256": args.prepared_plan_sha256,
            "wait_cap_seconds": args.wait_seconds,
            "endpoints_started": 0,
        }
        preflight_path = args.output / "preflight.json"
        write_new(preflight_path, preflight)
        print(json.dumps(preflight, allow_nan=False), flush=True)
        deadline = time.monotonic() + args.wait_seconds
        while not args.release_marker.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "300-second root release wait expired; no endpoint started"
                )
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        plan_path, plan_sha = final_plan(
            read_json(args.release_marker),
            prepared,
            sha(preflight_path),
            args.excluded_node,
        )
        write_new(
            args.output / "release-accepted.json",
            {
                "marker_sha256": sha(args.release_marker),
                "preflight_sha256": sha(preflight_path),
                "plan_path": str(plan_path),
                "plan_sha256": plan_sha,
            },
        )
        caps = read_json(plan_path)["caps"]
        for arm in ARMS:
            remaining = min(args.job_cap, caps["job_seconds"]) - (
                time.monotonic() - began
            )
            if remaining <= 30:
                raise TimeoutError("finite Job deadline reached before next setup arm")
            folder = args.output / (arm + "-process")
            folder.mkdir(exist_ok=False)
            command = [
                sys.executable,
                "-B",
                str(args.adapter),
                "--plan",
                str(plan_path),
                "--plan-sha256",
                plan_sha,
                "--arm",
                arm,
                "--output",
                str(folder / "endpoint"),
            ]
            outcome["endpoints_started"] += 1
            result = {
                "arm": arm,
                "command": command,
                **run_child(
                    command,
                    folder,
                    min(args.process_cap, caps["per_process_seconds"], remaining - 30),
                ),
            }
            write_new(folder / "outcome.json", result)
            outcome["attempts"].append(result)
            if result["timed_out"] or result["returncode"] < 0:
                raise RuntimeError(
                    "setup arm timed out or was signaled; no automatic retry"
                )
        outcome["status"] = (
            "PROCESSES_RETURNED"
            if all(x["returncode"] == 0 for x in outcome["attempts"])
            else "FAILED"
        )
    except BaseException as error:
        outcome.update(
            status="FAILED", error=redact(f"{type(error).__name__}: {error}")
        )
        raise
    finally:
        outcome["elapsed_seconds"] = time.monotonic() - began
        write_new(args.output / "job-outcome.json", outcome)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--prepared-plan", type=Path, required=True)
    parser.add_argument("--prepared-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-marker", type=Path, required=True)
    parser.add_argument("--excluded-node", default="qb-prod-gpu095")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--wait-seconds", type=int, default=300)
    parser.add_argument("--process-cap", type=int, default=3600)
    parser.add_argument("--job-cap", type=int, default=14400)
    args = parser.parse_args()
    result = gate(args, load_adapter(args.adapter, args.adapter_sha256))
    if result["status"] != "PROCESSES_RETURNED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Freeze the minimal V4 HOST proposal; never stage or run an attempt."""

from __future__ import annotations

from pathlib import Path

from transport import read, sha, write_new

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    base_path = here / "prepared-plan-v3.json"
    if (
        sha(base_path)
        != "6ceb8a28243193af3de562d8ea1fefd8ce0f18c93b293ec7bd0377853f3f6021"
    ):
        raise ValueError("original V3 plan changed")
    plan = read(base_path)
    original_task = plan["task_root"]
    new_task = original_task.rsplit("/", 1)[0] + "/v2"
    plan["expected_job_name"] = "h100-small-cbs-20261024-e7f7-v2"
    plan["task_root"] = new_task
    plan["output_root"] = new_task + "/output"
    for key, value in plan["runtime_controls"].items():
        if isinstance(value, str) and value.startswith(original_task + "/"):
            plan["runtime_controls"][key] = new_task + value[len(original_task) :]
    spec = read(here / "environment-derived-additions-v1.json")
    plan["environment_additions"] = {
        "file": "environment-derived-additions-v1.json",
        "sha256": sha(here / "environment-derived-additions-v1.json"),
        "base_manifest_sha256": plan["environment_inventory"]["sha256"],
        "runtime_cohort": spec["runtime_cohort"],
        "base_regular_files": 40661,
        "qualified_added_files": 1,
        "qualified_regular_files": 40662,
        "qualified_regular_bytes": 5717393830,
        "qualified_env_files": 16377,
    }
    plan["bytecode_write_control"] = (
        "outer PYTHONDONTWRITEBYTECODE=1 before any interpreter plus retained -B; bounded children inherit this targeted control"
    )
    plan["failed_prior_attempt"] = {
        "file": "v3-failed-preflight.json",
        "sha256": sha(here / "v3-failed-preflight.json"),
        "job_name": "h100-small-cbs-20261024-e7f7-v1",
        "created_at_raw": "1792861634000",
        "result": "FAILED strict env nameset, no release/output/attempts, zero scientific calls",
        "resource_submission_spent": True,
        "auto_retry": False,
    }
    plan["supersedes_host_plan"] = {
        "version": 3,
        "plan_sha256": sha(base_path),
        "packet_sha256": "25b07f18f30ee99216538ef087f7f107c707d8a1b0fce2620494238d8e8671f0",
        "host_linux_validation": "64/64 zero skip; reuse unaffected scientific and kernel behavior evidence",
        "state": "preserved original V3 and actual failed first Job; no retrospective science/SDK failure",
    }
    plan["manual_attempt_status"] = (
        "NOT_AUTHORIZED; root must separately approve one manual attempt2/new Job/observer/release, never automatic retry"
    )
    plan["new_adapter_status"] = (
        "V4_HOST_DERIVED_CACHE_CANDIDATE_ONLY; no staging/resource/science permission"
    )
    plan["new_transport_adapter_sha256"] = sha(here / "native_worker.py")
    plan["task_files"] = {
        p.name: sha(p) for p in sorted(here.iterdir()) if p.suffix in (".py", ".sh")
    }
    for name in (
        "inputs-v3.json",
        "reference-admission-v3.json",
        "test-model-fixture.json",
        "environment-derived-additions-v1.json",
        "derived-cache-proof.json",
        "v3-failed-preflight.json",
    ):
        plan["task_files"][name] = sha(here / name)
    plan["new_adapter_components"] = {
        name: value
        for name, value in plan["task_files"].items()
        if name.endswith((".py", ".sh"))
    }
    # Opt-in transition evidence, not a permanent production-refactor blocker.
    original = read(base_path)
    for name in (
        "native_worker.py",
        "observables.py",
        "model_contract.py",
        "replay.py",
        "process_guard.py",
        "finalize.py",
        "device_probe.py",
        "admission.py",
    ):
        if plan["task_files"][name] != original["task_files"][name]:
            raise ValueError(
                "V4 unexpectedly changes science/raw/kernel component: " + name
            )
    if (
        plan["calls"] != original["calls"]
        or plan["science_source"] != original["science_source"]
    ):
        raise ValueError("V4 unexpectedly changes scientific scope")
    write_new(here / "prepared-plan-v4.json", plan)
    print(sha(here / "prepared-plan-v4.json"))

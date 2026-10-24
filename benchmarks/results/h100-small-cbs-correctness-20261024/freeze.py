"""Freeze a new host-only packet; never replace a previous plan or run science."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transport import read, sha, write_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dependencies", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    inputs = read(here / "inputs-v2.json")
    if args.external_dependencies is not None:
        inputs["external_dependencies"] = read(args.external_dependencies)
    inputs["archive_auxiliary_basis_identities"] = [
        "d8a39cc61488847fad5380c1a5e1160ecda5d0d7202ec6f7a66f52ff0b66f191"
    ] * 2
    inputs["independent_composite_model"] = read(here / "reference-admission-v3.json")[
        "independent_composite_model"
    ]
    input_path = here / "inputs-v3.json"
    if input_path.exists():
        expected_bytes = (json.dumps(inputs, indent=2, allow_nan=False) + "\n").encode()
        if input_path.read_bytes() != expected_bytes:
            raise ValueError(
                "existing V3 input bytes differ; no normalization or replacement"
            )
    else:
        write_new(input_path, inputs)
    plan = read(here / "prepared-plan-v1.json")
    plan["input_bundle_file"] = "inputs-v3.json"
    plan["input_bundle_sha256"] = sha(here / "inputs-v3.json")
    plan["reference_admission_sha256"] = sha(here / "reference-admission-v3.json")
    plan["archive_auxiliary_basis_identities"] = inputs[
        "archive_auxiliary_basis_identities"
    ]
    plan["expected_python_executable"] = (
        plan["public_readonly_runtime"] + "/env/bin/python"
    )
    plan["expected_python_version"] = (
        "3.11.16 (main, Sep  1 2026, 14:18:37) [Clang 22.1.3 ]"
    )
    plan["environment_inventory"] = {
        "path": "/inspire/ssd/project/project-public/czxs25220150/h100-paper-environment-public-20261024-9dcc/incoming-v1/target-independent-readback-after-device-v1.json",
        "bytes": 9564261,
        "sha256": "07b07209ea8fb999b2258d000cf10ce388e0d9b28c15a1ccdf0ee1db712bdf41",
        "regular_files": 40661,
        "regular_bytes": 5717381795,
        "symlinks": 8,
    }
    plan["allocation_helper"] = {
        "path": plan["public_readonly_runtime"]
        + "/scripts/public-route-runtime-v2/qz_smoke.py",
        "sha256": "36a14e43c1a0dd01f34b4ef076e2c9abe9250c7f49a6722616bdeca346f3bdb4",
    }
    plan["activation"] = {
        "path": "/opt/generativeqc-reuse/activate.sh",
        "bytes": 398,
        "mode": "0o644",
        "sha256": "f788e97155d31786a79d719b33ecded54a7e87a39ed452720c12f88bcfba21ee",
        "provenance": "CPU observation in this preparation; historical smoke GPU activation bytes UNAVAILABLE",
        "base_venv_activate": "not sourced by this entrypoint; optional CPU capture is not a transitive-closure proof",
    }
    plan["qz_scheduling"] = {
        "authority": "ROOT_ONLY; no authorization in this packet",
        "supported_project": "公共科研项目",
        "workspace": "分布式训练空间",
        "group": "开发区-H100-cuda13.2版本-183核",
        "required_resource": {
            "gpu": 1,
            "nodes": 1,
            "cpu": 20,
            "memory_gib": 200,
            "shared_memory_gib": 32,
        },
        "required_GPU": "H100 80GB SM90, exactly one CUDA-visible device; fixed pinned helper supports this resource schema",
        "priority_and_image": "root must verify live policy/quota/image/budget; no default priority or availability assertion",
    }
    plan["caps"].update(
        metadata_phase_seconds=600,
        device_probe_seconds=60,
        all_task_stop_bytes=11 << 30,
        new_tmp_cache_stop_bytes=2 << 30,
        python_invocations_on_H100=14,
        control_python_invocations=10,
        science_python_invocations=4,
        directly_invoked_sccache_version_processes=2,
        nvidia_smi_queries=10,
        runtime_internal_compiler_child_count="UNAVAILABLE; owned finite process group, not assumed zero",
        bounded_invocations=6,
        guard_ready_handshake_seconds=10,
        cleanup_term_grace_seconds=30,
        cleanup_kill_reap_seconds=5,
        cleanup_report_seconds=2,
        cleanup_extra_seconds_per_bounded=37,
        cleanup_extra_seconds_total=222,
    )
    plan["process_accounting"] = {
        "bootstrap": 1,
        "supervisor": 1,
        "zero_science_device_probes": 2,
        "science_workers": 4,
        "lifecycle_guards": 6,
        "root_CPU_finalizer": "one separate host invocation only after root authorizes the real allocation",
        "concurrency": "one worker/probe at a time under the supervisor; no simultaneous scientific workers",
        "note": "14 Python invocations includes 10 non-scientific control invocations; 32 attempt slots remain 22 EF +10 E-only, not 32 EF",
    }
    plan["storage_accounting"] = (
        "512MiB output/8GiB JIT/11GiB all-task are monitored stop thresholds, not filesystem quota; 64MiB RLIMIT_FSIZE applies per science file. Polling can overshoot a threshold; 12GiB free-space admission retains margin."
    )
    for name, leaf in [
        ("TMPDIR", "tmp"),
        ("PIP_CACHE_DIR", "pip-cache"),
        ("CUPY_CACHE_DIR", "cupy-cache"),
        ("XDG_CACHE_HOME", "cache"),
    ]:
        plan["runtime_controls"][name] = plan["task_root"] + "/" + leaf
    plan["new_adapter_status"] = (
        "HOST_IMPLEMENTED_NOT_EXECUTED; frozen proposal only, root independent review and SCI/resource authorization still required"
    )
    plan["new_transport_adapter_sha256"] = sha(here / "native_worker.py")
    plan["task_files"] = {
        p.name: sha(p) for p in sorted(here.iterdir()) if p.suffix in (".py", ".sh")
    }
    for name in (
        "inputs-v3.json",
        "reference-admission-v3.json",
        "test-model-fixture.json",
    ):
        plan["task_files"][name] = sha(here / name)
    plan["new_adapter_components"] = {
        name: value
        for name, value in plan["task_files"].items()
        if name.endswith((".py", ".sh"))
    }
    plan["required_admission_before_root_allocation"][0] = (
        "root independently reviews frozen scripts and runs applicable host failure tests; Windows skips are not POSIX cleanup PASS"
    )
    plan["failure_policy"] = (
        "Every requested/failed/refused attempt consumes its one fixed slot; no replacement or call33. A scientific failure stops its dependent workflow but independent model workers may continue. Identity, storage, timeout or missing cleanup stops later workers; all NOT_RUN remain explicit."
    )
    plan["offline_replay"] = (
        "replay.py reconstructs IEEE/index, each constituent/composite error and native parity/directions from originals; source authentication and complete precision/work remain separate axes"
    )
    plan["remaining"]["runtime_metadata_copy"] = (
        "RESOLVED exact original schema and SHA; no new scan was used to recover it"
    )
    plan["remaining"]["activation_pin"] = (
        "NEW CPU-observed byte pin; verify before source on future actual GPU image, never relabel old smoke activation provenance"
    )
    plan["supersedes_host_plan"] = {
        "version": 2,
        "plan_sha256": "b29114762acd79ec0a11685f3e0719d59cbfd87df07a37afa1ed302bf19aff3b",
        "packet_sha256": "0ad6b48505c06ba05db697709b4173d756dff140019e6f7da5720e2728a0f0b8",
        "state": "NOT_ADMITTED; original immutable packet retained",
    }
    plan["root_release_contract"] = "generativeqc.h100-small-cbs-root-release.v2"
    plan["creation_binding"] = (
        "root must separately pin actual new Job name+expected creation identity+submission window; canonical comparison never rewrites raw CLI JSON"
    )
    plan["independent_composite_model"] = inputs["independent_composite_model"]
    plan["raw_first_contract"] = (
        "minimal composite/available constituent E/F and complete outer seconds are fsynced in observables before optional metadata; enrichment/IO/integrity failures stop later models, never counted as continuable scientific errors"
    )
    plan["scientific_failure_continuation"] = (
        "only SCI_NATIVE/SCIENTIFIC_REQUEST or SCI_NUMERIC/SCIENTIFIC_NUMERIC with no timeout/storage/cleanup trigger; all control/transport/integrity failures halt"
    )
    plan["new_adapter_status"] = (
        "V3_HOST_REPAIRED_NOT_EXECUTED; independent root admission and separate actual SCI/resource authorization remain required"
    )
    write_new(here / "prepared-plan-v3.json", plan)
    print(sha(here / "prepared-plan-v3.json"))

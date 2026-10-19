"""Offline source/work/activity qualification; global unknowns stay explicit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from benchmarks.cc_cupti_report import OBSERVER_FENCE_ID, exclusive_ledger
from benchmarks.cc_source_census import validate_cc_census
from benchmarks.cc_tile_environment import (
    canonical_hash,
    execution_environment,
    validate_child,
)
from benchmarks.cc_tile_pipeline_campaign import CONTROLS, WORK, check_science
from benchmarks.cc_trace_reader import (
    activity_inventory,
    open_trace,
    regular_stream,
    require,
    source_inventory,
)


def closed_bytes(path: Path) -> bytes:
    """Read a finite, stable opened regular object without following or blocking."""
    with regular_stream(path) as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(512 * 1024 + 1)
        after = os.fstat(stream.fileno())
    require(len(raw) <= 512 * 1024, "JSON evidence exceeds admission bound")
    require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        and len(raw) == before.st_size,
        "closed evidence changed during read",
    )
    return raw


def json_record(path: Path, expected_hash: str | None = None) -> tuple[dict, str]:
    """Bind finite closed JSON bytes; expected hashes come from the frozen packet."""
    raw = closed_bytes(path)
    digest = hashlib.sha256(raw).hexdigest()
    require(
        expected_hash is None or digest == expected_hash, "JSON evidence hash mismatch"
    )
    value = json.loads(raw)
    require(isinstance(value, dict), "JSON evidence must be an object")
    return value, digest


# CUPTI12.9.79/API28, from the pinned NVIDIA wheel headers. These are vendor
# ABI definitions, not hashes or spellings of production implementation bodies.
RUNTIME = 5
STREAM_SYNCHRONIZE = 131
DEVICE_SYNCHRONIZE = 165
GPU_SITES = frozenset((3, 6, 7, 10, 11, 12, 19, 20, 22, 23, 24, 27, 29, 30, 31))
KERNEL_SITES = GPU_SITES - {24, 31}
HOST_METADATA_SITES = frozenset((5, 9, 26, 32))
EXTRA_WORK = (
    "ccsd_replay_evaluations",
    "df_source_scans",
    "df_source_reads",
    "df_transform_stages",
    "df_block_gemms",
    "df_cuda_transform_calls",
    "df_transform_fmas",
    "triples_panel_calls",
    "triples_epilogue_kernels",
    "triples_panel_capacity",
    "triples_reduction_kernels",
    "triples_moment_products",
    "triples_fp64_gemms",
)


def validate_frozen_child(
    spec: dict, role: str, receipt: dict, plan: dict, witness: dict
) -> None:
    """Bind self-consistent child documents to the external frozen recipe.

    This consumer covers the declared strict-FP64 three-role recipe, not an
    arbitrary campaign whose three children merely agree with each other.
    """
    require(role in ("clean-1", "clean-2", "diagnostic"), "unknown child role")
    run = spec["run"]
    directory = PurePosixPath(run["input"]).parent / role
    argv = [
        run["executable"],
        run["input"],
        str(directory / "endpoint.json"),
        *CONTROLS,
    ]
    diagnostic = role == "diagnostic"
    if diagnostic:
        argv += ["--cc-cupti-v1", run["backend"], str(directory / "capture")]
    require(
        plan["argv"] == receipt["command"] == argv
        and receipt["diagnostic"] is diagnostic,
        "child differs from complete frozen role argv",
    )
    expected_environment = execution_environment(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":16:8",
            "PATH": "/opt/generativeqc-reuse/venv/bin:/opt/generativeqc-reuse/cuda-12.9/bin:/usr/bin:/bin",
            "LD_LIBRARY_PATH": spec["library_path"],
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "LC_CTYPE": "C.UTF-8",
        }
    )
    require(
        plan["environment"] == expected_environment,
        "child differs from frozen semantic/observation environment",
    )
    process = receipt["native_process"]
    require(
        type(process.get("pid")) is int
        and process["pid"] > 0
        and isinstance(process.get("starttime"), str)
        and process["starttime"].isdigit()
        and int(process["starttime"]) > 0
        and witness.get("process") == process,
        "child witness native PID/start identity mismatch",
    )
    validate_child(witness, plan, process["pid"], process)


def audit_campaign(root: Path, spec_sha256: str, worker_sha256: str) -> dict[str, Any]:
    """Recompute the closed three-child evidence with external frozen anchors.

    Loader and allocation attestations remain explicitly producer observations;
    this audit independently checks their frozen bindings and the actual raw
    source/work/activity records. It does not fabricate a remote independent
    DSO rehash or retroactively repair execution supervision.
    """
    spec, _ = json_record(root / "spec.json", spec_sha256)
    require(
        hashlib.sha256(closed_bytes(root / "worker.py")).hexdigest() == worker_sha256,
        "worker differs from frozen source",
    )
    require(
        spec["roles"] == ["clean-1", "clean-2", "diagnostic"]
        and spec["scientific_sample_limit"] == 3
        and spec["automatic_retries"] == 0,
        "unadmitted sample plan",
    )
    oracle, oracle_hash = json_record(
        root / "oracle.json", spec["run"]["sha256"]["oracle"]
    )
    require(
        oracle_hash
        == "ba40ed0b759eb27a9f3909377a6da0795fd292019a9b079e58c82929941a059b",
        "independent reference identity changed",
    )
    require(
        spec["run"]["sha256"]["input"]
        == "9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e",
        "frozen scientific input changed",
    )
    require(
        hashlib.sha256(closed_bytes(root / "input.input")).hexdigest()
        == spec["run"]["sha256"]["input"],
        "actual molecular input differs",
    )
    campaign, campaign_hash = json_record(root / "campaign-result.json")
    require(len(campaign["children"]) == 3, "missing/extra scientific children")
    rows = []
    receipts = []
    child_evidence = []
    profiles = []
    identities = set()
    for role, declared in zip(spec["roles"], campaign["children"], strict=True):
        require(role == declared["role"], "scientific child order changed")
        directory = root / role
        receipt, receipt_hash = json_record(
            directory / "receipt.json", declared["receipt_sha256"]
        )
        endpoint, endpoint_hash = json_record(directory / "endpoint.json")
        witness, _ = json_record(directory / "child_environment.json")
        plan, _ = json_record(directory / "execution_plan.json")
        process = receipt["native_process"]
        validate_frozen_child(spec, role, receipt, plan, witness)
        require(
            receipt["native_exit"] == 0 and receipt["cleanup_errors"] == [],
            "native process/lifetime failed",
        )
        require(
            receipt["inputs_before"] == receipt["inputs_after"]
            and receipt["frozen_inputs_unchanged"] is True,
            "native inputs drifted",
        )
        require(
            receipt["semantic_record"] == endpoint,
            "receipt differs from actual endpoint bytes",
        )
        require(
            receipt["allocation_receipt_sha256"]
            == spec["run"]["allocation_receipt_sha256"],
            "allocation receipt changed",
        )
        numerical = check_science(endpoint, oracle)
        require(
            numerical["status"] == "PASS", "original independent numerical gates failed"
        )
        bindings = receipt["native_bindings"]["libraries"]
        expected = {**spec["run"]["mapped_libraries"]}
        for path, driver in spec["run"]["driver_libraries"].items():
            expected["actual_driver:" + path] = {
                "path": path,
                "sha256": driver["sha256"],
            }
        if role == "diagnostic":
            for name in ("backend", "cupti"):
                expected["observer_" + name] = {
                    "path": spec["run"][name],
                    "sha256": spec["run"]["sha256"][name],
                }
        require(set(bindings) == set(expected), "observed module inventory changed")
        require(
            all(
                bindings[name]["path"] == value["path"]
                and bindings[name]["sha256"] == value["sha256"]
                for name, value in expected.items()
            ),
            "observed module identity changed",
        )
        identity = (process["pid"], process["starttime"])
        require(
            identity not in identities, "reused native process is not a fresh child"
        )
        identities.add(identity)
        profiles.append(canonical_hash(plan["environment"]["profile"]))
        require(
            type(receipt["supervised_child_wall_ns"]) is int
            and receipt["supervised_child_wall_ns"] > 0,
            "complete child wall unavailable",
        )
        child_evidence.append(
            {
                "role": role,
                "receipt_sha256": receipt_hash,
                "endpoint_sha256": endpoint_hash,
                "native_process": process,
                "independent_science": numerical,
                "supervised_child_wall_ns": receipt["supervised_child_wall_ns"],
            }
        )
        rows.append(endpoint)
        receipts.append(receipt)
    require(len(set(profiles)) == 1, "scientific semantic environments differ")
    for key in (*WORK, *EXTRA_WORK):
        require(
            all(type(row.get(key)) is int and row[key] >= 0 for row in rows)
            and len({row[key] for row in rows}) == 1,
            "semantic work differs/unavailable: " + key,
        )
    for key in (
        "total_energy",
        "triples_energy",
        "ccsd_replay_r1_max",
        "ccsd_replay_r2_max",
    ):
        require(
            len({struct.pack("!d", row[key]) for row in rows}) == 1,
            "numerical bits differ: " + key,
        )
    traces = {
        kind: open_trace(root / "diagnostic" / f"capture.{kind}.bin")
        for kind in ("activity", "source", "graph", "legacy")
    }
    require(
        len({trace.identity[:2] for trace in traces.values()}) == 4,
        "raw trace aliasing",
    )
    for kind, trace in traces.items():
        require(
            trace.sha256 == receipts[2]["raw_artifacts"][kind]["sha256"],
            "raw artifact differs from producer receipt",
        )
    graph, legacy = traces["graph"].metadata, traces["legacy"].metadata
    require(
        all(graph[i] == 0 for i in (2, 3, 7)) and graph[6] == 1,
        "graph loss/error/open lifetime",
    )
    require(
        all(legacy[i] == 0 for i in (2, 3, 4)) and legacy[5:8] == (1, 28, 2),
        "legacy source loss/error/tail",
    )
    source = source_inventory(traces["source"])
    require(
        source["pid"] == receipts[2]["native_process"]["pid"]
        and str(source["pid_starttime"])
        == str(receipts[2]["native_process"]["starttime"]),
        "source journal process mismatch",
    )
    before, after = receipts[2]["parent_before"], receipts[2]["parent_after"]
    require(
        before["clock"] == after["clock"] == "cupti28-normalized-default-ns"
        and before["library_sha256"]
        == after["library_sha256"]
        == spec["run"]["sha256"]["cupti"],
        "caller clock source mismatch",
    )
    require(
        before["cupti_ns"]
        <= traces["source"].metadata[17]
        <= traces["source"].metadata[18]
        <= after["cupti_ns"],
        "source clock envelope outside caller",
    )
    selected = qualify_targets(
        source,
        activity_inventory(traces["activity"]),
        rows[2],
        (oracle["nocc"], oracle["nvir"], oracle["naux"]),
        (before["cupti_ns"], after["cupti_ns"]),
        (0,),
    )
    guard, _ = json_record(root / "outer-guard-receipt.json")
    return {
        "schema": "generativeqc.cc-cupti-campaign-audit.v1",
        "science": "PASS_THREE_FRESH_NATIVE_CHILDREN_ORIGINAL_INDEPENDENT_GATES_BITS_AND_WORK",
        "selected_phase_qualification": selected,
        "frozen_packet": {
            "spec_sha256": spec_sha256,
            "worker_sha256": worker_sha256,
            "production_source": spec["production_source"],
            "library_sha256": spec["run"]["sha256"]["library"],
        },
        "campaign_result_sha256": campaign_hash,
        "children": child_evidence,
        "raw_files": {
            key: {
                "sha256": value.sha256,
                "records": value.count,
                "bytes": value.identity[2],
            }
            for key, value in traces.items()
        },
        "loader_attestation_scope": "producer observed actual PID/maps/inodes/digests; frozen bindings checked here, not an independent remote DSO rehash",
        "execution_supervision": {
            "native_deadline_seconds": 1800,
            "outer_guard_attached_after_launch": guard["attached_after_launch"],
            "outer_guard_seen_actors": len(guard["seen"]),
            "outer_guard_drained": guard.get("drained"),
            "limitation": "No prelaunch outer controller bound covered this run; a late empty-session observation is not execution-time kill/reap evidence.",
        },
        "parent_issue": "OPEN_REMAINING_OPTIMIZATION_RESPONSE_FORCE_AND_GLOBAL_SCOPE",
    }


def qualify_targets(
    source: dict[str, Any],
    activity: dict[str, Any],
    semantic: dict[str, int],
    geometry: tuple[int, int, int],
    parent_window: tuple[int, int],
    allowed_lowerings: tuple[int, ...],
) -> dict[str, Any]:
    """Require each admitted target, source join and explicit wait independently."""
    census = validate_cc_census(source, semantic, geometry, allowed_lowerings)
    start, end = parent_window
    require(
        type(start) is int and type(end) is int and 0 < start < end,
        "invalid caller window",
    )
    phases = source["phases"]
    root = source["root"]
    require(
        start <= root["start"] <= root["end"] <= end,
        "source root outside full caller window",
    )
    per_phase: dict[int, Counter[str]] = defaultdict(Counter)
    by_site: dict[int, Counter[str]] = defaultdict(Counter)
    unknown: Counter[str] = Counter()
    intervals = []
    fence = []
    sync_api: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    sync_device: dict[int, list[tuple[int, ...]]] = defaultdict(list)

    for row in activity["gpu"] + activity["waits"] + activity["api"]:
        identity = activity["external"].get((3, row[5]))
        phase = phases.get(identity)
        kind = (
            "kernel"
            if row[0] == 10
            else "memset"
            if row[0] == 2
            else "copy"
            if row[0] in (1, 22)
            else "device_wait"
            if row[0] == 38
            else "host_api"
        )
        require(
            start <= row[3] <= row[4] <= end,
            "activity outside independent caller window",
        )
        if kind != "host_api":
            intervals.append((row[3], row[4], kind))
        if phase is not None:
            if phase["words"][4] in GPU_SITES | {13}:
                require(
                    row[3] >= phase["start"],
                    "target activity precedes its real source submission",
                )
            per_phase[identity][kind] += 1
            by_site[phase["words"][4]][kind] += 1
        elif identity == OBSERVER_FENCE_ID:
            require(
                kind not in ("kernel", "copy", "memset"),
                "GPU work incorrectly attributed to observer fence",
            )
            unknown["explicit_observer_" + kind] += 1
        else:
            unknown["unattributed_" + kind] += 1
        if (
            identity == OBSERVER_FENCE_ID
            and row[0] == RUNTIME
            and row[1] == DEVICE_SYNCHRONIZE
        ):
            require(
                row[13] == 0 and row[14] == source["tid"],
                "observer fence failed or wrong thread",
            )
            fence.append(row)
        if phase is not None and phase["words"][4] == 13:
            if row[0] == RUNTIME and row[1] in (STREAM_SYNCHRONIZE, DEVICE_SYNCHRONIZE):
                require(
                    row[13] == 0 and row[14] == source["tid"],
                    "target wait API failed or wrong thread",
                )
                sync_api[identity].append(row)
            elif row[0] == 38:
                require(row[13] == 0, "target device wait failed")
                sync_device[identity].append(row)

    require(len(fence) == 1, "missing/ambiguous exact observer completion fence")
    require(
        root["end"] <= fence[0][3],
        "observer fence begins before scientific source scope ends",
    )
    require(
        all(row[4] <= fence[0][4] for row in activity["gpu"] + activity["waits"]),
        "asynchronous activity extends past actual completion fence",
    )
    target_ids = []
    wait_ids = []
    for identity, phase in phases.items():
        site = phase["words"][4]
        if site in GPU_SITES:
            require(phase["status"] == 0, "target submission did not complete")
            require(
                per_phase[identity]["kernel" if site in KERNEL_SITES else "copy"] > 0,
                "required target lacks actual GPU activity",
            )
            target_ids.append(identity)
        if site in (24, 31):
            copies = [
                row
                for row in activity["gpu"]
                if row[0] in (1, 22)
                and activity["external"].get((3, row[5])) == identity
            ]
            require(
                sum(row[2] for row in copies) == phase["words"][6],
                "publication transfer bytes differ from source domain",
            )
        if site == 13:
            require(
                sync_api[identity] and sync_device[identity],
                "required explicit wait lacks host API or device wait record",
            )
            require(
                {row[5] for row in sync_device[identity]}
                <= {row[5] for row in sync_api[identity]},
                "device wait not joined to its real explicit API",
            )
            wait_ids.append(identity)
        if site in HOST_METADATA_SITES:
            intervals.append(
                (phase["start"], phase["end"], "host_metadata_preparation")
            )
    require(wait_ids, "explicit wait coverage absent")
    downloads = [phase for phase in phases.values() if phase["words"][4] == 31]
    require(
        len(downloads) == 1 and downloads[0]["words"][6] == 12,
        "missing/extra strict-FP64 triples energy/error download",
    )
    ledger = exclusive_ledger(start, end, intervals)
    return {
        "schema": "generativeqc.cc-cupti-target-qualification.v1",
        "selected_phase_coverage": "PASS_SOURCE_WORK_ACTIVITY_AND_EXPLICIT_WAIT_JOINS",
        "source_work_census": census,
        "target_submission_phases": len(target_ids),
        "explicit_wait_phases": len(wait_ids),
        "activity_by_source_site": {
            str(key): dict(value) for key, value in by_site.items()
        },
        "full_caller_cupti_window_ns": {
            "start": start,
            "end": end,
            "duration": end - start,
        },
        "exclusive_ledger_ns": ledger,
        "unattributed_activity": dict(unknown),
        "source_root_ns": {"start": root["start"], "end": root["end"]},
        "actual_completion_fence_ns": {
            "start": fence[0][3],
            "end": fence[0][4],
            "correlation": fence[0][5],
        },
        "whole_program_fine_attribution": "INCOMPLETE_UNATTRIBUTED_RESIDUAL_PRESERVED",
        "implicit_library_wait_coverage": "UNAVAILABLE",
        "causal_hideable_time": "UNAVAILABLE",
        "performance_profitability": "UNAVAILABLE",
        "note": "Host metadata is not CPU tensor production; observed overlap is not causality. Caller/source/supervised monotonic boundaries are never subtracted.",
    }


def main() -> int:
    """Audit closed artifacts against reviewed external packet hashes."""
    from benchmarks.cc_tile_environment import atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_campaign(args.campaign, args.spec_sha256, args.worker_sha256)
    atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

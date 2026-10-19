"""Independent admission/joins/tail controls, never hardware/science evidence."""

from __future__ import annotations

import copy

import pytest
from test_cc_source_census import fixture

from benchmarks.cc_cupti_qualification import (
    GPU_SITES,
    qualify_targets,
    validate_frozen_child,
)
from benchmarks.cc_cupti_report import OBSERVER_FENCE_ID
from benchmarks.cc_tile_environment import (
    CHILD_SCHEMA,
    canonical_hash,
    execution_environment,
)
from benchmarks.cc_tile_pipeline_campaign import CONTROLS
from benchmarks.cc_trace_reader import PHASE, InvalidTrace


@pytest.mark.parametrize("role", ["clean-1", "clean-2", "diagnostic"])
@pytest.mark.parametrize(
    "mutation", [None, "extra_argument", "workspace", "starttime", "injection"]
)
def test_child_documents_bind_frozen_recipe(role: str, mutation: str | None) -> None:
    spec = {
        "run": {
            "executable": "/compute/endpoint",
            "input": "/campaign/input.input",
            "backend": "/compute/observer.so",
        },
        "library_path": "/compute/runtime:/cuda/lib64",
    }
    argv = [
        "/compute/endpoint",
        "/campaign/input.input",
        f"/campaign/{role}/endpoint.json",
        *CONTROLS,
    ]
    if role == "diagnostic":
        argv += ["--cc-cupti-v1", "/compute/observer.so", f"/campaign/{role}/capture"]
    environment = {
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
    if mutation == "extra_argument":
        argv += ["--unadmitted-rebound-argument"]
    elif mutation == "workspace":
        del environment["CUBLAS_WORKSPACE_CONFIG"]
    elif mutation == "injection":
        environment["LD_PRELOAD"] = "/unadmitted/inject.so"
    plan = {"argv": argv, "environment": execution_environment(environment)}
    plan["profile_sha256"] = canonical_hash(plan["environment"]["profile"])
    process = {"pid": 101, "starttime": "123"}
    receipt = {
        "command": argv,
        "native_process": process,
        "diagnostic": role == "diagnostic",
    }
    witness = {
        "schema": CHILD_SCHEMA,
        "status": "PREPARED_FOR_EXEC",
        "pid": 101,
        "argv": argv,
        "environment": plan["environment"],
        "environment_sha256": canonical_hash(plan["environment"]),
        "process": {**process, "starttime": "999999999999"}
        if mutation == "starttime"
        else process,
    }
    if mutation is None:
        validate_frozen_child(spec, role, receipt, plan, witness)
    else:
        with pytest.raises(InvalidTrace):
            validate_frozen_child(spec, role, receipt, plan, witness)


def protocol_case() -> tuple:
    source, semantic, geometry = fixture()
    root = next(p for p in source["phases"].values() if p["words"][4] == 1)
    root["end"] = 100000
    source.update(root=root, tid=77)
    wait_id = max(source["phases"]) + 1
    source["phases"][wait_id] = {
        "words": (PHASE, 1, wait_id, root["words"][2], 13, 0, 0, 0, 0, 0, 77, 0),
        "start": 80000,
        "end": 80010,
        "status": 0,
    }
    source["begin_order"].append(wait_id)
    download_id = wait_id + 1
    source["phases"][download_id] = {
        "words": (PHASE, 1, download_id, root["words"][2], 31, 0, 12, 0, 0, 0, 77, 0),
        "start": 49000,
        "end": 49001,
        "status": 0,
    }
    source["begin_order"].append(download_id)
    activity = {"gpu": [], "waits": [], "api": [], "external": {}}

    def row(
        kind: int, identity: int, cid: int, *, size: int = 0, cbid: int = 0
    ) -> tuple:
        words = [0] * 20
        words[0], words[1], words[2] = kind, cbid, size
        words[3:6] = (50000, 50001, cid)
        words[14] = 77
        activity["external"][(3, cid)] = identity
        return tuple(words)

    for identity, phase in source["phases"].items():
        site = phase["words"][4]
        if site in GPU_SITES:
            activity["gpu"].append(
                row(
                    1 if site in (24, 31) else 10,
                    identity,
                    identity,
                    size=phase["words"][6] if site in (24, 31) else 0,
                )
            )
    activity["api"].append(row(5, wait_id, 900001, cbid=131))
    activity["waits"].append(row(38, wait_id, 900001))
    for name in ("api", "waits"):
        adjusted = list(activity[name][-1])
        adjusted[3:5] = (80001, 80002)
        activity[name][-1] = tuple(adjusted)
    fence = list(row(5, OBSERVER_FENCE_ID, 900002, cbid=165))
    fence[3:5] = (100010, 100020)
    activity["api"].append(tuple(fence))
    return source, activity, semantic, geometry


def test_selected_targets_pass_with_explicit_global_unknown_residual() -> None:
    source, activity, semantic, geometry = protocol_case()
    result = qualify_targets(source, activity, semantic, geometry, (1, 110000), (0,))
    assert result["selected_phase_coverage"].startswith("PASS")
    assert sum(result["exclusive_ledger_ns"].values()) == 109999
    assert result["exclusive_ledger_ns"]["unattributed"] > 0
    assert result["whole_program_fine_attribution"].startswith("INCOMPLETE")
    assert result["causal_hideable_time"] == "UNAVAILABLE"


@pytest.mark.parametrize(
    "failure",
    (
        "missing_gpu",
        "missing_device_wait",
        "failed_wait",
        "wrong_tid",
        "fence_gpu",
        "past_fence",
        "wrong_copy_bytes",
        "kernel_replaced_by_memset",
        "download_replaced_by_kernel",
        "activity_before_submission",
        "fence_before_root_end",
        "joint_missing_download",
    ),
)
def test_target_activity_wait_and_tail_fail_closed(failure: str) -> None:
    source, activity, semantic, geometry = copy.deepcopy(protocol_case())
    if failure == "missing_gpu":
        activity["gpu"].pop(0)
    elif failure == "missing_device_wait":
        activity["waits"].clear()
    elif failure in ("failed_wait", "wrong_tid"):
        row = list(activity["api"][0])
        row[13 if failure == "failed_wait" else 14] = 99
        activity["api"][0] = tuple(row)
    elif failure == "fence_gpu":
        row = list(activity["gpu"][0])
        row[5] = 900002
        activity["gpu"].append(tuple(row))
    elif failure == "past_fence":
        row = list(activity["gpu"][0])
        row[4] = 100021
        activity["gpu"][0] = tuple(row)
    elif failure in ("kernel_replaced_by_memset", "activity_before_submission"):
        index = next(i for i, row in enumerate(activity["gpu"]) if row[0] == 10)
        row = list(activity["gpu"][index])
        if failure == "kernel_replaced_by_memset":
            row[0] = 2
        else:
            identity = activity["external"][(3, row[5])]
            row[3] = source["phases"][identity]["start"] - 1
        activity["gpu"][index] = tuple(row)
    elif failure == "fence_before_root_end":
        source["root"]["end"] = 100011
    elif failure in ("download_replaced_by_kernel", "joint_missing_download"):
        identity = next(i for i, p in source["phases"].items() if p["words"][4] == 31)
        index = next(
            i
            for i, row in enumerate(activity["gpu"])
            if activity["external"][(3, row[5])] == identity
        )
        if failure == "download_replaced_by_kernel":
            row = list(activity["gpu"][index])
            row[0] = 10
            activity["gpu"][index] = tuple(row)
        else:
            activity["gpu"].pop(index)
            del source["phases"][identity]
            source["begin_order"].remove(identity)
    else:
        index = next(i for i, row in enumerate(activity["gpu"]) if row[0] == 1)
        row = list(activity["gpu"][index])
        row[2] += 8
        activity["gpu"][index] = tuple(row)
    with pytest.raises(InvalidTrace):
        qualify_targets(source, activity, semantic, geometry, (1, 110000), (0,))

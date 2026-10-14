"""Hand-clock and discrete independent oracles for attribution, not CC math."""

import random
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from benchmarks.cc_tile_pipeline_attribution import (
    KINDS,
    SCHEMA,
    checksum,
    read_nsys,
    summarize,
)
from benchmarks.cc_tile_pipeline_campaign import WORK, check_science


def fixture() -> tuple[dict, dict]:
    identity = {
        "source_commit": "a" * 40,
        "library_sha256": "b" * 64,
        "executable_sha256": "c" * 64,
        "input_sha256": "d" * 64,
        "environment_sha256": "e" * 64,
        "device_uuid": "GPU-test",
        "run_id": "hand-clock",
    }
    window = {"start_ns": 0, "end_ns": 30, "wall_ns": 30}
    trace = {
        "schema": SCHEMA,
        "identity": identity,
        "endpoint": window,
        "coverage": dict.fromkeys(KINDS, True),
        "capture": {
            "complete_tail": True,
            "drained": True,
            "export_complete": True,
            "lossless": True,
            "outcome": "success",
            "profiler_version": "fixture-v1",
        },
        "events": [],
    }
    receipt = {
        "schema": SCHEMA,
        "identity": deepcopy(identity),
        "endpoint": deepcopy(window),
    }
    return trace, receipt


def event(name: str, kind: str, start: int, end: int, **extra: Any) -> dict:
    return dict(id=name, kind=kind, start_ns=start, end_ns=end, **extra)


def test_cross_stream_nested_unordered_hand_oracle() -> None:
    trace, receipt = fixture()
    trace["events"] = [
        event("p", "producer", 0, 8),
        event("p-nested", "producer", 2, 4),
        event("k1", "kernel", 5, 15, stream=1),
        event("k2", "kernel", 10, 20, stream=2),
        event("c", "copy", 12, 22, stream=3, direction="H2D", bytes=512),
        event("w", "wait", 18, 26),
        event("m", "memset", 27, 29, stream=0),
    ]
    random.Random(12).shuffle(trace["events"])
    report = summarize(trace, receipt)
    assert report["status"] == "PASS"
    assert report["union_ns"]["kernel"] == 15
    assert report["duration_sum_ns_explanatory_only"]["kernel"] == 20
    assert report["union_ns"]["producer"] == 8
    assert report["intersection_ns"]["kernel+copy"] == 8
    assert report["no_recorded_gpu_activity_ns"] == 11
    assert report["producer_during_gpu_gap_ns"] == 5
    assert report["wait_during_gpu_gap_ns"] == 4
    assert report["observed_dma_bytes"] == {"H2D": 512, "D2H": 0, "D2D": 0}
    assert sum(report["observed_exclusive_wall_ns"].values()) == 30
    assert report["critical_path_hideable_ns"] is None


def test_discrete_oracle_does_not_use_interval_sweep() -> None:
    trace, receipt = fixture()
    rng = random.Random(1765)
    trace["events"] = [
        event(
            str(i),
            rng.choice(KINDS),
            a,
            rng.randrange(a + 1, 31),
            stream=i % 3,
            direction="D2D",
            bytes=i,
        )
        for i in range(50)
        for a in [rng.randrange(30)]
    ]
    report = summarize(trace, receipt)
    # Independently enumerate each integer clock cell, grouping set membership.
    ledger = {}
    for tick in range(30):
        kinds = sorted(
            {
                row["kind"]
                for row in trace["events"]
                if row["start_ns"] <= tick < row["end_ns"]
            }
        )
        key = "+".join(kinds) or "unattributed"
        ledger[key] = ledger.get(key, 0) + 1
    assert report["observed_exclusive_wall_ns"] == ledger


@pytest.mark.parametrize(
    "flag", ["complete_tail", "drained", "export_complete", "lossless"]
)
def test_missing_tail_loss_or_drain_is_incomplete(flag: str) -> None:
    trace, receipt = fixture()
    trace["capture"][flag] = False
    result = summarize(trace, receipt)
    assert result["status"] == "INCOMPLETE"
    assert result["no_recorded_gpu_activity_ns"] is None
    assert result["critical_path_hideable_ns"] is None


def test_failure_is_retained_after_successful_drain() -> None:
    trace, receipt = fixture()
    trace["capture"]["outcome"] = "failure"
    result = summarize(trace, receipt)
    assert result["status"] == "INCOMPLETE"
    assert "endpoint failed" in result["issues"]


def test_opaque_producer_never_becomes_zero() -> None:
    trace, receipt = fixture()
    trace["coverage"]["producer"] = False
    result = summarize(trace, receipt)
    assert result["status"] == "INCOMPLETE"
    assert result["union_ns"]["producer"] is None
    assert result["producer_during_gpu_gap_ns"] is None
    assert result["counts"]["producer"] is None


@pytest.mark.parametrize(
    "key", ["source_commit", "library_sha256", "input_sha256", "run_id"]
)
def test_identity_mismatch_rejected(key: str) -> None:
    trace, receipt = fixture()
    trace["identity"][key] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        summarize(trace, receipt)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "outside",
        "reversed",
        "float",
        "bool",
        "unknown-copy",
        "negative-bytes",
    ],
)
def test_invalid_activity_rejected(change: str) -> None:
    trace, receipt = fixture()
    row = event("a", "copy", 2, 5, stream=0, direction="D2H", bytes=8)
    trace["events"] = [row]
    if change == "duplicate":
        trace["events"].append(deepcopy(row))
    elif change == "outside":
        row["end_ns"] = 31
    elif change == "reversed":
        row["end_ns"] = 1
    elif change == "float":
        row["start_ns"] = 2.0
    elif change == "bool":
        row["start_ns"] = True
    elif change == "unknown-copy":
        row["direction"] = "unknown"
    else:
        row["bytes"] = -1
    with pytest.raises(ValueError):
        summarize(trace, receipt)


def test_partial_boundary_cannot_replace_complete_receipt() -> None:
    trace, receipt = fixture()
    trace["endpoint"] = {"start_ns": 0, "end_ns": 20, "wall_ns": 20}
    with pytest.raises(ValueError, match="boundary"):
        summarize(trace, receipt)


def test_no_material_observed_gap_negative_control() -> None:
    trace, receipt = fixture()
    trace["events"] = [event("compute", "kernel", 0, 30, stream=0)]
    result = summarize(trace, receipt)
    assert result["no_recorded_gpu_activity_ns"] == 0
    assert result["producer_during_gpu_gap_ns"] == 0
    assert result["critical_path_hideable_ns"] is None


def test_missing_copy_table_never_implies_zero_transfers() -> None:
    trace, receipt = fixture()
    trace["coverage"]["copy"] = False
    result = summarize(trace, receipt)
    assert result["observed_dma_bytes"] == {"H2D": None, "D2H": None, "D2D": None}
    assert result["no_recorded_gpu_activity_ns"] is None


def sqlite_fixture(tmp_path: Path) -> tuple[Path, dict]:
    trace, receipt = fixture()
    path = tmp_path / "nsys.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
            create table StringIds(id integer, value text);
            insert into StringIds values(1,'cudaStreamWaitEvent');
            create table NVTX_EVENTS(start integer,end integer,globalTid integer,text text,textId integer);
            insert into NVTX_EVENTS values(0,30,7,'endpoint',null),(0,4,8,'prepare',null);
            create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,
                deviceId integer,streamId integer,correlationId integer);
            insert into CUPTI_ACTIVITY_KIND_KERNEL values(5,20,1,0,2,1),(1,100,2,0,2,1);
            create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer,end integer,globalPid integer,
                deviceId integer,streamId integer,correlationId integer,bytes integer,copyKind integer);
            insert into CUPTI_ACTIVITY_KIND_MEMCPY values(10,22,1,0,3,2,512,1);
            create table CUPTI_ACTIVITY_KIND_MEMSET(start integer,end integer,globalPid integer,
                deviceId integer,streamId integer,correlationId integer);
            create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,
                correlationId integer,nameId integer);
            insert into CUPTI_ACTIVITY_KIND_RUNTIME values(24,25,8,3,1);
        """)
    receipt["capture"] = trace["capture"]
    receipt["nsys"] = {
        "sqlite_sha256": checksum(path),
        "sqlite_size": path.stat().st_size,
        "cuda_global_pid": 1,
        "device_id": 0,
        "runtime_global_tids": [8],
        "producer_labels": ["prepare"],
        "producer_coverage_complete": True,
        "endpoint_label": "endpoint",
        "endpoint_global_tid": 7,
    }
    return path, receipt


def test_sqlite_extraction_preserves_async_tail_and_unknown_device_wait(
    tmp_path: Path,
) -> None:
    path, receipt = sqlite_fixture(tmp_path)
    trace = read_nsys(path, receipt)
    result = summarize(trace, receipt)
    assert result["status"] == "INCOMPLETE"  # API enqueue is not device event wait.
    assert result["union_ns"]["wait"] is None
    assert result["union_ns"]["copy"] == 12
    assert result["observed_dma_bytes"]["H2D"] == 512
    assert len(trace["events"]) == 4  # Other process is excluded before bounds check.
    assert trace["events"][2]["api"] == "cudaStreamWaitEvent"


@pytest.mark.parametrize(
    "defect", ["digest", "tail", "duplicate", "copy-kind", "missing-column"]
)
def test_sqlite_negative_controls(tmp_path: Path, defect: str) -> None:
    path, receipt = sqlite_fixture(tmp_path)
    if defect == "digest":
        receipt["nsys"]["sqlite_sha256"] = "0" * 64
    else:
        with sqlite3.connect(path) as db:
            if defect == "tail":
                db.execute("update NVTX_EVENTS set end=null where text='endpoint'")
            elif defect == "duplicate":
                db.execute(
                    "insert into CUPTI_ACTIVITY_KIND_KERNEL select * from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=1"
                )
            elif defect == "copy-kind":
                db.execute("update CUPTI_ACTIVITY_KIND_MEMCPY set copyKind=0")
            else:
                db.execute(
                    "alter table CUPTI_ACTIVITY_KIND_MEMCPY rename column bytes to guessed_bytes"
                )
        receipt["nsys"].update(
            sqlite_sha256=checksum(path), sqlite_size=path.stat().st_size
        )
    with pytest.raises(ValueError):
        read_nsys(path, receipt)


@pytest.mark.parametrize("defect", [None, "energy", "nan", "mixed", "work", "response"])
def test_campaign_scientific_gates_preserve_negative_samples(
    defect: str | None,
) -> None:
    oracle = {
        "nocc": 9,
        "nvir": 221,
        "naux": 488,
        "total_energy": -79.0,
        "triples_energy": -0.01,
    }
    record = dict.fromkeys(WORK, 1)
    record.update(
        nbf=230,
        naux=488,
        forces_requested=0,
        total_energy=-79.0,
        triples_energy=-0.01,
        ccsd_replay_r1_max=1e-12,
        ccsd_replay_r2_max=1e-12,
        ccsd_mixed_evaluations=0,
        ccsd_fp32_gemm_calls=0,
        triples_fp32_gemms=0,
    )
    if defect == "energy":
        record["total_energy"] += 1e-7
    elif defect == "nan":
        record["total_energy"] = float("nan")
    elif defect == "mixed":
        record["ccsd_fp32_gemm_calls"] = 1
    elif defect == "work":
        record["ccsd_gemm_calls"] = None
    elif defect == "response":
        record["forces_requested"] = 1
    if defect not in (None, "energy"):
        with pytest.raises(ValueError):
            check_science(record, oracle)
    else:
        result = check_science(record, oracle)
        assert result["status"] == ("PASS" if defect is None else "FAIL")
        assert result["gates"]["total_energy_error"]["limit"] == 1e-8

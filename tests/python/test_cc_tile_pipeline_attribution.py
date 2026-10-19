"""Hand-clock and discrete independent oracles for attribution, not CC math."""

import gzip
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest

from benchmarks import cc_tile_pipeline_campaign as campaign
from benchmarks.cc_tile_environment import (
    CHILD_SCHEMA,
    atomic_json,
    canonical_hash,
    execution_environment,
    observation_policy,
    profile_environment,
)
from benchmarks.cc_tile_pipeline_attribution import (
    KINDS,
    SCHEMA,
    checksum,
    read_nsys,
    summarize,
)
from benchmarks.cc_tile_pipeline_attribution import main as attribution_main
from benchmarks.cc_tile_pipeline_campaign import WORK, check_science
from benchmarks.cc_tile_pipeline_campaign import main as campaign_main
from benchmarks.cc_tile_pipeline_campaign_report import compare_runs

CAMPAIGN_ARTIFACTS = {
    "native.log",
    "endpoint.json",
    "oracle.json",
    "execution_plan.json",
    "child_environment.json",
    "cc_tile_child.py",
    "cc_tile_environment.py",
}


def fake_endpoint(command: list[str], options: dict, endpoint: bytes) -> None:
    """Model the documented launcher/child evidence at the subprocess boundary."""
    plan = json.loads(Path(command[-2]).read_text())
    environment = execution_environment(options["env"])
    atomic_json(
        Path(command[-1]),
        {
            "schema": CHILD_SCHEMA,
            "status": "PREPARED_FOR_EXEC",
            "pid": 1765,
            "argv": plan["argv"],
            "environment": environment,
            "environment_sha256": canonical_hash(environment),
        },
    )
    Path(plan["argv"][2]).write_bytes(endpoint)


def mock_native_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign,
        "observe_native",
        lambda pid, executable, libraries: {
            "executable": str(executable.resolve()),
            "libraries": libraries,
        },
    )


def fixture() -> tuple[dict, dict]:
    policy = observation_policy(
        execution_environment({})["observation"],
        execution_environment({})["observation"],
    )
    identity = {
        "source_commit": "a" * 40,
        "library_sha256": "b" * 64,
        "executable_sha256": "c" * 64,
        "input_sha256": "d" * 64,
        "environment_sha256": canonical_hash(profile_environment({})),
        "environment_profile": profile_environment({}),
        "device_uuid": "GPU-test",
        "run_id": "hand-clock",
        "observation_policy": policy,
        "observation_policy_sha256": canonical_hash(policy),
    }
    window = {"start_ns": 0, "end_ns": 30, "wall_ns": 30}
    trace = {
        "diagnostic": True,
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
        "execution_environment": execution_environment({}),
        "execution_environment_sha256": canonical_hash(execution_environment({})),
    }
    receipt = {
        "diagnostic": True,
        "schema": SCHEMA,
        "identity": deepcopy(identity),
        "endpoint": deepcopy(window),
        "execution_environment": execution_environment({}),
        "execution_environment_sha256": canonical_hash(execution_environment({})),
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
            create table DIAGNOSTIC_EVENT(timestamp integer,timestampType integer,source integer,
                severity integer,text text,globalPid integer);
            insert into DIAGNOSTIC_EVENT values(0,1,3,1,'fixture profiling info',-1);
            create table ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id integer,name text,label text);
            insert into ENUM_DIAGNOSTIC_SEVERITY_LEVEL values
                (0,'UnknownLevel','Unknown'),(1,'Info','Info'),(2,'Warning','Warning'),
                (3,'Error','Error'),(4,'Verbose','Verbose');
            create table ENUM_DIAGNOSTIC_SOURCE_TYPE(id integer,name text,label text);
            insert into ENUM_DIAGNOSTIC_SOURCE_TYPE values
                (0,'UnknownSource','Unknown'),(1,'Injection','Injection'),
                (2,'Daemon','Daemon'),(3,'Analysis','Analysis');
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
            create table ENUM_CUPTI_SYNC_TYPE(id integer,name text,label text);
            insert into ENUM_CUPTI_SYNC_TYPE values(3,'CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT','Stream wait event');
            create table CUPTI_ACTIVITY_KIND_SYNCHRONIZATION(start integer,end integer,deviceId integer,
                streamId integer,correlationId integer,globalPid integer,syncType integer,
                eventId integer,eventSyncId integer);
            insert into CUPTI_ACTIVITY_KIND_SYNCHRONIZATION values(23,24,0,2,3,1,3,11,12);
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


def test_sqlite_extraction_preserves_async_tail_and_device_wait(
    tmp_path: Path,
) -> None:
    path, receipt = sqlite_fixture(tmp_path)
    trace = read_nsys(path, receipt)
    result = summarize(trace, receipt)
    assert result["status"] == "PASS"
    assert result["union_ns"]["wait"] == 1
    assert result["union_ns"]["copy"] == 12
    assert result["observed_dma_bytes"]["H2D"] == 512
    assert len(trace["events"]) == 4  # Other process is excluded before bounds check.
    assert trace["extraction"]["host_wait_api"] == {
        "coverage": True,
        "count": 1,
        "duration_sum_ns": 1,
        "union_ns": 1,
    }


def test_missing_device_synchronization_table_is_not_host_wait_zero(
    tmp_path: Path,
) -> None:
    path, receipt = sqlite_fixture(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("drop table CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
    receipt["nsys"].update(
        sqlite_sha256=checksum(path), sqlite_size=path.stat().st_size
    )
    result = summarize(read_nsys(path, receipt), receipt)
    assert result["status"] == "INCOMPLETE"
    assert result["union_ns"]["wait"] is None
    assert result["extraction"]["host_wait_api"]["count"] == 1


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
        assert result["gates"]["triples_energy_error"]["limit"] == 1e-10
        assert result["gates"]["physical_replay_r1"]["limit"] == 1e-10
        assert result["gates"]["physical_replay_r2"]["limit"] == 1e-10


def test_cli_retains_deterministic_bounded_segment_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace, receipt = fixture()
    trace["events"] = [event("compute", "kernel", 0, 30, stream=0)]
    trace_path, receipt_path = tmp_path / "trace.json", tmp_path / "receipt.json"
    trace_path.write_text(json.dumps(trace))
    receipt_path.write_text(json.dumps(receipt))
    output, segments = tmp_path / "report.json", tmp_path / "segments.json.gz"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cc_tile_pipeline_attribution.py",
            "--trace",
            str(trace_path),
            "--receipt",
            str(receipt_path),
            "--output",
            str(output),
            "--segments-output",
            str(segments),
        ],
    )
    assert attribution_main() == 0
    report = json.loads(output.read_text())
    with gzip.open(segments, "rt") as stream:
        ledger = json.load(stream)
    assert ledger == [{"start_ns": 0, "end_ns": 30, "kinds": ["kernel"]}]
    assert report["segments"]["count"] == 1
    assert report["segments"]["sha256"] == checksum(segments)


def test_cli_rejects_report_overwriting_segment_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace, receipt = fixture()
    trace_path, receipt_path = tmp_path / "trace.json", tmp_path / "receipt.json"
    trace_path.write_text(json.dumps(trace))
    receipt_path.write_text(json.dumps(receipt))
    shared = tmp_path / "shared-output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cc_tile_pipeline_attribution.py",
            "--trace",
            str(trace_path),
            "--receipt",
            str(receipt_path),
            "--output",
            str(shared),
            "--segments-output",
            str(shared),
        ],
    )
    with pytest.raises(ValueError, match="distinct"):
        attribution_main()


def campaign_run(tmp_path: Path, name: str, *, diagnostic: bool) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    endpoint = dict.fromkeys(WORK, 1)
    endpoint.update(
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
    oracle = {
        "nocc": 9,
        "nvir": 221,
        "naux": 488,
        "total_energy": -79.0,
        "triples_energy": -0.01,
    }
    (directory / "endpoint.json").write_text(json.dumps(endpoint))
    (directory / "oracle.json").write_text(json.dumps(oracle))
    (directory / "native.log").write_text("retained\n")
    identity = fixture()[0]["identity"]
    identity["oracle_sha256"] = checksum(directory / "oracle.json")
    identity["oracle_input_sha256"] = identity["input_sha256"]
    environment = execution_environment(
        {
            key: "1"
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        }
    )
    identity["environment_profile"] = environment["profile"]
    identity["observation_policy"] = observation_policy(
        environment["observation"], environment["observation"]
    )
    identity["observation_policy_sha256"] = canonical_hash(
        identity["observation_policy"]
    )
    identity["environment_sha256"] = canonical_hash(environment["profile"])
    command = [
        "/native",
        "/input",
        str(directory / "endpoint.json"),
        *campaign.CONTROLS,
    ]
    plan = {
        "argv": command,
        "environment": environment,
        "profile_sha256": identity["environment_sha256"],
    }
    atomic_json(directory / "execution_plan.json", plan)
    atomic_json(
        directory / "child_environment.json",
        {
            "schema": CHILD_SCHEMA,
            "status": "PREPARED_FOR_EXEC",
            "pid": 1765,
            "argv": command,
            "environment": environment,
            "environment_sha256": canonical_hash(environment),
        },
    )
    for script in ("cc_tile_child.py", "cc_tile_environment.py"):
        (directory / script).write_bytes(
            Path(campaign.__file__).with_name(script).read_bytes()
        )
    libraries = {
        "libgenerativeqc.so.0": {
            "path": "/mock-library",
            "sha256": identity["library_sha256"],
            "file_identity": {
                "st_dev": 0,
                "st_ino": 1,
                "st_size": 4,
                "st_mtime_ns": 1,
                "st_ctime_ns": 1,
            },
        }
    }
    artifacts = {
        path.name: {"sha256": checksum(path), "size": path.stat().st_size}
        for path in directory.iterdir()
    }
    receipt = {
        "schema": "generativeqc.cc-tile-campaign.v2",
        "clock_schema": campaign.CLOCK_SCHEMA,
        "identity": identity,
        "diagnostic": diagnostic,
        "command": command,
        "native_pid": 1765,
        "execution_environment": environment,
        "execution_environment_sha256": canonical_hash(environment),
        "child_environment_verified": True,
        "loader_libraries": libraries,
        "native_bindings": {"executable": "/native", "libraries": libraries},
        "launcher_sha256": {
            script: checksum(directory / script)
            for script in ("cc_tile_child.py", "cc_tile_environment.py")
        },
        "outcome": "PASS",
        "exit_code": 0,
        "binary_identity_unchanged": True,
        "input_identity_unchanged": True,
        "process_wall_ns": 100 if diagnostic else 80 + len(name),
        "science": check_science(endpoint, oracle),
        "work": {key: endpoint[key] for key in WORK},
        "artifacts": artifacts,
    }
    (directory / "receipt.json").write_text(json.dumps(receipt))
    return directory


def test_campaign_report_keeps_diagnostic_out_of_clean_timing(tmp_path: Path) -> None:
    runs = [
        campaign_run(tmp_path, "clean-a", diagnostic=False),
        campaign_run(tmp_path, "clean-bb", diagnostic=False),
        campaign_run(tmp_path, "diagnostic", diagnostic=True),
    ]
    result = compare_runs(runs)
    assert result["status"] == "PASS"
    assert result["clean_process_wall_ns"] == [87, 88]
    assert result["clean_process_wall_median_ns"] == 87.5
    assert result["diagnostic_process_wall_ns"] == 100
    assert result["diagnostic_wall_used_for_timing"] is False


def test_campaign_report_rejects_the_same_clean_run_twice(tmp_path: Path) -> None:
    clean = campaign_run(tmp_path, "clean", diagnostic=False)
    diagnostic = campaign_run(tmp_path, "diagnostic", diagnostic=True)
    with pytest.raises(ValueError, match="distinct"):
        compare_runs([clean, clean, diagnostic])


@pytest.mark.parametrize("defect", ["identity", "work", "energy", "artifact", "shape"])
def test_campaign_report_rejects_mismatched_or_incomplete_runs(
    tmp_path: Path, defect: str
) -> None:
    runs = [
        campaign_run(tmp_path, "clean-a", diagnostic=False),
        campaign_run(tmp_path, "clean-b", diagnostic=False),
        campaign_run(tmp_path, "diagnostic", diagnostic=True),
    ]
    receipt = json.loads((runs[1] / "receipt.json").read_text())
    if defect == "identity":
        receipt["identity"]["run_id"] = "different"
    elif defect == "work":
        receipt["work"]["ccsd_gemm_calls"] += 1
    elif defect == "artifact":
        (runs[1] / "native.log").write_text("changed\n")
    elif defect == "shape":
        runs[1] = campaign_run(tmp_path, "another-diagnostic", diagnostic=True)
    else:
        endpoint = json.loads((runs[1] / "endpoint.json").read_text())
        endpoint["total_energy"] += 1e-12
        (runs[1] / "endpoint.json").write_text(json.dumps(endpoint))
        receipt["artifacts"]["endpoint.json"] = {
            "sha256": checksum(runs[1] / "endpoint.json"),
            "size": (runs[1] / "endpoint.json").stat().st_size,
        }
    if defect in ("identity", "work", "energy"):
        (runs[1] / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        compare_runs(runs)


@pytest.mark.parametrize("changed", [None, "input", "oracle", "library", "executable"])
def test_campaign_runner_retains_frozen_oracle_and_rejects_changed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str | None
) -> None:
    fixture_run = campaign_run(tmp_path, "fixture", diagnostic=False)
    oracle = fixture_run / "oracle.json"
    oracle_bytes = oracle.read_bytes()
    endpoint = (fixture_run / "endpoint.json").read_bytes()
    paths = {"oracle": oracle}
    for key in ("input", "library", "executable"):
        paths[key] = tmp_path / key
        paths[key].write_bytes(key.encode())
    environment = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": str(tmp_path),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    identity = json.loads((fixture_run / "receipt.json").read_text())["identity"]
    for key, path in paths.items():
        identity[f"{key}_sha256"] = checksum(path)
    identity["oracle_input_sha256"] = identity["input_sha256"]
    identity["environment_profile"] = profile_environment(os.environ)
    observation = execution_environment(os.environ)["observation"]
    identity["observation_policy"] = observation_policy(observation, observation)
    identity["observation_policy_sha256"] = canonical_hash(
        identity["observation_policy"]
    )
    identity["environment_sha256"] = canonical_hash(identity["environment_profile"])
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity))
    output = tmp_path / "run"
    monkeypatch.setattr(
        "sys.argv",
        [
            "campaign.py",
            *(
                argument
                for key, path in paths.items()
                for argument in (f"--{key}", str(path))
            ),
            "--identity",
            str(identity_path),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        "benchmarks.cc_tile_pipeline_campaign.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"libgenerativeqc.so.0 => {paths['library'].resolve()} (0x0)\n"
        ),
    )

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            fake_endpoint(command, kwargs, endpoint)

        def wait(self, timeout: int) -> int:
            if changed:
                paths[changed].write_bytes(b"changed during native execution")
            return 0

    monkeypatch.setattr("benchmarks.cc_tile_pipeline_campaign.subprocess.Popen", Native)
    mock_native_bindings(monkeypatch)
    if changed:
        with pytest.raises(ValueError, match="changed during endpoint"):
            campaign_main()
    else:
        assert campaign_main() == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["outcome"] == ("FAIL" if changed else "PASS")
    assert receipt["input_identity_unchanged"] is (changed not in ("input", "oracle"))
    assert receipt["binary_identity_unchanged"] is (
        changed not in ("library", "executable")
    )
    assert (output / "oracle.json").read_bytes() == oracle_bytes
    assert (
        receipt["artifacts"]["oracle.json"]["sha256"]
        == hashlib.sha256(oracle_bytes).hexdigest()
    )


def campaign_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Path], Path, bytes]:
    """Tiny frozen inputs; endpoint bytes have an independent science fixture."""
    fixture_run = campaign_run(tmp_path, "fixture", diagnostic=False)
    mock_native_bindings(monkeypatch)
    paths = {"oracle": fixture_run / "oracle.json"}
    for key in ("input", "library", "executable"):
        paths[key] = tmp_path / key
        paths[key].write_bytes(key.encode())
    environment = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": str(tmp_path),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    identity = json.loads((fixture_run / "receipt.json").read_text())["identity"]
    for key, path in paths.items():
        identity[f"{key}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity["oracle_input_sha256"] = identity["input_sha256"]
    identity["environment_profile"] = profile_environment(os.environ)
    observation = execution_environment(os.environ)["observation"]
    identity["observation_policy"] = observation_policy(observation, observation)
    identity["observation_policy_sha256"] = canonical_hash(
        identity["observation_policy"]
    )
    identity["environment_sha256"] = canonical_hash(identity["environment_profile"])
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity))
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign.py",
            *(arg for key, path in paths.items() for arg in (f"--{key}", str(path))),
            "--identity",
            str(identity_path),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"libgenerativeqc.so.0 => {paths['library'].resolve()} (0x0)\n"
        ),
    )
    return paths, output, (fixture_run / "endpoint.json").read_bytes()


@pytest.mark.parametrize(
    "failure_type",
    [KeyboardInterrupt, SystemExit, RuntimeError, subprocess.TimeoutExpired],
)
def test_campaign_interrupt_reaps_only_owned_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_type: type[BaseException]
) -> None:
    _, output, endpoint = campaign_invocation(tmp_path, monkeypatch)
    original = (
        failure_type("native", 1800)
        if failure_type is subprocess.TimeoutExpired
        else failure_type("original wait failure")
    )
    real_popen = subprocess.Popen
    children = []
    unrelated = real_popen([sys.executable, "-c", "import time; time.sleep(60)"])

    class Native:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            fake_endpoint(command, kwargs, endpoint)
            self.child = real_popen(
                [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
            )
            children.append(self.child)
            self.pid = self.child.pid
            self.failed = False

        def wait(self, timeout: int | None = None) -> int:
            if not self.failed:
                self.failed = True
                raise original
            assert timeout is not None and 0 < timeout <= 10
            return self.child.wait(timeout=timeout)

        def kill(self) -> None:
            self.child.kill()

    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    try:
        with pytest.raises(failure_type) as caught:
            campaign_main()
        assert caught.value is original
        assert children and all(child.poll() is not None for child in children)
        assert unrelated.poll() is None
        receipt = json.loads((output / "receipt.json").read_text())
        assert receipt["outcome"] == (
            "TIMEOUT" if failure_type is subprocess.TimeoutExpired else "FAIL"
        )
        assert receipt["failure"]["type"] == failure_type.__name__
        assert receipt["elapsed_wall_ns"] > 0
        assert set(receipt["artifacts"]) == CAMPAIGN_ARTIFACTS
        for name, artifact in receipt["artifacts"].items():
            raw = (output / name).read_bytes()
            assert artifact == {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
    finally:
        # Also leave the old implementation's deliberately failing control clean.
        for child in [*children, unrelated]:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)


@pytest.mark.parametrize("wait_fails", [False, True])
@pytest.mark.parametrize("defect", ["missing-library", "artifact-hash", "log-close"])
def test_campaign_finalization_retains_failure_and_remaining_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wait_fails: bool, defect: str
) -> None:
    paths, output, endpoint = campaign_invocation(tmp_path, monkeypatch)
    original = RuntimeError("original wait failure")
    final_error = OSError("injected finalization failure")
    real_open = Path.open
    real_checksum = campaign.checksum
    alive = False

    class Log:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

        def close(self) -> None:
            self.stream.close()
            raise final_error

    def open_file(path: Path, *args: Any, **kwargs: Any) -> Any:
        stream = real_open(path, *args, **kwargs)
        if path == output / "native.log" and args == ("wb",):
            return Log(stream)
        return stream

    def hash_file(path: Path) -> str:
        if path == output / "native.log":
            raise final_error
        return real_checksum(path)

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            nonlocal alive
            fake_endpoint(command, kwargs, endpoint)
            alive = True

        def wait(self, timeout: int | None = None) -> int:
            nonlocal alive
            if not alive:
                assert timeout is not None and 0 < timeout <= 10
                return -9
            if defect == "missing-library":
                paths["library"].unlink()
            if wait_fails:
                raise original
            alive = False
            return 0

        def kill(self) -> None:
            nonlocal alive
            alive = False

    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    if defect == "log-close":
        monkeypatch.setattr(Path, "open", open_file)
    elif defect == "artifact-hash":
        monkeypatch.setattr(campaign, "checksum", hash_file)
    expected = (
        RuntimeError
        if wait_fails
        else (FileNotFoundError if defect == "missing-library" else OSError)
    )
    with pytest.raises(expected) as caught:
        campaign_main()
    if wait_fails:
        assert caught.value is original
    elif defect != "missing-library":
        assert caught.value is final_error
    assert not alive
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["outcome"] == "FAIL"
    assert receipt["elapsed_wall_ns"] > 0
    assert receipt["binary_identity_unchanged"] is (defect != "missing-library")
    assert receipt["input_identity_unchanged"] is True
    assert receipt["finalization_errors"]
    if wait_fails:
        assert receipt["failure"] == {"type": "RuntimeError", "message": str(original)}
    assert {"endpoint.json", "oracle.json"} <= set(receipt["artifacts"])
    assert ("native.log" in receipt["artifacts"]) is (defect != "artifact-hash")


def test_campaign_cleanup_failures_preserve_original_and_attempt_bounded_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, endpoint = campaign_invocation(tmp_path, monkeypatch)
    original = KeyboardInterrupt("original interrupt")
    attempted_reap = False

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            fake_endpoint(command, kwargs, endpoint)
            self.failed = False

        def wait(self, timeout: int | None = None) -> int:
            nonlocal attempted_reap
            if not self.failed:
                self.failed = True
                raise original
            attempted_reap = True
            assert timeout is not None and 0 < timeout <= 10
            raise subprocess.TimeoutExpired("native cleanup", timeout)

        def kill(self) -> None:
            raise OSError("kill unavailable")

    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    with pytest.raises(KeyboardInterrupt) as caught:
        campaign_main()
    assert caught.value is original
    assert attempted_reap
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["outcome"] == "FAIL"
    assert {row["type"] for row in receipt["finalization_errors"]} == {
        "OSError",
        "TimeoutExpired",
    }
    assert CAMPAIGN_ARTIFACTS == set(receipt["artifacts"])


@pytest.mark.parametrize("wait_fails", [False, True])
@pytest.mark.parametrize("phase", ["after-spawn", "final-write", "final-replace"])
def test_campaign_receipt_io_failure_preserves_original_and_durable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wait_fails: bool, phase: str
) -> None:
    _, output, endpoint = campaign_invocation(tmp_path, monkeypatch)
    original = RuntimeError("original wait failure")
    io_error = OSError("receipt storage unavailable")
    real_write = Path.write_text
    real_replace = Path.replace
    alive = False
    previous_receipt = None

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            nonlocal alive
            fake_endpoint(command, kwargs, endpoint)
            alive = True

        def wait(self, timeout: int | None = None) -> int:
            nonlocal alive
            if not alive:
                assert timeout is not None and 0 < timeout <= 10
                return -9
            if wait_fails:
                raise original
            alive = False
            return 0

        def kill(self) -> None:
            nonlocal alive
            alive = False

    def write_file(path: Path, data: str, *args: Any, **kwargs: Any) -> int:
        nonlocal previous_receipt
        if path.name in ("receipt.json", "receipt.json.tmp"):
            receipt = json.loads(data)
            failing = (
                phase == "after-spawn"
                and "native_pid" in receipt
                and "artifacts" not in receipt
            ) or (phase == "final-write" and "artifacts" in receipt)
            if failing:
                previous_receipt = (output / "receipt.json").read_bytes()
                # Model a partial disk write, not just a failure before open.
                real_write(path, "partial")
                raise io_error
        return real_write(path, data, *args, **kwargs)

    def replace_file(path: Path, target: Path) -> Path:
        nonlocal previous_receipt
        if phase == "final-replace" and "artifacts" in json.loads(path.read_text()):
            previous_receipt = (output / "receipt.json").read_bytes()
            raise io_error
        return real_replace(path, target)

    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    monkeypatch.setattr(Path, "write_text", write_file)
    monkeypatch.setattr(Path, "replace", replace_file)
    expected = io_error if phase == "after-spawn" or not wait_fails else original
    with pytest.raises(type(expected)) as caught:
        campaign_main()
    assert caught.value is expected
    assert not alive
    assert previous_receipt is not None
    if phase == "after-spawn":
        receipt = json.loads((output / "receipt.json").read_text())
        assert receipt["outcome"] == "FAIL"
        assert "artifacts" in receipt
        assert "receipt.json.tmp" not in receipt["artifacts"]
    else:
        assert (output / "receipt.json").read_bytes() == previous_receipt
        assert json.loads(previous_receipt)["outcome"] == "INCOMPLETE"
        if wait_fails:
            assert any(
                "receipt finalization failed" in note for note in original.__notes__
            )


@pytest.mark.parametrize("wait_fails", [False, True])
def test_campaign_diagnostic_close_preserves_wait_failure_and_teardown_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wait_fails: bool
) -> None:
    _, output, endpoint = campaign_invocation(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--diagnostic"])
    original = RuntimeError("original wait failure")
    close_error = OSError("diagnostic close failed")
    alive = False

    class Scope:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            assert not alive, "native teardown must stay inside diagnostic scope"
            raise close_error

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            nonlocal alive
            fake_endpoint(command, kwargs, endpoint)
            alive = True

        def wait(self, timeout: int | None = None) -> int:
            nonlocal alive
            if wait_fails and alive:
                raise original
            alive = False
            return 0

        def kill(self) -> None:
            nonlocal alive
            alive = False

    monkeypatch.setitem(
        sys.modules, "nvtx", SimpleNamespace(annotate=lambda *a, **k: Scope())
    )
    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    expected = original if wait_fails else close_error
    with pytest.raises(type(expected)) as caught:
        campaign_main()
    assert caught.value is expected
    assert not alive
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["outcome"] == "FAIL"
    assert receipt["failure"]["message"] == str(expected)
    if wait_fails:
        assert any(
            row["message"] == str(close_error) for row in receipt["finalization_errors"]
        )


def test_campaign_changed_identity_does_not_replace_wait_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, output, endpoint = campaign_invocation(tmp_path, monkeypatch)
    original = RuntimeError("original wait failure")

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            fake_endpoint(command, kwargs, endpoint)
            self.finished = False

        def wait(self, timeout: int | None = None) -> int:
            if not self.finished:
                paths["input"].write_bytes(b"changed input")
                raise original
            return -9

        def kill(self) -> None:
            self.finished = True

    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    with pytest.raises(RuntimeError) as caught:
        campaign_main()
    assert caught.value is original
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["outcome"] == "FAIL"
    assert receipt["input_identity_unchanged"] is False
    assert receipt["binary_identity_unchanged"] is True
    assert "finalization_errors" not in receipt
    assert CAMPAIGN_ARTIFACTS == set(receipt["artifacts"])


@pytest.mark.parametrize(
    "defect",
    [
        "nonfinite-energy",
        "wrong-energy",
        "nonfinite-residual",
        "failed-residual",
        "mixed-precision",
        "force",
        "missing-work",
        "forged-work",
        "missing-gate",
        "forged-gate",
        "widened-limit",
        "bool-gate-value",
        "nonfinite-limit",
        "missing-oracle",
        "wrong-oracle",
        "empty-identity",
        "invalid-source",
        "wrong-oracle-input",
        "bool-dimension",
        "float-dimension",
        "zero-dimension",
        "bool-force",
        "failed-exit",
        "bool-exit",
        "changed-input",
        "truthy-binary-identity",
    ],
)
def test_campaign_report_recomputes_even_identical_forged_passes(
    tmp_path: Path, defect: str
) -> None:
    """Cross-run agreement and fresh hashes cannot validate false PASS claims."""
    runs = [
        campaign_run(tmp_path, "clean-a", diagnostic=False),
        campaign_run(tmp_path, "clean-b", diagnostic=False),
        campaign_run(tmp_path, "diagnostic", diagnostic=True),
    ]
    for directory in runs:
        receipt = json.loads((directory / "receipt.json").read_text())
        endpoint = json.loads((directory / "endpoint.json").read_text())
        if defect == "nonfinite-energy":
            endpoint["total_energy"] = float("nan")
        elif defect == "wrong-energy":
            endpoint["total_energy"] += 1e-7
        elif defect == "nonfinite-residual":
            endpoint["ccsd_replay_r1_max"] = float("nan")
        elif defect == "failed-residual":
            endpoint["ccsd_replay_r2_max"] = 1e-6
        elif defect == "mixed-precision":
            endpoint["ccsd_fp32_gemm_calls"] = 1
        elif defect == "force":
            endpoint["forces_requested"] = 1
        elif defect == "missing-work":
            endpoint.pop("ccsd_q_operations")
        elif defect == "forged-work":
            receipt["work"]["ccsd_gemm_calls"] += 1
        elif defect == "missing-gate":
            receipt["science"]["gates"].pop("physical_replay_r2")
        elif defect == "forged-gate":
            receipt["science"]["gates"]["physical_replay_r2"]["value"] = 0.0
        elif defect == "widened-limit":
            receipt["science"]["gates"]["total_energy_error"]["limit"] = 1.0
        elif defect == "bool-gate-value":
            receipt["science"]["gates"]["total_energy_error"]["value"] = False
        elif defect == "nonfinite-limit":
            receipt["science"]["gates"]["total_energy_error"]["limit"] = float("nan")
        elif defect == "missing-oracle":
            (directory / "oracle.json").unlink()
        elif defect == "wrong-oracle":
            oracle = json.loads((directory / "oracle.json").read_text())
            oracle["total_energy"] -= 0.1
            (directory / "oracle.json").write_text(json.dumps(oracle))
        elif defect == "empty-identity":
            receipt["identity"] = {}
        elif defect == "invalid-source":
            receipt["identity"]["source_commit"] = "not-a-commit"
        elif defect == "wrong-oracle-input":
            receipt["identity"]["oracle_input_sha256"] = "0" * 64
        elif defect == "bool-dimension":
            endpoint["naux"] = True
        elif defect == "float-dimension":
            endpoint["nbf"] = 230.0
        elif defect == "zero-dimension":
            endpoint["naux"] = 0
        elif defect == "bool-force":
            endpoint["forces_requested"] = False
        elif defect == "failed-exit":
            receipt["exit_code"] = 1
        elif defect == "bool-exit":
            receipt["exit_code"] = False
        elif defect == "changed-input":
            receipt["input_identity_unchanged"] = False
        else:
            receipt["binary_identity_unchanged"] = 1
        (directory / "endpoint.json").write_text(json.dumps(endpoint))
        # Rehash deliberately: the negative control survives ordinary integrity
        # checks and differs only in its unsupported scientific claim.
        receipt["artifacts"] = {
            path.name: {"sha256": checksum(path), "size": path.stat().st_size}
            for path in directory.iterdir()
            if path.name != "receipt.json"
        }
        (directory / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        compare_runs(runs)


@pytest.mark.parametrize(
    ("name", "replacement", "wait_fails"),
    [
        ("native.log", "unchanged", False),
        ("endpoint.json", "missing", False),
        ("oracle.json", "missing", False),
        ("native.log", "missing", False),
        ("native.log", "symlink", False),
        ("native.log", "directory", False),
        ("native.log", "symlink", True),
        ("native.log", "directory", True),
    ],
)
def test_campaign_artifact_inventory_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: str,
    wait_fails: bool,
) -> None:
    paths, output, endpoint = campaign_invocation(tmp_path, monkeypatch)
    original = RuntimeError("original wait failure")
    real_open = Path.open

    class Log:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def __getattr__(self, name: str) -> Any:
            return getattr(self.stream, name)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

        def close(self) -> None:
            self.stream.close()
            if replacement == "unchanged":
                return
            artifact = output / name
            artifact.unlink()
            if replacement == "symlink":
                artifact.symlink_to(paths["input"])
            elif replacement == "directory":
                artifact.mkdir()

    def open_file(path: Path, *args: Any, **kwargs: Any) -> Any:
        stream = real_open(path, *args, **kwargs)
        if path == output / "native.log" and args == ("wb",):
            return Log(stream)
        return stream

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            fake_endpoint(command, kwargs, endpoint)
            self.killed = False

        def wait(self, timeout: int | None = None) -> int:
            if wait_fails and not self.killed:
                raise original
            return 0

        def kill(self) -> None:
            self.killed = True

    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    monkeypatch.setattr(Path, "open", open_file)
    if replacement == "unchanged":
        assert campaign_main() == 0
    else:
        with pytest.raises(RuntimeError if wait_fails else ValueError) as caught:
            campaign_main()
        if wait_fails:
            assert caught.value is original
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["outcome"] == ("PASS" if replacement == "unchanged" else "FAIL")
    assert receipt["binary_identity_unchanged"] is True
    assert receipt["input_identity_unchanged"] is True
    assert (name in receipt["artifacts"]) is (replacement == "unchanged")
    if replacement != "unchanged":
        assert any(name in row["stage"] for row in receipt["finalization_errors"])
    for artifact_name, metadata in receipt["artifacts"].items():
        raw = (output / artifact_name).read_bytes()
        assert metadata == {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def malformed_profile(profile: dict, defect: str) -> dict:
    changed = deepcopy(profile)
    values = changed["variables"]
    if defect == "missing":
        del values["CUBLAS_WORKSPACE_CONFIG"]
    elif defect == "presence":
        values["CUBLAS_WORKSPACE_CONFIG"]["present"] = 1
    elif defect == "absent-empty":
        values["CUBLAS_WORKSPACE_CONFIG"] = {"present": False, "value": ""}
    elif defect == "present-null":
        values["CUBLAS_WORKSPACE_CONFIG"] = {"present": True, "value": None}
    else:
        values["CUBLAS_WORKSPACE_CONFIG"]["unknown"] = "extra"
    return changed


@pytest.mark.parametrize(
    "defect", ["missing", "presence", "absent-empty", "present-null", "extra"]
)
def test_attribution_rejects_equal_malformed_environment_with_refreshed_hashes(
    defect: str,
) -> None:
    trace, receipt = fixture()
    for packet in (trace, receipt):
        profile = malformed_profile(packet["identity"]["environment_profile"], defect)
        packet["identity"]["environment_profile"] = profile
        packet["identity"]["environment_sha256"] = canonical_hash(profile)
        packet["execution_environment"]["profile"] = deepcopy(profile)
        packet["execution_environment_sha256"] = canonical_hash(
            packet["execution_environment"]
        )
    with pytest.raises(ValueError):
        summarize(trace, receipt)


@pytest.mark.parametrize("side", [0, 1])
def test_attribution_rejects_missing_actual_execution_environment(side: int) -> None:
    packets = fixture()
    del packets[side]["execution_environment"]
    with pytest.raises(ValueError):
        summarize(*packets)


def rewrite_execution(directory: Path, receipt: dict, execution: dict) -> None:
    receipt["execution_environment"] = execution
    receipt["execution_environment_sha256"] = canonical_hash(execution)
    for name in ("execution_plan.json", "child_environment.json"):
        item = json.loads((directory / name).read_text())
        item["environment"] = execution
        if name == "child_environment.json":
            item["environment_sha256"] = canonical_hash(execution)
        atomic_json(directory / name, item)
        receipt["artifacts"][name] = {
            "sha256": checksum(directory / name),
            "size": (directory / name).stat().st_size,
        }
    atomic_json(directory / "receipt.json", receipt)


def test_two_clean_samples_cannot_mix_logging_controls(tmp_path: Path) -> None:
    runs = [
        campaign_run(tmp_path, name, diagnostic=name == "diagnostic")
        for name in ["a", "bb", "diagnostic"]
    ]
    policy = json.loads((runs[0] / "receipt.json").read_text())["identity"][
        "observation_policy"
    ]
    policy["clean"]["CUBLAS_LOGINFO_DBG"] = {"present": True, "value": "0"}
    for i, directory in enumerate(runs):
        receipt = json.loads((directory / "receipt.json").read_text())
        receipt["identity"]["observation_policy"] = policy
        receipt["identity"]["observation_policy_sha256"] = canonical_hash(policy)
        execution = receipt["execution_environment"]
        if i < 2:
            execution["observation"]["CUBLAS_LOGINFO_DBG"] = {
                "present": True,
                "value": str(i),
            }
        rewrite_execution(directory, receipt, execution)
    with pytest.raises(ValueError, match="observation controls"):
        compare_runs(runs)


def test_clean_preload_is_rejected_even_if_declared_and_rehashed(
    tmp_path: Path,
) -> None:
    directory = campaign_run(tmp_path, "clean", diagnostic=False)
    receipt = json.loads((directory / "receipt.json").read_text())
    policy = receipt["identity"]["observation_policy"]
    policy["clean"]["LD_PRELOAD"] = {"present": True, "value": "/not-clean.so"}
    receipt["identity"]["observation_policy_sha256"] = canonical_hash(policy)
    receipt["execution_environment"]["observation"]["LD_PRELOAD"] = policy["clean"][
        "LD_PRELOAD"
    ]
    rewrite_execution(directory, receipt, receipt["execution_environment"])
    from benchmarks.cc_tile_pipeline_campaign_report import read_run

    with pytest.raises(ValueError, match="clean samples"):
        read_run(directory)


def test_declared_run_specific_log_paths_preserve_clean_recipe(tmp_path: Path) -> None:
    runs = [
        campaign_run(tmp_path, name, diagnostic=name == "diagnostic")
        for name in ["a", "bb", "diagnostic"]
    ]
    policy = json.loads((runs[0] / "receipt.json").read_text())["identity"][
        "observation_policy"
    ]
    for role in ("clean", "diagnostic"):
        policy[role]["CUSOLVER_LOG_FILE"] = {
            "present": True,
            "value": "$RUN_DIR/cusolver.log",
        }
    for directory in runs:
        receipt = json.loads((directory / "receipt.json").read_text())
        receipt["identity"]["observation_policy"] = policy
        receipt["identity"]["observation_policy_sha256"] = canonical_hash(policy)
        receipt["execution_environment"]["observation"]["CUSOLVER_LOG_FILE"] = {
            "present": True,
            "value": str(directory) + "/cusolver.log",
        }
        rewrite_execution(directory, receipt, receipt["execution_environment"])
    assert compare_runs(runs)["status"] == "PASS"


@pytest.mark.parametrize(
    "defect", ["missing", "presence", "absent-empty", "present-null", "extra"]
)
def test_campaign_reader_rejects_three_equal_malformed_environment_profiles(
    tmp_path: Path, defect: str
) -> None:
    runs = [
        campaign_run(tmp_path, name, diagnostic=name == "diagnostic")
        for name in ["a", "bb", "diagnostic"]
    ]
    for directory in runs:
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        profile = malformed_profile(receipt["identity"]["environment_profile"], defect)
        receipt["identity"]["environment_profile"] = profile
        receipt["identity"]["environment_sha256"] = canonical_hash(profile)
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        compare_runs(runs)


def test_workspace_absent_and_empty_have_distinct_frozen_identities() -> None:
    absent = profile_environment({})
    empty = profile_environment({"CUBLAS_WORKSPACE_CONFIG": ""})
    assert canonical_hash(absent) != canonical_hash(empty)
    assert absent["variables"]["CUBLAS_WORKSPACE_CONFIG"] == {
        "present": False,
        "value": None,
    }
    assert empty["variables"]["CUBLAS_WORKSPACE_CONFIG"] == {
        "present": True,
        "value": "",
    }


def test_campaign_observer_controls_are_separate_and_retained(tmp_path: Path) -> None:
    runs = [
        campaign_run(tmp_path, name, diagnostic=name == "diagnostic")
        for name in ["a", "bb", "diagnostic"]
    ]
    directory = runs[-1]
    receipt = json.loads((directory / "receipt.json").read_text())
    execution = receipt["execution_environment"]
    execution["observation"]["LD_PRELOAD"] = {
        "present": True,
        "value": "/qualified-observer.so",
    }
    policy = deepcopy(receipt["identity"]["observation_policy"])
    policy["diagnostic"] = deepcopy(execution["observation"])
    for other in runs[:-1]:
        clean_receipt = json.loads((other / "receipt.json").read_text())
        clean_receipt["identity"]["observation_policy"] = policy
        clean_receipt["identity"]["observation_policy_sha256"] = canonical_hash(policy)
        atomic_json(other / "receipt.json", clean_receipt)
    receipt["identity"]["observation_policy"] = policy
    receipt["identity"]["observation_policy_sha256"] = canonical_hash(policy)
    receipt["execution_environment_sha256"] = canonical_hash(execution)
    plan = json.loads((directory / "execution_plan.json").read_text())
    plan["environment"] = execution
    child = json.loads((directory / "child_environment.json").read_text())
    child["environment"] = execution
    child["environment_sha256"] = canonical_hash(execution)
    atomic_json(directory / "execution_plan.json", plan)
    atomic_json(directory / "child_environment.json", child)
    for name in ["execution_plan.json", "child_environment.json"]:
        receipt["artifacts"][name] = {
            "sha256": checksum(directory / name),
            "size": (directory / name).stat().st_size,
        }
    atomic_json(directory / "receipt.json", receipt)
    assert compare_runs(runs)["status"] == "PASS"
    assert (
        json.loads((directory / "child_environment.json").read_text())["environment"][
            "observation"
        ]["LD_PRELOAD"]["value"]
        == "/qualified-observer.so"
    )


def test_campaign_parent_change_after_snapshot_cannot_change_child_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, endpoint = campaign_invocation(tmp_path, monkeypatch)

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "changed-parent")
            fake_endpoint(command, kwargs, endpoint)

        def wait(self, timeout: int) -> int:
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    assert campaign_main() == 0
    receipt = json.loads((output / "receipt.json").read_text())
    child = json.loads((output / "child_environment.json").read_text())
    assert child["environment"] == receipt["execution_environment"]


def test_campaign_child_override_fails_even_with_refreshed_child_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output, endpoint = campaign_invocation(tmp_path, monkeypatch)

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            changed = dict(kwargs)
            changed["env"] = {
                **kwargs["env"],
                "CUBLAS_WORKSPACE_CONFIG": "changed-child",
            }
            fake_endpoint(command, changed, endpoint)

        def wait(self, timeout: int) -> int:
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    with pytest.raises(ValueError):
        campaign_main()
    assert json.loads((output / "receipt.json").read_text())["outcome"] == "FAIL"


@pytest.mark.parametrize(
    ("child_pid", "parent_pid"), [(1, True), (True, 1), (0, 0), (-1, -1)]
)
def test_child_binding_requires_two_positive_integer_pids(
    child_pid: object, parent_pid: object
) -> None:
    from benchmarks.cc_tile_environment import validate_child

    execution = execution_environment({})
    plan = {
        "argv": ["/native"],
        "environment": execution,
        "profile_sha256": canonical_hash(execution["profile"]),
    }
    child = {
        "schema": CHILD_SCHEMA,
        "status": "PREPARED_FOR_EXEC",
        "pid": child_pid,
        "argv": plan["argv"],
        "environment": execution,
        "environment_sha256": canonical_hash(execution),
    }
    with pytest.raises(ValueError):
        validate_child(child, plan, parent_pid)


@pytest.mark.parametrize(
    "defect", ["missing", "not-object", "bad-file-id", "wrong-executable"]
)
def test_campaign_reader_rejects_invalid_actual_execution_bindings(
    tmp_path: Path, defect: str
) -> None:
    from benchmarks.cc_tile_pipeline_campaign_report import read_run

    directory = campaign_run(tmp_path, "bound", diagnostic=False)
    receipt = json.loads((directory / "receipt.json").read_text())
    if defect == "missing":
        del receipt["native_bindings"]
    elif defect == "not-object":
        receipt["loader_libraries"] = []
        receipt["native_bindings"]["libraries"] = []
    elif defect == "bad-file-id":
        receipt["loader_libraries"]["libgenerativeqc.so.0"]["file_identity"][
            "st_ino"
        ] = True
        receipt["native_bindings"]["libraries"] = receipt["loader_libraries"]
    else:
        receipt["native_bindings"]["executable"] = "/different"
    atomic_json(directory / "receipt.json", receipt)
    with pytest.raises(ValueError):
        read_run(directory)


def test_slow_parent_attestation_is_excluded_from_child_completion_wall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    _, output, endpoint = campaign_invocation(tmp_path, monkeypatch)
    real_supervisor = campaign.NativeSupervisor
    state = {"main_clock": 100}
    observed = {}

    def supervisor(process: Any, deadline: float) -> Any:
        instance = real_supervisor(process, deadline)
        observed["supervisor"] = instance
        return instance

    def clock() -> int:
        return (
            state["main_clock"]
            if threading.current_thread() is threading.main_thread()
            else 200
        )

    def delayed_attestation(pid: int, executable: Path, libraries: dict) -> dict:
        assert observed["supervisor"].finished.wait(2)
        state["main_clock"] = 1000
        return {"executable": str(executable.resolve()), "libraries": libraries}

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            fake_endpoint(command, kwargs, endpoint)

        def wait(self, timeout: float) -> int:
            return 0

    monkeypatch.setattr(campaign, "NativeSupervisor", supervisor)
    monkeypatch.setattr(campaign, "observe_native", delayed_attestation)
    monkeypatch.setattr(campaign.time, "perf_counter_ns", clock)
    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    assert campaign_main() == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["process_wall_ns"] == 100
    assert receipt["attestation_wall_ns"] == 900


def test_child_watchdog_cleanup_does_not_wait_for_parent_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    _, output, endpoint = campaign_invocation(tmp_path, monkeypatch)
    killed = threading.Event()
    original = subprocess.TimeoutExpired("native", 1800)

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            fake_endpoint(command, kwargs, endpoint)

        def wait(self, timeout: float) -> int:
            if killed.is_set():
                assert timeout <= 10
                return -9
            assert 0 <= timeout <= 1800
            raise original

        def kill(self) -> None:
            killed.set()

    def delayed_attestation(pid: int, executable: Path, libraries: dict) -> dict:
        assert killed.wait(2), "parent attestation must not postpone child watchdog"
        return {"executable": str(executable.resolve()), "libraries": libraries}

    monkeypatch.setattr(campaign, "observe_native", delayed_attestation)
    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        campaign_main()
    assert caught.value is original
    assert json.loads((output / "receipt.json").read_text())["outcome"] == "TIMEOUT"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux same-PID exec contract")
@pytest.mark.parametrize("workspace", [None, "", ":16:8"])
def test_real_exec_workspace_inheritance(tmp_path: Path, workspace: str | None) -> None:
    from benchmarks.cc_tile_environment import validate_child, validate_execution

    supplied = {
        key: os.environ[key]
        for key in ("PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "LC_CTYPE")
        if key in os.environ
    }
    supplied.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if workspace is not None:
        supplied["CUBLAS_WORKSPACE_CONFIG"] = workspace
    environment = execution_environment(supplied)
    profile_hash = canonical_hash(environment["profile"])
    native = tmp_path / "actual.json"
    program = "import json,os,sys;json.dump({'pid':os.getpid(),'present':'CUBLAS_WORKSPACE_CONFIG' in os.environ,'value':os.environ.get('CUBLAS_WORKSPACE_CONFIG')},open(sys.argv[1],'w'))"
    plan = {
        "argv": [sys.executable, "-c", program, str(native)],
        "environment": environment,
        "profile_sha256": profile_hash,
    }
    for name in ("cc_tile_child.py", "cc_tile_environment.py"):
        (tmp_path / name).write_bytes(
            Path(campaign.__file__).with_name(name).read_bytes()
        )
    atomic_json(tmp_path / "plan.json", plan)
    witness = tmp_path / "child.json"
    process = subprocess.Popen(
        [
            sys.executable,
            str(tmp_path / "cc_tile_child.py"),
            str(tmp_path / "plan.json"),
            str(witness),
        ],
        env=validate_execution(environment, profile_hash),
    )
    assert process.wait(timeout=20) == 0
    validate_child(json.loads(witness.read_text()), plan, process.pid)
    assert json.loads(native.read_text()) == {
        "pid": process.pid,
        "present": workspace is not None,
        "value": workspace,
    }


@pytest.mark.parametrize("side", [0, 1])
def test_attribution_rejects_missing_observer_field_with_fresh_hash(side: int) -> None:
    packets = fixture()
    del packets[side]["execution_environment"]["observation"]["LD_PRELOAD"]
    packets[side]["execution_environment_sha256"] = canonical_hash(
        packets[side]["execution_environment"]
    )
    with pytest.raises(ValueError):
        summarize(*packets)


def session_fixture() -> tuple[dict, dict, dict, dict]:
    from benchmarks.cc_tile_environment import SESSION_KEY, SESSION_SCHEMA

    clean = execution_environment({})
    profiler = {
        "path": "/owned/nsys",
        "sha256": "a" * 64,
        "version": "fixture-nsys",
        "argv": [
            "/owned/nsys",
            "profile",
            "--session-new=owned-capture",
            "/owned/parent",
        ],
        "session_name": "owned-capture",
    }
    policy = observation_policy(
        clean["observation"], clean["observation"], nsys_profiler=profiler
    )
    execution = execution_environment({SESSION_KEY: "opaque-ID-1"})
    parent = {"pid": 44, "starttime": "12000"}
    query = "42 owned-capture Running\n"
    binding = {
        "schema": SESSION_SCHEMA,
        "policy_sha256": canonical_hash(policy),
        "profiler": profiler,
        "outer_process": {"pid": 22, "starttime": "11000"},
        "outer_session": {"present": False, "value": None},
        "profiler_process": {
            "process": {"pid": 33, "starttime": "11900"},
            "executable": "/owned/nsys",
            "sha256": "a" * 64,
        },
        "parent": {"process": parent, "session_id": "opaque-ID-1"},
        "session_query": {
            "argv": ["/owned/nsys", "sessions", "list", "--show-header=false"],
            "exit_code": 0,
            "stdout": query,
            "stdout_sha256": hashlib.sha256(query.encode()).hexdigest(),
        },
        "id_mapping": "UNAVAILABLE",
    }
    return policy, execution, binding, parent


def validate_session_fixture(
    policy: dict, execution: dict, binding: dict | None, parent: dict
) -> None:
    from benchmarks.cc_tile_environment import validate_execution, validate_observation

    validate_execution(execution, canonical_hash(execution["profile"]))
    validate_observation(
        execution,
        policy,
        canonical_hash(policy),
        True,
        "/owned/run",
        binding,
        canonical_hash(binding) if binding is not None else None,
        parent,
    )


def test_nsys_dynamic_id_retains_original_policy_and_semantic_profile() -> None:
    from benchmarks.cc_tile_environment import SESSION_KEY

    policy, execution, binding, parent = session_fixture()
    original = deepcopy(policy)
    validate_session_fixture(policy, execution, binding, parent)
    execution["observation"][SESSION_KEY]["value"] = "different.opaque.session"
    binding["parent"]["session_id"] = "different.opaque.session"
    validate_session_fixture(policy, execution, binding, parent)
    assert policy == original
    assert execution["profile"] == execution_environment({})["profile"]


@pytest.mark.parametrize("value", ["", "\0", "a" * 1025, "$NSYS_GENERATED_SESSION"])
def test_nsys_session_rejects_invalid_actual_values(value: str) -> None:
    from benchmarks.cc_tile_environment import SESSION_KEY

    policy, execution, binding, parent = session_fixture()
    execution["observation"][SESSION_KEY]["value"] = value
    binding["parent"]["session_id"] = value
    with pytest.raises(ValueError):
        validate_session_fixture(policy, execution, binding, parent)


@pytest.mark.parametrize(
    "defect",
    [
        "missing-key",
        "missing-binding",
        "wrong-source",
        "extra-rule",
        "not-bool",
        "disabled-tool",
        "disabled-present",
        "literal",
        "clean-empty",
        "other-key",
        "parent-id",
        "parent-pid",
        "parent-starttime",
        "profiler-exe",
        "profiler-sha",
        "profiler-argv",
        "query-exit",
        "query-hash",
        "query-name",
        "query-substring",
        "mapping-claim",
        "preset-empty",
        "preset-id",
        "missing-starttime",
        "old-schema",
    ],
)
def test_nsys_session_rejects_consistently_rehashed_invalid_evidence(
    defect: str,
) -> None:
    from benchmarks.cc_tile_environment import SESSION_KEY

    policy, execution, binding, parent = session_fixture()
    if defect == "missing-key":
        del execution["observation"][SESSION_KEY]
    elif defect == "missing-binding":
        binding = None
    elif defect == "wrong-source":
        policy["session_control"]["source"] = "manual"
    elif defect == "extra-rule":
        policy["session_control"]["other"] = "NSYS_*"
    elif defect == "not-bool":
        policy["session_control"]["enabled"] = 1
    elif defect == "disabled-tool":
        policy["session_control"]["enabled"] = False
    elif defect == "disabled-present":
        policy["session_control"].update(enabled=False, profiler=None)
    elif defect == "literal":
        policy["diagnostic"][SESSION_KEY]["value"] = "opaque-ID-1"
    elif defect == "clean-empty":
        policy["clean"][SESSION_KEY] = {"present": True, "value": ""}
    elif defect == "other-key":
        policy["diagnostic"]["LANG"] = deepcopy(policy["diagnostic"][SESSION_KEY])
    elif defect == "parent-id":
        binding["parent"]["session_id"] = "mismatch"
    elif defect == "parent-pid":
        parent = dict(parent, pid=45)
    elif defect == "parent-starttime":
        parent = dict(parent, starttime="12001")
    elif defect == "profiler-exe":
        binding["profiler_process"]["executable"] = "/wrong/nsys"
    elif defect == "profiler-sha":
        binding["profiler_process"]["sha256"] = "b" * 64
    elif defect == "profiler-argv":
        binding["profiler"] = dict(binding["profiler"], argv=["wrong"])
    elif defect == "query-exit":
        binding["session_query"]["exit_code"] = 1
    elif defect == "query-hash":
        binding["session_query"]["stdout_sha256"] = "b" * 64
    elif defect in ("query-name", "query-substring"):
        text = (
            "42 other Running\n"
            if defect == "query-name"
            else "42 owned-capture-extra Running\n"
        )
        binding["session_query"].update(
            stdout=text, stdout_sha256=hashlib.sha256(text.encode()).hexdigest()
        )
    elif defect == "mapping-claim":
        binding["id_mapping"] = "ASSUMED_FROM_NAME"
    elif defect in ("preset-empty", "preset-id"):
        binding["outer_session"] = {
            "present": True,
            "value": "" if defect == "preset-empty" else "manual",
        }
    elif defect == "missing-starttime":
        del binding["outer_process"]["starttime"]
    elif defect == "old-schema":
        execution["schema"] = "generativeqc.cc-execution-environment.v1"
    if binding is not None:
        binding["policy_sha256"] = canonical_hash(policy)
    with pytest.raises(ValueError):
        validate_session_fixture(policy, execution, binding, parent)


@pytest.mark.parametrize("value", ["", "manual-ID"])
def test_outer_launcher_rejects_preset_session_even_empty(value: str) -> None:
    from benchmarks.cc_tile_environment import SESSION_KEY, require_unprofiled_launch

    require_unprofiled_launch({})
    with pytest.raises(ValueError, match="preset session"):
        require_unprofiled_launch({SESSION_KEY: value})


def test_dynamic_session_still_requires_same_pid_child_value() -> None:
    from benchmarks.cc_tile_environment import SESSION_KEY, validate_child

    _, execution, _, _ = session_fixture()
    plan = {
        "argv": ["/owned/native"],
        "environment": execution,
        "profile_sha256": canonical_hash(execution["profile"]),
    }
    child = {
        "schema": CHILD_SCHEMA,
        "status": "PREPARED_FOR_EXEC",
        "pid": 50,
        "process": {"pid": 50, "starttime": "22000"},
        "argv": plan["argv"],
        "environment": deepcopy(execution),
        "environment_sha256": canonical_hash(execution),
    }
    validate_child(child, plan, 50, {"pid": 50, "starttime": "22000"})
    child["environment"]["observation"][SESSION_KEY]["value"] = "wrong"
    child["environment_sha256"] = canonical_hash(child["environment"])
    with pytest.raises(ValueError):
        validate_child(child, plan, 50, {"pid": 50, "starttime": "22000"})


def session_packets() -> tuple[dict, dict]:
    packets = fixture()
    policy, execution, binding, parent = session_fixture()
    capture = deepcopy(packets[0]["capture"])
    capture.update(
        profiler_version=binding["profiler"]["version"],
        profiler_sha256=binding["profiler"]["sha256"],
        profiler_argv=binding["profiler"]["argv"],
    )
    for packet in packets:
        packet["identity"]["observation_policy"] = deepcopy(policy)
        packet["identity"]["observation_policy_sha256"] = canonical_hash(policy)
        packet["execution_environment"] = deepcopy(execution)
        packet["execution_environment_sha256"] = canonical_hash(execution)
        packet["profiler_session_binding"] = deepcopy(binding)
        packet["profiler_session_binding_sha256"] = canonical_hash(binding)
        packet["parent_process"] = deepcopy(parent)
        packet["capture"] = deepcopy(capture)
    return packets


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize(
    "defect",
    ["missing-binding", "tool-version", "tool-hash", "tool-argv", "binding-hash"],
)
def test_attribution_independently_requires_actual_profiler_binding(
    side: int, defect: str
) -> None:
    packets = session_packets()
    summarize(*packets)
    packet = packets[side]
    if defect == "missing-binding":
        del packet["profiler_session_binding"]
    elif defect == "tool-version":
        packet["capture"]["profiler_version"] = "wrong"
    elif defect == "tool-hash":
        packet["capture"]["profiler_sha256"] = "b" * 64
    elif defect == "tool-argv":
        packet["capture"]["profiler_argv"] = ["wrong"]
    elif defect == "binding-hash":
        packet["profiler_session_binding_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        summarize(*packets)


def test_attribution_rejects_two_individually_valid_different_session_bindings() -> (
    None
):
    trace, receipt = session_packets()
    # Use a different valid process starttime, preserving each local hash/schema.
    receipt["profiler_session_binding"]["outer_process"]["starttime"] = "11001"
    receipt["profiler_session_binding_sha256"] = canonical_hash(
        receipt["profiler_session_binding"]
    )
    with pytest.raises(ValueError, match="session bindings differ"):
        summarize(trace, receipt)


@pytest.mark.parametrize("defect", ["missing", "different-starttime", "different-pid"])
def test_profiled_child_requires_actual_parent_observed_process_identity(
    defect: str,
) -> None:
    from benchmarks.cc_tile_environment import validate_child

    _, execution, _, _ = session_fixture()
    plan = {
        "argv": ["/owned/native"],
        "environment": execution,
        "profile_sha256": canonical_hash(execution["profile"]),
    }
    observed = {"pid": 50, "starttime": "22000"}
    child = {
        "schema": CHILD_SCHEMA,
        "status": "PREPARED_FOR_EXEC",
        "pid": 50,
        "process": dict(observed),
        "argv": plan["argv"],
        "environment": execution,
        "environment_sha256": canonical_hash(execution),
    }
    if defect == "missing":
        del child["process"]
    elif defect == "different-starttime":
        child["process"]["starttime"] = "22001"
    else:
        child["process"]["pid"] = 51
    with pytest.raises(ValueError):
        validate_child(child, plan, 50, observed)


def diagnostics_fixture(tmp_path: Path) -> tuple[list[dict], dict, dict]:
    from benchmarks.cc_tile_capture_diagnostics import (
        PARENT_WARNINGS,
        diagnostics_policy,
    )

    _, _, binding, parent = session_fixture()
    source = tmp_path / "parent.py"
    source.write_text("qualified wrapper fixture\n")
    severity = {"UnknownLevel": 0, "Info": 1, "Warning": 2, "Error": 3, "Verbose": 4}
    origins = {"UnknownSource": 0, "Injection": 1, "Daemon": 2, "Analysis": 3}
    policy = diagnostics_policy(
        binding["profiler"], checksum(source), severity, origins
    )
    rows = [
        {
            "timestamp": 0,
            "timestampType": 1,
            "source": 3,
            "severity": 1,
            "text": "Profiling has started.",
            "globalPid": -1,
        }
    ]
    rows.extend(
        {
            "timestamp": 1,
            "timestampType": 2,
            "source": 3,
            "severity": 2,
            "text": text,
            "globalPid": parent["pid"] * 0x1000000,
        }
        for text in PARENT_WARNINGS
    )
    arguments = {
        "profiler": binding["profiler"],
        "severity_codes": severity,
        "source_codes": origins,
        "binding": binding,
        "binding_hash": canonical_hash(binding),
        "parent": parent,
        "native": {"pid": 50, "starttime": "22000"},
        "parent_global_tid": 44 * 0x1000000 + 44,
        "native_global_pid": 50 * 0x1000000,
        "parent_cuda_activities": 0,
        "native_cuda_activities": 1,
        "parent_source": source,
    }
    return rows, policy, arguments


def test_predeclared_parent_warnings_are_retained_without_claiming_cuda_pass(
    tmp_path: Path,
) -> None:
    from benchmarks.cc_tile_capture_diagnostics import classify_diagnostics

    rows, policy, arguments = diagnostics_fixture(tmp_path)
    result = classify_diagnostics(rows, policy, canonical_hash(policy), **arguments)
    assert result["accepted_parent_warning_count"] == 2
    assert result["accepted_parent_warnings"] == rows[1:]
    assert result["records"] == rows
    assert "PASS" not in result["classification"]


@pytest.mark.parametrize(
    "defect",
    [
        "native-warning",
        "global-warning",
        "unknown-warning",
        "same-pid-other-domain",
        "extra-text",
        "duplicate",
        "wrong-source",
        "error",
        "unknown-severity",
        "unknown-source",
        "bool-severity",
        "bad-clock-type",
        "missing-rows",
        "parent-cuda",
        "unknown-parent-cuda",
        "native-no-cuda",
        "unknown-native-cuda",
        "parent-pid",
        "parent-starttime",
        "native-pid",
        "source-changed",
        "source-missing",
        "tool-version",
        "tool-hash",
        "binding-hash",
        "enum-drift",
        "unknown-policy-field",
        "extra-allowed-text",
        "policy-hash",
    ],
)
def test_diagnostics_parent_exception_cannot_hide_unknown_or_child_failure(
    tmp_path: Path, defect: str
) -> None:
    from benchmarks.cc_tile_capture_diagnostics import classify_diagnostics

    rows, policy, arguments = diagnostics_fixture(tmp_path)
    if defect == "native-warning":
        rows[1]["globalPid"] = arguments["native_global_pid"]
    elif defect == "global-warning":
        rows[1]["globalPid"] = -1
    elif defect == "unknown-warning":
        rows[1]["globalPid"] = 51 * 0x1000000
    elif defect == "same-pid-other-domain":
        rows[1]["globalPid"] += 1 << 48
    elif defect == "extra-text":
        rows[1]["text"] += " extra"
    elif defect == "duplicate":
        rows.append(deepcopy(rows[1]))
    elif defect == "wrong-source":
        rows[1]["source"] = 1
    elif defect == "error":
        rows[1]["severity"] = 3
    elif defect == "unknown-severity":
        rows[1]["severity"] = 0
    elif defect == "unknown-source":
        rows[1]["source"] = 0
    elif defect == "bool-severity":
        rows[1]["severity"] = True
    elif defect == "bad-clock-type":
        rows[1]["timestampType"] = 99
    elif defect == "missing-rows":
        rows = []
    elif defect == "parent-cuda":
        arguments["parent_cuda_activities"] = 1
    elif defect == "unknown-parent-cuda":
        arguments["parent_cuda_activities"] = None
    elif defect == "native-no-cuda":
        arguments["native_cuda_activities"] = 0
    elif defect == "unknown-native-cuda":
        arguments["native_cuda_activities"] = None
    elif defect == "parent-pid":
        arguments["parent"] = dict(arguments["parent"], pid=45)
    elif defect == "parent-starttime":
        arguments["parent"] = dict(arguments["parent"], starttime="12001")
    elif defect == "native-pid":
        arguments["native"]["pid"] = 51
    elif defect == "source-changed":
        arguments["parent_source"].write_text("changed\n")
    elif defect == "source-missing":
        arguments["parent_source"].unlink()
    elif defect == "tool-version":
        arguments["profiler"] = dict(arguments["profiler"], version="unqualified")
    elif defect == "tool-hash":
        arguments["profiler"] = dict(arguments["profiler"], sha256="b" * 64)
    elif defect == "binding-hash":
        arguments["binding_hash"] = "0" * 64
    elif defect == "enum-drift":
        arguments["source_codes"] = dict(arguments["source_codes"], Analysis=9)
    elif defect == "unknown-policy-field":
        policy["ignore_all_warnings"] = True
    elif defect == "extra-allowed-text":
        policy["parent_warnings"].append("GPU buffer overflow")
    policy_hash = "0" * 64 if defect == "policy-hash" else canonical_hash(policy)
    with pytest.raises(ValueError):
        classify_diagnostics(rows, policy, policy_hash, **arguments)


@pytest.mark.parametrize("defect", ["warnings", "missing-table", "unknown-enum"])
def test_sqlite_diagnostics_fail_closed_without_prior_parent_policy(
    tmp_path: Path, defect: str
) -> None:
    path, receipt = sqlite_fixture(tmp_path)
    with sqlite3.connect(path) as db:
        if defect == "warnings":
            db.execute(
                "update DIAGNOSTIC_EVENT set severity=2,text='No CUDA events collected. Does the process use CUDA?'"
            )
        elif defect == "missing-table":
            db.execute("drop table DIAGNOSTIC_EVENT")
        else:
            db.execute(
                "update ENUM_DIAGNOSTIC_SOURCE_TYPE set name='unrecognized' where id=3"
            )
    receipt["nsys"].update(
        sqlite_sha256=checksum(path), sqlite_size=path.stat().st_size
    )
    with pytest.raises(ValueError):
        read_nsys(path, receipt)


def sqlite_parent_warning_fixture(tmp_path: Path) -> tuple[Path, dict]:
    path, receipt = sqlite_fixture(tmp_path)
    policy, execution, binding, parent = session_fixture()
    diagnostic_rows, warning_policy, arguments = diagnostics_fixture(tmp_path)
    receipt["identity"]["observation_policy"] = policy
    receipt["identity"]["observation_policy_sha256"] = canonical_hash(policy)
    receipt["identity"]["capture_diagnostics_policy"] = warning_policy
    receipt["identity"]["capture_diagnostics_policy_sha256"] = canonical_hash(
        warning_policy
    )
    receipt["execution_environment"] = execution
    receipt["execution_environment_sha256"] = canonical_hash(execution)
    receipt["profiler_session_binding"] = binding
    receipt["profiler_session_binding_sha256"] = canonical_hash(binding)
    receipt["parent_process"] = parent
    receipt["native_process"] = arguments["native"]
    receipt["capture"].update(
        profiler_version=binding["profiler"]["version"],
        profiler_sha256=binding["profiler"]["sha256"],
        profiler_argv=binding["profiler"]["argv"],
    )
    parent_tid = arguments["parent_global_tid"]
    native_pid = arguments["native_global_pid"]
    native_tid = native_pid + arguments["native"]["pid"]
    with sqlite3.connect(path) as db:
        db.execute(
            "update NVTX_EVENTS set text='cc1765/complete-process',globalTid=? where text='endpoint'",
            (parent_tid,),
        )
        db.execute(
            "update NVTX_EVENTS set globalTid=? where text='prepare'", (native_tid,)
        )
        for statement in [
            "update CUPTI_ACTIVITY_KIND_KERNEL set globalPid=? where globalPid=1",
            "update CUPTI_ACTIVITY_KIND_MEMCPY set globalPid=? where globalPid=1",
            "update CUPTI_ACTIVITY_KIND_SYNCHRONIZATION set globalPid=? where globalPid=1",
        ]:
            db.execute(statement, (native_pid,))
        db.execute("update CUPTI_ACTIVITY_KIND_RUNTIME set globalTid=?", (native_tid,))
        db.execute("delete from DIAGNOSTIC_EVENT")
        db.executemany(
            "insert into DIAGNOSTIC_EVENT values(?,?,?,?,?,?)",
            [
                (
                    r["timestamp"],
                    r["timestampType"],
                    r["source"],
                    r["severity"],
                    r["text"],
                    r["globalPid"],
                )
                for r in diagnostic_rows
            ],
        )
    receipt["nsys"].update(
        sqlite_sha256=checksum(path),
        sqlite_size=path.stat().st_size,
        cuda_global_pid=native_pid,
        runtime_global_tids=[native_tid],
        endpoint_label="cc1765/complete-process",
        endpoint_global_tid=parent_tid,
        parent_source_artifact=str(arguments["parent_source"]),
    )
    return path, receipt


def test_sqlite_reader_binds_and_retains_only_parent_warning_exception(
    tmp_path: Path,
) -> None:
    path, receipt = sqlite_parent_warning_fixture(tmp_path)
    trace = read_nsys(path, receipt)
    diagnostics = trace["extraction"]["diagnostics"]
    assert diagnostics["accepted_parent_warning_count"] == 2
    assert diagnostics["observed_parent_cuda_activities"] == 0
    assert len(diagnostics["records"]) == 3
    assert summarize(trace, receipt)["status"] == "PASS"


@pytest.mark.parametrize(
    "defect",
    [
        "child-warning",
        "parent-cuda-api",
        "missing-parent-source",
        "missing-policy",
        "source-drift",
    ],
)
def test_sqlite_reader_parent_exception_is_not_a_global_warning_filter(
    tmp_path: Path, defect: str
) -> None:
    path, receipt = sqlite_parent_warning_fixture(tmp_path)
    if defect == "child-warning":
        with sqlite3.connect(path) as db:
            db.execute(
                "update DIAGNOSTIC_EVENT set globalPid=? where severity=2",
                (receipt["nsys"]["cuda_global_pid"],),
            )
    elif defect == "parent-cuda-api":
        with sqlite3.connect(path) as db:
            db.execute(
                "insert into CUPTI_ACTIVITY_KIND_RUNTIME values(1,2,?,99,1)",
                (receipt["nsys"]["endpoint_global_tid"],),
            )
    elif defect == "missing-parent-source":
        del receipt["nsys"]["parent_source_artifact"]
    elif defect == "missing-policy":
        del receipt["identity"]["capture_diagnostics_policy"]
        del receipt["identity"]["capture_diagnostics_policy_sha256"]
    else:
        Path(receipt["nsys"]["parent_source_artifact"]).write_text("changed")
    receipt["nsys"].update(
        sqlite_sha256=checksum(path), sqlite_size=path.stat().st_size
    )
    with pytest.raises(ValueError):
        read_nsys(path, receipt)


@pytest.mark.parametrize("lost", [False, True])
def test_campaign_retains_predeclared_parent_source_or_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost: bool
) -> None:
    from benchmarks.cc_tile_capture_diagnostics import diagnostics_policy

    _, output, endpoint = campaign_invocation(tmp_path, monkeypatch)
    identity_path = tmp_path / "identity.json"
    identity = json.loads(identity_path.read_text())
    _, _, binding, _ = session_fixture()
    source = tmp_path / "nvtx_parent.py"
    source.write_text("parent source fixture\n")
    profiler = dict(
        binding["profiler"], argv=[*binding["profiler"]["argv"][:-1], str(source)]
    )
    observation = execution_environment(os.environ)["observation"]
    identity["observation_policy"] = observation_policy(
        observation, observation, nsys_profiler=profiler
    )
    identity["observation_policy_sha256"] = canonical_hash(
        identity["observation_policy"]
    )
    identity["capture_diagnostics_policy"] = diagnostics_policy(
        profiler,
        checksum(source),
        {"UnknownLevel": 0, "Info": 1, "Warning": 2, "Error": 3, "Verbose": 4},
        {"UnknownSource": 0, "Injection": 1, "Daemon": 2, "Analysis": 3},
    )
    identity["capture_diagnostics_policy_sha256"] = canonical_hash(
        identity["capture_diagnostics_policy"]
    )
    atomic_json(identity_path, identity)

    class Native:
        pid = 1765

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            fake_endpoint(command, kwargs, endpoint)

        def wait(self, timeout: float) -> int:
            if lost:
                (output / "nsys_parent.py").unlink()
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", Native)
    if lost:
        with pytest.raises(ValueError, match="required campaign artifact"):
            campaign_main()
    else:
        assert campaign_main() == 0
        assert (output / "nsys_parent.py").read_bytes() == source.read_bytes()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["outcome"] == ("FAIL" if lost else "PASS")


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="same-PID exec and immutable run-directory contract is Linux-specific",
)
def test_real_child_launcher_keeps_immutable_run_directory_free_of_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use real Python exec with synthetic data and explicitly mocked native bindings."""
    import base64

    from benchmarks.cc_tile_pipeline_campaign_report import read_run

    paths, output, endpoint = campaign_invocation(tmp_path, monkeypatch)
    paths["input"].write_text(
        "import base64,sys\nfrom pathlib import Path\n"
        f"Path(sys.argv[1]).write_bytes(base64.b64decode({base64.b64encode(endpoint).decode()!r}))\n"
    )
    identity_path = tmp_path / "identity.json"
    identity = json.loads(identity_path.read_text())
    identity["input_sha256"] = checksum(paths["input"])
    identity["oracle_input_sha256"] = identity["input_sha256"]
    identity["executable_sha256"] = checksum(Path(sys.executable))
    atomic_json(identity_path, identity)
    arguments = list(sys.argv)
    arguments[arguments.index("--executable") + 1] = sys.executable
    monkeypatch.setattr(sys, "argv", arguments)
    assert campaign_main() == 0
    assert not (output / "__pycache__").exists()
    assert all(path.is_file() and not path.is_symlink() for path in output.iterdir())
    assert read_run(output)["receipt"]["outcome"] == "PASS"

"""Hand-clock and discrete independent oracles for attribution, not CC math."""

import gzip
import hashlib
import json
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
    artifacts = {
        path.name: {"sha256": checksum(path), "size": path.stat().st_size}
        for path in directory.iterdir()
    }
    identity = fixture()[0]["identity"]
    identity["oracle_sha256"] = checksum(directory / "oracle.json")
    identity["oracle_input_sha256"] = identity["input_sha256"]
    receipt = {
        "schema": "generativeqc.cc-tile-campaign.v1",
        "identity": identity,
        "diagnostic": diagnostic,
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
    identity["environment_sha256"] = hashlib.sha256(
        json.dumps(environment, sort_keys=True).encode()
    ).hexdigest()
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
            Path(command[2]).write_bytes(endpoint)

        def wait(self, timeout: int) -> int:
            if changed:
                paths[changed].write_bytes(b"changed during native execution")
            return 0

    monkeypatch.setattr("benchmarks.cc_tile_pipeline_campaign.subprocess.Popen", Native)
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
    identity["environment_sha256"] = hashlib.sha256(
        json.dumps(environment, sort_keys=True).encode()
    ).hexdigest()
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
            Path(command[2]).write_bytes(endpoint)
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
        assert set(receipt["artifacts"]) == {
            "native.log",
            "endpoint.json",
            "oracle.json",
        }
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
            Path(command[2]).write_bytes(endpoint)
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
            Path(command[2]).write_bytes(endpoint)
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
    assert {"endpoint.json", "oracle.json", "native.log"} == set(receipt["artifacts"])


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
            Path(command[2]).write_bytes(endpoint)
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
            Path(command[2]).write_bytes(endpoint)
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
            Path(command[2]).write_bytes(endpoint)
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
    assert {"endpoint.json", "oracle.json", "native.log"} == set(receipt["artifacts"])


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

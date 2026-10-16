"""Host orchestration and parser fixtures; these are not cold GPU measurements."""

import hashlib
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.performance import assess_comparison

from benchmarks.pbe0_xc_cold_pairs import digest, observe_process
from tools.generativeqc_validation.pbe0_xc_cold import verify_observation, verify_pairs
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/pbe0-xc-tiles-20261006"


def fixture_record() -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse immutable vectors/work as parser fixtures without relabelling data."""
    original = load_publication_record(BUNDLE, role="samples", name="pairs-48.json")
    reference = load_publication_record(BUNDLE, role="input", name="reference-48.json")
    observations = []
    for geometry in (0, 1):
        source = next(
            r
            for r in original["samples"]
            if r["selection"] == "baseline"
            and int(r["phase"] == "moved-warm") == geometry
        )
        for arm in ("baseline", "candidate"):
            diagnostic = deepcopy(source["diagnostics"])
            nonce = f"synthetic-{geometry}-{arm}"
            endpoint = {
                "nonce": nonce,
                "pid": 222 + len(observations),
                "ppid": 111,
                "selection": arm,
                "geometry": geometry,
                "energy": diagnostic["energy"],
                "forces": diagnostic["forces"],
                "synchronized": True,
            }
            worker = {
                k: diagnostic[k]
                for k in (
                    "iterations",
                    "fock_builds",
                    "native_ks_diagnostic",
                    "native_scf_ao_work",
                    "native_force_work_raw",
                    "native_force_components",
                )
            }
            worker["native_ks_diagnostic"]["coulomb_exchange_work"] = {
                "complete": True,
                "coulomb_applications": worker["fock_builds"],
                "exchange_applications": worker["fock_builds"],
            }
            worker.update(
                status="complete",
                endpoint=endpoint,
                protocol=reference["protocol"],
                native_build=deepcopy(original["native_build"]),
                warm_start_requested=False,
                item_status=0,
                converged=True,
                worker_sha256="synthetic-worker",
                sdk_source_head="synthetic-sdk",
                warm_start_used=False,
                warm_start_fallback=False,
                allocation={
                    "slurm_job_id": "host-fixture",
                    "inspire_job_name": None,
                    "cuda_visible_devices": "synthetic",
                    "nvidia_visible_devices": None,
                },
            )
            observations.append(
                {
                    "nonce": nonce,
                    "child_pid": endpoint["pid"],
                    "launcher_pid": endpoint["pid"],
                    "process_lineage": "direct",
                    "parent_pid": 111,
                    "launch_started": 10.0,
                    "endpoint_received": 11.0,
                    "process_completed": 12.0,
                    "seconds": 2.0,
                    "delivery_seconds": 1.0,
                    "process_seconds": 2.0,
                    "exit_code": 0,
                    "timeout": False,
                    "endpoint": endpoint,
                    "worker": worker,
                    "selection": arm,
                    "geometry": geometry,
                    "pair": 0,
                    "workload": "energy-plus-force",
                    "synchronized": True,
                    "inputs_hash": canonical_hash(
                        {
                            "protocol": reference["protocol"],
                            "geometry": geometry,
                            "density_scope": "fresh process, owner and density",
                        }
                    ),
                }
            )
    record = {
        "schema": "generativeqc.pbe0-xc-point-family-cold.v1",
        "stage": "complete",
        "protocol": reference["protocol"],
        "repeats": 1,
        "feasibility": True,
        "expected_native_identity": original["native_build"]["probe"][
            "source_identity"
        ],
        "observations": observations,
        "manifest": {
            "worker_sha256": "synthetic-worker",
            "sdk_source_head": "synthetic-sdk",
            "native_identity": original["native_build"]["probe"]["source_identity"],
            "library_path": original["native_build"]["library_path"],
            "library_sha256": original["native_build"]["library_sha256"],
        },
    }
    record["assessments"] = {
        str(g): assess_comparison([r for r in observations if r["geometry"] == g])
        for g in (0, 1)
    }
    return record, reference


def test_feasibility_does_not_satisfy_full_cold_gate() -> None:
    record, reference = fixture_record()
    verify_pairs(record, reference, allow_feasibility=True)
    with pytest.raises(AssertionError):
        verify_pairs(record, reference)
    with pytest.raises(AssertionError):
        verify_pairs(
            record, reference, allow_feasibility=True, require_improvement=True
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "energy",
        "force",
        "request",
        "work",
        "warm",
        "time-bool",
        "time-delivery",
        "source",
        "hardware",
        "exit",
        "sync",
        "nonce",
        "empty-history",
        "nonfinite-history",
        "item-status",
        "not-converged",
        "incomplete-jk",
        "unknown-jk",
        "worker-code",
        "sdk-source",
        "library",
    ),
)
def test_invalid_cold_observations_fail_closed(mutation: str) -> None:
    record, reference = fixture_record()
    row = record["observations"][0]
    worker = row["worker"]
    if mutation == "energy":
        row["endpoint"]["energy"] += 1e-3
    elif mutation == "force":
        row["endpoint"]["forces"][0][0] += 1e-3
    elif mutation == "request":
        worker["native_force_work_raw"]["grid_tile_points_requested"] = None
    elif mutation == "work":
        worker["native_force_components"]["becke_owners"]["stationary"][
            "work_counters"
        ]["becke_phase_points"] = 0
    elif mutation == "warm":
        worker["warm_start_used"] = True
    elif mutation == "time-bool":
        row["launch_started"] = True
    elif mutation == "time-delivery":
        row["seconds"] = row["delivery_seconds"]
    elif mutation == "source":
        worker["native_build"]["probe"]["source_identity"] = "changed"
    elif mutation == "hardware":
        worker["native_build"]["probe"]["device"]["major"] = 9
    elif mutation == "exit":
        row["exit_code"] = 1
    elif mutation == "sync":
        row["synchronized"] = False
    elif mutation == "nonce":
        row["nonce"] = "other"
    elif mutation == "empty-history":
        worker["native_ks_diagnostic"]["history"] = []
    elif mutation == "nonfinite-history":
        worker["native_ks_diagnostic"]["history"][0]["components"]["xc"] = float("nan")
    elif mutation == "item-status":
        worker["item_status"] = False
    elif mutation == "not-converged":
        worker["converged"] = False
    elif mutation == "incomplete-jk":
        worker["native_ks_diagnostic"]["coulomb_exchange_work"]["complete"] = False
    elif mutation == "unknown-jk":
        worker["native_ks_diagnostic"]["coulomb_exchange_work"][
            "coulomb_applications"
        ] = None
    elif mutation == "worker-code":
        worker["worker_sha256"] = "changed"
    elif mutation == "sdk-source":
        worker["sdk_source_head"] = "changed"
    elif mutation == "library":
        worker["native_build"]["library_sha256"] = "changed"
    with pytest.raises(AssertionError):
        verify_observation(row, reference, record["manifest"])


@pytest.mark.parametrize("work", ("jk", "non-becke", "tensor", "route"))
def test_unequal_semantic_work_is_rejected(work: str) -> None:
    record, reference = fixture_record()
    worker = record["observations"][1]["worker"]
    if work == "jk":
        worker["native_ks_diagnostic"]["coulomb_exchange_work"][
            "exchange_applications"
        ] += 1
    elif work == "non-becke":
        worker["native_force_components"]["work_counts"]["extra-executed-work"] = 1
    elif work == "tensor":
        worker["native_force_work_raw"]["tensor_work"]["extra-executed-work"] = 1
    else:
        worker["native_force_components"]["stationary_integral_derivative_route"] = (
            "other"
        )
    with pytest.raises(AssertionError):
        verify_pairs(record, reference, allow_feasibility=True)


@pytest.mark.parametrize("field", ("device", "allocation"))
def test_all_pairs_share_one_frozen_build_and_allocation(field: str) -> None:
    record, reference = fixture_record()
    for row in record["observations"][2:]:
        if field == "device":
            row["worker"]["native_build"]["probe"]["device"]["name"] = "another device"
        else:
            row["worker"]["allocation"]["slurm_job_id"] = "another job"
    with pytest.raises(AssertionError):
        verify_pairs(record, reference, allow_feasibility=True)


def test_large_artifact_hash_is_streamed(tmp_path: Path) -> None:
    payload = b"artifact\x00" * (1 << 19)
    path = tmp_path / "artifact.so"
    path.write_bytes(payload)
    assert digest(path) == hashlib.sha256(payload).hexdigest()


def command(tmp_path: Path, nonce: str, mode: str = "normal") -> list[str]:
    path = tmp_path / "synthetic_worker.py"
    path.write_text(
        """import json,os,sys,time
nonce,mode=sys.argv[1:]
if mode=='timeout':time.sleep(30)
if mode=='malformed':
 print('PBE_COLD_ENDPOINT={',flush=True);time.sleep(30)
time.sleep(.03)
e={'nonce':nonce,'pid':os.getpid(),'ppid':os.getppid(),'synchronized':True}
print('PBE_COLD_ENDPOINT='+json.dumps(e),flush=True)
time.sleep(.12)
print('PBE_COLD_RESULT='+json.dumps({'status':'complete','endpoint':e}),flush=True)
sys.exit(1 if mode=='failed' else 0)
""",
        encoding="utf-8",
    )
    return [sys.executable, str(path), nonce, mode]


def test_fresh_process_timing_includes_audit_and_cleanup(tmp_path: Path) -> None:
    rows = [
        observe_process(command(tmp_path, nonce), nonce=nonce, timeout=5)
        for nonce in ("first", "second")
    ]
    assert all(
        r["child_pid"] != os.getpid() and r["synchronized"] and r["exit_code"] == 0
        for r in rows
    )
    assert rows[0]["child_pid"] != rows[1]["child_pid"]
    assert all(r["seconds"] >= r["delivery_seconds"] + 0.08 for r in rows)


@pytest.mark.parametrize("mode", ("timeout", "malformed", "failed"))
def test_failed_child_is_reaped_and_cannot_claim_sync(
    tmp_path: Path, mode: str
) -> None:
    row = observe_process(
        command(tmp_path, "failure", mode),
        nonce="failure",
        timeout=0.3 if mode == "timeout" else 5,
    )
    assert row["exit_code"] is not None and row["synchronized"] is False
    if mode == "timeout":
        assert row["timeout"] is True
    elif mode == "malformed":
        assert row["parse_error"] == "malformed endpoint"
    else:
        assert row["exit_code"] == 1


def test_missing_executable_retains_failed_observation(tmp_path: Path) -> None:
    row = observe_process([str(tmp_path / "nonexistent")], nonce="missing", timeout=1)
    assert row["spawn_error"] and row["child_pid"] is None and not row["synchronized"]


def test_collector_consumer_is_not_shadowed_by_frozen_sdk(tmp_path: Path) -> None:
    sdk = tmp_path / "sdk"
    (sdk / "tools").mkdir(parents=True)
    (sdk / "tools/__init__.py").write_text("")
    script = """import importlib.util,sys
from pathlib import Path
from typing import Any
sdk,root=sys.argv[1:];sys.path[:0]=[sdk,str(Path(root)/'python'),root]
p=Path(root)/'benchmarks/pbe0_xc_cold_pairs.py'
s=importlib.util.spec_from_file_location('collector',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
c=m.load_consumer();assert Path(c.__file__)==Path(root)/'tools/generativeqc_validation/pbe0_xc_cold.py'
"""
    subprocess.run(
        [sys.executable, "-c", script, str(sdk), str(ROOT)], check=True, timeout=15
    )

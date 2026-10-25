"""Host orchestration and parser fixtures; these are not cold GPU measurements."""

import hashlib
import io
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.performance import assess_comparison

from benchmarks import pbe0_xc_cold_pairs as collector
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
            # The vectors are immutable fixtures. These process/solve receipts
            # are explicitly synthetic cold observations, not relabelled runs.
            worker["iterations"] = worker["fock_builds"] = 2
            worker["initial_guess"] = None
            worker["restart_origin"] = "cold"
            ks = worker["native_ks_diagnostic"]
            ks["initial_density_used"] = False
            ks["fock_builds"] = 2
            first = deepcopy(ks["history"][0])
            first.update(iteration=1, energy_change=None)
            last = deepcopy(first)
            last.update(iteration=2, energy_change=1e-13)
            ks["history"] = [first, last]
            ks["coulomb_exchange_work"].update(
                coulomb_applications=2, exchange_applications=2
            )
            worker["native_scf_ao_work"]["xc_evaluations"] = 2
            state = f"cuda-ks:{len(observations) + 1}:1:2"
            worker["precision"] = {
                "requested_mode": "fp64",
                "effective_bits": 64,
                "strict_refinement_applied": False,
                "complete": True,
                "operator_inventory_complete": True,
                "operator_work_counters_valid": True,
                "refinement_iterations": 0,
                "mixed_stage_fock_builds": 0,
                "execution_retries": 0,
                "conversion_count": 0,
                "fallback_count": 0,
                "strict_stage_fock_builds": 2,
                "post_scf_fock_builds": 0,
                "final_residual_audits": 1,
                "returned_state_identity": state,
                "scf_fock_timeline": [
                    {
                        "kind": "strict_fock",
                        "sequence": i - 1,
                        "iteration": i,
                        "state": state,
                        "phase": "scf",
                        "count": 1,
                    }
                    for i in (1, 2)
                ]
                + [
                    {
                        "kind": "final_audit",
                        "sequence": 2,
                        "iteration": 2,
                        "state": state,
                        "phase": "finalization",
                        "count": 1,
                    }
                ],
                "operators": [
                    {
                        "name": name,
                        "storage": "fp64",
                        "compute": "fp64",
                        "accumulation": "fp64",
                        "reduction": "fp64",
                        "arithmetic_mode": "strict",
                        "count": 2,
                    }
                    for name in ("coulomb_j", "exchange_k", "xc")
                ],
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


def collector_args(tmp_path: Path, output: Path) -> list[str]:
    _, reference = fixture_record()
    (tmp_path / "reference.json").write_text(json.dumps(reference))
    (tmp_path / "basis.json").write_text("synthetic basis")
    (tmp_path / "library.so").write_text("synthetic library")
    return [
        "collector",
        "--atoms",
        "48",
        "--basis-file",
        str(tmp_path / "basis.json"),
        "--reference",
        str(tmp_path / "reference.json"),
        "--sdk-source",
        str(ROOT),
        "--library",
        str(tmp_path / "library.so"),
        "--expected-native-identity",
        "synthetic-native",
        "--output",
        str(output),
        "--feasibility",
        "--repeats",
        "1",
    ]


def host_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "host-fixture")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "host-fixture")


@pytest.mark.parametrize(
    "alias",
    ("existing", "basis", "reference", "library", "symlink", "dangling", "hardlink"),
)
def test_output_reservation_preserves_existing_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    output = tmp_path / "output.json"
    argv = collector_args(tmp_path, output)
    targets = {
        "basis": tmp_path / "basis.json",
        "reference": tmp_path / "reference.json",
        "library": tmp_path / "library.so",
    }
    if alias in targets:
        argv[argv.index("--output") + 1] = str(targets[alias])
    elif alias == "symlink":
        output.symlink_to(tmp_path / "basis.json")
    elif alias == "dangling":
        output.symlink_to(tmp_path / "missing")
    elif alias == "hardlink":
        os.link(tmp_path / "basis.json", output)
    else:
        output.write_text("prior receipt")
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    host_allocation(monkeypatch)
    monkeypatch.setattr(sys, "argv", argv)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("output must be reserved before provenance or worker execution")

    monkeypatch.setattr(collector, "native_inventory", unexpected)
    monkeypatch.setattr(collector.subprocess, "check_output", unexpected)
    monkeypatch.setattr(collector, "observe_process", unexpected)
    with pytest.raises(SystemExit):
        collector.main()
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (tmp_path / "missing").exists()


def test_separate_collector_results_tree_is_protected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "collector"
    output = source / "benchmarks/results/new-receipt.json"
    argv = collector_args(tmp_path, output)
    host_allocation(monkeypatch)
    monkeypatch.setattr(sys, "argv", argv)
    consumer = collector.load_consumer()
    monkeypatch.setattr(collector, "load_consumer", lambda: consumer)
    monkeypatch.setattr(
        collector, "__file__", str(source / "benchmarks/pbe0_xc_cold_pairs.py")
    )
    with pytest.raises(SystemExit) as error:
        collector.main()
    assert error.value.code == 2
    assert "argument --output:" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("root_kind", ("collector", "sdk"))
@pytest.mark.parametrize("alias", ("direct", "symlink", "traversal"))
@pytest.mark.parametrize("output_first", (False, True))
def test_both_results_roots_are_rejected_before_allocation_or_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    root_kind: str,
    alias: str,
    output_first: bool,
) -> None:
    source, sdk = tmp_path / "collector", tmp_path / "sdk"
    protected = (source if root_kind == "collector" else sdk) / "benchmarks/results"
    protected.mkdir(parents=True)
    original = protected / "prior.json"
    original.write_text("retained evidence", encoding="utf-8")
    target = original
    if alias == "symlink":
        target = tmp_path / "alias.json"
        target.symlink_to(original)
    elif alias == "traversal":
        target = protected.parent.parent / ".artifacts/../benchmarks/results/prior.json"
    argv = collector_args(tmp_path, target)
    argv[argv.index("--sdk-source") + 1] = str(sdk)
    if output_first:
        index = argv.index("--output")
        output = argv[index : index + 2]
        del argv[index : index + 2]
        argv[1:1] = output
    monkeypatch.setattr(collector, "__file__", str(source / "benchmarks/collector.py"))
    monkeypatch.setattr(sys, "argv", argv)
    for name in (
        "SLURM_JOB_ID",
        "INSPIRE_JOB_NAME",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(name, raising=False)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid output reached scientific imports or provenance")

    monkeypatch.setattr(collector, "load_consumer", forbidden)
    monkeypatch.setattr(collector, "worker", forbidden)
    monkeypatch.setattr(collector, "native_inventory", forbidden)
    with pytest.raises(SystemExit) as error:
        collector.main()
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert "argument --output:" in diagnostic and "benchmarks/results" in diagnostic
    assert "allocation" not in diagnostic
    assert original.read_text(encoding="utf-8") == "retained evidence"


@pytest.mark.parametrize("root_kind", ("collector", "sdk"))
def test_direct_script_rejects_results_without_scientific_packages(
    tmp_path: Path,
    root_kind: str,
) -> None:
    source, sdk = tmp_path / "collector", tmp_path / "sdk"
    (source / "benchmarks").mkdir(parents=True)
    for name in ("pbe0_xc_cold_pairs.py", "_retention.py"):
        (source / "benchmarks" / name).write_bytes(
            (ROOT / "benchmarks" / name).read_bytes()
        )
    (sdk / "generativeqc").mkdir(parents=True)
    (sdk / "generativeqc/__init__.py").write_text(
        "raise RuntimeError('early science import')", encoding="utf-8"
    )
    target = (
        source if root_kind == "collector" else sdk
    ) / "benchmarks/results/old.json"
    argv = collector_args(tmp_path, target)
    argv[0] = str(source / "benchmarks/pbe0_xc_cold_pairs.py")
    argv[argv.index("--sdk-source") + 1] = str(sdk)
    environment = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "SLURM_JOB_ID",
            "INSPIRE_JOB_NAME",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "PYTHONPATH",
        )
    }
    result = subprocess.run(
        [sys.executable, *argv],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert (
        "argument --output:" in result.stderr and "benchmarks/results" in result.stderr
    )
    assert (
        "early science import" not in result.stderr
        and "allocation" not in result.stderr
    )
    assert not target.parent.exists()


@pytest.mark.parametrize("optimization", ("-O", "-OO"))
@pytest.mark.parametrize(
    "module",
    ("benchmarks.pbe0_xc_cold_pairs", "tools.generativeqc_validation.pbe0_xc_cold"),
)
def test_optimized_python_cannot_collect_or_validate(
    module: str, optimization: str
) -> None:
    result = subprocess.run(
        [sys.executable, optimization, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "requires Python without -O/-OO" in result.stderr


def test_consumer_exception_retains_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "receipt.json"
    argv = collector_args(tmp_path, output)
    host_allocation(monkeypatch)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        collector, "native_inventory", lambda _: (1, "synthetic-native")
    )
    monkeypatch.setattr(
        collector.subprocess,
        "check_output",
        lambda command, **kwargs: "synthetic-head" if command[-1] == "HEAD" else "",
    )
    row = {"endpoint": None, "worker": None, "exit_code": 1, "synchronized": False}
    monkeypatch.setattr(collector, "observe_process", lambda *args, **kwargs: dict(row))
    consumer = collector.load_consumer()

    def reject(*args: Any) -> None:
        raise StopIteration("missing independent geometry")

    monkeypatch.setattr(consumer, "verify_observation", reject)
    monkeypatch.setattr(collector, "load_consumer", lambda: consumer)
    with pytest.raises(StopIteration, match="missing independent geometry"):
        collector.main()
    retained = json.loads(output.read_text())
    assert retained["stage"] == "incomplete"
    assert retained["failure_type"] == "StopIteration"
    assert len(retained["observations"]) == 1
    assert retained["observations"][0]["exit_code"] == 1


def test_close_error_does_not_replace_collection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = collector_args(tmp_path, tmp_path / "receipt.json")
    host_allocation(monkeypatch)
    monkeypatch.setattr(sys, "argv", argv)

    class BrokenClose(io.StringIO):
        def close(self) -> None:
            super().close()
            raise OSError("close failed")

    stream = BrokenClose()
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: stream)

    def fail(*args: Any) -> None:
        raise RuntimeError("collection failed")

    monkeypatch.setattr(collector, "collect", fail)
    with pytest.raises(RuntimeError, match="collection failed") as captured:
        collector.main()
    assert isinstance(captured.value.__cause__, OSError)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-precision",
        "unknown-precision",
        "mode",
        "bits",
        "refinement",
        "mixed",
        "retry",
        "conversion",
        "fallback",
        "incomplete",
        "inventory",
        "counters",
        "strict-count",
        "post-count",
        "audit-count",
        "state",
        "other-solve",
        "mixed-event",
        "refinement-event",
        "missing-event",
        "operator-fp32",
        "operator-count",
        "missing-xc",
        "warm-density",
        "diagnostic-count",
        "provisional-history",
        "cold-baseline",
        "final-energy",
        "final-density",
        "final-residual",
        "exit-bool",
    ),
)
def test_cold_precision_and_history_receipts_fail_closed(mutation: str) -> None:
    record, reference = fixture_record()
    row = record["observations"][0]
    worker, precision = row["worker"], row["worker"]["precision"]
    ks = worker["native_ks_diagnostic"]
    changes = {
        "mode": ("requested_mode", "auto"),
        "bits": ("effective_bits", 32),
        "refinement": ("strict_refinement_applied", True),
        "mixed": ("mixed_stage_fock_builds", 1),
        "retry": ("execution_retries", 1),
        "conversion": ("conversion_count", 1),
        "fallback": ("fallback_count", 1),
        "incomplete": ("complete", False),
        "inventory": ("operator_inventory_complete", False),
        "counters": ("operator_work_counters_valid", False),
        "strict-count": ("strict_stage_fock_builds", 3),
        "post-count": ("post_scf_fock_builds", 1),
        "audit-count": ("final_residual_audits", 0),
        "state": ("returned_state_identity", None),
    }
    if mutation in changes:
        key, value = changes[mutation]
        precision[key] = value
    elif mutation == "missing-precision":
        del worker["precision"]
    elif mutation == "unknown-precision":
        worker["precision"] = None
    elif mutation == "other-solve":
        precision["scf_fock_timeline"][0]["state"] = "cuda-ks:999:999:1"
    elif mutation == "mixed-event":
        precision["scf_fock_timeline"][0]["kind"] = "mixed_fock"
    elif mutation == "refinement-event":
        precision["scf_fock_timeline"][0]["phase"] = "refinement"
    elif mutation == "missing-event":
        precision["scf_fock_timeline"].pop()
    elif mutation == "operator-fp32":
        precision["operators"][0]["compute"] = "fp32"
    elif mutation == "operator-count":
        precision["operators"][0]["count"] += 1
    elif mutation == "missing-xc":
        precision["operators"].pop()
    elif mutation == "warm-density":
        ks["initial_density_used"] = True
    elif mutation == "diagnostic-count":
        ks["fock_builds"] += 1
    elif mutation == "provisional-history":
        ks["history"][0]["operator_scope"] = "provisional-semilocal-rsh"
    elif mutation == "cold-baseline":
        ks["history"][0]["energy_change"] = 0.0
    elif mutation == "final-energy":
        ks["history"][-1]["energy_change"] = 1.0
    elif mutation == "final-density":
        ks["density_change_max"] = 1.0
    elif mutation == "final-residual":
        ks["physical_residual_max"] = 1.0
    elif mutation == "exit-bool":
        row["exit_code"] = False
    with pytest.raises((AssertionError, KeyError, TypeError)):
        verify_observation(row, reference, record["manifest"])


def test_equal_precision_work_ignores_only_process_owned_state() -> None:
    record, reference = fixture_record()
    verify_pairs(record, reference, allow_feasibility=True)
    record["observations"][1]["worker"]["precision"]["operators"][2]["count"] += 1
    with pytest.raises(AssertionError):
        verify_pairs(record, reference, allow_feasibility=True)


def test_finalization_work_is_retained_without_inventing_an_scf_stage() -> None:
    record, reference = fixture_record()
    for row in record["observations"]:
        worker, precision = row["worker"], row["worker"]["precision"]
        worker["fock_builds"] = 3
        worker["native_ks_diagnostic"]["fock_builds"] = 3
        worker["native_ks_diagnostic"]["coulomb_exchange_work"].update(
            coulomb_applications=3, exchange_applications=3
        )
        precision["post_scf_fock_builds"] = 1
        for operator in precision["operators"][:2]:
            operator["count"] = 3
        precision["scf_fock_timeline"].append(
            {
                "kind": "post_scf_fock",
                "sequence": 3,
                "iteration": 2,
                "state": precision["returned_state_identity"],
                "phase": "finalization",
                "count": 1,
            }
        )
    verify_pairs(record, reference, allow_feasibility=True)


def test_worker_retains_actual_precision_and_closes_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, reference = fixture_record()
    receipt = record["observations"][0]["worker"]
    precision = deepcopy(receipt["precision"])
    item = SimpleNamespace(
        status=0,
        converged=True,
        status_message="success",
        energy=receipt["endpoint"]["energy"],
        forces=SimpleNamespace(tolist=lambda: receipt["endpoint"]["forces"]),
        warm_start_used=False,
        warm_start_fallback=False,
        iterations=2,
        fock_builds=2,
        precision=precision,
        initial_guess=None,
        restart_origin="cold",
        ks_diagnostic=SimpleNamespace(
            to_payload=lambda: receipt["native_ks_diagnostic"]
        ),
    )

    class Owner:
        closed = False

        def _public_dft_cuda_force(self, *args: Any, **kwargs: Any) -> Any:
            assert kwargs["tile_points"] == 256
            return None, receipt["native_force_work_raw"]

        def execute(self, **kwargs: Any) -> Any:
            self._public_dft_cuda_force()
            return SimpleNamespace(items=[item])

        def close(self) -> None:
            self.closed = True

    owner = Owner()

    class Calculator:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["precision"] == "fp64"

        def prepare_batch(self, *args: Any, **kwargs: Any) -> Any:
            assert kwargs["warm_start"] is False
            return owner

    modules = {
        "cupy": SimpleNamespace(
            cuda=SimpleNamespace(
                Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))
            )
        ),
        "generativeqc": SimpleNamespace(
            Calculator=Calculator,
            GridSpec=lambda **kwargs: kwargs,
            KsOptions=lambda **kwargs: kwargs,
        ),
        "benchmarks._support": SimpleNamespace(
            cuda_accelerator_metadata=lambda _: {},
            environment_metadata=lambda **kwargs: {},
        ),
        "benchmarks.compare_gpu4pyscf_batch": SimpleNamespace(
            native_build_metadata=lambda _: receipt["native_build"]
        ),
        "benchmarks.dft_force_components": SimpleNamespace(
            normalize_force_work=lambda _: receipt["native_force_components"]
        ),
        "benchmarks.ks_preliminary_density": SimpleNamespace(
            read_ao_work=lambda _: receipt["native_scf_ao_work"]
        ),
        "benchmarks.readme_hf_scaling": SimpleNamespace(
            scaling_cases=lambda: {"water-48": object()}
        ),
        "benchmarks.readme_omol25": SimpleNamespace(
            load_comparison_basis=lambda *args, **kwargs: (object(), None),
            protocol=lambda *args, **kwargs: reference["protocol"],
        ),
        "benchmarks.readme_pbe0": SimpleNamespace(PBE0=object()),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        collector.subprocess, "check_output", lambda *args, **kwargs: "synthetic-head"
    )
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    args = SimpleNamespace(
        sdk_source=ROOT,
        library=tmp_path / "library.so",
        selection="baseline",
        atoms=48,
        basis_file=tmp_path / "basis.json",
        reference=path,
        geometry=0,
        nonce="synthetic-worker",
    )
    # The real worker owns a fresh process; isolate its environment in this host test.
    environment = dict(os.environ)
    with monkeypatch.context() as worker_process:
        worker_process.setattr(os, "environ", dict(environment))
        observed = collector.worker(args)
    assert dict(os.environ) == environment
    assert observed["precision"] is precision
    assert observed["native_ks_diagnostic"] is receipt["native_ks_diagnostic"]
    assert observed["initial_guess"] is None and observed["restart_origin"] == "cold"
    assert owner.closed is True


@pytest.mark.parametrize("kind", ("minao", "huckel"))
def test_fresh_native_guess_is_admitted_and_its_work_is_compared(kind: str) -> None:
    record, reference = fixture_record()
    for index, row in enumerate(record["observations"]):
        worker = row["worker"]
        worker["native_ks_diagnostic"]["initial_density_used"] = True
        worker["initial_guess"] = {
            "kind": kind,
            "outcome": "used",
            "work_counters_complete": True,
            "target_attempts": 1,
            "discarded_target_iterations": 0,
            "discarded_target_fock_builds": 0,
            "preliminary_iterations": 4,
            "preliminary_fock_builds": 4,
            "preparation_numeric_capacity": 1024,
            "preparation_seconds": index / 10,
        }
    verify_pairs(record, reference, allow_feasibility=True)
    record["observations"][1]["worker"]["initial_guess"]["preliminary_fock_builds"] += 1
    with pytest.raises(AssertionError):
        verify_pairs(record, reference, allow_feasibility=True)


@pytest.mark.parametrize(
    "mutation", ("origin", "retained", "retry", "incomplete", "attempts", "discarded")
)
def test_retained_or_incomplete_initial_guesses_are_rejected(mutation: str) -> None:
    record, reference = fixture_record()
    worker = record["observations"][0]["worker"]
    worker["native_ks_diagnostic"]["initial_density_used"] = True
    guess = {
        "kind": "huckel",
        "outcome": "used",
        "work_counters_complete": True,
        "target_attempts": 1,
        "discarded_target_iterations": 0,
        "discarded_target_fock_builds": 0,
        "preliminary_iterations": 4,
        "preliminary_fock_builds": 4,
        "preparation_numeric_capacity": 1024,
        "preparation_seconds": 0.1,
    }
    worker["initial_guess"] = guess
    if mutation == "origin":
        worker["restart_origin"] = "persistent_restart"
    elif mutation == "retained":
        guess["outcome"] = "existing_density"
    elif mutation == "retry":
        guess["outcome"] = "target_retried"
    elif mutation == "incomplete":
        guess["work_counters_complete"] = False
    elif mutation == "attempts":
        guess["target_attempts"] = 2
    else:
        guess["discarded_target_fock_builds"] = 1
    with pytest.raises(AssertionError):
        verify_pairs(record, reference, allow_feasibility=True)


@pytest.mark.parametrize("token", ("NaN", "Infinity", "-Infinity", "1e999"))
def test_nonfinite_worker_json_retains_serializable_failure(token: str) -> None:
    script = (
        "import time;print('PBE_COLD_ENDPOINT={\"bad\":' + "
        + repr(token)
        + " + '}',flush=True);time.sleep(30)"
    )
    row = observe_process([sys.executable, "-c", script], nonce="bad", timeout=5)
    assert row["parse_error"] == "malformed endpoint"
    assert row["synchronized"] is False and row["exit_code"] is not None
    json.dumps(row, allow_nan=False)


def test_journal_failure_does_not_replace_consumer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collector_args(tmp_path, tmp_path / "unused.json")
    args = SimpleNamespace(
        atoms=48,
        basis_file=tmp_path / "basis.json",
        reference=tmp_path / "reference.json",
        sdk_source=ROOT,
        library=tmp_path / "library.so",
        expected_native_identity="synthetic-native",
        feasibility=True,
        repeats=1,
        timeout=1,
    )
    monkeypatch.setattr(
        collector, "native_inventory", lambda _: (1, "synthetic-native")
    )
    monkeypatch.setattr(
        collector.subprocess,
        "check_output",
        lambda command, **kwargs: "synthetic-head" if command[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(
        collector, "observe_process", lambda *args, **kwargs: {"exit_code": 1}
    )
    consumer = collector.load_consumer()

    def reject(*args: Any) -> None:
        raise StopIteration("missing independent geometry")

    monkeypatch.setattr(consumer, "verify_observation", reject)

    class BrokenJournal(io.StringIO):
        writes = 0

        def write(self, value: str) -> int:
            self.writes += 1
            if self.writes > 1:
                raise OSError("journal unavailable")
            return super().write(value)

    with pytest.raises(StopIteration, match="missing independent geometry") as captured:
        collector.collect(args, BrokenJournal(), consumer)
    assert isinstance(captured.value.__cause__, OSError)


def test_full_synthetic_pairs_require_complete_profitable_cold_windows() -> None:
    record, reference = fixture_record()
    originals = record["observations"]
    record.update(repeats=5, feasibility=False, observations=[])
    for geometry in (0, 1):
        for pair in range(5):
            arms = (
                ("baseline", "candidate")
                if pair % 2 == 0
                else ("candidate", "baseline")
            )
            for arm in arms:
                row = deepcopy(
                    next(
                        x
                        for x in originals
                        if x["geometry"] == geometry and x["selection"] == arm
                    )
                )
                row["nonce"] = row["endpoint"]["nonce"] = (
                    f"full-synthetic-{geometry}-{pair}-{arm}"
                )
                row["pair"] = pair
                row["seconds"] = row["process_seconds"] = (
                    2.0 if arm == "baseline" else 1.5
                )
                row["process_completed"] = row["launch_started"] + row["seconds"]
                record["observations"].append(row)
    record["assessments"] = {
        str(g): assess_comparison(
            [r for r in record["observations"] if r["geometry"] == g]
        )
        for g in (0, 1)
    }
    verify_pairs(record, reference, require_improvement=True)
    record["observations"][0]["seconds"] = record["observations"][0]["delivery_seconds"]
    with pytest.raises(AssertionError):
        verify_pairs(record, reference, require_improvement=True)

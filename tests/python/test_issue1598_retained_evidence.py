"""The reduced historical incremental capsule fails closed without CUDA."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tools import verify_issue1598_history as history
from tools.verify_issue1598_history import restore

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "benchmarks/results/issue1598-incremental-direct-jk-20261011"
POINT = "new/rhf/direct-48-incremental-1.json.gz"
FAILED_POINT = "old/rhf/direct-24-incremental-1.outcome.gz"


@pytest.fixture(scope="module")
def historical_capsule(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("issue1598-original")
    restore(destination, fetch=True)
    return destination


def refresh_manifest(root: Path) -> None:
    """Re-sign storage mutations to reach semantic, rather than hash, gates."""
    path = root / "capsule.json"
    manifest = json.loads(path.read_text())
    for entry in manifest["files"]:
        raw = (root / entry["path"]).read_bytes()
        entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    path.write_text(json.dumps(manifest))


def mutate_record(
    root: Path, relative: str, mutation: Callable[[dict], object]
) -> None:
    path = root / relative
    record = json.loads(gzip.decompress(path.read_bytes()))
    mutation(record)
    path.write_bytes(gzip.compress(json.dumps(record).encode(), mtime=0))
    refresh_manifest(root)


def shift_results(record: dict, *, first_sample_only: bool = False) -> None:
    """Keep independent-reference gates passing while breaking on/off gates."""
    for engine in ("generativeqc", "gpu4pyscf"):
        result = record[engine]
        rows = (
            result["warm_samples"][:1]
            if first_sample_only
            else [result, *result["warm_samples"]]
        )
        for row in rows:
            row["energies_hartree"][0] += 1.0


def make_warm_faster(record: dict) -> None:
    result = record["generativeqc"]
    for sample in result["warm_samples"]:
        sample["seconds"] /= 1000
    result["warm_seconds"] = [s["seconds"] for s in result["warm_samples"]]
    result["warm_median_seconds"] /= 1000


def verify(root: Path, *, optimized: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(root / "verify.py")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("intact", ""),
        ("missing_manifest", "capsule.json"),
        ("missing_success", "missing retained member"),
        ("missing_outcome", "missing retained member"),
        ("missing_manifest_member", "retained member inventory changed"),
        ("drop_all_selected_pairs", "retained member inventory changed"),
        ("duplicate_member", "duplicate manifest member"),
        ("extra_payload", "unlisted compressed records"),
        ("checksum", "retained checksum/size mismatch"),
        ("noninteger_bytes", "retained checksum/size mismatch"),
        ("source", "source identity mismatch"),
        ("dirty", "source identity mismatch"),
        ("source_receipt", "source receipt mismatch"),
        ("library_receipt", "library receipt mismatch"),
        ("native_source", "native source identity mismatch"),
        ("library", "library identity mismatch"),
        ("device", "device identity mismatch"),
        ("probe_device", "device identity mismatch"),
        ("uuid", "device UUID mismatch"),
        ("workload", "measured workload identity changed"),
        ("mode", "incremental mode mismatch"),
        ("inactive", "incremental activation mismatch"),
        ("iterations", "invalid iteration count"),
        ("work_count", "invalid incremental work count"),
        ("retry", "retry disposition mismatch"),
        ("retry_attempt_bool", "retry disposition mismatch"),
        ("retry_exit_bool", "retry disposition mismatch"),
        ("retry_retried_int", "retry disposition mismatch"),
        ("success_exit_bool", "retry disposition mismatch"),
        ("success_attempt_bool", "retry disposition mismatch"),
        ("false_gate", "recorded gate failed"),
        ("gate_failures", "recorded gate failed"),
        ("nan", "non-finite retained value"),
        ("infinity", "non-finite retained value"),
        ("nan_time", "non-finite retained value"),
        ("on_off_pair", "energies_hartree gate failed"),
        ("warm_on_off_pair", "energies_hartree gate failed"),
        ("negative_time", "invalid cold timing"),
        ("bool_time", "invalid cold timing"),
        ("negative_warm_time", "invalid warm timing"),
        ("reference_negative_time", "invalid cold timing"),
        ("cold_direction", "claimed cold regression missing"),
        ("warm_direction", "claimed warm regression missing"),
        ("median", "warm timing summary mismatch"),
        ("warm_seconds", "warm timing summary mismatch"),
        ("reference_median", "warm timing summary mismatch"),
        ("missing_sample", "warm pair inventory changed"),
        ("sequence_bool", "warm pair inventory changed"),
        ("last_summary", "last-sample summary mismatch"),
        ("empty_result", "physical result shape mismatch"),
        ("weakened_tolerance", "energy tolerance changed"),
        ("weakened_force_tolerance", "force tolerance changed"),
        ("promotion_status", "capsule must reject default promotion"),
        ("promotion_scope", "capsule must reject default promotion"),
        ("accepted_schema", "wrong historical capsule schema"),
        ("evidence_scope", "evidence scope changed"),
        ("campaign_identity", "campaign identities changed"),
    ],
)
def test_historical_capsule_gates(
    tmp_path: Path, historical_capsule: Path, optimized: bool, mutation: str, error: str
) -> None:
    shutil.copytree(historical_capsule, tmp_path, dirs_exist_ok=True)
    if mutation == "missing_manifest":
        (tmp_path / "capsule.json").unlink()
    elif mutation in {"missing_success", "missing_outcome"}:
        relative = POINT if mutation == "missing_success" else FAILED_POINT
        (tmp_path / relative).unlink()
    elif mutation in {
        "missing_manifest_member",
        "drop_all_selected_pairs",
        "duplicate_member",
        "noninteger_bytes",
        "promotion_status",
        "promotion_scope",
        "accepted_schema",
        "evidence_scope",
        "campaign_identity",
    }:
        path = tmp_path / "capsule.json"
        manifest = json.loads(path.read_text())
        if mutation == "missing_manifest_member":
            manifest["files"] = [e for e in manifest["files"] if e["path"] != POINT]
        elif mutation == "drop_all_selected_pairs":
            removed = [e for e in manifest["files"] if "/direct-48-" in e["path"]]
            for entry in removed:
                (tmp_path / entry["path"]).unlink()
            manifest["files"] = [e for e in manifest["files"] if e not in removed]
        elif mutation == "duplicate_member":
            manifest["files"].append(manifest["files"][0])
        elif mutation == "noninteger_bytes":
            manifest["files"][0]["bytes"] = float(manifest["files"][0]["bytes"])
        elif mutation == "promotion_status":
            manifest["decision"]["status"] = "accepted"
        elif mutation == "promotion_scope":
            manifest["decision"]["scope"] = "performance"
        elif mutation == "accepted_schema":
            manifest["schema"] = "generativeqc.benchmark-publication.v1"
        elif mutation == "evidence_scope":
            manifest["evidence_scope"]["failure_causes_verified"] = True
        else:
            manifest["campaigns"]["new"]["revision"] = "0" * 40
        path.write_text(json.dumps(manifest))
    elif mutation == "extra_payload":
        (tmp_path / "extra.json.gz").write_bytes(gzip.compress(b"{}", mtime=0))
    elif mutation == "checksum":
        path = tmp_path / POINT
        path.write_bytes(path.read_bytes() + b"unexpected")
    elif mutation in {"source_receipt", "library_receipt"}:
        name = "source" if mutation == "source_receipt" else "library"
        (tmp_path / f"old/{name}-sha.txt.gz").write_bytes(
            gzip.compress(("0" * 64 + "\n").encode(), mtime=0)
        )
        refresh_manifest(tmp_path)
    elif mutation in {"on_off_pair", "warm_on_off_pair"}:
        mutate_record(
            tmp_path,
            POINT,
            lambda d: shift_results(
                d, first_sample_only=mutation == "warm_on_off_pair"
            ),
        )
    elif mutation.startswith(("retry", "success_")):
        mutations = {
            "retry": {"exit_code": 0, "attempt": 1, "retried": False},
            "retry_attempt_bool": {"attempt": True},
            "retry_exit_bool": {"exit_code": True},
            "retry_retried_int": {"retried": 1},
            "success_exit_bool": {"exit_code": False},
            "success_attempt_bool": {"attempt": True},
        }
        relative = (
            POINT.replace(".json.gz", ".outcome.gz")
            if mutation.startswith("success_")
            else FAILED_POINT
        )
        mutate_record(tmp_path, relative, lambda d: d.update(mutations[mutation]))
    elif mutation != "intact":
        mutations = {
            "source": lambda d: d["environment"]["git"].update(commit="0" * 40),
            "dirty": lambda d: d["environment"]["git"].update(dirty=True),
            "native_source": lambda d: d["native_build"]["probe"].update(
                source_identity="0" * 64
            ),
            "library": lambda d: d["native_build"].update(library_sha256="0" * 64),
            "device": lambda d: d["environment"]["accelerator"].update(name="CPU"),
            "probe_device": lambda d: d["native_build"]["probe"]["device"].update(
                name="CPU"
            ),
            "uuid": lambda d: d["generativeqc"]["eigensolver_diagnostics"][0].update(
                device_uuid="another-device"
            ),
            "workload": lambda d: d["workload"].update(method="uhf"),
            "mode": lambda d: d["generativeqc"]["cold_convergence"][0][
                "incremental_direct_jk"
            ].update(requested=False),
            "inactive": lambda d: d["generativeqc"]["cold_convergence"][0][
                "incremental_direct_jk"
            ].update(active=False),
            "iterations": lambda d: d["generativeqc"]["cold_convergence"][0].update(
                iterations=0
            ),
            "work_count": lambda d: d["generativeqc"]["cold_convergence"][0][
                "incremental_direct_jk"
            ].update(delta_builds=-1),
            "false_gate": lambda d: d["gate"].update(passed="false"),
            "gate_failures": lambda d: d["gate"].update(failures=["unverified"]),
            "nan": lambda d: d["generativeqc"]["forces_hartree_per_bohr"][0][
                0
            ].__setitem__(1, float("nan")),
            "infinity": lambda d: d["generativeqc"]["warm_samples"][1][
                "energies_hartree"
            ].__setitem__(0, float("inf")),
            "nan_time": lambda d: d["gpu4pyscf"]["warm_samples"][0].update(
                seconds=float("nan")
            ),
            "negative_time": lambda d: d["generativeqc"].update(cold_seconds=-1),
            "bool_time": lambda d: d["generativeqc"].update(cold_seconds=True),
            "negative_warm_time": lambda d: d["generativeqc"]["warm_samples"][0].update(
                seconds=-1
            ),
            "reference_negative_time": lambda d: d["gpu4pyscf"].update(cold_seconds=-1),
            "cold_direction": lambda d: d["generativeqc"].update(cold_seconds=0.1),
            "warm_direction": make_warm_faster,
            "median": lambda d: d["generativeqc"].update(warm_median_seconds=0.1),
            "warm_seconds": lambda d: d["generativeqc"]["warm_seconds"].__setitem__(
                0, 0.1
            ),
            "reference_median": lambda d: d["gpu4pyscf"].update(
                warm_median_seconds=0.1
            ),
            "missing_sample": lambda d: d["generativeqc"]["warm_samples"].pop(),
            "sequence_bool": lambda d: d["generativeqc"]["warm_samples"][0].update(
                sequence_index=False
            ),
            "last_summary": lambda d: d["generativeqc"]["energies_hartree"].__setitem__(
                0, 0
            ),
            "empty_result": lambda d: d["generativeqc"]["warm_samples"][0].update(
                energies_hartree=[]
            ),
            "weakened_tolerance": lambda d: d["settings"]["gates"].update(
                maximum_energy_error_hartree=1
            ),
            "weakened_force_tolerance": lambda d: d["settings"]["gates"].update(
                maximum_force_error_hartree_per_bohr=1
            ),
        }
        mutate_record(tmp_path, POINT, mutations[mutation])
    result = verify(tmp_path, optimized=optimized)
    if mutation == "intact":
        assert result.returncode == 0, result.stderr
        summary = json.loads(result.stdout)
        assert summary["status"] == "verified-reduced-historical-negative-evidence"
        old, new = summary["campaigns"]["old"], summary["campaigns"]["new"]
        assert len(old["points"]) == len(old["pairs"]) == 0
        assert len(new["points"]) == 2
        assert len(new["pairs"]) == 1
        assert len(old["final_unsuccessful_retries"]) == 3
        assert len(new["final_unsuccessful_retries"]) == 4
        assert len(old["not_retained"]) == 13
        assert len(new["not_retained"]) == 6
        assert new["pairs"][0]["ratios"] == pytest.approx(
            {"cold": 1.4096138925884272, "warm": 2.028080233109676}
        )
    else:
        assert result.returncode != 0, f"{mutation} was accepted: {result.stdout}"
        assert error in result.stderr, result.stderr


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("engine", ["generativeqc", "gpu4pyscf"])
@pytest.mark.parametrize("sample_index", [0, 1, 2])
@pytest.mark.parametrize("field", ["energies_hartree", "forces_hartree_per_bohr"])
def test_every_warm_independent_reference_pair(
    tmp_path: Path,
    historical_capsule: Path,
    optimized: bool,
    engine: str,
    sample_index: int,
    field: str,
) -> None:
    shutil.copytree(historical_capsule, tmp_path, dirs_exist_ok=True)

    def mutate(record: dict) -> None:
        result = record[engine]
        sample = result["warm_samples"][sample_index]
        if field == "energies_hartree":
            sample[field][0] += 1
        else:
            sample[field][0][0][0] += 1
        if sample_index == 2:
            result[field] = sample[field]

    mutate_record(tmp_path, POINT, mutate)
    result = verify(tmp_path, optimized=optimized)
    assert result.returncode != 0, result.stdout
    assert f"{field} gate failed" in result.stderr, result.stderr


@pytest.mark.parametrize("damage", ["tree", "manifest", "member"])
def test_history_recovery_rejects_tampered_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    original = history.git

    def altered(*args: str) -> bytes:
        raw = original(*args)
        if (
            (damage == "tree" and args[0] == "rev-parse")
            or (
                damage == "manifest"
                and args[0] == "show"
                and args[1].endswith("/capsule.json")
            )
            or (damage == "member" and args[0] == "show" and args[1].endswith(POINT))
        ):
            return b"tampered"
        return raw

    monkeypatch.setattr(history, "git", altered)
    with pytest.raises(ValueError, match="identity|checksum"):
        restore(tmp_path)


def test_current_history_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, str(CAPSULE / "verify.py")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["historical_commit"] == history.COMMIT
    assert payload["cold_timing_scope"].startswith("prepared-plan-execute-only")

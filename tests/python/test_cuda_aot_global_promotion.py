"""Global AOT qualification is not a renamed workload-local warm gate.

The campaigns here are deliberately synthetic checker fixtures, not GPU or
scientific qualification evidence. Actual retained campaigns require raw,
source-bound acquisition and compile/resource receipts.
"""

from __future__ import annotations

import copy
import json
import lzma
import shutil
from pathlib import Path

import pytest
from generativeqc_compiler.common.evidence import canonical_hash, file_hash
from generativeqc_compiler.integral.ir import KernelConsumer
from generativeqc_compiler.integral.production_profile import resolve_production_profile

from tools.generativeqc_validation.cuda_aot_policy import (
    MANIFEST,
    PHASES,
    ROOT,
    capability_audit,
    check_defaults,
    default_policies,
    promotion_gate,
)

SOURCE = "a" * 64
BASELINE = "b" * 64
CANDIDATE = "c" * 64


def campaign() -> dict:
    """Build a small complete positive fixture without running a device."""
    oracle = {
        "engine": "independent-test-oracle",
        "version": "synthetic-fixture-only",
        "input_sha256": "d" * 64,
        "scientific_contract_sha256": "e" * 64,
        "geometries": {
            name: {
                "geometry_sha256": digest * 64,
                "converged": True,
                "energies": [-1.0],
                "forces": [[0.0, 0.0, 0.0]],
            }
            for name, digest in (("initial", "1"), ("moved", "2"))
        },
    }
    pairs = []
    for repeat in range(4):
        pair = {}
        for arm, policy in (("baseline", BASELINE), ("candidate", CANDIDATE)):
            pair[arm] = {
                "process_id": f"{arm}-{repeat}",
                "source_identity": SOURCE,
                "policy_sha256": policy,
                "scientific_contract_sha256": "e" * 64,
                "production_oracle_calls": 0,
                "nondefault_controls": {},
                "scheduler": {
                    "job_id": "test",
                    "node": "test-only",
                    "cuda_visible_devices": "0",
                    "time_limit": "00:10:00",
                },
                "device": {
                    "uuid": "synthetic",
                    "name": "synthetic-test-device",
                    "architecture": "sm_120",
                    "driver": "test-driver",
                },
                "build": {
                    "release": True,
                    "fast_compile": False,
                    "compiler": "test-nvcc",
                    "cuda_images": ["120"],
                    "library_sha256": ("3" if arm == "baseline" else "4") * 64,
                    "resource_status": "passed",
                    "resource_receipt_sha256": "5" * 64,
                    "resource_usage": {
                        "registers_per_thread": 32,
                        "stack_bytes": 0,
                        "local_bytes": 0,
                        "shared_bytes": 0,
                    },
                },
                "acquisition_receipt_sha256": "6" * 64,
                "module_first_use_seconds": 1.0,
                "package_bytes": 100,
                "peak_device_bytes": 1024,
                "phases": {},
            }
            for phase in PHASES:
                reference = oracle["geometries"][
                    "moved" if phase.startswith("moved") else "initial"
                ]
                pair[arm]["phases"][phase] = {
                    **copy.deepcopy(reference),
                    "converged": True,
                    "backend": "cuda",
                    "scope": "complete-energy-and-analytic-forces",
                    "seconds": 10.0 if arm == "baseline" else 8.0,
                    "iterations": [5],
                    "semantic_work": {
                        "scf_iterations": 5,
                        "fock_builds": 6,
                        "force_evaluations": 1,
                        "quartet_tasks": 32,
                    },
                }
        pairs.append(pair)
    return {
        "schema": "generativeqc.cuda-aot-global-promotion",
        "schema_version": 1,
        "source_identity": SOURCE,
        "architecture": "sm_120",
        "baseline_policy_sha256": BASELINE,
        "candidate_policy_sha256": CANDIDATE,
        "scientific_contract_sha256": "e" * 64,
        "oracle": oracle,
        "pairs": pairs,
    }


def gate(payload: dict) -> dict:
    return promotion_gate(
        payload,
        source_identity=SOURCE,
        architecture="sm_120",
        baseline_policy=BASELINE,
        candidate_policy=CANDIDATE,
    )


def test_complete_fresh_global_campaign_passes_without_a_local_profile() -> None:
    result = gate(campaign())
    assert result["passed"] and result["scope"] == "global-default"
    assert set(result["phases"]) == {*PHASES, "module_first_use"}


@pytest.mark.parametrize("phase", PHASES)
def test_fast_warm_does_not_hide_any_endpoint_regression(phase: str) -> None:
    payload = campaign()
    for pair in payload["pairs"]:
        pair["candidate"]["phases"][phase]["seconds"] = 11.0
    with pytest.raises(ValueError, match="nonregression"):
        gate(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "stale",
        "warm-only",
        "same-process",
        "wrong-card",
        "wrong-driver",
        "science",
        "reference-in-production",
        "oracle",
        "full-forces",
        "energy",
        "forces",
        "nan",
        "iterations",
        "work",
        "missing-work",
        "memory",
        "package",
        "first-use",
        "resources",
        "debug",
        "fast-compile",
        "unconverged",
        "cpu",
        "no-move",
        "no-receipt",
        "compiler",
        "controls",
        "partial-timing",
        "no-improvement",
    ),
)
def test_incomplete_or_regressive_global_campaign_is_rejected(mutation: str) -> None:
    payload = campaign()
    run = payload["pairs"][0]["candidate"]
    sample = run["phases"]["warm"]
    if mutation == "stale":
        payload["source_identity"] = "0" * 64
    elif mutation == "warm-only":
        for pair in payload["pairs"]:
            for arm in pair.values():
                arm["phases"] = {"warm": arm["phases"]["warm"]}
    elif mutation == "same-process":
        run["process_id"] = payload["pairs"][0]["baseline"]["process_id"]
    elif mutation in ("wrong-card", "wrong-driver"):
        run["device"]["name" if mutation == "wrong-card" else "driver"] = (
            "another-device"
        )
    elif mutation == "science":
        run["scientific_contract_sha256"] = "0" * 64
    elif mutation == "reference-in-production":
        run["production_oracle_calls"] = 1
    elif mutation == "oracle":
        payload["oracle"]["engine"] = "generativeqc"
    elif mutation == "full-forces":
        sample["forces"] = [[0.0]]
    elif mutation == "energy":
        sample["energies"][0] += 1e-6
    elif mutation == "forces":
        sample["forces"][0][0] = 1e-5
    elif mutation == "nan":
        sample["seconds"] = float("nan")
    elif mutation == "iterations":
        sample["iterations"] = [6]
    elif mutation == "work":
        sample["semantic_work"]["quartet_tasks"] += 1
    elif mutation == "missing-work":
        del sample["semantic_work"]["quartet_tasks"]
    elif mutation == "memory":
        run["peak_device_bytes"] += 1
    elif mutation == "package":
        run["package_bytes"] += 1
    elif mutation == "first-use":
        for pair in payload["pairs"]:
            pair["candidate"]["module_first_use_seconds"] = 1.1
    elif mutation == "resources":
        run["build"]["resource_usage"]["stack_bytes"] = 4
    elif mutation in ("debug", "fast-compile"):
        run["build"]["release" if mutation == "debug" else "fast_compile"] = (
            mutation != "debug"
        )
    elif mutation == "unconverged":
        sample["converged"] = False
    elif mutation == "cpu":
        sample["backend"] = "cpu"
    elif mutation == "no-move":
        payload["oracle"]["geometries"]["moved"] = copy.deepcopy(
            payload["oracle"]["geometries"]["initial"]
        )
    elif mutation == "no-receipt":
        del run["build"]["resource_receipt_sha256"]
    elif mutation == "compiler":
        run["build"]["compiler"] = "other-nvcc"
    elif mutation == "controls":
        run["nondefault_controls"] = {"GENERATIVEQC_AOT_FOCK_SHELL_CLASSES": "none"}
    elif mutation == "partial-timing":
        sample["scope"] = "kernel-only"
    else:
        for pair in payload["pairs"]:
            for phase in PHASES:
                pair["candidate"]["phases"][phase]["seconds"] = 10.0
    with pytest.raises((ValueError, KeyError)):
        gate(payload)


def fixture_root(tmp_path: Path) -> dict:
    inventory = json.loads(
        (ROOT / "manifests/maintenance/cuda_aot_default_decisions.json").read_text()
    )
    for relative in (
        MANIFEST,
        Path(inventory["retained_baseline"]["manifest"]),
        Path(inventory["retained_baseline"]["effective_policies"]),
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return inventory


def test_stock_defaults_are_retained_not_requalified(tmp_path: Path) -> None:
    result = check_defaults(tmp_path, fixture_root(tmp_path), source_identity=SOURCE)
    assert result == {
        "changed_targets": [],
        "retained_defaults": True,
        "promotions": {},
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "packaging",
        "preference",
        "derivatives",
        "class",
        "recurrence",
        "schedule",
        "new-target",
    ),
)
def test_a_global_manifest_change_cannot_be_laundered_by_refreshing_an_audit(
    tmp_path: Path, mutation: str
) -> None:
    inventory = fixture_root(tmp_path)
    path = tmp_path / MANIFEST
    manifest = json.loads(path.read_text())
    profile = manifest["architectures"]["sm_120"]
    portable = manifest["architectures"]["portable_cuda"]
    if mutation == "packaging":
        portable["optional_fock_variants"] = True
    elif mutation == "preference":
        profile["prefer_rys_task_fock"] = False
    elif mutation == "derivatives":
        portable["compile_derivative_shells"] = True
    elif mutation == "class":
        profile["preferred_rys_task_fock_shell_classes"] = []
    elif mutation == "recurrence":
        row = next(
            item for item in portable["kernels"] if item["shell_class"] == "psss"
        )
        row["recurrence"] = "subset_wick"
    elif mutation == "schedule":
        portable["kernels"][0]["schedule"]["unroll_pair_terms"] = not portable[
            "kernels"
        ][0]["schedule"]["unroll_pair_terms"]
    else:
        manifest["architectures"]["test-unmeasured"] = copy.deepcopy(profile)
        manifest["architectures"]["test-unmeasured"]["compatible_architectures"] = [
            "sm_90"
        ]
        manifest["architectures"]["test-unmeasured"]["prefer_rys_task_fock"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fresh global-default"):
        check_defaults(tmp_path, inventory, source_identity=SOURCE)


def test_manifest_formatting_is_not_a_global_default_change(tmp_path: Path) -> None:
    inventory = fixture_root(tmp_path)
    path = tmp_path / MANIFEST
    path.write_text(json.dumps(json.loads(path.read_text()), sort_keys=True, indent=4))
    assert check_defaults(tmp_path, inventory, source_identity=SOURCE)[
        "retained_defaults"
    ]


def test_retained_baseline_identity_is_a_real_integrity_check(tmp_path: Path) -> None:
    inventory = fixture_root(tmp_path)
    baseline = tmp_path / inventory["retained_baseline"]["manifest"]
    baseline.write_text("{}")
    with pytest.raises(ValueError, match="baseline integrity"):
        check_defaults(tmp_path, inventory, source_identity=SOURCE)


def test_repacking_a_different_baseline_cannot_change_retained_identity(
    tmp_path: Path,
) -> None:
    inventory = fixture_root(tmp_path)
    baseline = tmp_path / inventory["retained_baseline"]["manifest"]
    baseline.write_bytes(lzma.compress(b"{}"))
    inventory["retained_baseline"]["sha256"] = file_hash(baseline)
    with pytest.raises(ValueError, match="baseline uncompressed integrity"):
        check_defaults(tmp_path, inventory, source_identity=SOURCE)


def test_audit_covers_all_extra_and_precision_consumers_without_promoting() -> None:
    report = capability_audit(ROOT / MANIFEST)
    tuned = resolve_production_profile(ROOT / MANIFEST, "sm_120")
    portable = resolve_production_profile(ROOT / MANIFEST, "sm_120", "portable")
    expected = {
        f"extra-{consumer.value}:{item.spec.name}"
        for item in tuned.selections
        for consumer in item.consumers
        if not any(
            item.spec.name == other.spec.name and consumer in other.consumers
            for other in portable.selections
        )
    }
    features = {row["feature"] for row in report["rows"]}
    assert expected <= features
    assert {
        f"mixed-fock:{item.spec.name}"
        for item in tuned.selections
        if item.has_capability("mixed_fock")
    } <= features
    assert {
        f"exact-derivative:{item.spec.name}"
        for item in tuned.selections
        if KernelConsumer.FORCE in item.consumers
    } <= features
    assert set(report["routes"]) == {"sm_80", "sm_90", "sm_120"}
    for row in report["rows"]:
        for target in row["targets"].values():
            assert target["independent_device_numerics"] == "not-run-for-this-audit"
            assert set(target["full_endpoint"].values()) == {"not-run-for-this-audit"}
    assert report["unmatched_registry"]["expected_status"] == "no_matching_kernel_set"


def test_policy_identity_is_semantic_and_keeps_precision_contracts() -> None:
    policies = default_policies(ROOT / MANIFEST)
    assert policies["sm_80"]["compiled_masks"]["mixed_fock"] == 0
    assert policies["sm_120"]["compiled_masks"]["mixed_fock"] > 0
    assert canonical_hash(policies["sm_80"]) != canonical_hash(policies["sm_90"])


def retained_campaign(
    tmp_path: Path,
    *,
    corrupt_resources: bool = False,
    hidden_resource_regression: bool = False,
) -> dict:
    """Exercise the entire receipt chain using explicitly synthetic records."""
    inventory = fixture_root(tmp_path)
    path = tmp_path / MANIFEST
    manifest = json.loads(path.read_text())
    manifest["architectures"]["sm_120"]["prefer_rys_task_fock"] = False
    path.write_text(json.dumps(manifest))
    baseline = json.loads(
        lzma.decompress(
            (
                tmp_path / inventory["retained_baseline"]["effective_policies"]
            ).read_bytes()
        )
    )["sm_120"]
    candidate = default_policies(path)["sm_120"]
    policies = {"baseline": baseline, "candidate": candidate}
    payload = campaign()
    payload["baseline_policy_sha256"] = canonical_hash(baseline)
    payload["candidate_policy_sha256"] = canonical_hash(candidate)
    artifacts = []

    def retain(name: str, data: str) -> str:
        relative = f"evidence/{name}"
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data)
        digest = file_hash(target)
        artifacts.append({"path": relative, "sha256": digest})
        return digest

    log = retain(
        "resources.txt",
        "no resources"
        if corrupt_resources
        else "Function synthetic_kernel:\n REG:32 STACK:0 SHARED:0 LOCAL:0\n",
    )
    payload["oracle"]["input_sha256"] = retain(
        "reference-input.json", json.dumps({"fixture_only": True})
    )
    payload["oracle"]["reference_receipt_sha256"] = retain(
        "reference.json", json.dumps(payload["oracle"])
    )
    for index, pair in enumerate(payload["pairs"]):
        for arm, run in pair.items():
            run["policy_sha256"] = canonical_hash(policies[arm])
            run["compiled_masks"] = policies[arm]["compiled_masks"]
            build = run["build"]
            selected_log = log
            if hidden_resource_regression:
                registers = 40 if arm == "candidate" else 32
                selected_log = retain(
                    f"resources-{index}-{arm}.txt",
                    f"Function synthetic_kernel:\n REG:{registers} STACK:0 SHARED:0 LOCAL:0\nFunction unrelated_peak:\n REG:64 STACK:0 SHARED:0 LOCAL:0\n",
                )
                build["resource_usage"]["registers_per_thread"] = 64
            resource = {
                "schema": "generativeqc.cuda-aot-resource-receipt/v1",
                "source_identity": SOURCE,
                "library_sha256": build["library_sha256"],
                "architecture": "sm_120",
                "compiler": build["compiler"],
                "cuda_images": build["cuda_images"],
                "exit_code": 0,
                "commands": [["synthetic-test-command"]],
                "cuobjdump_sha256": selected_log,
            }
            build["resource_receipt_sha256"] = retain(
                f"resource-{index}-{arm}.json", json.dumps(resource)
            )
            record = {
                key: value
                for key, value in run.items()
                if key != "acquisition_receipt_sha256"
            }
            run["acquisition_receipt_sha256"] = retain(
                f"acquisition-{index}-{arm}.json",
                json.dumps(
                    {
                        "schema": "generativeqc.cuda-aot-acquisition-receipt/v1",
                        "run_sha256": canonical_hash(record),
                    }
                ),
            )
    campaign_hash = retain("campaign.json", json.dumps(payload))
    inventory["promotions"].append(
        {
            "architecture": "sm_120",
            "candidate_policy_sha256": canonical_hash(candidate),
            "campaign": "evidence/campaign.json",
            "sha256": campaign_hash,
            "artifacts": artifacts,
        }
    )
    return inventory


def test_full_global_decision_recomputes_raw_resources_and_acquisition(
    tmp_path: Path,
) -> None:
    result = check_defaults(
        tmp_path, retained_campaign(tmp_path), source_identity=SOURCE
    )
    assert result["changed_targets"] == ["sm_120"]
    assert result["promotions"]["sm_120"]["passed"]


def test_a_passed_string_cannot_substitute_for_actual_kernel_resources(
    tmp_path: Path,
) -> None:
    inventory = retained_campaign(tmp_path, corrupt_resources=True)
    with pytest.raises(ValueError, match="no kernel resources"):
        check_defaults(tmp_path, inventory, source_identity=SOURCE)


def test_aggregate_maxima_cannot_hide_a_regressing_kernel(tmp_path: Path) -> None:
    inventory = retained_campaign(tmp_path, hidden_resource_regression=True)
    with pytest.raises(ValueError, match="per-kernel compiled resource"):
        check_defaults(tmp_path, inventory, source_identity=SOURCE)


def test_frozen_density_warm_replay_retains_real_zero_scf_work() -> None:
    payload = campaign()
    for pair in payload["pairs"]:
        for run in pair.values():
            for phase in ("warm", "moved_warm"):
                sample = run["phases"][phase]
                sample["reused_converged_state"] = True
                sample["iterations"] = [0]
                sample["semantic_work"]["scf_iterations"] = 0
                sample["semantic_work"]["fock_builds"] = 0
    assert gate(payload)["passed"]


def test_fresh_cold_process_cannot_claim_warm_state_reuse() -> None:
    payload = campaign()
    payload["pairs"][0]["candidate"]["phases"]["cold"]["reused_converged_state"] = True
    with pytest.raises(ValueError, match="warm-phase only"):
        gate(payload)


def test_raw_acquisition_tampering_is_rejected(tmp_path: Path) -> None:
    inventory = retained_campaign(tmp_path)
    (tmp_path / "evidence/acquisition-0-candidate.json").write_text("{}")
    with pytest.raises(ValueError, match="raw receipt integrity"):
        check_defaults(tmp_path, inventory, source_identity=SOURCE)


def test_evidence_cannot_escape_repository_root(tmp_path: Path) -> None:
    inventory = fixture_root(tmp_path)
    inventory["retained_baseline"]["manifest"] = "../outside.json"
    with pytest.raises(ValueError, match="escapes root"):
        check_defaults(tmp_path, inventory, source_identity=SOURCE)


@pytest.mark.parametrize("time_limit", ("UNLIMITED", "0", "00:00:00", "01:99:00", True))
def test_global_campaign_requires_a_real_finite_scheduler_limit(
    time_limit: object,
) -> None:
    payload = campaign()
    payload["pairs"][0]["candidate"]["scheduler"]["time_limit"] = time_limit
    with pytest.raises(ValueError, match="finite scheduler"):
        gate(payload)


def test_unconverged_independent_reference_cannot_qualify_a_default() -> None:
    payload = campaign()
    payload["oracle"]["geometries"]["moved"]["converged"] = False
    with pytest.raises(ValueError, match="independent reference is unconverged"):
        gate(payload)


def test_implicit_compiler_default_change_cannot_rewrite_its_own_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.generativeqc_validation import cuda_aot_policy

    inventory = fixture_root(tmp_path)
    actual = default_policies(tmp_path / MANIFEST)
    actual["sm_80"]["default_preferences"]["streaming_fock"] ^= 1
    monkeypatch.setattr(cuda_aot_policy, "default_policies", lambda *_: actual)
    with pytest.raises(ValueError, match="fresh global-default"):
        check_defaults(tmp_path, inventory, source_identity=SOURCE)


@pytest.mark.parametrize("capability", ("8.1", "sm_81", "13.x", "sm_13x"))
def test_new_compatibility_domains_cannot_escape_the_default_gate(
    tmp_path: Path, capability: str
) -> None:
    inventory = fixture_root(tmp_path)
    path = tmp_path / MANIFEST
    payload = json.loads(path.read_text())
    payload["architectures"]["sm_120"]["compatible_compute_capabilities"] = [capability]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        check_defaults(tmp_path, inventory, source_identity=SOURCE)


def test_named_experiment_does_not_promote_an_auto_default(tmp_path: Path) -> None:
    inventory = fixture_root(tmp_path)
    path = tmp_path / MANIFEST
    payload = json.loads(path.read_text())
    experiment = copy.deepcopy(payload["architectures"]["sm_120"])
    experiment["compatible_architectures"] = ["sm_120"]
    experiment["prefer_rys_task_fock"] = False
    payload["architectures"]["sm_120_experimental"] = experiment
    path.write_text(json.dumps(payload))
    assert check_defaults(tmp_path, inventory, source_identity=SOURCE)[
        "retained_defaults"
    ]

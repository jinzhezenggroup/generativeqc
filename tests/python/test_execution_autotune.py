"""Native policy searches are isolated recipes, not global Calculator defaults."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import autotune, execution_tune, profiles, tune_cli


def endpoint_sample(workload: dict, policy: dict) -> dict:
    """Synthetic nonexecutable evidence used only in a temporary test directory."""
    schedule = {"component": "jk", "environment": policy}
    return {
        "seconds": 0.75 if policy else 1.0,
        "iterations": [1],
        "energies": [-1.0],
        "forces": [[[0.0, 0.0, 0.0]] * 2],
        "converged": True,
        "backend": ["cuda"],
        "endpoint_kind": "energy_force",
        "synchronized": True,
        "scientific_identity": profiles.canonical_hash(workload),
        "source_hash": workload["source_identity"],
        "schedule": schedule,
        "schedule_identity": profiles.canonical_hash(schedule),
        "native_xc_schedule": {"point_tile": 256},
        "dft_work": [
            {
                "grid_points": 100,
                "fock_builds": 1,
                "coulomb_exchange_work": {
                    "complete": True,
                    "coulomb_applications": 1,
                    "exchange_applications": 1,
                },
            }
        ],
        "resources": {
            "peak_bytes": 4096,
            "rejected_allocations": 0,
            "admission": {"limit_bytes": 8192, "combined_peak_bytes": 6144},
        },
    }


@pytest.fixture
def search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nBohr\nH 0 0 -0.7\nH 0 0 0.7\n")
    binary = tmp_path / "nonexecutable.so"
    binary.write_bytes(b"synthetic test artifact, never a production library")
    state = SimpleNamespace(
        args=SimpleNamespace(
            component="jk",
            method="pbe0-rks",
            input=xyz,
            units="bohr",
            basis="sto-3g",
            grid_level=0,
            charge=0,
            multiplicity=1,
            representation="spherical",
            device_id=2,
            batch=1,
            full=False,
            max_candidates=1,
            repeats=5,
            budget_seconds=60,
        ),
        probe={"source_identity": "a" * 64, "device": {"major": 12, "minor": 0}},
        library=SimpleNamespace(_name=str(binary)),
        calls=[],
        failure=None,
    )
    monkeypatch.setattr(profiles, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(
        profiles,
        "select_library",
        lambda *args: (state.library, {"source": "portable"}),
    )

    def worker(
        library: Path,
        workload: dict,
        component: str,
        policy: dict,
        directory: Path,
        **kwargs: Any,
    ) -> dict:
        execution_tune.validate_policy(component, policy)
        state.calls.append((copy.deepcopy(policy), str(directory)))
        sample = endpoint_sample(workload, policy)
        if state.failure == "slow":
            sample["seconds"] = 1.25 if policy else 1.0
        if policy and state.failure == "memory":
            sample["resources"]["rejected_allocations"] = 1
        if policy and state.failure == "work":
            sample["dft_work"][0]["coulomb_exchange_work"]["exchange_applications"] = 0
        if policy and state.failure == "force":
            sample["forces"][0][0][0] = 0.01
        if policy and state.failure == "holdout" and "holdout" in str(directory):
            sample["seconds"] = 1.1
        return sample

    monkeypatch.setattr(execution_tune, "worker", worker)
    monkeypatch.setattr(
        autotune,
        "_worker",
        lambda *args, **kwargs: {
            "independent_reference": "synthetic unit-test reference, not hardware evidence",
            "energies": [-1.0],
            "forces": [[[0.0, 0.0, 0.0]] * 2],
        },
    )
    state.run = lambda: execution_tune.run(
        state.args,
        base_library=state.library,
        probe=state.probe,
        tools={},
        target=SimpleNamespace(architecture="sm_120"),
    )
    monkeypatch.setattr(execution_tune.ctypes, "CDLL", lambda path: state.library)
    monkeypatch.setattr(profiles, "probe_device", lambda *args: state.probe)
    return state


def test_search_and_replay_revalidate_all_raw_gates_without_global_activation(
    search: Any,
) -> None:
    report = search.run()
    assert report["qualified_recipe"] and report["installed"] is None
    assert execution_tune.passed(report["holdout"])
    assert len(report["holdout"]["endpoint"]["candidate"]) >= 5
    output = Path(report["directory"]) / "replayed.json"
    result = execution_tune.replay(Path(report["qualified_recipe"]), output)
    assert result["converged"] and output.is_file()
    assert not (Path(report["directory"]).parent / "active.json").exists()


@pytest.mark.parametrize("failure", ("slow", "memory", "work", "force", "holdout"))
def test_failed_gate_never_publishes_recipe(search: Any, failure: str) -> None:
    search.failure = failure
    report = search.run()
    assert report["qualified_recipe"] is None and report["installed"] is None
    assert not (Path(report["directory"]) / "recipe.json").exists()


@pytest.mark.parametrize(
    "corruption", ("binary", "identity", "policy", "workload", "receipt", "raw")
)
def test_replay_rejects_changed_inputs_before_execution(
    search: Any, corruption: str
) -> None:
    report = search.run()
    path = Path(report["qualified_recipe"])
    recipe = json.loads(path.read_text())
    if corruption == "binary":
        Path(recipe["library"]).write_bytes(b"changed nonexecutable bytes")
    elif corruption == "identity":
        search.probe["device"]["major"] = 9
    elif corruption == "policy":
        recipe["policy"] = {"GENERATIVEQC_DISABLE_MD_J": "1"}
    elif corruption == "workload":
        recipe["workload"]["basis"] = "def2-svp"
    elif corruption == "receipt":
        report["holdout"]["resources"]["passed"] = False
    else:
        report["holdout"]["endpoint"]["candidate"][0]["forces"][0][0][0] = 0.01
        recipe["evidence_hash"] = profiles.canonical_hash(report["holdout"])
    profiles.atomic_json(path, recipe)
    profiles.atomic_json(path.with_name("report.json"), report)
    count = len(search.calls)
    with pytest.raises(ValueError):
        execution_tune.replay(path, path.with_name("must-not-exist.json"))
    assert len(search.calls) == count


def test_subprocess_environment_preserves_slurm_visibility_without_mutating_parent(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-slurm-device")
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setenv("GENERATIVEQC_EXPERIMENTAL_PBE_POINT_FP32", "1")
    monkeypatch.setenv("GENERATIVEQC_DIRECT_K_TASK_SCHEDULE", "incumbent")
    before = dict(os.environ)
    policy = {"GENERATIVEQC_DIRECT_K_TASK_SCHEDULE": "fill"}
    environment = execution_tune.worker_environment(
        Path("/nonexecutable.so"), "jk", policy
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-slurm-device"
    assert environment["SLURM_JOB_ID"] == "fixture"
    assert environment["GENERATIVEQC_DIRECT_K_TASK_SCHEDULE"] == "fill"
    assert "GENERATIVEQC_EXPERIMENTAL_PBE_POINT_FP32" not in environment
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "policy",
    (
        {"CUDA_VISIBLE_DEVICES": "0"},
        {"GENERATIVEQC_KS_MIXED_MAX_ITERATIONS": "1"},
        {"GENERATIVEQC_DIRECT_K_TASK_SCHEDULE": "invalid"},
    ),
)
def test_policy_allowlist_excludes_scientific_or_visibility_changes(
    policy: dict,
) -> None:
    with pytest.raises(ValueError, match="policy"):
        execution_tune.worker_environment(Path("/nonexecutable.so"), "jk", policy)


def test_semilocal_search_does_not_propose_absent_exchange() -> None:
    policies = execution_tune.candidate_policies("jk", hybrid=False, full=True)
    assert all(not any("_K_" in key for key in policy) for policy in policies)


@pytest.mark.parametrize(
    "extra", (["--method", "rhf"], ["--method", "pbe-rks", "--xc-tiles", "512"])
)
def test_jk_cli_rejects_ambiguous_or_nonks_searches(extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        tune_cli.main(["unused.xyz", "--component", "jk", *extra])


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)
@pytest.mark.parametrize("method", ("pbe-rks", "pbe0-rks", "pbe0-uks"))
def test_native_jk_policy_candidates_match_independent_forces(
    tmp_path: Path, method: str
) -> None:
    """All proposed controls retain full-method J/K weights and spin semantics."""
    from generativeqc import _native
    from generativeqc._dft_autotune import (
        numerical_gate,
        reference_workload,
        resource_gate,
    )

    assert os.environ.get("SLURM_JOB_ID")
    library = _native.load_library()
    probe = profiles.probe_device(library, 0)
    workload = {
        "kind": "dft",
        "method": method,
        "basis": "def2-svp",
        "grid_level": 3,
        "atoms": [
            ["O", [0.0, 0.0, 0.0]],
            ["H", [0.0, -1.432336, 1.107153]],
            ["H", [0.0, 1.432336, 1.107153]],
        ],
        "charge": 1 if method.endswith("uks") else 0,
        "multiplicity": 2 if method.endswith("uks") else 1,
        "representation": "spherical",
        "device_id": 0,
        "batch": 1,
        "architecture": f"sm_{probe['device']['major']}{probe['device']['minor']}",
        "source_identity": probe["source_identity"],
    }
    reference = reference_workload(workload)
    for index, policy in enumerate(
        [
            {},
            *execution_tune.candidate_policies(
                "jk", hybrid=method != "pbe-rks", full=True
            ),
        ]
    ):
        sample = execution_tune.worker(
            Path(library._name),
            workload,
            "jk",
            policy,
            tmp_path / f"policy-{index}",
            timeout=180,
        )
        assert numerical_gate([sample], reference)["passed"]
        assert resource_gate(
            [sample],
            schedule_hash=sample["schedule_identity"],
            source_hash=probe["source_identity"],
        )["passed"]

"""Fitted HF tuning keeps the auxiliary metric and all numerical inputs fixed."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import _autotune_worker, _df_autotune, execution_tune, tune_cli


@pytest.fixture
def workload() -> dict:
    return {
        "kind": "df-hf",
        "method": "rhf",
        "atoms": [
            ["O", [0.0, 0.0, 0.0]],
            ["H", [0.0, -1.432336, 1.107153]],
            ["H", [0.0, 1.432336, 1.107153]],
        ],
        "basis": "sto-3g",
        "auxiliary_basis": "def2-svp",
        "representation": "spherical",
        "charge": 0,
        "multiplicity": 1,
        "batch": 1,
        "device_id": 0,
        "metric_relative_threshold": 1e-10,
        "df_memory_bytes": 64 << 20,
        "source_identity": "a" * 64,
        "architecture": "sm_120",
    }


def test_df_policy_allowlist_never_changes_precision_auxiliary_model_or_cutoff() -> (
    None
):
    for policy in _df_autotune.candidate_policies(full=True):
        execution_tune.validate_policy("df", policy)
    for policy in (
        {"GENERATIVEQC_DF_VALUE_STORAGE": "packed-single"},
        {"GENERATIVEQC_DF_METRIC_THRESHOLD": "0.1"},
        {"GENERATIVEQC_KS_RESIDENT_VALUES": "1"},
    ):
        with pytest.raises(ValueError):
            execution_tune.validate_policy("df", policy)


def test_df_numerical_gate_is_stricter_than_grid_discretization_floors() -> None:
    reference = {
        "independent_reference": "synthetic only",
        "energies": [-1.0],
        "forces": [[[0.0, 0.0, 0.0]] * 3],
    }
    sample = {"energies": [-1.0], "forces": [[[0.0, 0.0, 0.0]] * 3]}
    assert _df_autotune.numerical_gate([sample], reference)["passed"]
    for changed in (
        {**sample, "energies": [-1.0 + 2e-9]},
        {**sample, "forces": [[[2e-8, 0.0, 0.0]] * 3]},
    ):
        assert not _df_autotune.numerical_gate([changed], reference)["passed"]


def test_df_work_gate_rejects_changed_or_missing_metric_subspaces_and_work() -> None:
    reference = {"energies": [-1.0], "metric": {"naux": 24}}
    baseline = {
        "df_work": {
            "fock_builds": [1],
            "metrics": [
                {"system_index": 0, "effective_rank": 24, "absolute_threshold": 1e-8},
            ],
        }
    }
    assert _df_autotune.work_gate([baseline, baseline], reference)["passed"]
    for corruption in ("rank", "count", "missing"):
        candidate = copy.deepcopy(baseline)
        if corruption == "rank":
            candidate["df_work"]["metrics"][0]["effective_rank"] = 23
        elif corruption == "count":
            candidate["df_work"]["fock_builds"] = [2]
        else:
            candidate["df_work"]["metrics"] = []
        assert not _df_autotune.work_gate([baseline, candidate], reference)["passed"]


def test_df_worker_keeps_native_budget_and_never_requests_direct_hf_profiling(
    workload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SimpleNamespace(
        items=[
            SimpleNamespace(
                iterations=1,
                fock_builds=1,
                energy=-1.0,
                forces=np.zeros((3, 3)),
                converged=True,
                executed_backend="cuda",
            )
        ]
    )
    ledger = {
        "peak_bytes": 4096,
        "rejected_allocations": 0,
        "admission": {"limit_bytes": 8192, "combined_peak_bytes": 6144},
    }
    metric = {"system_index": 0, "effective_rank": 24, "absolute_threshold": 1e-8}

    class Prepared:
        def __init__(self) -> None:
            self.resource_diagnostics = {
                "phase": "execution",
                "execution": {"device_ledger": ledger},
            }

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, **kwargs: Any) -> Any:
            return result

        def set_warm_start_updates(self, enabled: bool) -> None:
            assert not enabled

        def last_density_fitting_metric_diagnostics(self) -> tuple:
            return (SimpleNamespace(to_dict=lambda: metric),)

    calls = []

    def calculator(**kwargs: Any) -> Any:
        calls.append(kwargs)

        def prepare(*args: Any, **kwargs: Any) -> Any:
            assert kwargs["shell_class_profiling"] is False
            return Prepared()

        return SimpleNamespace(profile_diagnostics={}, prepare_batch=prepare)

    monkeypatch.setattr(_autotune_worker, "Calculator", calculator)
    sample = _autotune_worker.execute_workload(workload, profile=True)
    assert calls[0]["density_fitting"] == "cuda"
    assert calls[0]["auxiliary_basis"] == workload["auxiliary_basis"]
    assert calls[0]["density_fitting_relative_threshold"] == 1e-10
    assert calls[0]["density_fitting_memory_budget_bytes"] == 64 << 20
    assert sample["resources"] == ledger
    assert sample["df_work"]["metrics"] == [metric]


@pytest.mark.parametrize("method", ("rhf", "uhf"))
def test_independent_df_reference_includes_full_metric_and_auxiliary_response(
    workload: dict, method: str
) -> None:
    pytest.importorskip("pyscf")
    workload.update(
        method=method,
        charge=int(method == "uhf"),
        multiplicity=2 if method == "uhf" else 1,
    )
    reference = _df_autotune.reference_workload(workload)
    assert (
        reference["metric"]["minimum_eigenvalue"]
        > reference["metric"]["full_rank_floor"]
    )
    assert reference["metric"]["naux"] == 24
    assert np.isfinite(reference["energies"]).all()
    assert np.max(np.abs(np.asarray(reference["forces"]).sum(axis=1))) < 1e-9


def test_reference_never_substitutes_pyscf_named_basis_constants(
    workload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyscf")
    from pyscf import gto

    original = gto.basis.load

    def load(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.lower().replace("_", "-") in ("sto-3g", "def2-svp"):
            pytest.fail("reference must use immutable input tables, not a named alias")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(gto.basis, "load", load)
    assert np.isfinite(_df_autotune.reference_workload(workload)["energies"]).all()


@pytest.mark.parametrize(
    "extra",
    (
        ["--method", "pbe-rks"],
        ["--method", "uhf"],
        ["--df-memory-mib", "0"],
        ["--repeats", "4"],
    ),
)
def test_df_cli_refuses_unqualified_or_unbounded_inputs(extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        tune_cli.main(["unused.xyz", "--component", "df", *extra])


def test_df_search_binds_auxiliary_model_and_memory_without_global_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc import autotune, profiles

    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nBohr\nH 0 0 -0.7\nH 0 0 0.7\n")
    binary = tmp_path / "nonexecutable.so"
    binary.write_bytes(b"synthetic fixture, not a production library")
    library = SimpleNamespace(_name=str(binary))
    args = SimpleNamespace(
        component="df",
        method="rhf",
        repeats=5,
        input=xyz,
        units="bohr",
        basis="sto-3g",
        auxiliary_basis="sto-3g",
        df_memory_mib=20,
        charge=0,
        multiplicity=1,
        representation="spherical",
        device_id=0,
        batch=1,
        grid_level=3,
        full=False,
        max_candidates=1,
        budget_seconds=60,
    )
    probe = {"source_identity": "a" * 64, "device": {"major": 12, "minor": 0}}
    reference = {
        "independent_reference": "synthetic unit test only",
        "reference_family": "df-hf",
        "energies": [-1.0],
        "forces": [[[0.0, 0.0, 0.0]] * 2],
        "metric": {"naux": 2},
    }
    monkeypatch.setattr(profiles, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(
        profiles, "select_library", lambda *args: (library, {"source": "portable"})
    )
    monkeypatch.setattr(autotune, "_worker", lambda *args, **kwargs: reference)
    observed = []

    def worker(
        library: Path,
        workload: dict,
        component: str,
        policy: dict,
        directory: Path,
        **kwargs: Any,
    ) -> dict:
        observed.append(copy.deepcopy(workload))
        schedule = {"component": component, "environment": policy}
        return {
            "seconds": 0.7 if policy else 1.0,
            "iterations": [1],
            "energies": [-1.0],
            "forces": [[[0.0, 0.0, 0.0]] * 2],
            "converged": True,
            "backend": ["cuda"],
            "synchronized": True,
            "endpoint_kind": "energy_force",
            "scientific_identity": profiles.canonical_hash(workload),
            "schedule_identity": profiles.canonical_hash(schedule),
            "source_hash": "a" * 64,
            "df_work": {
                "fock_builds": [1],
                "metrics": [
                    {"system_index": 0, "effective_rank": 2, "absolute_threshold": 1e-8}
                ],
            },
            "resources": {
                "peak_bytes": 4096,
                "rejected_allocations": 0,
                "admission": {"limit_bytes": 8192, "combined_peak_bytes": 6144},
            },
        }

    monkeypatch.setattr(execution_tune, "worker", worker)
    report = execution_tune.run(
        args,
        base_library=library,
        probe=probe,
        tools={},
        target=SimpleNamespace(architecture="sm_120"),
    )
    assert report["qualified_recipe"] and report["installed"] is None
    assert execution_tune.passed(report["holdout"])
    assert all(
        item["auxiliary_basis"] == "sto-3g"
        and item["metric_relative_threshold"] == 1e-10
        and item["df_memory_bytes"] == 20 << 20
        for item in observed
    )
    monkeypatch.setattr(execution_tune.ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(profiles, "probe_device", lambda *args: probe)
    execution_tune.replay(Path(report["qualified_recipe"]), tmp_path / "replayed.json")
    assert not (tmp_path / "cache/active.json").exists()


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)
@pytest.mark.parametrize("method", ("rhf",))
def test_real_df_policy_candidates_keep_full_reference_and_resource_contract(
    tmp_path: Path,
    workload: dict,
    method: str,
) -> None:
    """Qualify all proposed DF mappings independently, not against native copies."""
    from generativeqc import _native, profiles
    from generativeqc._dft_autotune import resource_gate

    assert os.environ.get("SLURM_JOB_ID")
    library = _native.load_library()
    probe = profiles.probe_device(library, 0)
    workload.update(
        method=method,
        charge=int(method == "uhf"),
        multiplicity=2 if method == "uhf" else 1,
        source_identity=probe["source_identity"],
    )
    reference = _df_autotune.reference_workload(workload)
    for index, policy in enumerate([{}, *_df_autotune.candidate_policies(full=True)]):
        sample = execution_tune.worker(
            Path(library._name),
            workload,
            "df",
            policy,
            tmp_path / f"policy-{index}",
            timeout=180,
        )
        assert _df_autotune.numerical_gate([sample], reference)["passed"]
        assert _df_autotune.work_gate([sample], reference)["passed"]
        assert resource_gate(
            [sample],
            schedule_hash=sample["schedule_identity"],
            source_hash=probe["source_identity"],
        )["passed"]

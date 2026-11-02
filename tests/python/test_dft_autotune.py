"""DFT search contracts; synthetic GPU evidence never leaves temporary storage."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import _autotune_worker, _dft_autotune, autotune, profiles
from generativeqc.ks import GridSpec, KsOptions, resolve_ks_options
from generativeqc_compiler.dft.xc_schedule import (
    DEVICE_FUSED,
    HOST_UNFUSED,
    molecular_grid_xc_workload,
)


def ledger() -> dict:
    """A bounded unit-test receipt, not a real-device resource measurement."""
    return {
        "peak_bytes": 4096,
        "rejected_allocations": 0,
        "admission": {"limit_bytes": 8192, "combined_peak_bytes": 6144},
    }


def test_endpoint_context_restricts_basis_batch_and_solver_controls(
    workload: dict,
) -> None:
    options = resolve_ks_options("pbe-rks")
    arguments = {
        "basis": "sto-3g",
        "representation": "spherical",
        "batch": 1,
        "method_identity": options.method_ir.identity,
        "controls": (100, 8, 1e-12, 1e-10, 1e-14),
    }
    context = profiles.dft_selection_context(**arguments)
    scientific = sample(workload)["scientific_workload"]
    diagnostics = {
        "source": "local",
        "dft_schedules": [
            {
                "workload_hash": profiles.canonical_hash(scientific),
                "selection_context": context,
                "schedule": {"point_tile": 512},
            }
        ],
    }
    assert profiles.select_dft_schedule(diagnostics, scientific) is None
    assert profiles.select_dft_schedule(
        diagnostics, scientific, selection_context=context
    )
    for changed in (
        {"basis": "def2-svp"},
        {"batch": 2},
        {"controls": (100, 8, 1e-9, 1e-5, 1e-14)},
    ):
        other = profiles.dft_selection_context(**{**arguments, **changed})
        assert other != context
        assert (
            profiles.select_dft_schedule(
                diagnostics, scientific, selection_context=other
            )
            is None
        )


def sample(workload: dict, seconds: float = 1.0) -> dict:
    options = resolve_ks_options(
        workload["method"], KsOptions(grid=GridSpec.treutler(workload["grid_level"]))
    )
    scientific = molecular_grid_xc_workload(
        architecture=workload["architecture"],
        functional=options.functional,
        atoms=workload["atoms"],
        grid_spec=options.grid,
        charge=workload["charge"],
        multiplicity=workload["multiplicity"],
        source_identity=workload["source_identity"],
    )
    schedule = DEVICE_FUSED.resolved(workload.get("tile_points", 256))
    return {
        "seconds": seconds,
        "iterations": [1] * workload["batch"],
        "energies": [-1.0] * workload["batch"],
        "forces": np.zeros((workload["batch"], len(workload["atoms"]), 3)).tolist(),
        "converged": True,
        "backend": ["cuda"] * workload["batch"],
        "endpoint_kind": "energy_force",
        "synchronized": True,
        "scientific_identity": scientific.identity,
        "scientific_workload": scientific.to_payload(),
        "schedule_identity": schedule.identity,
        "schedule": schedule.to_payload(),
        "source_hash": workload["source_identity"],
        "resources": ledger(),
    }


@pytest.fixture
def workload() -> dict:
    return {
        "kind": "dft",
        "method": "pbe-rks",
        "basis": "sto-3g",
        "grid_level": 0,
        "device_id": 2,
        "representation": "spherical",
        "batch": 1,
        "charge": 0,
        "multiplicity": 1,
        "architecture": "sm_120",
        "source_identity": "a" * 64,
        "atoms": [["H", [0.0, 0.0, -0.7]], ["H", [0.0, 0.0, 0.7]]],
    }


def test_tiles_are_bounded_distinct_and_do_not_repeat_baseline() -> None:
    args = SimpleNamespace(
        xc_tiles=[256, 128, 128, 512, 1024], full=False, max_candidates=2
    )
    assert _dft_autotune.candidate_tiles(args, 256) == [128, 512]
    args.xc_tiles, args.full, args.max_candidates = None, True, 3
    assert _dft_autotune.candidate_tiles(args, 128) == [64, 256, 512]
    args.xc_tiles = [True]
    with pytest.raises(ValueError, match="tiles"):
        _dft_autotune.candidate_tiles(args, 256)


def test_compiler_inventory_includes_new_owners_but_not_wheel_assets(
    tmp_path: Path,
) -> None:
    for name in (
        "__init__.py",
        "common/policy.py",
        "new_owner/schedule.json",
        "assets/upstream/reference.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    inventory = autotune._compiler_inventory(tmp_path)
    assert set(inventory) == {
        "python/generativeqc_compiler/__init__.py",
        "python/generativeqc_compiler/common/policy.py",
        "python/generativeqc_compiler/new_owner/schedule.json",
    }
    (tmp_path / "new_owner/schedule.json").write_text('{"changed": true}\n')
    assert autotune._compiler_inventory(tmp_path) != inventory


def test_loaded_compiler_and_matching_checkout_have_identical_complete_inventories() -> (
    None
):
    from generativeqc_compiler.common.paths import PACKAGE

    checkout = Path(__file__).resolve().parents[2] / "python/generativeqc_compiler"
    assert autotune._compiler_inventory(PACKAGE) == autotune._compiler_inventory(
        checkout
    )


def test_worker_uses_ks_options_and_observed_resources_not_hf_counters(
    workload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    options = resolve_ks_options("pbe-rks", KsOptions(grid=GridSpec.treutler(0)))
    diagnostic = SimpleNamespace(
        tile_points=256, to_payload=lambda: {"grid_points": 1000, "fock_builds": 1}
    )
    result = SimpleNamespace(
        items=[
            SimpleNamespace(
                iterations=1,
                energy=-1.0,
                forces=np.zeros((2, 3)),
                converged=True,
                executed_backend="cuda",
                ks_diagnostic=diagnostic,
            )
        ]
    )

    class Prepared:
        def __init__(self) -> None:
            self.resource_diagnostics = {
                "phase": "observation",
                "observation": {"device_ledger": ledger()},
            }

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def execute(self, **kwargs: Any) -> Any:
            assert kwargs == {"properties": ("energy", "forces"), "strict": True}
            return result

        def set_warm_start_updates(self, enabled: bool) -> None:
            assert enabled is False

        def last_shell_class_profile(self) -> None:
            pytest.fail("DFT must not ask for HF-specific shell counters")

    def prepare(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["shell_class_profiling"] is False
        return Prepared()

    def calculator(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            ks_options=options, profile_diagnostics={}, prepare_batch=prepare
        )

    monkeypatch.setattr(_autotune_worker, "Calculator", calculator)
    measured = _autotune_worker.execute_workload(workload, profile=True)
    assert calls[0]["device_id"] == 2
    assert calls[0]["resource_budget"] is not None
    assert measured["synchronized"] and measured["endpoint_kind"] == "energy_force"
    assert measured["scientific_identity"] == sample(workload)["scientific_identity"]
    assert measured["resources"] == ledger()
    assert measured["dft_work"] == [{"grid_points": 1000, "fock_builds": 1}]


@pytest.mark.parametrize(
    "corruption",
    ("missing", "over_limit", "rejected", "wrong_source", "wrong_schedule"),
)
def test_resource_gate_rejects_incomplete_or_unbounded_evidence(
    workload: dict, corruption: str
) -> None:
    measured = sample(workload)
    if corruption == "missing":
        measured["resources"] = {}
    elif corruption == "over_limit":
        measured["resources"]["admission"]["combined_peak_bytes"] = 8193
    elif corruption == "rejected":
        measured["resources"]["rejected_allocations"] = 1
    elif corruption == "wrong_source":
        measured["source_hash"] = "b" * 64
    else:
        measured["schedule_identity"] = "c" * 64
    result = _dft_autotune.resource_gate(
        [measured],
        schedule_hash=sample(workload)["schedule_identity"],
        source_hash="a" * 64,
    )
    assert not result["passed"]


def test_independent_gate_checks_every_sample_and_rejects_broadcasting(
    workload: dict,
) -> None:
    measured = sample(workload)
    reference = {
        "independent_reference": "test-only oracle",
        "energies": measured["energies"],
        "forces": measured["forces"],
    }
    assert _dft_autotune.numerical_gate([measured], reference)["passed"]
    changed = copy.deepcopy(measured)
    changed["forces"][0][0][0] = 1e-3
    assert not _dft_autotune.numerical_gate([measured, changed], reference)["passed"]
    changed["forces"] = [[[0.0, 0.0, 0.0]]]
    with pytest.raises(ValueError, match="layout"):
        _dft_autotune.numerical_gate([changed], reference)
    reference["energies"] = [float("nan")]
    with pytest.raises(ValueError, match="reference"):
        _dft_autotune.numerical_gate([measured], reference)


@pytest.fixture
def search(tmp_path: Path, workload: dict, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Exercise real bundle validation with temporary, nonexecutable byte artifacts."""
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nfixture in Bohr\nH 0 0 -0.7\nH 0 0 0.7\n")
    args = SimpleNamespace(
        method="pbe-rks",
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
        max_candidates=4,
        xc_tiles=[128, 512],
        budget_seconds=60,
        repeats=5,
        retune=False,
        portable_baseline=False,
        export=None,
        source_dir=Path(__file__).resolve().parents[2],
    )
    probe = {
        "source_identity": "a" * 64,
        "native_abi": 1,
        "schema": profiles.PROFILE_SCHEMA,
        "policy": profiles.POLICY,
        "device": {"major": 12, "minor": 0, "release_build": 1, "fast_compile": 0},
    }
    binary = tmp_path / "fixture-library.so"
    binary.write_bytes(b"unit-test only, not a native executable")
    base = SimpleNamespace(_name=str(binary))
    state = SimpleNamespace(
        args=args,
        base=base,
        probe=probe,
        tools={"nvcc": "test NVCC", "ptxas": "test PTXAS"},
        target=SimpleNamespace(architecture="sm_120", generator_abi=1),
        diagnostics={"source": "portable", "dft_schedules": []},
        incumbent=base,
        speeds={256: 1.0, 128: 0.8, 512: 1.2},
        installed=[],
        calls=[],
        reference_delta=0.0,
        refuse_resources=False,
    )
    monkeypatch.setattr(profiles, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(
        profiles, "select_library", lambda *args: (state.incumbent, state.diagnostics)
    )

    def worker(library: Path, path: Path, output: Path, **kwargs: Any) -> dict:
        payload = json.loads(path.read_text())
        state.calls.append((payload, kwargs))
        measured = sample(payload, state.speeds[payload.get("tile_points", 256)])
        if kwargs.get("reference"):
            measured = {
                "independent_reference": "synthetic unit-test oracle, not hardware evidence",
                "energies": [
                    value + state.reference_delta for value in measured["energies"]
                ],
                "forces": measured["forces"],
            }
        elif state.refuse_resources:
            measured["resources"]["rejected_allocations"] = 1
        profiles.atomic_json(output, measured)
        return measured

    def install(directory: Path, actual_probe: dict, tools: dict) -> Path:
        validated = profiles.validate_bundle(directory, actual_probe, tools)
        state.installed.append(validated)
        return directory

    monkeypatch.setattr(autotune, "_worker", worker)
    monkeypatch.setattr(profiles, "install_bundle", install)

    def run() -> dict:
        return _dft_autotune.run_dft(
            args, base_library=base, probe=probe, tools=state.tools, target=state.target
        )

    state.run = run
    return state


def test_search_promotes_only_complete_independent_bounded_winner(search: Any) -> None:
    report = search.run()
    assert "failure" not in report
    assert report["installed"]
    assert [row["accepted"] for row in report["candidates"]] == [True, False]
    profile = search.installed[0]
    assert profile["kernels"] == []
    assert profile["dft_schedules"][0]["schedule"]["point_tile"] == 128
    evidence = json.loads((Path(report["installed"]) / "evidence.json").read_text())
    assert evidence["dft_schedules"][0]["endpoint"]["complete_energy_force"]
    assert all(call[0]["device_id"] == 2 for call in search.calls)


@pytest.mark.parametrize("failure", ("slower", "reference", "memory", "budget"))
def test_failed_search_never_activates_a_profile(search: Any, failure: str) -> None:
    if failure == "slower":
        search.speeds[128] = 1.1
    elif failure == "reference":
        search.reference_delta = 1.0
    elif failure == "memory":
        search.refuse_resources = True
    else:
        search.args.budget_seconds = 0
    report = search.run()
    assert report["installed"] is None and search.installed == []
    assert "reason" in report or "failure" in report
    assert (Path(report["directory"]) / "report.json").is_file()


def test_retune_preserves_other_workloads_and_skips_exact_incumbent(
    search: Any,
) -> None:
    original = search.run()
    original_profile = copy.deepcopy(search.installed[-1])
    search.incumbent = SimpleNamespace(
        _name=str(Path(original["installed"]) / "libgenerativeqc.so")
    )
    search.diagnostics = {
        "source": "local",
        "directory": original["installed"],
        "dft_schedules": original_profile["dft_schedules"],
    }
    count = len(search.calls)
    skipped = search.run()
    assert skipped["installed"] is None and "--retune" in skipped["reason"]
    assert len(search.calls) == count
    search.args.input.write_text("2\nmoved geometry\nH 0 0 -0.8\nH 0 0 0.8\n")
    moved = search.run()
    assert moved["installed"]
    assert len(search.installed[-1]["dft_schedules"]) == 2
    assert original_profile["dft_schedules"][0] in search.installed[-1]["dft_schedules"]
    search.args.input.write_text("2\noriginal geometry\nH 0 0 -0.7\nH 0 0 0.7\n")
    search.args.retune, search.args.xc_tiles, search.speeds[512] = True, [512], 0.6
    retuned = search.run()
    assert retuned["installed"]
    assert search.installed[-1]["dft_schedules"][0]["schedule"]["point_tile"] == 512


def test_selection_context_must_match_retained_evidence(search: Any) -> None:
    result = search.run()
    directory = Path(result["installed"])
    profile_path = directory / "profile.json"
    profile = json.loads(profile_path.read_text())
    profile["dft_schedules"][0]["selection_context"] = "f" * 64
    profiles.atomic_json(profile_path, profile)
    with pytest.raises(ValueError, match="selection context"):
        profiles.validate_bundle(directory)


def test_host_incumbent_is_not_misrepresented_as_a_device_fused_baseline(
    search: Any,
) -> None:
    original = search.run()
    winner = copy.deepcopy(search.installed[-1]["dft_schedules"][0])
    winner["schedule"] = HOST_UNFUSED.resolved(256).to_payload()
    winner["schedule_hash"] = profiles.canonical_hash(winner["schedule"])
    search.diagnostics = {
        "source": "local",
        "directory": original["installed"],
        "dft_schedules": [winner],
    }
    search.args.retune = True
    count = len(search.calls)
    report = search.run()
    assert report["installed"] is None and "host-XC" in report["reason"]
    assert len(search.calls) == count


def test_portable_search_must_also_beat_actual_incumbent(search: Any) -> None:
    search.args.portable_baseline = True
    report = search.run()
    assert report["installed"]
    assert report["incumbent_endpoint"]["passed"]
    assert any(kwargs.get("generic") for _, kwargs in search.calls)
    final_pairs = search.calls[-10:]
    assert all(not kwargs.get("generic") for _, kwargs in final_pairs)


def test_portable_only_improvement_cannot_replace_faster_incumbent(
    search: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = autotune._worker
    search.args.portable_baseline, search.args.xc_tiles = True, [128]

    def execute(library: Path, path: Path, output: Path, **kwargs: Any) -> dict:
        measured = worker(library, path, output, **kwargs)
        if (
            "endpoint" in output.name
            and not kwargs.get("generic")
            and measured["schedule"]["point_tile"] == 256
        ):
            measured["seconds"] = 0.7
        return measured

    monkeypatch.setattr(autotune, "_worker", execute)
    report = search.run()
    assert report["installed"] is None and search.installed == []
    assert not report["incumbent_endpoint"]["passed"]


def test_candidate_timeout_keeps_working_configuration(
    search: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = autotune._worker

    def execute(library: Path, path: Path, output: Path, **kwargs: Any) -> dict:
        if "endpoint" in output.name:
            raise subprocess.TimeoutExpired("unit-test worker", kwargs["timeout"])
        return worker(library, path, output, **kwargs)

    monkeypatch.setattr(autotune, "_worker", execute)
    report = search.run()
    assert report["installed"] is None and search.installed == []
    assert all("timed out" in row["failure"] for row in report["candidates"])


def test_dft_extension_retains_validated_hf_kernels_and_their_evidence(
    search: Any,
) -> None:
    search.probe["device"].update(
        name="unit-test target", runtime_version=12090, driver_version=13000
    )
    original = search.run()
    directory = Path(original["installed"])
    profile_path, evidence_path = (
        directory / "profile.json",
        directory / "evidence.json",
    )
    profile = json.loads(profile_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    schedule = {"kind": "shell_task", "block_threads": 128}
    kernel = {
        "shell_class": "pppp",
        "consumer": "force",
        "schedule": schedule,
        "schedule_hash": profiles.canonical_hash(schedule),
        "source_hash": "b" * 64,
        "gates": {
            name: "pass"
            for name in ("target", "resources", "numerical", "performance", "endpoint")
        },
    }
    row = {
        "shell_class": "pppp",
        "consumer": "force",
        "accepted": True,
        "selected_trial": "fixture-only",
        "endpoint": {"passed": True},
        "tuning": {
            "winners": [
                {
                    "trial_key": "fixture-only",
                    "schedule": schedule,
                    "production_validation": {"accepted": True},
                }
            ]
        },
        "isolated": {
            "passed": True,
            "source_hash": kernel["source_hash"],
            "schedule_hash": kernel["schedule_hash"],
            "errors": {"unit-test fixture": {"passed": True}},
            "device": {
                "name": "unit-test target",
                "major": 12,
                "minor": 0,
                "runtime": 12090,
                "driver": 13000,
            },
        },
    }
    profile["kernels"] = [kernel]
    evidence.update(candidates=[row], endpoint={"passed": True})
    profiles.atomic_json(evidence_path, evidence)
    profile["artifacts"]["evidence.json"] = profiles.file_hash(evidence_path)
    profiles.atomic_json(profile_path, profile)
    profiles.validate_bundle(directory, search.probe, search.tools)
    search.incumbent = SimpleNamespace(_name=str(directory / "libgenerativeqc.so"))
    search.diagnostics = {
        "source": "local",
        "directory": str(directory),
        "dft_schedules": profile["dft_schedules"],
    }
    search.args.input.write_text("2\ndifferent DFT workload\nH 0 0 -0.8\nH 0 0 0.8\n")
    extended = search.run()
    assert extended["installed"]
    assert search.installed[-1]["kernels"] == [kernel]
    retained = json.loads((Path(extended["installed"]) / "evidence.json").read_text())
    assert retained["candidates"] == [row] and retained["endpoint"] == {"passed": True}


@pytest.mark.parametrize(
    "method",
    _dft_autotune.DFT_FUNCTIONALS,
)
def test_reference_selectors_match_declared_functional_composition(method: str) -> None:
    pytest.importorskip("pyscf")
    from generativeqc_compiler.method import ExactExchangePrimitive
    from pyscf.dft import libxc

    options = resolve_ks_options(method)
    functional = options.functional
    hybrid, components = libxc.parse_xc(_dft_autotune.DFT_FUNCTIONALS[method])
    expected = {
        libxc.XC_CODES[name]: float(weight) for name, weight in functional.components
    }
    assert dict(components) == expected
    exact_exchange = sum(
        float(primitive.coefficient)
        for primitive in options.method_ir.primitives
        if isinstance(primitive, ExactExchangePrimitive)
    )
    assert hybrid[0] == exact_exchange


@pytest.mark.parametrize(
    "method", ("lda-rks", "pbe-rks", "pbe-uks", "pbe0-rks", "b3lyp-rks", "r2scan-rks")
)
def test_real_cpu_reference_has_analytic_grid_response_and_batch_layout(
    workload: dict,
    method: str,
) -> None:
    pytest.importorskip("pyscf")
    workload.update(batch=2, method=method)
    if method.endswith("-uks"):
        workload.update(charge=1, multiplicity=2)
    reference = _dft_autotune.reference_workload(workload)
    assert np.isfinite(reference["energies"]).all()
    assert np.asarray(reference["forces"]).shape == (2, 2, 3)
    assert reference["energies"][0] == reference["energies"][1]
    assert np.max(np.abs(np.asarray(reference["forces"]).sum(axis=1))) < 1e-9


def test_reference_subprocess_never_requires_native_or_gpu_library(
    workload: dict,
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyscf")
    path, output = tmp_path / "workload.json", tmp_path / "reference.json"
    profiles.atomic_json(path, workload)
    reference = autotune._worker(
        tmp_path / "missing-native.so", path, output, reference=True, timeout=30
    )
    assert reference["independent_reference"]
    assert np.isfinite(reference["energies"]).all()


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires an allocated Slurm CUDA device",
)
@pytest.mark.parametrize(
    "method", ("lda-rks", "pbe-rks", "pbe-uks", "pbe0-rks", "b3lyp-rks", "r2scan-rks")
)
def test_real_dft_worker_and_independent_reference(workload: dict, method: str) -> None:
    """Qualify the actual native E+F replay and ledger without publishing a profile."""
    assert os.environ.get("SLURM_JOB_ID")
    from generativeqc import _native

    probe = profiles.probe_device(_native.load_library(), 0)
    workload.update(
        method=method,
        grid_level=3,
        device_id=0,
        source_identity=probe["source_identity"],
        architecture=f"sm_{probe['device']['major']}{probe['device']['minor']}",
    )
    if method.endswith("-uks"):
        workload.update(charge=1, multiplicity=2)
    baseline = _autotune_worker.execute_workload(workload, profile=True)
    candidate = _autotune_worker.execute_workload(
        {**workload, "tile_points": 128}, profile=True
    )
    reference = _dft_autotune.reference_workload(workload)
    assert _dft_autotune.numerical_gate([baseline, candidate], reference)["passed"]
    assert _dft_autotune.resource_gate(
        [candidate],
        schedule_hash=candidate["schedule_identity"],
        source_hash=probe["source_identity"],
    )["passed"]

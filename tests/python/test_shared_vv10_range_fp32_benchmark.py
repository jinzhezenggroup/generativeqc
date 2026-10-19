"""Host fault injection for live qualification; never loads a GPU library."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO

import pytest
from generativeqc import GridSpec

from benchmarks import shared_vv10_range_fp32 as runner


@pytest.fixture(scope="module")
def science() -> dict:
    return runner.protocol("water-def2-svp-spherical", GridSpec(48, 16, 32))


def oracle(science: dict) -> dict:
    return {
        "converged": True,
        "reference_backend": "gpu4pyscf",
        "functional": "WB97M-V",
        "density_fitting": False,
        "grid_response": True,
        "threads": 4,
        "conv_tol": 1e-12,
        "conv_tol_grad": 1e-10,
        "direct_scf_tol": 1e-14,
        "vv10_density_threshold": {
            "scf": science["vv10_density_threshold"]["reference_scf_and_forces"],
            "forces": science["vv10_density_threshold"]["reference_scf_and_forces"],
        },
        "aos": science["ao_count"],
        "grid_points": len(science["atoms"]) * 48 * 16 * 32,
        "energy": -1.0,
        "forces": [[0.0, 0.0, 0.0] for _ in science["atoms"]],
    }


def test_protocol_retains_actual_science(science: dict) -> None:
    assert science["coordinate_units"] == "Bohr"
    assert len(science["atoms"]) == 3
    assert science["basis"]["elements"][0]["shells"][0]["exponents"]
    assert science["basis_representation"] == "spherical"
    assert science["auxiliary_basis"] is None and not science["density_fitting"]
    assert science["method_ir"]["primitives"]
    for grid in ("semilocal_grid", "nonlocal_grid"):
        assert science[grid]["partition_iterations"] == 3
        assert science[grid]["pruning"] == "none"
    assert science["gates_Eh_maxF_RMS_Eh_per_bohr"] == runner.GATES
    assert science["vv10_density_policy"] == "vv10-molecular-rho-ge-1e-10-v2"
    assert science["vv10_density_threshold"] == {
        "native_scf_and_forces": 1e-10,
        "reference_scf_and_forces": 1e-10,
    }


@pytest.mark.parametrize(
    "field",
    [
        "atoms",
        "ao_count",
        "coordinate_units",
        "charge",
        "multiplicity",
        "method_ir",
        "basis",
        "basis_representation",
        "density_fitting",
        "auxiliary_basis",
        "metric_threshold",
        "ecp",
        "semilocal_grid",
        "nonlocal_grid",
        "grid_response",
        "vv10_density_threshold",
        "vv10_density_policy",
        "native_recipe",
        "reference_recipe",
        "gates_Eh_maxF_RMS_Eh_per_bohr",
    ],
)
def test_reused_oracle_requires_full_protocol(
    tmp_path: Path, science: dict, field: str
) -> None:
    invalid = copy.deepcopy(science)
    invalid[field] = {"changed": True}
    path = tmp_path / "oracle.json"
    path.write_text(
        json.dumps(
            {
                "status": "reference-completed",
                "protocol": invalid,
                "reference": oracle(science),
            }
        )
    )
    with pytest.raises(ValueError, match="complete protocol"):
        runner.load_oracle(path, science)


@pytest.mark.parametrize(
    "field,value",
    [
        ("functional", "PBE0"),
        ("aos", 999),
        ("grid_points", 1),
        ("density_fitting", True),
        ("reference_backend", "native"),
        ("converged", False),
        ("grid_response", False),
        ("conv_tol", 1e-6),
        ("threads", 8),
        ("forces", [[0, 0, 0]]),
        ("energy", float("nan")),
    ],
)
def test_reference_rejects_contradictory_inner_metadata(
    science: dict, field: str, value: object
) -> None:
    invalid = oracle(science)
    invalid[field] = value
    with pytest.raises(ValueError):
        runner.validate_reference(invalid, science)


@pytest.mark.parametrize("leg", ["scf", "forces", "missing"])
def test_reference_rejects_unmatched_vv10_domains(science: dict, leg: str) -> None:
    invalid = oracle(science)
    if leg == "missing":
        del invalid["vv10_density_threshold"]
    else:
        invalid["vv10_density_threshold"][leg] = 1e-8
    with pytest.raises(ValueError, match="contradicts protocol"):
        runner.validate_reference(invalid, science)


def test_old_vv10_receipt_is_rejected_without_modification(
    tmp_path: Path, science: dict
) -> None:
    old = copy.deepcopy(science)
    old.pop("vv10_density_policy")
    old["vv10_density_threshold"] = {
        "native_scf_and_forces": 1e-8,
        "reference_scf_and_forces": 1e-8,
    }
    path = tmp_path / "old-oracle.json"
    payload = json.dumps(
        {
            "schema": "generativeqc.shared-vv10-range-fp32.v1",
            "status": "reference-completed",
            "protocol": old,
            "reference": oracle(old),
        }
    ).encode()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="exact complete protocol"):
        runner.load_oracle(path, science)
    assert path.read_bytes() == payload


def setup_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science: dict) -> tuple:
    output, reference = tmp_path / "run.json", tmp_path / "oracle.json"
    reference.write_text(
        json.dumps(
            {
                "status": "reference-completed",
                "protocol": science,
                "reference": oracle(science),
            }
        )
    )
    monkeypatch.setenv("SLURM_JOB_ID", "host-only-fault-injection")
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", "/deliberately/wrong/requested-library")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--engine",
            "native",
            "--case",
            "fixture",
            "--oracle",
            str(reference),
            "--output",
            str(output),
            "--repeats",
            "1",
        ],
    )
    case = SimpleNamespace(
        atoms=science["atoms"],
        generativeqc_basis="basis",
        basis_representation="spherical",
        charge=0,
        multiplicity=1,
    )
    basis = object()
    monkeypatch.setattr(runner, "benchmark_cases", lambda: {"fixture": case})
    monkeypatch.setattr(runner, "snapshot_basis", lambda *a: basis)
    monkeypatch.setattr(runner, "protocol", lambda *a, **k: copy.deepcopy(science))
    source = {"source_identity": "c" * 64, "benchmark_sha256": {"runner": "d" * 64}}
    monkeypatch.setattr(runner, "source_receipt", lambda: copy.deepcopy(source))
    build = {
        "library_path": "/actually/selected/profile.so",
        "library_sha256": "a" * 64,
        "probe": {"source_identity": "c" * 64},
    }

    def sample(mode: str, settings: dict, case_name: str, **kwargs: object) -> dict:
        stored = json.loads(output.read_text())
        assert stored["records"][-1]["attempt_status"] == "running"
        assert kwargs == {"include_build_metadata": True, "case_snapshot": case}
        assert settings["basis"] is basis
        return {
            **oracle(science),
            "mode": mode,
            "seconds": 2.0 if mode == "fp64" else 1.0,
            "native_build": copy.deepcopy(build),
            "native_build_before": copy.deepcopy(build),
            "precision_evidence_error": None,
            "precision_validation": {"status": "qualified_budget_fock", "solves": [{}]},
            "precision": {
                "operators": [
                    {"name": n, "compute": "fp32", "accumulation": "fp64", "count": 1}
                    for n in ("coulomb_j", "exchange_k", "nonlocal_correlation")
                ]
            },
        }

    monkeypatch.setattr(runner, "native_sample", sample)
    return output, sample, build, source


def test_success_attests_each_actual_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science: dict
) -> None:
    output, _, build, _ = setup_run(tmp_path, monkeypatch, science)
    runner.main()
    report = json.loads(output.read_text())
    assert (
        report["status"] == "numerically-qualified" and not report["performance_claim"]
    )
    assert report["performance_observation_available"]
    assert report["performance_acceptance"].startswith("unassessed")
    assert report["medians_seconds"] == {"fp64": 2.0, "fast": 1.0}
    assert report["speedup_fp64_over_fast"] == 2.0
    assert [r["mode"] for r in report["records"]] == ["fp64", "fast", "fast", "fp64"]
    assert [r["first_call"] for r in report["records"]] == [True, True, False, False]
    assert all(r["native_build"] == build for r in report["records"])


def test_live_reference_receives_the_declared_vv10_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science: dict
) -> None:
    output, _, _, _ = setup_run(tmp_path, monkeypatch, science)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--engine",
            "reference",
            "--case",
            "fixture",
            "--output",
            str(output),
        ],
    )

    def reference(*args: object, **kwargs: object) -> dict:
        assert kwargs["vv10_density_threshold"] == 1e-10
        return oracle(science)

    monkeypatch.setattr(runner, "reference", reference)
    runner.main()
    report = json.loads(output.read_text())
    assert report["schema"] == "generativeqc.shared-vv10-range-fp32.v2"
    assert report["status"] == "reference-completed"


@pytest.mark.parametrize(
    "defect",
    [
        "exception",
        "interrupt",
        "nan",
        "energy",
        "max_force",
        "rms_force",
        "no_work",
        "extra_solve",
        "wrong_source",
        "changed_binary",
        "changed_source",
        "bad_time",
    ],
)
def test_failed_attempts_never_become_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science: dict, defect: str
) -> None:
    output, sample, _, source = setup_run(tmp_path, monkeypatch, science)
    calls = 0

    def failing(*args: object, **kwargs: object) -> dict:
        nonlocal calls
        calls += 1
        if defect == "exception":
            raise RuntimeError("injected device failure")
        if defect == "interrupt":
            raise KeyboardInterrupt("injected interruption")
        row = sample(*args, **kwargs)
        if defect == "nan":
            row["energy"] = float("nan")
        if defect == "energy":
            row["energy"] += 1.0
        if defect == "max_force":
            row["forces"][0][0] = 1e-3
        if defect == "rms_force":
            row["forces"] = [[5e-5] * 3 for _ in science["atoms"]]
        if defect == "no_work":
            row["precision"]["operators"][-1]["count"] = 0
        if defect == "extra_solve":
            row["precision_validation"]["solves"].append({})
        if defect == "wrong_source":
            row["native_build"]["probe"]["source_identity"] = "b" * 64
            row["native_build_before"]["probe"]["source_identity"] = "b" * 64
        if defect == "changed_binary" and calls > 1:
            row["native_build"]["library_sha256"] = "b" * 64
            row["native_build_before"]["library_sha256"] = "b" * 64
        if defect == "changed_source":
            source["source_identity"] = "b" * 64
        if defect == "bad_time":
            row["seconds"] = float("nan")
        return row

    monkeypatch.setattr(runner, "native_sample", failing)
    with pytest.raises((RuntimeError, ValueError, KeyboardInterrupt)):
        runner.main()
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and not report["performance_claim"]
    assert report["records"][-1]["attempt_status"] == "failed"
    assert report["error"] and "speedup_fp64_over_fast" not in report
    if defect == "nan":
        assert report["records"][-1]["energy"] is None
        assert report["nonfinite_values"]["$report.records[0].energy"] == "nan"


@pytest.mark.parametrize("kind", ["existing", "dangling-symlink", "concurrent"])
def test_exclusive_output_precedes_scientific_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science: dict, kind: str
) -> None:
    output, _, _, _ = setup_run(tmp_path, monkeypatch, science)
    target = tmp_path / "target.json"
    if kind == "existing":
        output.write_text("existing evidence")
    elif kind == "dangling-symlink":
        output.symlink_to(target)
    else:
        original = Path.open

        def competing(path: Path, mode: str = "r", *a: object, **k: object) -> TextIO:
            if path == output and mode == "x+":
                with original(path, "x") as stream:
                    stream.write("competing evidence")
            return original(path, mode, *a, **k)

        monkeypatch.setattr(Path, "open", competing)
    monkeypatch.setattr(
        runner, "protocol", lambda *a, **k: pytest.fail("science began")
    )
    with pytest.raises(SystemExit):
        runner.main()
    if kind == "dangling-symlink":
        assert output.is_symlink() and not target.exists()
    else:
        assert "evidence" in output.read_text()


@pytest.mark.parametrize("kind", ["retained", "alias", "traversal"])
def test_raw_output_rejects_retained_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from benchmarks import _retention

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    destination = retained / "new.json"
    if kind == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(retained)
        destination = alias / "new.json"
    if kind == "traversal":
        destination = tmp_path / ".artifacts/../benchmarks/results/new.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--engine",
            "reference",
            "--case",
            "unused",
            "--output",
            str(destination),
        ],
    )
    monkeypatch.setattr(
        runner, "protocol", lambda *a, **k: pytest.fail("science began")
    )
    with pytest.raises(SystemExit):
        runner.main()
    assert not list(retained.iterdir())


@pytest.mark.parametrize("mode", ["fp64", "fast"])
def test_shared_sample_attests_executed_owner_outside_timer(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from benchmarks import pbe0_precision as shared

    case = SimpleNamespace(atoms=[["H", [0, 0, 0]]], charge=0, multiplicity=1)
    observed = SimpleNamespace(
        forces=[[0, 0, 0]],
        energy=-1.0,
        converged=True,
        executed_backend="cuda",
        precision={},
        iterations=1,
        ks_diagnostic=None,
        precision_validation=None,
        resource_diagnostics={
            "generated_force": [{"index": 0, "work": {"grid_points": 1}}]
        },
    )
    actual = SimpleNamespace(
        singlepoint=lambda *a, **k: observed, clear_cache=lambda: None
    )
    wrapper = SimpleNamespace(
        _candidate=actual,
        singlepoint=actual.singlepoint,
        clear_cache=actual.clear_cache,
    )
    monkeypatch.setattr(shared, "Calculator", lambda **k: actual)
    monkeypatch.setattr(shared, "PrecisionCalculator", lambda **k: wrapper)
    monkeypatch.setattr(
        shared.ctypes,
        "CDLL",
        lambda _: SimpleNamespace(cudaDeviceSynchronize=lambda: 0),
    )
    monkeypatch.setattr(shared, "precision_from_result", lambda _: None)
    monkeypatch.setattr(
        shared, "benchmark_cases", lambda: pytest.fail("case snapshot was ignored")
    )
    times = iter([10.0, 10.5, 11.0, 14.5])
    monkeypatch.setattr(shared, "perf_counter", lambda: next(times))

    def metadata(owner: object) -> dict:
        assert owner is actual
        return {"library_sha256": "a" * 64}

    monkeypatch.setattr(shared, "native_build_metadata", metadata)
    row = shared.native_sample(
        mode, {}, "unused", include_build_metadata=True, case_snapshot=case
    )
    assert row["seconds"] == 4.0 and row["native_build"]["library_sha256"] == "a" * 64
    assert row["native_build_before"] == row["native_build"]
    assert row["excluded_provenance_seconds"] == 0.5
    assert row["observed_span_seconds"] == 4.5
    assert row["force_work"] == observed.resource_diagnostics["generated_force"]
    with pytest.raises(StopIteration):
        next(times)


def test_fast_rms_gate_is_independent_of_maximum(science: dict) -> None:
    row = {
        **oracle(science),
        "seconds": 1.0,
        "precision_evidence_error": None,
        "forces": [[5e-5] * 3 for _ in science["atoms"]],
    }
    measured = runner.errors(row, oracle(science))
    assert measured["max_force_Eh_per_bohr"] < runner.GATES["fast"][1]
    assert measured["rms_force_Eh_per_bohr"] > runner.GATES["fast"][2]
    with pytest.raises(ValueError, match="independent energy/force gate"):
        runner.check_sample(row, oracle(science), "fast")


def test_source_receipt_rejects_imports_from_another_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "generativeqc.foreign_fixture",
        SimpleNamespace(__file__="/other/checkout/foreign.py"),
    )
    with pytest.raises(ValueError, match="outside the attested source tree"):
        runner.source_receipt()


def test_oracle_hash_and_parse_share_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science: dict
) -> None:
    import hashlib

    path = tmp_path / "oracle.json"
    payload = json.dumps(
        {
            "status": "reference-completed",
            "protocol": science,
            "reference": oracle(science),
        }
    ).encode()
    calls = []

    def read_bytes(selected: Path) -> bytes:
        assert selected == path
        calls.append(True)
        return payload if len(calls) == 1 else b"changed while reading"

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    _, digest = runner.load_oracle(path, science)
    assert len(calls) == 1 and digest == hashlib.sha256(payload).hexdigest()


def test_slower_candidate_is_retained_without_performance_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science: dict
) -> None:
    output, sample, _, _ = setup_run(tmp_path, monkeypatch, science)

    def slower(*args: object, **kwargs: object) -> dict:
        row = sample(*args, **kwargs)
        if row["mode"] == "fast":
            row["seconds"] = 4.0
        return row

    monkeypatch.setattr(runner, "native_sample", slower)
    runner.main()
    report = json.loads(output.read_text())
    assert report["status"] == "numerically-qualified"
    assert report["speedup_fp64_over_fast"] == 0.5
    assert not report["performance_claim"] and report[
        "performance_acceptance"
    ].startswith("unassessed")


@pytest.mark.parametrize("replacement", ["file", "symlink"])
def test_updates_keep_reserved_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science: dict, replacement: str
) -> None:
    output, sample, _, _ = setup_run(tmp_path, monkeypatch, science)
    reserved, unrelated = tmp_path / "reserved.json", tmp_path / "unrelated.json"
    unrelated.write_text("unrelated evidence")
    changed = False

    def replace(*args: object, **kwargs: object) -> dict:
        nonlocal changed
        # The helper reads the current attempted record before replacement.
        row = (
            sample(*args, **kwargs)
            if not changed
            else {**oracle(science), "mode": args[0], "seconds": 1.0}
        )
        if not changed:
            output.rename(reserved)
            if replacement == "file":
                output.write_text("replacement evidence")
            else:
                output.symlink_to(unrelated)
            changed = True
            raise RuntimeError("stop after path replacement")
        return row

    monkeypatch.setattr(runner, "native_sample", replace)
    with pytest.raises(RuntimeError, match="path replacement"):
        runner.main()
    assert json.loads(reserved.read_text())["status"] == "failed"
    assert unrelated.read_text() == "unrelated evidence"
    assert output.read_text() == (
        "replacement evidence" if replacement == "file" else "unrelated evidence"
    )

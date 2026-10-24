"""Scientific identities and complete timing boundaries for README comparisons."""

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import GridSpec, load_r2scan3c_basis

from benchmarks.readme_dft import SCHEMA, specification
from benchmarks.readme_omol25 import (
    check_record,
    protocol,
    reference_baselines,
    reference_energy_forces,
    retain_endpoint_record,
    source_hashes,
)
from benchmarks.readme_wb97mv import _bind_reference_mixed_rsh_df
from tools.render_omol25_benchmarks import independent_phase_status
from tools.render_readme_dft import collect_point, phase_statistics, timing_cell


def test_precision_consumer_provenance_is_retained() -> None:
    """Dirty admission changes remain identifiable beside the native binary."""
    sources = source_hashes()
    for filename in (
        "precision_policy.py",
        "_df_precision.py",
        "_composite_precision.py",
    ):
        assert len(sources[f"python/generativeqc/{filename}"]) == 64


def test_force_schedule_consumer_provenance_is_retained() -> None:
    """Bind runtime schedule experiments to their actual native consumers."""
    sources = source_hashes()
    for filename in (
        "direct_force_class_domains.cu",
        "direct_force_class_domains.hpp",
        "direct_force_class_pages.cuh",
        "direct_force_diagnostic.hpp",
        "direct_spd_force_schedule.hpp",
    ):
        assert len(sources[f"src/scf/cuda/{filename}"]) == 64
    compiler_owner = (
        "python/generativeqc_compiler/integral/direct_cooperative_gradient_cuda.py"
    )
    assert len(sources[compiler_owner]) == 64
    for path in (
        "src/dft/cuda_ks.cpp",
        "src/dft/cuda_ks_final_validation_policy.hpp",
        "python/generativeqc/resources_hf.py",
    ):
        assert len(sources[path]) == 64


def test_hf_and_df_protocols_have_distinct_scientific_identities() -> None:
    spec = specification("hf")
    assert spec.native_method == "rhf"
    basis = load_r2scan3c_basis()
    direct = protocol(3, basis, GridSpec(), 5, benchmark=spec)
    auxiliary = SimpleNamespace(name="explicit-jkfit", identity="auxiliary-identity")
    fitted = protocol(
        3,
        basis,
        GridSpec(),
        5,
        benchmark=replace(spec, density_fitting=True),
        auxiliary_basis=auxiliary,
    )
    assert direct["method"] == "RHF"
    assert direct["grid"] is None
    assert not direct["density_fitting"]
    assert fitted["density_fitting"]
    assert fitted["auxiliary_basis_identity"] == auxiliary.identity
    assert fitted["reference_auxbasis_response"]
    with pytest.raises(ValueError, match="auxiliary basis identity"):
        protocol(3, basis, GridSpec(), 5, benchmark=replace(spec, density_fitting=True))


def test_independent_vv10_grid_is_part_of_the_benchmark_identity() -> None:
    """A point-count collision must not pool scientifically distinct protocols."""
    basis = load_r2scan3c_basis()
    primary = GridSpec(8, 4, 8)
    secondary = GridSpec(4, 8, 8)
    benchmark = specification("wb97m-v")
    original = protocol(3, basis, primary, 5, benchmark=benchmark)
    for shared in (primary, replace(primary, element_radii=((1, 1.0),))):
        assert (
            protocol(3, basis, primary, 5, benchmark=benchmark, nonlocal_grid=shared)
            == original
        )
    distinct = protocol(
        3, basis, primary, 5, benchmark=benchmark, nonlocal_grid=secondary
    )
    assert distinct.pop("nonlocal_grid") == asdict(secondary) | {"element_radii": []}
    assert distinct == original
    with pytest.raises(ValueError, match="requires a VV10"):
        protocol(
            3,
            basis,
            primary,
            5,
            benchmark=specification("pbe"),
            nonlocal_grid=secondary,
        )


def test_df_workspace_hint_is_explicit_in_the_benchmark_protocol() -> None:
    """A memory-limited DF population must not be mislabeled as automatic DF."""
    basis = load_r2scan3c_basis()
    auxiliary = SimpleNamespace(name="explicit-jkfit", identity="auxiliary-identity")
    fitted = replace(specification("hf"), density_fitting=True)
    automatic = protocol(
        3, basis, GridSpec(), 5, benchmark=fitted, auxiliary_basis=auxiliary
    )
    bounded = protocol(
        3,
        basis,
        GridSpec(),
        5,
        benchmark=fitted,
        auxiliary_basis=auxiliary,
        density_fitting_memory_budget_bytes=12 << 30,
    )
    assert automatic["density_fitting_memory_budget_bytes"] == 0
    assert bounded.pop("density_fitting_memory_budget_bytes") == 12 << 30
    automatic.pop("density_fitting_memory_budget_bytes")
    assert bounded == automatic
    with pytest.raises(ValueError, match="DF workspace allowance"):
        protocol(
            3,
            basis,
            GridSpec(),
            5,
            benchmark=specification("hf"),
            density_fitting_memory_budget_bytes=12 << 30,
        )


@pytest.mark.parametrize("budget", [-1, 1 << 41, True, 1.5])
def test_df_workspace_hint_rejects_invalid_capacity(budget: Any) -> None:
    """Reject unrepresentable budgets before any native or GPU preparation."""
    with pytest.raises(ValueError, match="DF workspace allowance"):
        protocol(
            3,
            load_r2scan3c_basis(),
            GridSpec(),
            5,
            benchmark=replace(specification("hf"), density_fitting=True),
            auxiliary_basis=SimpleNamespace(name="aux", identity="identity"),
            density_fitting_memory_budget_bytes=budget,
        )


def test_native_df96_extension_preserves_historical_references(
    monkeypatch: Any,
) -> None:
    """A new native size must not fabricate references or rewrite existing points."""
    import runpy

    publication = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/readme-refresh-20261015"
    )
    refresh = runpy.run_path(str(publication / "refresh.py"))
    before = refresh["combined_summary"]("df-summary.json.xz")
    campaign = refresh["read_summary"](publication / "native-summary.json.xz")
    populations = campaign["populations"]["cuda"]["wb97m-v"]
    populations[:] = [point for point in populations if point["atoms"] != 96]
    native = {"status": "completed_unchecked", "records": []}
    populations.append(
        {"atoms": 96, "protocol": refresh["df96_protocol"](), "native": native}
    )
    monkeypatch.setitem(
        refresh["combined_summary"].__globals__, "read_summary", lambda path: campaign
    )
    after = refresh["combined_summary"]("df-summary.json.xz")
    for method, points in before["methods"].items():
        current = {point["atoms"]: point for point in after["methods"][method]}
        for point in points:
            if (method, point["atoms"]) != ("wb97m-v", 96):
                assert current[point["atoms"]] == point
    added = after["methods"]["wb97m-v"][-1]
    assert added["atoms"] == 96
    assert added["protocol"]["density_fitting_memory_budget_bytes"] == 12 << 30
    assert (
        added["protocol"]["reference_rsh_exchange"]["long_range_correction_k"]
        == "exact-direct"
    )
    assert added["engines"]["native"] == native
    assert added["engines"]["reference"] == {"status": "not_run", "outcome": None}
    assert "independent_references" not in added


def test_varying_work_keeps_endpoint_median_but_failed_points_remain_gaps() -> None:
    entry = {
        "status": "measured",
        "records": [
            {
                "phase": "warm",
                "iterations": iterations,
                "scf_jk_builds": iterations + 1,
                "complete_seconds": seconds,
            }
            for iterations, seconds in [
                (1, 1.0),
                (2, 3.0),
                (1, 2.0),
                (3, 9.0),
                (1, 1.5),
            ]
        ],
    }
    assert phase_statistics(entry, "warm") == (2.0, 1.0, 7.0)
    assert timing_cell(entry, "warm") == "2.000 [1.000–9.000]†"
    assert np.isnan(phase_statistics(entry, "cold")[0])
    entry["status"] = "timeout"
    assert np.isnan(phase_statistics(entry, "warm")[0])


@pytest.mark.parametrize(
    "reason", ["operator_cancelled", "stopped_after_12_atom_energy_gate_failure"]
)
def test_operator_stop_retains_completed_cold_but_not_incomplete_warm(
    tmp_path: Path, reason: str
) -> None:
    directory = tmp_path / "96"
    directory.mkdir()
    row = dict(
        completed_sample(),
        complete_seconds=1220.0,
        forces=[[0.0, 0.0, 0.0]] * 96,
        gate=None,
        execution_gate=True,
    )
    (directory / "native.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "running",
                "stage": "warm/0",
                "protocol": {
                    "atoms": 96,
                    "repeats": 5,
                    "force_return": "host_array_in_timed_endpoint",
                },
                "records": [row],
            }
        )
    )
    outcome = {"exit_code": 143, "reason": reason}
    (directory / "native.outcome").write_text(json.dumps(outcome))
    entry = collect_point(tmp_path, 96)["engines"]["native"]
    assert entry["status"] == "interrupted"
    assert entry["outcome"] == outcome
    assert entry["records"] == [
        {key: value for key, value in row.items() if key != "forces"}
    ]
    assert phase_statistics(entry, "cold")[0] == 1220.0
    assert timing_cell(entry, "cold") == "1220.000‡"
    assert np.isnan(phase_statistics(entry, "warm")[0])


def completed_sample(phase: str = "cold", geometry: int = 0) -> dict:
    """A converged host E+F result, independent of any native implementation."""
    return {
        "phase": phase,
        "geometry": geometry,
        "repeat": 0,
        "energy": -76.0,
        "forces": [[0.0, 0.0, 0.0]] * 3,
        "converged": True,
        "status": 0,
        "iterations": 1,
        "complete_seconds": 1.0,
        "gate": True,
    }


def test_fast_budget_does_not_relabel_strict_parity_or_relax_other_norms() -> None:
    """Fast acceptance is explicit and separately reports the unchanged FP64 gate."""
    budget = {"energy_abs": 1e-5, "force_max_abs": 1e-4, "force_rms": 3e-5}
    oracle = completed_sample()
    row = completed_sample()
    row["forces"] = [[1e-6, 0, 0]] * 3
    assert not check_record(row, oracle)["gate"]
    checked = check_record(row, oracle, budget=budget)
    assert checked["gate"] and not checked["fp64_gate"]
    assert checked["force_rms_error"] > 0
    row["forces"] = [[4e-5, 4e-5, 4e-5]] * 3
    assert not check_record(row, oracle, budget=budget)["gate"]
    row["forces"] = [[0, 0, 0]] * 3
    row["energy"] += 2e-5
    assert not check_record(row, oracle, budget=budget)["gate"]


@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "failed"])
def test_fast_budget_retains_complete_endpoint_safety(invalid: str) -> None:
    budget = {"energy_abs": 1e-5, "force_max_abs": 1e-4, "force_rms": 3e-5}
    row = completed_sample()
    if invalid == "shape":
        row["forces"] = [[0, 0, 0]]
    elif invalid == "nonfinite":
        row["energy"] = float("nan")
    else:
        row["status"] = 5
    checked = check_record(row, completed_sample(), budget=budget)
    assert not checked["gate"] and checked["force_rms_error"] is None


def test_readme_runner_fast_cli_rejects_ambiguous_experiments(monkeypatch: Any) -> None:
    """Reject unsupported comparisons before loading a GPU or requiring Slurm."""
    from benchmarks.readme_omol25 import main

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    required = [
        "--atoms",
        "3",
        "--basis-file",
        "unused.json",
        "--output",
        ".artifacts/unused.json",
    ]
    for engine, options in (
        ("reference", ["--policy", "fast"]),
        ("native", ["--fast-budget", "1e-5", "1e-4", "3e-5"]),
        ("native", ["--policy", "fast", "--force-active-ao"]),
    ):
        with pytest.raises(SystemExit) as caught:
            main(specification("pbe0"), [engine, *required, *options])
        assert caught.value.code == 2


def test_partial_reference_checks_only_its_completed_geometry() -> None:
    scientific = {"atoms": 3}
    oracle = {
        "protocol": scientific,
        "status": "running",
        "records": [completed_sample()],
    }
    baselines = reference_baselines(oracle, scientific, allow_missing=True)
    assert set(baselines) == {0}
    record = {"protocol": scientific, "stage": "cold/0", "records": []}
    checked = completed_sample()
    retain_endpoint_record(record, checked, baselines[0], lambda _: None)
    assert checked["gate"] is True
    unchecked = completed_sample("moved", 1)
    retain_endpoint_record(record, unchecked, baselines.get(1), lambda _: None)
    assert unchecked["execution_gate"] is True
    assert unchecked["gate"] is None
    assert unchecked["energy_error"] is None
    assert unchecked["force_error"] is None
    with pytest.raises(ValueError, match="incomplete"):
        reference_baselines(oracle, scientific)
    with pytest.raises(ValueError, match="protocol differs"):
        reference_baselines(oracle, {"atoms": 12}, allow_missing=True)
    assert reference_baselines(None, scientific, allow_missing=True) == {}


@pytest.mark.parametrize(
    "failure", ["forces", "convergence", "status", "timing", "energy"]
)
def test_missing_reference_never_accepts_an_incomplete_endpoint(failure: str) -> None:
    row = completed_sample()
    if failure == "forces":
        row["forces"] = [[0.0, 0.0, 0.0]]
    elif failure == "convergence":
        row["converged"] = False
    elif failure == "status":
        row["status"] = 5
    elif failure == "timing":
        row["complete_seconds"] = float("inf")
    else:
        row["energy"] = float("nan")
    record = {"protocol": {"atoms": 3}, "stage": "cold/0", "records": []}
    with pytest.raises(RuntimeError, match="gate failed"):
        retain_endpoint_record(record, row, None, lambda _: None)
    assert record["records"][0]["gate"] is False


def test_phase_reduction_checks_partial_oracles_and_every_requested_repeat() -> None:
    scientific = {
        "atoms": 3,
        "repeats": 2,
        "force_return": "host_array_in_timed_endpoint",
    }
    reference = {
        "schema": SCHEMA,
        "protocol": scientific,
        "status": "running",
        "records": [completed_sample()],
    }
    rows = [
        dict(completed_sample(phase, geometry), repeat=repeat)
        for phase, geometry, count in (
            ("cold", 0, 1),
            ("warm", 0, 2),
            ("moved", 1, 1),
            ("moved-warm", 1, 2),
        )
        for repeat in range(count)
    ]
    for row in rows:
        if row["geometry"] == 1:
            row.update(gate=None, execution_gate=True)
    native = {"schema": SCHEMA, "protocol": scientific, "records": rows}
    expected = {
        "cold": "measured",
        "warm": "measured",
        "moved": "completed_unchecked",
        "moved-warm": "completed_unchecked",
    }
    assert (
        independent_phase_status(native, reference, atoms=3, schema=SCHEMA) == expected
    )
    rows.pop()
    expected["moved-warm"] = "incomplete"
    assert (
        independent_phase_status(native, reference, atoms=3, schema=SCHEMA) == expected
    )
    rows[1]["energy"] += 0.01
    expected["warm"] = "failed"
    assert (
        independent_phase_status(native, reference, atoms=3, schema=SCHEMA) == expected
    )
    rows.append(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        independent_phase_status(native, reference, atoms=3, schema=SCHEMA)


@pytest.mark.parametrize("partial_file", [False, True])
def test_reference_timeout_never_skips_native_runner(
    tmp_path: Path, partial_file: bool
) -> None:
    """Exercise the real shell orchestration with a CPU-only process fixture."""
    helper = tmp_path / "endpoint-probe"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "if sys.argv[1] == '-':\n"
        "    pathlib.Path(sys.argv[2]).write_text('{}'); sys.exit(0)\n"
        "engine = sys.argv[sys.argv.index('--method') + 2]\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "if engine == 'reference':\n"
        + (
            "    output.write_text(json.dumps({'status': 'running', 'records': []}))\n"
            if partial_file
            else ""
        )
        + "    sys.exit(124)\n"
        "assert '--allow-missing-reference' in sys.argv\n"
        "output.write_text(json.dumps({'native_ran': True}))\n"
    )
    helper.chmod(0o700)
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", str(root / "benchmarks/run_readme_dft_benchmarks.sh"), "both"],
        cwd=root,
        env={
            **os.environ,
            "SLURM_JOB_ID": "host-only-control-flow",
            "GENERATIVEQC_LIBRARY": "not-loaded",
            "README_BENCHMARK_PYTHON": str(helper),
            "README_BENCHMARK_OUTPUT": str(output),
            "README_BENCHMARK_METHODS": "wb97m-v",
            "README_BENCHMARK_SIZES": "48",
            "README_BENCHMARK_POINT_TIMEOUT": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    point = output / "wb97m-v/48"
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads((point / "reference.outcome").read_text())["exit_code"] == 124
    assert json.loads((point / "native.outcome").read_text())["exit_code"] == 0
    assert json.loads((point / "native.json").read_text())["native_ran"]


@pytest.mark.parametrize("density_fitting", (False, True))
def test_composite_protocol_retains_basis_and_corrections(
    density_fitting: bool,
) -> None:
    """Both README panels retain the defining basis and full 3c corrections."""
    spec = replace(specification("r2scan-3c"), density_fitting=density_fitting)
    result = protocol(
        3,
        load_r2scan3c_basis(),
        GridSpec(),
        5,
        benchmark=spec,
        auxiliary_basis=SimpleNamespace(
            name="explicit-jkfit", identity="auxiliary-identity"
        ),
    )
    assert spec.native_method == "r2scan-3c-rks"
    assert spec.reference_xc == "R2SCAN"
    assert result["basis"] == "def2-mTZVPP"
    assert result["representation"] == "spherical"
    assert result["density_fitting"] == density_fitting
    assert result["aos"] == 34
    assert result["d4_parameters"]["s9"] == 2.0
    assert "included in every endpoint" in result["reference_corrections"]
    assert not specification("r2scan").is_r2scan3c


@pytest.mark.parametrize("name,xc", [("b3lyp", "B3LYP"), ("m06-2x", "M06_2X")])
def test_hybrid_identity(name: str, xc: str) -> None:
    spec = specification(name)
    assert spec.reference_xc == xc
    assert spec.native_method == name + "-rks"
    assert spec.basis == "def2-SVP"


def test_composite_correction_and_host_export_are_inside_endpoint_timer() -> None:
    events = []
    gradient = SimpleNamespace(kernel=lambda: np.ones((1, 3)))
    engine = SimpleNamespace(
        kernel=lambda **kwargs: -10.0,
        converged=True,
        nuc_grad_method=lambda: gradient,
    )

    def clock() -> float:
        events.append("clock")
        return float(len(events))

    def export(value: np.ndarray) -> np.ndarray:
        events.append("export")
        return value

    def correction() -> tuple[float, np.ndarray]:
        events.append("correction")
        return -0.5, np.full((1, 3), 0.25)

    energy, forces, seconds = reference_energy_forces(
        engine,
        None,
        export=export,
        synchronize=lambda: None,
        clock=clock,
        correction=correction,
    )
    assert energy == -10.5
    np.testing.assert_array_equal(forces, np.full((1, 3), -1.25))
    assert events == ["clock", "export", "correction", "clock"]
    assert seconds == 3.0


def test_wb97mv_df_protocol_records_the_mixed_range_exchange_oracle() -> None:
    spec = specification("wb97m-v")
    auxiliary = SimpleNamespace(name="explicit-jkfit", identity="auxiliary-identity")
    basis = load_r2scan3c_basis()
    direct = protocol(3, basis, GridSpec(), 5, benchmark=spec)
    matched = protocol(
        3,
        basis,
        GridSpec(),
        5,
        benchmark=replace(spec, density_fitting=True),
        auxiliary_basis=auxiliary,
    )
    assert "reference_rsh_exchange" not in direct
    assert matched["reference_rsh_exchange"] == {
        "primary_j": "density-fitted-full-range",
        "primary_k": "density-fitted-full-range",
        "long_range_correction_k": "exact-direct",
        "long_range_correction_gradient": "exact-direct-analytic",
    }


@pytest.mark.parametrize("allow_missing", [False, True])
def test_mixed_wb97mv_rejects_plain_df_reference_even_when_optional(
    allow_missing: bool,
) -> None:
    scientific = protocol(
        3,
        load_r2scan3c_basis(),
        GridSpec(),
        5,
        benchmark=replace(specification("wb97m-v"), density_fitting=True),
        auxiliary_basis=SimpleNamespace(name="jkfit", identity="explicit-auxiliary"),
    )
    old_protocol = dict(scientific)
    del old_protocol["reference_rsh_exchange"]
    oracle = {
        "protocol": old_protocol,
        "status": "measured",
        "records": [completed_sample(), completed_sample("moved", 1)],
    }
    with pytest.raises(ValueError, match="protocol differs"):
        reference_baselines(oracle, scientific, allow_missing=allow_missing)


def test_mixed_wb97mv_partial_reference_preserves_strict_and_unchecked_gates() -> None:
    scientific = protocol(
        3,
        load_r2scan3c_basis(),
        GridSpec(),
        5,
        benchmark=replace(specification("wb97m-v"), density_fitting=True),
        auxiliary_basis=SimpleNamespace(name="jkfit", identity="explicit-auxiliary"),
    )
    oracle = {
        "protocol": scientific,
        "status": "running",
        "records": [completed_sample()],
    }
    with pytest.raises(ValueError, match="incomplete"):
        reference_baselines(oracle, scientific)
    baselines = reference_baselines(oracle, scientific, allow_missing=True)
    assert set(baselines) == {0}
    record = {"protocol": scientific, "stage": "cold/0", "records": []}
    checked = completed_sample()
    retain_endpoint_record(record, checked, baselines[0], lambda _: None)
    assert checked["gate"] is True
    unchecked = completed_sample("moved", 1)
    retain_endpoint_record(record, unchecked, baselines.get(1), lambda _: None)
    assert unchecked["execution_gate"] is True
    assert unchecked["gate"] is None
    assert unchecked["energy_error"] is None
    assert unchecked["force_error"] is None


def test_mixed_wb97mv_oracle_keeps_full_df_and_routes_lr_energy_and_force() -> None:
    """No GPU needed: verify both method-dispatch boundaries and signed weights."""
    calls = []

    def fitted_k(**kwargs: Any) -> str:
        calls.append(("df_k", kwargs))
        return "df full K"

    def fitted_gradient(**kwargs: Any) -> str:
        calls.append(("df_gradient", kwargs))
        return "df full JK derivative"

    engine = SimpleNamespace(
        mol=SimpleNamespace(spin=0),
        get_k=fitted_k,
        make_rdm1=lambda: "D0",
        nuc_grad_method=lambda: SimpleNamespace(jk_energy_per_atom=fitted_gradient),
    )

    def direct_k(owner: Any, mol: Any, dm: Any, **kwargs: Any) -> str:
        assert owner is engine
        assert mol is engine.mol
        calls.append(("exact_k", dict(dm=dm, **kwargs)))
        return "exact LR K"

    def direct_gradient(gradient: Any, **kwargs: Any) -> str:
        assert gradient is not None
        calls.append(("exact_gradient", kwargs))
        return "exact LR K derivative"

    _bind_reference_mixed_rsh_df(engine, direct_k, direct_gradient)

    assert engine.get_k(dm="D1", omega=0) == "df full K"
    assert engine.get_k(dm="D1", omega=0.3) == "exact LR K"
    assert engine.get_k(omega=0.3) == "exact LR K"
    gradient = engine.nuc_grad_method()
    assert (
        gradient.jk_energy_per_atom(dm="D1", j_factor=1, k_factor=0.15)
        == "df full JK derivative"
    )
    assert (
        gradient.jk_energy_per_atom(dm="D1", j_factor=0, k_factor=0.85, omega=0.3)
        == "exact LR K derivative"
    )
    assert [call[0] for call in calls] == [
        "df_k",
        "exact_k",
        "exact_k",
        "df_gradient",
        "exact_gradient",
    ]
    assert calls[1][1]["dm"] == "D1"
    assert calls[2][1]["dm"] == "D0"
    assert calls[-1][1]["k_factor"] == 0.85
    assert calls[-1][1]["omega"] == 0.3
    with pytest.raises(ValueError, match="positive long-range omega"):
        engine.get_k(omega=-0.3)
    with pytest.raises(ValueError, match="only permits exact LR K"):
        gradient.jk_energy_per_atom(j_factor=1, omega=0.3)


@pytest.mark.parametrize(
    "xc,spin,with_df,only_dfj",
    [
        ("PBE0", 0, object(), False),
        ("WB97M_V", 1, object(), False),
        ("WB97M_V", 0, None, False),
        ("WB97M_V", 0, object(), True),
    ],
)
def test_mixed_wb97mv_oracle_rejects_unmatched_reference(
    xc: str, spin: int, with_df: object, only_dfj: bool
) -> None:
    from benchmarks.readme_wb97mv import configure_reference_mixed_rsh_df

    engine = SimpleNamespace(
        xc=xc, mol=SimpleNamespace(spin=spin), with_df=with_df, only_dfj=only_dfj
    )
    with pytest.raises(ValueError, match="requires restricted full-J/K DF"):
        configure_reference_mixed_rsh_df(engine)


def test_omission_does_not_hide_completed_native_without_reference(
    tmp_path: Path,
) -> None:
    """Only the incomplete size is absent; an unchecked native size stays visible."""
    from tools.render_readme_dft import render

    rows = [
        {"phase": phase, "iterations": 1, "fock_builds": 1, "complete_seconds": seconds}
        for phase, seconds in [("cold", 10.0)] + [("warm", 1.0)] * 5
    ]
    native = {"status": "completed_unchecked", "records": rows}
    reference = {"status": "not_run", "records": []}
    annotation = "96: incomplete E+F omitted"
    summary = {
        "sizes": [3, 12, 48, 96],
        "density_fitting": True,
        "panel_annotations": {"wb97m-v": annotation},
        "methods": {
            "wb97m-v": [
                {
                    "atoms": 48,
                    "protocol": {"aos": 384},
                    "engines": {"native": native, "reference": reference},
                }
            ]
        },
    }
    render(summary, tmp_path)
    assert annotation in (tmp_path / "df-cold-warm.svg").read_text()
    assert (
        "GPU4PySCF cold/warm unavailable: 48 atoms"
        in (tmp_path / "df-cold-warm.svg").read_text()
    )
    table = (tmp_path / "df-timings.md").read_text()
    assert annotation in table
    assert "10.000‡ | not run | 1.000‡ | not run" in table
    assert "| ωB97M-V | 48 |" in table
    assert "| ωB97M-V | 96 |" not in table


def test_readme_chart_reports_missing_plotted_phases_not_moved_timeouts(
    tmp_path: Path,
) -> None:
    """Direct/48 can be complete in the plot despite a later moved timeout."""
    from tools.render_readme_dft import render

    rows = [
        {"phase": phase, "iterations": 1, "fock_builds": 1, "complete_seconds": secs}
        for phase, secs in [("cold", 20.0)] + [("warm", 2.0)] * 5
    ]
    native = {"status": "measured", "records": rows}
    reference_48 = {
        "status": "timeout",
        "phase_status": {"cold": "measured", "warm": "measured", "moved": "timeout"},
        "records": rows,
    }
    reference_96 = {"status": "timeout", "records": []}
    summary = {
        "sizes": [48, 96],
        "density_fitting": False,
        "methods": {
            "wb97m-v": [
                {
                    "atoms": atoms,
                    "protocol": {"aos": atoms * 8},
                    "engines": {
                        "native": native,
                        "reference": reference,
                    },
                }
                for atoms, reference in [(48, reference_48), (96, reference_96)]
            ]
        },
    }
    render(summary, tmp_path)
    svg = (tmp_path / "dft-cold-warm.svg").read_text()
    assert "GPU4PySCF cold/warm unavailable: 96 atoms" in svg
    assert "GPU4PySCF cold/warm unavailable: 48 atoms" not in svg
    assert "Reference sequence incomplete" not in svg


def test_readme_chart_identifies_only_missing_warm_phase() -> None:
    from tools.render_readme_dft import missing_plot_phase_annotations

    native = {
        "status": "measured",
        "records": [{"phase": phase} for phase in ("cold", "warm")],
    }
    reference = {
        "status": "timeout",
        "phase_status": {"cold": "measured", "warm": "timeout"},
        "records": [{"phase": "cold"}],
    }
    point = {"atoms": 96, "engines": {"native": native, "reference": reference}}
    assert missing_plot_phase_annotations([point]) == [
        "GPU4PySCF warm unavailable: 96 atoms"
    ]


@pytest.mark.parametrize("priming_fails", [False, True])
def test_cache_priming_precedes_fresh_measurements(
    tmp_path: Path, priming_fails: bool
) -> None:
    """Only compilation artifacts persist; a failed primer admits no measurement."""
    script = Path("benchmarks/results/readme-wb97mv-20261014/reproduce.sh").resolve()
    runner = tmp_path / "benchmarks/run_readme_dft_benchmarks.sh"
    runner.parent.mkdir()
    runner.write_text(
        'printf "%s\\t%s\\t%s\\t%s\\n" "$README_BENCHMARK_DENSITY_FITTING" '
        '"$README_BENCHMARK_SIZES" "$README_BENCHMARK_OUTPUT" '
        '"$README_BENCHMARK_POINT_TIMEOUT" >> calls.tsv\n'
        'mkdir -p "$README_BENCHMARK_OUTPUT"\n'
        'if [[ "$FAIL_PRIMING" == 1 && "$README_BENCHMARK_OUTPUT" == */compiler-cache-priming/* ]]; then exit 7; fi\n'
    )
    environment = os.environ | {
        "SLURM_JOB_ID": "mock-no-gpu",
        "GENERATIVEQC_LIBRARY": "mock-not-loaded.so",
        "README_BENCHMARK_MODES": "none cuda",
        "README_BENCHMARK_SIZES": "3 12 48 96",
        "FAIL_PRIMING": "1" if priming_fails else "0",
    }
    result = subprocess.run(
        ["bash", str(script), "output"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = [
        line.split("\t") for line in (tmp_path / "calls.tsv").read_text().splitlines()
    ]
    assert calls[0] == ["none", "3 12", "output/compiler-cache-priming/none", "1800"]
    if priming_fails:
        assert result.returncode != 0
        assert len(calls) == 1
        return
    assert result.returncode == 0, result.stderr
    assert calls[1:5] == [
        ["none", str(atoms), "output/none", "3600" if atoms == 96 else "1800"]
        for atoms in (3, 12, 48, 96)
    ]
    assert calls[5] == ["cuda", "3 12", "output/compiler-cache-priming/cuda", "1800"]
    assert calls[6:] == [
        ["cuda", str(atoms), "output/cuda", "1800"] for atoms in (3, 12, 48)
    ]


def test_svg_compaction_preserves_geometry_and_text_whitespace() -> None:
    """Presentation reuse must not alter path coordinates or text spacing."""
    import runpy
    from xml.etree import ElementTree

    compact = runpy.run_path("benchmarks/results/readme-wb97mv-20261014/refresh.py")[
        "compact_svg_presentation"
    ]
    paths = "".join(
        f'<path d="M {index} 0 L {index} 1" fill="#087e8b" stroke="#087e8b" stroke-width="1.7" stroke-linecap="square"/>'
        for index in range(20)
    )
    svg = f'<svg xmlns="http://www.w3.org/2000/svg"><defs/><g>{paths}</g><text>cold <tspan>3</tspan> atoms</text></svg>'
    result = compact(svg)
    assert len(result) < len(svg)
    tree = ElementTree.fromstring(result)
    retained = tree.findall(".//{http://www.w3.org/2000/svg}path")
    assert [element.attrib["d"] for element in retained] == [
        f"M {index} 0 L {index} 1" for index in range(20)
    ]
    assert all(element.attrib["stroke-linecap"] == "square" for element in retained)
    assert all("class" in element.attrib for element in retained)
    assert "stroke-width:1.7px" in result
    text = tree.find("{http://www.w3.org/2000/svg}text")
    assert text is not None and "".join(text.itertext()) == "cold 3 atoms"


def test_small_cold_audit_rejects_a_reused_measurement_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful primer must not qualify a warmed Calculator as cold."""
    import runpy

    module = runpy.run_path("benchmarks/results/readme-wb97mv-20261014/refresh.py")
    original_reader = module["read_summary"]

    def patched_reader(path: Path) -> dict[str, Any]:
        summary = original_reader(path)
        if (
            path.parent.name == "readme-wb97mv-20261014"
            and path.name == "summary.json.xz"
        ):
            summary["methods"]["wb97m-v"][0]["cache_priming"][
                "fresh_measurement_process"
            ] = False
        return summary

    monkeypatch.setitem(module["verify"].__globals__, "read_summary", patched_reader)
    with pytest.raises(ValueError, match="separate cache primer"):
        module["verify"]()

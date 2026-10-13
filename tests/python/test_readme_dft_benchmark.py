"""Scientific identities and complete timing boundaries for README comparisons."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import GridSpec, load_r2scan3c_basis

from benchmarks.readme_dft import specification
from benchmarks.readme_omol25 import protocol, reference_energy_forces
from benchmarks.readme_wb97mv import _bind_reference_mixed_rsh_df
from tools.render_readme_dft import collect_point, phase_statistics, timing_cell


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
def test_operator_stop_retains_partial_evidence_without_publishing_timings(
    tmp_path: Path, reason: str
) -> None:
    directory = tmp_path / "96"
    directory.mkdir()
    row = {"phase": "cold", "complete_seconds": 1220.0, "gate": True}
    (directory / "native.json").write_text(
        json.dumps(
            {"status": "running", "stage": "warm/0", "protocol": {}, "records": [row]}
        )
    )
    outcome = {"exit_code": 143, "reason": reason}
    (directory / "native.outcome").write_text(json.dumps(outcome))
    entry = collect_point(tmp_path, 96)["engines"]["native"]
    assert entry["status"] == "interrupted"
    assert entry["outcome"] == outcome
    assert entry["records"] == [row]
    assert np.isnan(phase_statistics(entry, "cold")[0])


def test_composite_protocol_retains_basis_and_corrections() -> None:
    spec = specification("r2scan-3c")
    result = protocol(3, load_r2scan3c_basis(), GridSpec(), 5, benchmark=spec)
    assert spec.native_method == "r2scan-3c-rks"
    assert spec.reference_xc == "R2SCAN"
    assert result["basis"] == "def2-mTZVPP"
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

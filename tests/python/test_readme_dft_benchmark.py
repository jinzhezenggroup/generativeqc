"""Scientific identities and complete timing boundaries for README comparisons."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import GridSpec, load_r2scan3c_basis

from benchmarks.readme_dft import specification
from benchmarks.readme_omol25 import protocol, reference_energy_forces
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

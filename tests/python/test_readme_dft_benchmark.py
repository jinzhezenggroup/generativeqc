"""Scientific identities and complete timing boundaries for README comparisons."""

from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import GridSpec, load_r2scan3c_basis

from benchmarks.readme_dft import specification
from benchmarks.readme_omol25 import protocol, reference_energy_forces


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

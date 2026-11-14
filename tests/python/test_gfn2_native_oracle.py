"""Complete native endpoints against independent molecular tblite fixtures.

The fixtures include Cl/Si d shells; they qualify more than the scalar H0
lowering alone. Production never imports tblite to execute these tests.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator

FIXTURE = Path(__file__).resolve().parents[1] / "data/gfn2_native_tblite.json"
CASES = json.loads(FIXTURE.read_text())["cases"]
CUTOFF_FIXTURE = json.loads(
    (FIXTURE.parent / "gfn2_sharp_cutoff_tblite.json").read_text()
)


@pytest.mark.parametrize(
    "case", [*CASES, CUTOFF_FIXTURE["case"]], ids=lambda case: case["name"]
)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_gfn2_native_energy_force_against_tblite(case: dict, device: str) -> None:
    if device == "cuda":
        if os.environ.get("GENERATIVEQC_TEST_GFN2_CUDA") != "1":
            pytest.skip("explicit GFN2 CUDA qualification is disabled")
        if not os.environ.get("SLURM_JOB_ID"):
            pytest.fail("GFN2 CUDA qualification requires a Slurm allocation")
    calculator = Calculator(
        method="gfn2-xtb",
        device=device,
        max_iterations=300,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    atoms = list(zip(case["symbols"], case["positions"], strict=True))
    result = calculator.singlepoint(atoms)
    assert result.converged
    assert result.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    assert result.energy == pytest.approx(case["energy"], rel=0, abs=5e-7)
    np.testing.assert_allclose(result.forces, case["forces"], rtol=0, atol=5e-7)
    np.testing.assert_allclose(np.sum(result.forces, axis=0), 0, rtol=0, atol=1e-10)


def test_gfn2_cutoff_fixture_distinguishes_tblite_default_model() -> None:
    """A small independent control reproduces the large campaign's force failure."""
    sharp = CUTOFF_FIXTURE["case"]
    smoothed = CUTOFF_FIXTURE["smoothed_control"]
    difference = np.asarray(sharp["forces"]) - np.asarray(smoothed["forces"])
    assert np.max(np.abs(difference)) > 5e-7


def test_gfn2_cutoff_fixture_against_live_independent_oracle() -> None:
    """Regenerate both model controls when the pinned optional oracle is installed."""
    import importlib.metadata

    tblite = pytest.importorskip("tblite.interface")
    if importlib.metadata.version("tblite") != "0.6.0":
        pytest.skip("cutoff control requires pinned tblite 0.6.0")
    case = CUTOFF_FIXTURE["case"]
    for configuration, reference in (
        (CUTOFF_FIXTURE["xtb_config"], case),
        (
            CUTOFF_FIXTURE["smoothed_control"]["xtb_config"],
            CUTOFF_FIXTURE["smoothed_control"],
        ),
    ):
        calculator = tblite.Calculator(
            "GFN2-xTB",
            np.asarray([8, 1, 1] * 3),
            np.asarray(case["positions"]),
            xtb_config=configuration,
        )
        calculator.set("accuracy", CUTOFF_FIXTURE["accuracy"])
        calculator.set("temperature", CUTOFF_FIXTURE["temperature_hartree"])
        calculator.set("max-iter", 300)
        calculator.set("verbosity", 0)
        result = calculator.singlepoint()
        assert result.get("energy") == pytest.approx(
            reference["energy"], rel=0, abs=1e-11
        )
        np.testing.assert_allclose(
            -result.get("gradient"), reference["forces"], rtol=0, atol=5e-10
        )

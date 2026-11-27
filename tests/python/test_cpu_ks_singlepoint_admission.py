"""Budgeted CPU KS singlepoints use the same admission as prepared batches."""

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, ResourceBudget


@pytest.mark.parametrize("provider", ("dense", "packed", "df"))
@pytest.mark.parametrize("properties", (("energy",), ("energy", "forces")))
def test_internal_cpu_ks_admission(
    provider: str, properties: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admit the same properties and budget without quoting an external plan."""
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    options = {
        "method": "pbe0-rks" if provider == "packed" else "pbe-rks",
        "basis_representation": "spherical",
        "ks_options": KsOptions(grid=GridSpec(12, 5, 10)),
        "initial_guess": None,
        "density_fitting": "cpu" if provider == "df" else "none",
        "cpu_packed_eri_maximum_bytes": 48 if provider == "packed" else 0,
    }
    probe = Calculator(**options)
    plan = probe.estimate_resources([atoms], properties=properties).require_feasible()
    cap = plan.peak_bytes["host"]
    with probe.prepare_batch([atoms], resource_plan=plan, warm_start=False) as batch:
        reference = batch.execute(strict=True, properties=properties).items[0]

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("singlepoint left prepared admission to quote its own plan")

    monkeypatch.setattr(Calculator, "estimate_resources", forbidden)
    short = Calculator(**options, resource_budget=ResourceBudget(host_bytes=cap - 1))
    with pytest.raises(MemoryError, match="no supported plan fits.*host"):
        short.singlepoint(atoms, properties=properties)
    exact = Calculator(**options, resource_budget=ResourceBudget(host_bytes=cap))
    result = exact.singlepoint(atoms, properties=properties)
    assert result.converged and result.iterations == reference.iterations
    assert result.energy == reference.energy
    assert (
        result.resource_diagnostics["observation"]["sampled_item_peak_host_bytes"] > 0
    )
    if "forces" in properties:
        np.testing.assert_array_equal(result.forces, reference.forces)
    else:
        assert result.forces is None

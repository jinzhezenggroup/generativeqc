"""Force admission with the native dense ERI capacity owner."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, InitialGuessSpec, ResourceBudget
from generativeqc.ks import KsOptions
from generativeqc.resources_ks import ks_resource_request, native_dense_force_request
from test_r1_native_forces import forbid_python_force

HE = [("He", (0.03, -0.02, 0.01))]
HEH = [*HE, ("H", (0.15, 0.13, 1.5))]
FORCES = ("energy", "forces")


@pytest.mark.parametrize("method", ("pbe-rks", "pbe0-rks", "b3lyp-d3(bj)-uks"))
def test_dense_native_force_budget(
    method: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Admit exact capacity and keep native derivatives across coordinate replay."""
    unrestricted = method.endswith("-uks")
    systems = [HEH] if unrestricted else [HE, HEH]
    controls = {"multiplicities": [2]} if unrestricted else {"charges": [0, 1]}
    options = {
        "method": method,
        "basis_representation": "spherical" if unrestricted else "cartesian",
        "ks_options": KsOptions(grid=GridSpec(8, 4, 8)),
        "initial_guess": InitialGuessSpec(kind="hf") if method == "pbe-rks" else None,
    }
    probe = Calculator(**options)
    plan = probe.estimate_resources(systems, properties=FORCES, **controls)
    if not any(map(native_dense_force_request, plan.requests)):
        pytest.skip("requires native dense CPU stationary force capacity")
    peak = plan.require_feasible().peak_bytes["host"]
    assert peak < 128 << 20

    class LegacyLibrary:
        """Older libraries may still use the conservative consumer reservation."""

        def __getattr__(self, name: str) -> object:
            if name == "generativeqc_resource_ks_cpu_force_fleet_v1":
                raise AttributeError(name)
            return getattr(probe._library, name)

    legacy = Calculator(**options)
    legacy._library = LegacyLibrary()
    legacy_plan = legacy.estimate_resources(systems, properties=FORCES, **controls)
    assert not any(map(native_dense_force_request, legacy_plan.requests))
    assert legacy_plan.peak_bytes["host"] > peak

    trace = tmp_path / "dense-force.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
    short = ResourceBudget(host_bytes=peak - 1)
    plan = probe.estimate_resources(
        systems, budget=short, properties=FORCES, **controls
    )
    assert plan.status == "infeasible"
    with pytest.raises(MemoryError, match="no supported plan fits.*host"):
        probe.prepare_batch(systems, resource_plan=plan, **controls)
    assert not trace.exists()

    forbid_python_force(monkeypatch)
    exact = Calculator(**options, resource_budget=ResourceBudget(host_bytes=peak))
    plan = exact.estimate_resources(systems, properties=FORCES, **controls)
    coordinates = [None] * (len(systems) - 1) + [[HEH[0][1], (0.16, 0.13, 1.51)]]
    with (
        exact.prepare_batch(systems, resource_plan=plan, **controls) as batch,
        probe.prepare_batch(systems, **controls) as reference,
    ):
        assert batch._native_dft_force_eligible()
        for positions in (None, None, coordinates):
            actual = batch.execute(positions, strict=True, properties=FORCES)
            expected = reference.execute(positions, strict=True, properties=FORCES)
            assert actual.energies == pytest.approx(expected.energies, abs=2e-10)
            for item, control in zip(actual.items, expected.items, strict=True):
                np.testing.assert_allclose(
                    item.forces, control.forces, atol=2e-8, rtol=0
                )
                assert {
                    key: value
                    for key, value in (item.initial_guess or {}).items()
                    if key != "preparation_seconds"
                } == {
                    key: value
                    for key, value in (control.initial_guess or {}).items()
                    if key != "preparation_seconds"
                }
                if method == "pbe-rks":
                    assert item.initial_guess is not None
            observed = batch.resource_diagnostics["observation"]
            assert 0 < observed["sampled_item_peak_host_bytes"] <= peak
            assert batch.resource_diagnostics["generated_force"] == []
        assert (
            batch.execute(coordinates, strict=True, properties=("energy",))
            .items[0]
            .forces
            is None
        )
        assert (
            batch.execute(coordinates, strict=True, properties=FORCES).items[0].forces
            is not None
        )
        if trace.exists():
            trace.unlink()
        monkeypatch.setattr(
            batch, "_native_dft_force_eligible", lambda index=None: False
        )
        with pytest.raises(NotImplementedError, match="cannot fund compatibility"):
            batch.execute(strict=True, properties=FORCES)
        assert not trace.exists()

    scalar_controls = {"multiplicity": 2} if unrestricted else {"charge": 1}
    single = exact.singlepoint(HEH, properties=FORCES, **scalar_controls)
    unbounded = probe.singlepoint(HEH, properties=FORCES, **scalar_controls)
    np.testing.assert_allclose(single.forces, unbounded.forces, atol=2e-8, rtol=0)
    assert single.resource_diagnostics["generated_force"] == []


def test_implicit_dense_force_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Internal budgets also certify the prepared force entry point."""
    options = {
        "method": "pbe0-rks",
        "ks_options": KsOptions(grid=GridSpec(8, 4, 8)),
        "initial_guess": None,
    }
    plan = Calculator(**options).estimate_resources([HE])
    if not native_dense_force_request(plan.requests[0]):
        pytest.skip("requires native dense CPU stationary force capacity")
    calculator = Calculator(
        **options, resource_budget=ResourceBudget(host_bytes=plan.peak_bytes["host"])
    )
    forbid_python_force(monkeypatch)
    with calculator.prepare_batch([HE]) as batch:
        assert batch._dense_force_plan and batch._fixed_ks_resource_replay is not None
        assert batch.execute(strict=True, properties=FORCES).items[0].forces is not None
    from generativeqc.batch import PreparedBatch

    monkeypatch.setattr(
        PreparedBatch, "_native_dft_force_eligible", lambda self, index=None: False
    )
    with pytest.raises(NotImplementedError, match="every prepared item"):
        calculator.prepare_batch([HE])


def test_energy_only_replay_cannot_claim_dense_force_admission() -> None:
    """A force upgrade needs the conservative budget without force preparation."""
    options = {"method": "pbe-rks", "ks_options": KsOptions(grid=GridSpec(8, 4, 8))}
    probe = Calculator(**options)
    force_plan = probe.estimate_resources([HE], properties=FORCES)
    if not native_dense_force_request(force_plan.requests[0]):
        pytest.skip("requires native dense CPU stationary force capacity")
    cap = ResourceBudget(host_bytes=force_plan.peak_bytes["host"])
    calculator = Calculator(**options, resource_budget=cap)
    with calculator.prepare_batch([HE]) as batch:
        assert batch.execute(strict=True).items[0].forces is None
        with pytest.raises(
            MemoryError, match="forces exceed the prepared resource budget"
        ):
            batch.execute(strict=True, properties=FORCES)
        assert batch.execute(strict=True).items[0].forces is None
    with calculator.prepare_batch([HE], ks_export=False) as batch:
        assert batch.execute(strict=True).items[0].forces is None


@pytest.mark.parametrize("status", (0, -1, 2))
def test_dense_force_standalone_query_status(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only absence of an inventory can trigger the legacy force allowance."""
    probe = Calculator(method="pbe-rks")
    plan = probe.estimate_resources([HE], properties=FORCES)
    if not native_dense_force_request(plan.requests[0]):
        pytest.skip("requires native dense CPU stationary force capacity")

    def query(*_args: object) -> int:
        return status

    monkeypatch.setattr(
        probe._library, "generativeqc_resource_ks_cpu_force_fleet_v1", query
    )
    if status == 2:
        request = ks_resource_request([HE], include_forces=True, library=probe._library)
        assert not native_dense_force_request(request)
    else:
        with pytest.raises(ValueError, match="native CPU KS fleet inventory"):
            ks_resource_request([HE], include_forces=True, library=probe._library)
        with pytest.raises(ValueError, match="native CPU KS fleet inventory"):
            probe.prepare_batch([HE], resource_plan=plan)


def test_installed_cli_and_python_admit_the_same_dense_force_bytes(
    tmp_path: Path,
) -> None:
    """Compare exact and one-byte-short admission under identical controls."""
    prefix = os.environ.get("GENERATIVEQC_NATIVE_SDK_PREFIX")
    if prefix is None:
        pytest.skip("requires an explicitly installed native SDK")
    calculator = Calculator(
        method="pbe0-rks",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GridSpec(8, 4, 8)),
        initial_guess=None,
        energy_tolerance=1e-9,
        density_tolerance=1e-5,
    )
    cli = Path(prefix) / "bin" / "generativeqc"
    assert (
        Path(calculator._library._name).resolve().is_relative_to(Path(prefix).resolve())
    )
    plan = calculator.estimate_resources([HEH], charges=[1], properties=FORCES)
    if not any(map(native_dense_force_request, plan.requests)):
        pytest.skip("requires native dense CPU stationary force capacity")
    peak = plan.require_feasible().peak_bytes["host"]
    xyz = tmp_path / "heh.xyz"
    xyz.write_text("2\nBohr\nHe 0.03 -0.02 0.01\nH 0.15 0.13 1.5\n")
    trace = tmp_path / "scf.jsonl"
    runtime = dict(
        os.environ,
        PATH="/nonexistent",
        PYTHONHOME="/nonexistent",
        PYTHONPATH="/nonexistent",
        GENERATIVEQC_DF_PROGRESS_TRACE=str(trace),
    )
    args = (
        str(xyz),
        "--json",
        "--units",
        "bohr",
        "--method",
        "pbe0-rks",
        "--representation",
        "spherical",
        "--charge",
        "1",
        "--forces",
        "--grid-radial-points",
        "8",
        "--grid-polar-points",
        "4",
        "--grid-azimuth-points",
        "8",
    )
    estimate = json.loads(
        subprocess.check_output([cli, "resources", *args], env=runtime, timeout=30)
    )
    assert estimate["peak_bytes"]["host"] == peak
    refused = subprocess.run(
        [cli, "run", *args, "--host-bytes", str(peak - 1)],
        env=runtime,
        timeout=30,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0 and "exceeds host budget" in refused.stderr
    assert not trace.exists()
    admitted = json.loads(
        subprocess.check_output(
            [cli, "run", *args, "--host-bytes", str(peak)],
            env=runtime,
            timeout=30,
        )
    )
    result = calculator.singlepoint(HEH, charge=1, properties=FORCES)
    assert admitted["energy_hartree"] == pytest.approx(result.energy, abs=2e-10)
    np.testing.assert_allclose(
        admitted["forces_hartree_per_bohr"], result.forces, atol=2e-8, rtol=0
    )

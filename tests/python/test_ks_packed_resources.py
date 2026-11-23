"""Native value storage must agree across planning, preparation, and replay."""

import json
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, ResourceBudget, estimate_ks_resources
from generativeqc.ks import KsOptions
from test_r1_native_forces import forbid_python_force

HE = [("He", (0.03, -0.02, 0.01))]
HEH = [*HE, ("H", (0.15, 0.13, 1.5))]
GRID = KsOptions(grid=GridSpec(8, 4, 8))


@pytest.mark.parametrize("method", ("pbe0-rks", "b3lyp-d3(bj)-uks"))
def test_exact_packed_batch_budget(
    method: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unrestricted = method.endswith("-uks")
    systems = [HEH] if unrestricted else [HE, HEH]
    controls = {"multiplicities": [2]} if unrestricted else {"charges": [0, 1]}
    # One spherical s AO per atom: 1 or 6 FP64 pair-pair orbits per item.
    exact_provider = 48 if unrestricted else 56
    options = {
        "method": method,
        "basis_representation": "spherical",
        "ks_options": GRID,
        "initial_guess": None,
        "cpu_packed_eri_maximum_bytes": exact_provider,
    }
    probe = Calculator(**options).estimate_resources(systems, **controls)
    peak = probe.require_feasible().peak_bytes["host"]
    exact = Calculator(**options, resource_budget=ResourceBudget(host_bytes=peak))
    dense = Calculator(**dict(options, cpu_packed_eri_maximum_bytes=0))
    with pytest.raises(ValueError, match="inputs differ from the global resource plan"):
        dense.prepare_batch(systems, resource_plan=probe, **controls)
    more_provider = Calculator(
        **dict(options, cpu_packed_eri_maximum_bytes=exact_provider + 1)
    )
    assert (
        more_provider.estimate_resources(systems, **controls).peak_bytes["host"] == peak
    )
    with pytest.raises(ValueError, match="inputs differ from the global resource plan"):
        more_provider.prepare_batch(systems, resource_plan=probe, **controls)

    trace = tmp_path / "packed.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
    short_host = Calculator(
        **options, resource_budget=ResourceBudget(host_bytes=peak - 1)
    )
    short_provider = Calculator(
        **dict(options, cpu_packed_eri_maximum_bytes=exact_provider - 1),
        resource_budget=ResourceBudget(host_bytes=peak),
    )
    for calculator, message in (
        (short_host, "no supported plan fits.*host"),
        (short_provider, "packed spherical ERI capacity exceeded"),
    ):
        assert calculator.estimate_resources(systems, **controls).status == "infeasible"
        with pytest.raises(MemoryError, match=message):
            calculator.prepare_batch(systems, **controls)
        assert not trace.exists()

    coordinates = [None] * (len(systems) - 1) + [[HEH[0][1], (0.16, 0.13, 1.51)]]
    with exact.prepare_batch(systems, **controls) as actual:
        prep = actual.resource_diagnostics["preparation"]
        assert 0 < prep["sampled_item_peak_host_bytes"] <= peak
        with dense.prepare_batch(systems, **controls) as reference:
            for geometry in (None, coordinates, coordinates):
                observed = actual.execute(geometry, strict=True)
                expected = reference.execute(
                    geometry, strict=True, properties=("energy",)
                )
                assert observed.energies == pytest.approx(expected.energies, abs=2e-10)
                assert all(item.forces is None for item in observed.items)
                replay = actual.resource_diagnostics["observation"]
                assert 0 < replay["sampled_item_peak_host_bytes"] <= peak
        with pytest.raises(NotImplementedError, match="energy-only plans"):
            actual.execute(properties=("energy", "forces"))
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [
        event["value"]
        for event in events
        if event.get("key") == "packed_eri_batch_planned_bytes"
    ] == [exact_provider]

    with Calculator(**options).prepare_batch(
        systems, resource_plan=probe, **controls
    ) as external:
        assert all(item.forces is None for item in external.execute(strict=True).items)


def test_packed_plan_does_not_retry_dense_inventory() -> None:
    dense = Calculator()
    # The default symbolic/portable lane remains available. Explicit packed
    # storage must be sized by the current library, not by that lane.
    assert estimate_ks_resources([HEH], charges=[1]).status == "feasible"
    with pytest.raises(NotImplementedError, match="native KS program and inventory"):
        estimate_ks_resources(
            [HEH],
            charges=[1],
            library=object(),
            basis_representation="spherical",
            cpu_packed_eri_maximum_bytes=48,
        )
    with pytest.raises(ValueError, match="real spherical"):
        estimate_ks_resources(
            [HEH],
            charges=[1],
            library=dense._library,
            basis_representation="cartesian",
            cpu_packed_eri_maximum_bytes=48,
        )


@pytest.mark.parametrize("method", ("pbe0-rks", "b3lyp-d3(bj)-uks"))
def test_packed_native_force_plan(
    method: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A counted packed batch uses native derivatives after cold and moved SCF."""
    controls = {"multiplicities": [2]} if method.endswith("-uks") else {"charges": [1]}
    options = {
        "method": method,
        "basis_representation": "spherical",
        "ks_options": GRID,
        "initial_guess": None,
    }
    systems = [HEH]
    forces = ("energy", "forces")
    probe = Calculator(**options, cpu_packed_eri_maximum_bytes=48)
    try:
        draft = probe.estimate_resources(systems, properties=forces, **controls)
    except NotImplementedError as error:
        if "this build omits native CPU stationary forces" not in str(error):
            raise
        pytest.skip(str(error))
    peak = draft.require_feasible().peak_bytes["host"]
    assert peak > probe.estimate_resources(systems, **controls).peak_bytes["host"]

    trace = tmp_path / "packed-force.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
    refuse = (
        (peak - 1, 48, "no supported plan fits.*host"),
        (peak, 47, "packed spherical ERI capacity exceeded"),
    )
    for host_cap, provider_cap, message in refuse:
        short = Calculator(
            **options,
            cpu_packed_eri_maximum_bytes=provider_cap,
            resource_budget=ResourceBudget(host_bytes=host_cap),
        )
        plan = short.estimate_resources(systems, properties=forces, **controls)
        assert plan.status == "infeasible"
        with pytest.raises(MemoryError, match=message):
            short.prepare_batch(systems, resource_plan=plan, **controls)
        assert not trace.exists()

    forbid_python_force(monkeypatch)
    exact = Calculator(
        **options,
        cpu_packed_eri_maximum_bytes=48,
        resource_budget=ResourceBudget(host_bytes=peak),
    )
    dense = Calculator(**options)
    plan = exact.estimate_resources(systems, properties=forces, **controls)
    with pytest.raises(ValueError, match="require ks_export=True"):
        exact.prepare_batch(systems, resource_plan=plan, ks_export=False, **controls)
    with (
        exact.prepare_batch(systems, resource_plan=plan, **controls) as prepared,
        dense.prepare_batch(systems, **controls) as reference,
    ):
        assert prepared._native_dft_force_eligible()
        resource_requests = []
        original_request = exact._resource_request

        def record_request(*args: object, **kwargs: object) -> object:
            resource_requests.append(kwargs["properties"])
            return original_request(*args, **kwargs)

        monkeypatch.setattr(exact, "_resource_request", record_request)
        value_only = prepared.execute(strict=True, properties=("energy",))
        assert value_only.items[0].forces is None
        assert prepared.resource_plan == plan
        for coordinates in (None, [[HEH[0][1], (0.16, 0.13, 1.51)]]):
            result = prepared.execute(coordinates, strict=True).items[0]
            expected = reference.execute(
                coordinates, strict=True, properties=forces
            ).items[0]
            assert result.energy == pytest.approx(expected.energy, abs=2e-10)
            np.testing.assert_allclose(
                result.forces, expected.forces, atol=2e-8, rtol=0
            )
            observation = prepared.resource_diagnostics["observation"]
            assert 0 < observation["sampled_item_peak_host_bytes"] <= peak
            assert prepared.resource_diagnostics["generated_force"] == []
        assert not resource_requests
        monkeypatch.setenv("GQC_TEST_REPLAY_RESOURCE_INPUTS", "changed")
        prepared.execute(strict=True)
        prepared.execute(strict=True)
        assert len(resource_requests) == 1
        with monkeypatch.context() as changed:
            changed.setattr(exact, "_cpu_packed_eri_maximum_bytes", 49)
            with pytest.raises(ValueError, match="resource inputs .* changed"):
                prepared.execute(strict=True)

    scalar_controls = (
        {"multiplicity": controls["multiplicities"][0]}
        if "multiplicities" in controls
        else {"charge": controls["charges"][0]}
    )
    single = exact.singlepoint(HEH, properties=forces, **scalar_controls)
    expected_single = dense.singlepoint(HEH, properties=forces, **scalar_controls)
    assert single.energy == pytest.approx(expected_single.energy, abs=2e-10)
    np.testing.assert_allclose(single.forces, expected_single.forces, atol=2e-8, rtol=0)

"""Direct CPU cold-start inventory follows native policy and admission."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from generativeqc import (
    Calculator,
    ElementBasis,
    GridSpec,
    InitialGuessSpec,
    KsOptions,
    ResourceBudget,
)
from generativeqc._model_resolution import snapshot_basis

SYMBOL = "generativeqc_resource_preliminary_cpu_shape_v1"
AUTO_SYMBOL = "generativeqc_resource_preliminary_cpu_shape_v2"
H2 = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))]
WATER = [
    ("O", (0.01, -0.04, 0.02)),
    ("H", (1.42, 0.13, 1.06)),
    ("H", (-1.37, -0.09, 1.10)),
]
PROPERTIES = ("energy", "forces")


@pytest.mark.parametrize("kind,provider", [("minao", 3), ("huckel", 5)])
def test_automatic_ragged_layout_and_runtime_agree(
    kind: str, provider: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Domain fallback, local skip and total refusal have distinct effects."""
    basis = snapshot_basis("sto-3g")
    # A synthetic one-shell Ca18+ keeps the out-of-domain calculation small
    # while preserving the real all-electron Z=20 topology and singlet spin.
    basis = replace(
        basis,
        name="cold-start-topology-fixture",
        elements=(*basis.elements, ElementBasis(20, basis.by_element[2].shells)),
    )
    helium = [("He", (0.0, 0.0, 0.0))]
    systems = [helium, WATER, [("Ca", (0.0, 0.0, 0.0))]]
    charges = (0, 0, 18)
    core = Calculator("rhf", basis, initial_guess=None)
    if not hasattr(core._library, AUTO_SYMBOL):
        pytest.skip("requires the native automatic CPU preliminary inventory")
    explicit = Calculator("rhf", basis, initial_guess=InitialGuessSpec(kind))
    local = explicit.singlepoint(helium).initial_guess["preparation_numeric_capacity"]
    target = core.estimate_resources(systems, charges=charges).require_feasible()
    target_peak = target.peak_bytes["host"]
    calc = Calculator("rhf", basis)
    # Select both already-supported native policies with the same local limit.
    # Public default discovery is deliberately unchanged by this inventory.
    calc._initial_guess = replace(
        calc.initial_guess, kind=kind, maximum_numeric_bytes=local
    )

    def forbidden(*_args: Any) -> int:
        pytest.fail("automatic inventory re-entered the split system query")

    monkeypatch.setattr(
        calc._library,
        f"generativeqc_initial_guess_auto_system_kind_v{1 if kind == 'minao' else 2}",
        forbidden,
    )
    plan = calc.estimate_resources(systems, charges=charges).require_feasible()
    peak = plan.peak_bytes["host"]
    schedule = json.loads(plan.requests[0].identity.schedule)
    assert schedule["initial_guess_inventory"] == "native-cpu-direct-preliminary-v2"
    assert schedule["initial_guess_dispositions"] == [0, 1, 3]
    assert schedule["initial_guess_resolved_kinds"] == [provider, provider, 0]
    assert peak == target_peak + local
    calc._initial_guess = replace(calc.initial_guess, maximum_numeric_bytes=local - 1)
    skipped = calc.estimate_resources(systems, charges=charges).require_feasible()
    assert skipped.peak_bytes["host"] == target_peak
    assert json.loads(skipped.requests[0].identity.schedule)[
        "initial_guess_dispositions"
    ] == [1, 1, 3]
    calc._initial_guess = replace(calc.initial_guess, maximum_numeric_bytes=local)
    below = calc.estimate_resources(
        systems, charges=charges, budget=ResourceBudget(host_bytes=peak - 1)
    )
    with pytest.raises(MemoryError, match="exceed|fits"):
        below.require_feasible()
    exact = calc.estimate_resources(
        systems, charges=charges, budget=ResourceBudget(host_bytes=peak)
    ).require_feasible()
    moved = [(x, y, z + 0.02 * i) for i, (_, (x, y, z)) in enumerate(WATER)]
    with calc.prepare_batch(systems, charges=charges, resource_plan=exact) as batch:
        for coordinates in (None, [None, None, None], [None, moved, None]):
            results = batch.execute(coordinates, strict=True)
            assert all(item.converged for item in results.items)
            if coordinates is None:
                used, skipped, declined = (item.initial_guess for item in results.items)
                assert used["outcome"] == "used"
                assert used["preparation_numeric_capacity"] == local
                assert skipped["outcome"] == "budget_skipped"
                assert skipped["preliminary_fock_builds"] == 0
                assert declined is None or declined["outcome"] == "disabled"
            geometries = (
                systems
                if coordinates is None
                else [
                    original
                    if updated is None
                    else list(zip((atom[0] for atom in original), updated, strict=True))
                    for original, updated in zip(systems, coordinates, strict=True)
                ]
            )
            for item, geometry, charge in zip(
                results.items, geometries, charges, strict=True
            ):
                assert item.energy == pytest.approx(
                    core.singlepoint(geometry, charge=charge).energy, abs=2e-9
                )


def test_bad_automatic_inventory_cannot_restart_the_legacy_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calc = Calculator("rhf")
    library = calc._library
    if not hasattr(library, AUTO_SYMBOL):
        pytest.skip("requires the native automatic CPU preliminary inventory")
    native = calc.estimate_resources([H2]).require_feasible()
    assert "initial_guess_inventory" in json.loads(native.requests[0].identity.schedule)

    def reject(*_args: Any) -> int:
        return -1

    def malformed(*args: Any) -> int:
        for index, value in enumerate(
            (0, calc.initial_guess.maximum_numeric_bytes, 0, 0, 3, 5)
        ):
            args[-2][index] = value
        return 0

    for query in (reject, malformed):
        monkeypatch.setattr(library, AUTO_SYMBOL, query)
        with pytest.raises(ValueError, match="native CPU preliminary inventory"):
            calc.estimate_resources([H2])


def test_ragged_explicit_layout_excludes_budget_skipped_members() -> None:
    """At a budget that admits He but not water, retain only the admitted seed."""
    helium = [[("He", (0.0, 0.0, 0.0))]]
    systems = [*helium, WATER]
    target = Calculator("rhf", initial_guess=None)
    if not hasattr(target._library, SYMBOL):
        pytest.skip("requires the native explicit CPU preliminary inventory")
    probe = Calculator("rhf", initial_guess=InitialGuessSpec("huckel"))
    capacity = probe.singlepoint(helium[0]).initial_guess[
        "preparation_numeric_capacity"
    ]
    policy = InitialGuessSpec("huckel", maximum_numeric_bytes=capacity)
    calc = Calculator("rhf", initial_guess=policy)
    plan = calc.estimate_resources(systems).require_feasible()
    schedule = json.loads(plan.requests[0].identity.schedule)
    assert schedule["initial_guess_dispositions"] == [0, 1]
    peak = plan.peak_bytes["host"]
    assert peak == target.estimate_resources(systems).peak_bytes["host"] + capacity
    bounded = Calculator(
        "rhf", initial_guess=policy, resource_budget=ResourceBudget(host_bytes=peak)
    )
    bounded_plan = bounded.estimate_resources(systems).require_feasible()
    with bounded.prepare_batch(systems, resource_plan=bounded_plan) as batch:
        for coordinates in (
            None,
            [None, None],
            [None, [(x, y, z + 0.001 * i) for i, (_, (x, y, z)) in enumerate(WATER)]],
        ):
            result = batch.execute(coordinates, strict=True)
            assert all(item.converged for item in result.items)
            if coordinates is None:
                assert [item.initial_guess["outcome"] for item in result.items] == [
                    "used",
                    "budget_skipped",
                ]


def test_legacy_symbol_absence_is_distinct_from_native_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an older image may keep the conservative preliminary planner."""
    calc = Calculator(
        "rhf", initial_guess=InitialGuessSpec("hf", maximum_numeric_bytes=1)
    )
    library = calc._library
    if not hasattr(library, SYMBOL):
        pytest.skip("requires the native explicit CPU preliminary inventory")
    native = calc.estimate_resources([H2]).require_feasible()

    class Legacy:
        def __getattr__(self, name: str) -> Any:
            if name == SYMBOL:
                raise AttributeError(name)
            return getattr(library, name)

    calc._library = Legacy()
    legacy = calc.estimate_resources([H2]).require_feasible()
    assert legacy.peak_bytes["host"] > native.peak_bytes["host"]
    assert "initial_guess_inventory" not in json.loads(
        legacy.requests[0].identity.schedule
    )
    calc._library = library

    def reject(*_args: Any) -> int:
        return -1

    def malformed(*args: Any) -> int:
        for index, value in enumerate((8, 1, 8, 16, 1)):
            args[-2][index] = value
        return 0

    for query in (reject, malformed):
        monkeypatch.setattr(library, SYMBOL, query)
        with pytest.raises(ValueError, match="native CPU preliminary inventory"):
            calc.estimate_resources([H2])


@pytest.mark.parametrize("kind", ("hf", "lda", "minao", "huckel", "auto"))
def test_installed_cli_python_preliminary_capacity_matches(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both frontends use the same total host cutoff and preserve local fallback."""
    prefix = os.environ.get("GENERATIVEQC_NATIVE_SDK_PREFIX")
    if prefix is None:
        pytest.skip("requires an explicitly installed native SDK")
    options = {
        "method": "pbe-rks",
        "ks_options": KsOptions(grid=GridSpec(12, 6, 12)),
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    probe = (
        Calculator(**options)
        if kind == "auto"
        else Calculator(**options, initial_guess=InitialGuessSpec(kind))
    )
    library_path = Path(probe._library._name).resolve()
    assert library_path.is_relative_to(Path(prefix).resolve())
    if not hasattr(probe._library, SYMBOL):
        pytest.skip("requires the native explicit CPU preliminary inventory")
    if kind == "auto" and not hasattr(probe._library, AUTO_SYMBOL):
        pytest.skip("requires the native automatic CPU preliminary inventory")
    cli = Path(prefix) / "bin/generativeqc"
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nBohr\nH 0 0 0\nH 0 0 1.4\n")
    args = (
        str(xyz),
        "--units",
        "bohr",
        "--method",
        "pbe-rks",
        "--forces",
        "--json",
        "--grid-radial-points",
        "12",
        "--grid-polar-points",
        "6",
        "--grid-azimuth-points",
        "12",
        "--energy-tolerance",
        "1e-12",
        "--density-tolerance",
        "1e-10",
        "--initial-guess",
        kind,
    )
    trace = tmp_path / "scf.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
    runtime = dict(
        os.environ,
        PATH="/nonexistent",
        PYTHONPATH="/nonexistent",
        PYTHONHOME="/nonexistent",
    )

    def cli_json(command: str, *limits: str) -> dict:
        return json.loads(
            subprocess.check_output(
                [cli, command, *args, *limits], env=runtime, timeout=30
            )
        )

    required = cli_json("resources")["initial_guess_inventory"][
        "required_numeric_bytes"
    ]
    for local_cap, expected in ((required, "used"), (required - 1, "budget_skipped")):
        limits = ("--initial-guess-bytes", str(local_cap))
        cli_plan = cli_json("resources", *limits)
        peak = cli_plan["peak_bytes"]["host"]
        guess_options = (
            {}
            if kind == "auto"
            else {
                "initial_guess": InitialGuessSpec(kind, maximum_numeric_bytes=local_cap)
            }
        )
        calc = Calculator(
            **options,
            **guess_options,
            resource_budget=ResourceBudget(host_bytes=peak),
        )
        if kind == "auto":
            # Python exposes a default automatic candidate, not a public knob
            # for its local cap. Set only that cap to exercise both boundaries.
            calc._initial_guess = replace(
                calc.initial_guess, maximum_numeric_bytes=local_cap
            )
            assert (
                cli_plan["initial_guess_inventory"]["resolved_kind"]
                == calc.initial_guess.kind
            )
        plan = calc.estimate_resources([H2], properties=PROPERTIES).require_feasible()
        assert plan.peak_bytes["host"] == peak
        short = calc.estimate_resources(
            [H2], budget=ResourceBudget(host_bytes=peak - 1), properties=PROPERTIES
        )
        with pytest.raises(MemoryError, match="exceed|fits"):
            short.require_feasible()
        assert not trace.exists()
        result = calc.singlepoint(H2, properties=PROPERTIES)
        match = cli_json("run", *limits, "--host-bytes", str(peak))
        if kind == "auto":
            assert match["initial_guess"]["resolved_kind"] == calc.initial_guess.kind
        assert match["energy_hartree"] == pytest.approx(result.energy, abs=2e-10)
        assert tuple(map(tuple, match["forces_hartree_per_bohr"])) == tuple(
            pytest.approx(vector, abs=1e-9) for vector in result.forces
        )
        assert (
            match["initial_guess"]["outcome"]
            == result.initial_guess["outcome"]
            == expected
        )
        assert result.initial_guess["preparation_numeric_capacity"] == required
        trace.unlink(missing_ok=True)

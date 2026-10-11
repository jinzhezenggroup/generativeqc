"""Guarded defaults share admission with planning and retain an explicit rollback."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import (
    BasisProvenance,
    BasisSet,
    BasisShell,
    Calculator,
    ElementBasis,
    GridSpec,
    InitialGuessSpec,
    KsOptions,
    Primitive,
    ResourceBudget,
    Shell,
    _native,
)
from generativeqc.initial_guess import (
    initial_guess_for_systems,
    with_initial_guess_resources,
)

if TYPE_CHECKING:
    from pathlib import Path

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]
HEAVY = [("Ca", (0.0, 0.0, 0.0))]


@pytest.fixture(scope="module")
def native() -> None:
    library = _native.load_library(device="cpu")
    if not hasattr(library, "generativeqc_initial_guess_options_version"):
        pytest.skip("MINAO native build required")


@pytest.mark.parametrize("systems", [[WATER], [HEAVY], [WATER, HEAVY]])
def test_batch_element_admission_is_conservative(systems: list) -> None:
    policy = InitialGuessSpec("minao")
    calc = SimpleNamespace(
        _initial_guess=policy, _automatic_initial_guess=True, _basis="sto-3g"
    )
    assert initial_guess_for_systems(calc, systems) is (
        policy if systems == [WATER] else None
    )
    if systems != [WATER]:
        target = object()
        assert with_initial_guess_resources(target, calc, systems, None, None) is target
    calc._automatic_initial_guess = False
    assert initial_guess_for_systems(calc, systems) is policy


def test_automatic_ecp_admission_keeps_hcore(monkeypatch: pytest.MonkeyPatch) -> None:
    from generativeqc import ecp

    monkeypatch.setattr(ecp, "resolve_ecp", lambda basis, atoms: ((2, 0, 0), ()))
    calc = SimpleNamespace(
        _initial_guess=InitialGuessSpec("minao"),
        _automatic_initial_guess=True,
        _basis="sto-3g",
    )
    assert initial_guess_for_systems(calc, [WATER]) is None


@pytest.mark.parametrize("method", ["rhf", "pbe-rks", "pbe0-rks"])
def test_cpu_default_and_explicit_hcore(native: None, method: str) -> None:
    options = (
        {"ks_options": KsOptions(grid=GridSpec(16, 8, 16))}
        if method.endswith("-rks")
        else {}
    )
    auto = Calculator(method, "sto-3g", device="cpu", **options)
    core = Calculator(method, "sto-3g", device="cpu", initial_guess=None, **options)
    assert auto.initial_guess == InitialGuessSpec("minao")
    assert core.initial_guess is None
    assert auto.capabilities == core.capabilities
    assert auto._method_descriptor(systems=[WATER]).initial_guess.contents.kind == 4
    assert not core._method_descriptor(systems=[WATER]).initial_guess
    assert (
        auto._method_descriptor(systems=[WATER, HEAVY]).initial_guess.contents.kind == 4
    )
    automatic_plan = auto._resource_request(
        [WATER], properties=("energy",), ks_options=auto.ks_options
    )
    explicit = Calculator(
        method,
        "sto-3g",
        device="cpu",
        initial_guess=InitialGuessSpec("minao"),
        **options,
    )
    explicit_plan = explicit._resource_request(
        [WATER], properties=("energy",), ks_options=explicit.ks_options
    )
    assert automatic_plan.candidates == explicit_plan.candidates
    schedule = json.loads(automatic_plan.identity.schedule)
    assert schedule["initial_guess_execution"] == "native-auto-cpu-v1"
    assert schedule["initial_guess_resolved_kinds"] == [3]


@pytest.mark.parametrize(
    "method,options", [("uhf", {}), ("rhf", {"density_fitting": "cpu"}), ("mp2", {})]
)
def test_unsupported_cpu_domain_preserves_hcore(
    native: None, method: str, options: dict
) -> None:
    auto = Calculator(method, "sto-3g", device="cpu", **options)
    core = Calculator(method, "sto-3g", device="cpu", initial_guess=None, **options)
    assert auto.initial_guess is None
    assert auto.capabilities == core.capabilities


@pytest.mark.parametrize("method", ["rhf", "pbe-rks", "pbe0-rks"])
def test_mixed_native_batch_resolves_each_item(native: None, method: str) -> None:
    """Both two-electron ions fit the same one-AO owner, but Ca is outside MINAO."""
    basis = (Shell(0, 0, (Primitive(1.0, 1.0),)),)
    systems = [[("He", (0.0, 0.0, 0.0))], HEAVY]
    options = (
        {"ks_options": KsOptions(grid=GridSpec(16, 8, 16))}
        if method.endswith("-rks")
        else {}
    )
    calc = Calculator(method, basis, device="cpu", **options)
    with Calculator(
        method, basis, device="cpu", initial_guess=None, **options
    ).prepare_batch(systems, charges=[0, 18]) as core:
        reference = core.execute(strict=True, properties=("energy",))
    with calc.prepare_batch(systems, charges=[0, 18]) as batch:
        result = batch.execute(strict=True, properties=("energy",))
        warm = batch.execute(strict=True, properties=("energy",))
        moved = batch.execute(
            [[(0.1, 0.2, 0.3)], [(0.2, 0.3, 0.4)]],
            strict=True,
            properties=("energy",),
        )
        failed = batch.execute([[(float("nan"), 0, 0)], None], properties=("energy",))
        replay = batch.execute(strict=True, properties=("energy",))
    assert all(item.converged for item in result.items)
    assert result.items[0].initial_guess["kind"] == "minao"
    assert result.items[0].initial_guess["outcome"] == "used"
    assert result.items[1].initial_guess is None
    if method == "rhf":
        # Independent normalized exp(-r^2) oracle: two one-electron expectations
        # plus the closed-shell Gaussian self-Coulomb integral, with no SCF loop.
        analytical = [
            3 - 4 * charge * np.sqrt(2 / np.pi) + 2 / np.sqrt(np.pi)
            for charge in (2, 20)
        ]
        assert [item.energy for item in result.items] == pytest.approx(
            analytical, abs=1e-10
        )
    for current in (result, warm, moved, replay):
        assert [item.energy for item in current.items] == pytest.approx(
            [item.energy for item in reference.items], abs=1e-9
        )
    for current in (warm, moved, replay):
        assert all(item.warm_start_used for item in current.items)
        assert current.items[0].initial_guess["outcome"] == "existing_density"
        assert current.items[1].initial_guess is None
    assert not failed.items[0].converged and failed.items[1].converged


@pytest.mark.parametrize("method", ["rhf", "pbe-rks", "pbe0-rks"])
def test_mixed_resource_plan_and_exact_budget(native: None, method: str) -> None:
    """Charge only the admitted seed and retain the complete endpoint byte gate."""
    basis = (Shell(0, 0, (Primitive(1.0, 1.0),)),)
    systems = [[("He", (0, 0, 0))], HEAVY]
    options = (
        {"ks_options": KsOptions(grid=GridSpec(8, 6, 12))}
        if method.endswith("-rks")
        else {}
    )
    automatic = Calculator(method, basis, device="cpu", **options)
    core = Calculator(method, basis, device="cpu", initial_guess=None, **options)
    request = automatic._resource_request(
        systems,
        charges=[0, 18],
        properties=("energy",),
        ks_options=automatic.ks_options,
    )
    target = core._resource_request(
        systems, charges=[0, 18], properties=("energy",), ks_options=core.ks_options
    )
    assert json.loads(request.identity.schedule)["initial_guess_resolved_kinds"] == [
        3,
        0,
    ]
    for candidate, baseline in zip(request.candidates, target.candidates, strict=True):
        assert candidate.estimates[:-2] == baseline.estimates
        assert candidate.estimates[-2].bytes == 8
        assert candidate.estimates[-1].bytes > 0
    peak = automatic.estimate_resources(systems, charges=[0, 18]).peak_bytes["host"]
    exact = Calculator(
        method,
        basis,
        device="cpu",
        resource_budget=ResourceBudget(host_bytes=peak),
        **options,
    )
    with exact.prepare_batch(systems, charges=[0, 18]) as batch:
        result = batch.execute(strict=True, properties=("energy",))
        observation = batch.resource_diagnostics["observation"]
        assert 0 < observation["sampled_item_peak_host_bytes"] <= peak
    assert result.items[0].initial_guess["outcome"] == "used"
    assert result.items[1].initial_guess is None
    below = Calculator(
        method,
        basis,
        device="cpu",
        resource_budget=ResourceBudget(host_bytes=peak - 1),
        **options,
    )
    with pytest.raises(MemoryError):
        below.prepare_batch(systems, charges=[0, 18])


def test_all_ineligible_default_keeps_target_inventory(native: None) -> None:
    """A native Hcore selection has no hidden or unnecessary seed allowance."""
    basis = (Shell(0, 0, (Primitive(1.0, 1.0),)),)
    systems = [HEAVY]
    automatic = Calculator("rhf", basis, device="cpu")
    core = Calculator("rhf", basis, device="cpu", initial_guess=None)
    request = automatic._resource_request(systems, charges=[18], properties=("energy",))
    target = core._resource_request(systems, charges=[18], properties=("energy",))
    assert request.candidates == target.candidates
    assert json.loads(request.identity.schedule)["initial_guess_resolved_kinds"] == [0]
    reference = core.singlepoint(HEAVY, charge=18, properties=("energy",))
    with automatic.prepare_batch(systems, charges=[18]) as batch:
        result = batch.execute(strict=True, properties=("energy",)).items[0]
    assert result.converged and result.energy == pytest.approx(
        reference.energy, abs=1e-10
    )
    assert result.initial_guess is None


@pytest.mark.parametrize("method", ["rhf", "pbe-rks", "pbe0-rks"])
def test_mixed_ecp_basis_does_not_disable_an_all_electron_neighbor(
    native: None, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unused ECP metadata is not a model-wide veto on a per-system policy."""
    potential = {
        "ecp_type": "scalar_ecp",
        "angular_momentum": [0],
        "r_exponents": [2],
        "gaussian_exponents": ["0.8"],
        "coefficients": [["-2"]],
    }
    shells = (BasisShell(0, ("1",), (("1",),)),)
    basis = BasisSet(
        "mixed automatic-policy fixture",
        (
            ElementBasis(2, shells),
            ElementBasis(
                11, shells, ecp_core_electrons=10, ecp_data=json.dumps([potential])
            ),
        ),
        BasisProvenance("synthetic test", "1", "CC0", "0" * 64),
    )
    options = (
        {"ks_options": KsOptions(grid=GridSpec(8, 6, 12))}
        if method.endswith("-rks")
        else {}
    )
    systems = [[("He", (0, 0, 0))], [("Na", (0, 0, 0))]]
    automatic = Calculator(method, basis, device="cpu", **options)
    core = Calculator(method, basis, device="cpu", initial_guess=None, **options)
    assert automatic.initial_guess == InitialGuessSpec("minao")
    assert automatic.capabilities == core.capabilities
    request = automatic._resource_request(
        systems,
        charges=[0, -1],
        properties=("energy",),
        ks_options=automatic.ks_options,
    )
    assert json.loads(request.identity.schedule)["initial_guess_resolved_kinds"] == [
        3,
        0,
    ]
    with core.prepare_batch(systems, charges=[0, -1]) as batch:
        reference = batch.execute(strict=True, properties=("energy",))
    with automatic.prepare_batch(systems, charges=[0, -1]) as batch:
        result = batch.execute(strict=True, properties=("energy",))
        warm = batch.execute(strict=True, properties=("energy",))
    assert result.items[0].initial_guess["outcome"] == "used"
    assert warm.items[0].initial_guess["outcome"] == "existing_density"
    assert result.items[1].initial_guess is None and warm.items[1].initial_guess is None
    assert [item.energy for item in result.items] == pytest.approx(
        [item.energy for item in reference.items], abs=1e-9
    )
    with pytest.raises(NotImplementedError, match="MINAO requires"):
        Calculator(
            method,
            basis,
            device="cpu",
            initial_guess=InitialGuessSpec("minao"),
            **options,
        )
    monkeypatch.setattr(
        automatic._library,
        "generativeqc_initial_guess_capabilities_v1",
        lambda: _native.INITIAL_GUESS_CAPABILITY_MINAO,
    )
    monkeypatch.setattr(
        _native, "load_library", lambda *args, **kwargs: automatic._library
    )
    legacy = Calculator(method, basis, device="cpu", **options)
    assert legacy.initial_guess is None


def test_default_checkpoint_density_skips_preparation(
    native: None, tmp_path: Path
) -> None:
    path = tmp_path / "core.bin"
    with Calculator("rhf", "sto-3g", device="cpu", initial_guess=None).prepare_batch(
        [WATER]
    ) as source:
        reference = source.execute(strict=True, properties=("energy",)).items[0]
        source.save_checkpoint(path)
    with Calculator("rhf", "sto-3g", device="cpu").prepare_batch([WATER]) as target:
        target.load_checkpoint(path)
        imported = target.execute(strict=True, properties=("energy",)).items[0]
    assert imported.warm_start_used and imported.energy == pytest.approx(
        reference.energy, abs=1e-9
    )
    assert imported.initial_guess["outcome"] == "existing_density"
    assert imported.initial_guess["preparation_seconds"] == 0
    assert imported.initial_guess["preparation_numeric_capacity"] == 0


@pytest.mark.parametrize("method", ["rhf", "pbe0-rks"])
def test_cpu_default_endpoint_and_warm_precedence(native: None, method: str) -> None:
    options = {
        "method": method,
        "basis": "sto-3g",
        "device": "cpu",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    if method.endswith("-rks"):
        options["ks_options"] = KsOptions(grid=GridSpec(16, 8, 16))
    baseline = Calculator(**options, initial_guess=None).singlepoint(
        WATER, properties=("energy", "forces")
    )
    with Calculator(**options).prepare_batch([WATER]) as batch:
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
    assert cold.initial_guess["kind"] == "minao"
    assert cold.initial_guess["outcome"] == "used"
    assert cold.initial_guess["preliminary_fock_builds"] == 0
    assert cold.initial_guess["target_attempts"] == 1
    assert warm.initial_guess["outcome"] == "existing_density"
    assert warm.initial_guess["preparation_seconds"] == 0
    assert warm.initial_guess["preparation_numeric_capacity"] == 0
    for result in (cold, warm):
        assert result.energy == pytest.approx(baseline.energy, abs=1e-9)
        np.testing.assert_allclose(result.forces, baseline.forces, rtol=0, atol=1e-7)


def test_cpu_default_matches_independent_rhf_oracle(native: None) -> None:
    """Match oracle coefficients exactly rather than comparing rounded basis tables."""
    pyscf = pytest.importorskip("pyscf")
    molecule = pyscf.gto.M(atom=WATER, basis="sto-3g", unit="Bohr", verbose=0)
    shells = tuple(
        Shell(
            atom_index,
            shell[0],
            tuple(Primitive(row[0], row[1]) for row in shell[1:]),
        )
        for atom_index, (element, _) in enumerate(WATER)
        for shell in pyscf.gto.basis.load("sto-3g", element)
    )
    oracle = pyscf.scf.RHF(molecule)
    oracle.conv_tol = 1e-12
    oracle.conv_tol_grad = 1e-10
    oracle.kernel()
    assert oracle.converged
    result = Calculator(
        "rhf", shells, device="cpu", energy_tolerance=1e-12, density_tolerance=1e-10
    ).singlepoint(WATER, properties=("energy", "forces"))
    assert result.converged and result.initial_guess["outcome"] == "used"
    assert result.energy == pytest.approx(oracle.e_tot, abs=1e-9)
    np.testing.assert_allclose(
        result.forces, -oracle.nuc_grad_method().kernel(), rtol=0, atol=1e-7
    )


@pytest.mark.skipif(
    not os.environ.get("SLURM_JOB_ID"), reason="real CUDA work requires Slurm"
)
def test_cuda_default_endpoint_and_unsupported_domains(native: None) -> None:
    assert "CUDA_VISIBLE_DEVICES" in os.environ
    options = {
        "method": "pbe0-rks",
        "basis": "sto-3g",
        "device": "cuda",
        "ks_options": KsOptions(grid=GridSpec(16, 8, 16)),
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    auto = Calculator(**options)
    explicit = Calculator(**options, initial_guess=InitialGuessSpec("minao"))
    core = Calculator(**options, initial_guess=None)
    assert auto.initial_guess == InitialGuessSpec("minao")
    for calculator in (auto, explicit):
        with pytest.raises(NotImplementedError, match="global-hybrid"):
            calculator._resource_request([WATER], ks_options=calculator.ks_options)
    resource_options = options | {"method": "pbe-rks"}
    resource_auto = Calculator(**resource_options)
    resource_explicit = Calculator(
        **resource_options, initial_guess=InitialGuessSpec("minao")
    )
    assert resource_auto._resource_request(
        [WATER], ks_options=resource_auto.ks_options
    ) == resource_explicit._resource_request(
        [WATER], ks_options=resource_explicit.ks_options
    )
    assert not auto._method_descriptor(systems=[HEAVY]).initial_guess
    baseline = core.singlepoint(WATER, properties=("energy", "forces"))
    with auto.prepare_batch([WATER]) as batch:
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
    assert cold.initial_guess["outcome"] == "used"
    assert cold.initial_guess["preliminary_fock_builds"] == 0
    assert warm.initial_guess["outcome"] == "existing_density"
    for result in (cold, warm):
        assert result.energy == pytest.approx(baseline.energy, abs=1e-9)
        np.testing.assert_allclose(result.forces, baseline.forces, rtol=0, atol=1e-7)
    for overrides in (
        {"method": "pbe-uks"},
        {"density_fitting": "cuda"},
        {"precision": "auto"},
    ):
        assert Calculator(**(options | overrides)).initial_guess is None

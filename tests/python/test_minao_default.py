"""Guarded defaults share admission with planning and retain an explicit rollback."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    InitialGuessSpec,
    KsOptions,
    Primitive,
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
    assert auto._method_descriptor(systems=[WATER]).initial_guess.contents.kind == 3
    assert not core._method_descriptor(systems=[WATER]).initial_guess
    assert not auto._method_descriptor(systems=[WATER, HEAVY]).initial_guess
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
    assert automatic_plan == explicit_plan


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


def test_mixed_native_batch_uses_hcore(native: None) -> None:
    """Both two-electron ions fit the same one-AO owner, but Ca is outside MINAO."""
    basis = (Shell(0, 0, (Primitive(1.0, 1.0),)),)
    systems = [[("He", (0.0, 0.0, 0.0))], HEAVY]
    calc = Calculator("rhf", basis, device="cpu")
    with calc.prepare_batch(systems, charges=[0, 18]) as batch:
        result = batch.execute(strict=True, properties=("energy",))
    assert all(item.converged and item.initial_guess is None for item in result.items)


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

"""Native unrestricted CC user endpoints and independent molecular oracles."""

from __future__ import annotations

import builtins
import ctypes
import json
import os
import subprocess
import typing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc import (
    Atom,
    Calculator,
    Primitive,
    ResolvedModel,
    Shell,
    _native,
    method_capabilities,
)
from generativeqc_compiler.dft import NativeAO


def _calculator(
    method: str, basis: typing.Any = "sto-3g", **options: typing.Any
) -> Calculator:
    """Use explicit cold UHF, strict CC acceptance and no implicit forces."""
    defaults: dict[str, typing.Any] = {
        "device": "cpu",
        "basis": basis,
        "initial_guess": None,
        "max_iterations": 300,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "ccsd_max_iterations": 200,
        "ccsd_energy_tolerance": 1e-13,
        "ccsd_residual_tolerance": 1e-11,
    }
    return Calculator(method=method, **(defaults | options))


@pytest.mark.parametrize("method", ("uccsd", "uccsd(t)"))
def test_native_unrestricted_capabilities_and_model(method: str) -> None:
    """Keep unrestricted names, identities and energy-only capabilities distinct."""
    capabilities = method_capabilities(method)
    assert capabilities.available and capabilities.family == "coupled_cluster"
    assert capabilities.supported_properties == frozenset({"energy"})
    assert not capabilities.supports_batch
    model = _calculator(method).resolved_model(
        [("Li", (0.0, 0.0, 0.0))], multiplicity=2
    )
    assert model.method == method and model.multiplicity == 2
    assert ResolvedModel.from_dict(model.to_dict()) == model
    with pytest.raises(ValueError, match="conventional integrals"):
        replace(model, approximation="density-fitting")


@pytest.mark.parametrize("method", ("uccsd", "uccsd(t)"))
@pytest.mark.parametrize(
    "atoms,spin,basis",
    [
        ([("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))], 0, "sto-3g"),
        ([("Li", (0.0, 0.0, 0.0))], 1, "sto-3g"),
        ([("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.83))], 1, "sp-polarized"),
        ([("He", (0.0, 0.0, 0.0))], 0, "spd-custom"),
        ([("He", (0.0, 0.0, 0.0))], 0, "spd-spherical"),
    ],
)
def test_native_unrestricted_endpoint_matches_independent_pyscf(
    monkeypatch: pytest.MonkeyPatch, method: str, atoms: list, spin: int, basis: str
) -> None:
    """Match shell inputs, not native intermediates, to independent UHF/CC.

    The nine-AO OH case admits all four triples sectors with unequal spin
    populations. Imports of reference/prototype/compiler modules are forbidden
    during the actual public execution after the independent oracle is solved.
    """
    pyscf = pytest.importorskip("pyscf")
    from pyscf import cc

    labels = [f"{symbol}{index}" for index, (symbol, _) in enumerate(atoms)]
    matched_basis: dict[str, list] = {label: [] for label in labels}
    representation = "spherical" if basis == "spd-spherical" else "cartesian"
    if basis.startswith("spd-"):
        actual_basis = tuple(
            Shell(0, angular, (Primitive(exponent, 1.0),))
            for angular, exponent in ((0, 2.3), (0, 0.4), (1, 0.8), (2, 0.7))
        )
    else:
        with NativeAO(atoms, basis="sto-3g", multiplicity=spin + 1) as source:
            actual_basis = tuple(source.shells)
        if basis == "sp-polarized":
            actual_basis = (*actual_basis, Shell(1, 1, (Primitive(0.8, 1.0),)))
    for shell in actual_basis:
        matched_basis[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[
                    (primitive.exponent, primitive.coefficient)
                    for primitive in shell.primitives
                ],
            ]
        )
    molecule = pyscf.gto.M(
        atom=[
            (label, position)
            for label, (_, position) in zip(labels, atoms, strict=True)
        ],
        basis=matched_basis,
        spin=spin,
        unit="Bohr",
        cart=representation == "cartesian",
        verbose=0,
    )
    reference = pyscf.scf.UHF(molecule)
    reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = 1e-13, 1e-11, 300
    reference.kernel()
    assert reference.converged
    oracle = cc.UCCSD(reference)
    oracle.conv_tol, oracle.conv_tol_normt, oracle.max_cycle = 1e-13, 1e-11, 300
    correlation, _, _ = oracle.kernel()
    assert oracle.converged
    triples = oracle.ccsd_t() if method == "uccsd(t)" else 0.0
    original_import = builtins.__import__

    def guarded_import(
        name: str, *arguments: typing.Any, **keywords: typing.Any
    ) -> typing.Any:
        """Neither a Python method owner nor a reference library may execute."""
        if name.startswith(
            (
                "pyscf",
                "tools.generativeqc_cc",
                "generativeqc_compiler.cc",
                "generativeqc_compiler.tensor",
            )
        ):
            raise AssertionError(
                f"native public endpoint attempted runtime import {name}"
            )
        return original_import(name, *arguments, **keywords)

    calculator = _calculator(method, actual_basis, basis_representation=representation)
    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", guarded_import)
        result = calculator.singlepoint(atoms, multiplicity=spin + 1)
    assert result.converged and result.forces is None
    assert result.executed_backend == "cpu_reference"
    np.testing.assert_allclose(
        result.energy, reference.e_tot + correlation + triples, atol=3e-9, rtol=0
    )
    diagnostic = result.correlation
    assert diagnostic is not None
    np.testing.assert_allclose(
        diagnostic.ccsd_correlation_energy, correlation, atol=3e-9, rtol=0
    )
    np.testing.assert_allclose(
        diagnostic.ccsd_t_triples_energy, triples, atol=5e-10, rtol=0
    )
    assert diagnostic.ccsd_replay_singles_residual_max <= 1e-11
    assert diagnostic.ccsd_replay_doubles_residual_max <= 1e-11
    assert 0 < diagnostic.numeric_capacity_bytes <= 256 << 20
    assert len(diagnostic.equation_hash) == 64
    assert result.cc_performance is not None
    assert result.cc_performance.replay_graph_calls > 0
    if len(atoms) > 1 or atoms[0][0] != "H":
        assert result.cc_performance.source_reads > 0
    if method == "uccsd(t)" and basis == "sp-polarized":
        assert diagnostic.ccsd_t_virtual_triples > 0
        assert abs(diagnostic.ccsd_t_triples_energy) > 1e-8
        assert len(diagnostic.ccsd_t_equation_hash) == 64


@pytest.mark.parametrize(
    "restricted,unrestricted", (("rccsd", "uccsd"), ("rccsd(t)", "uccsd(t)"))
)
def test_native_unrestricted_restricted_limit(
    restricted: str, unrestricted: str
) -> None:
    """Independent native restricted owner is a parity oracle, not a fallback."""
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, -1.43, 1.1)), ("H", (0.0, 1.43, 1.1))]
    actual = _calculator(unrestricted).singlepoint(atoms)
    expected = _calculator(restricted).singlepoint(atoms, properties=("energy",))
    np.testing.assert_allclose(actual.energy, expected.energy, atol=3e-9, rtol=0)


@pytest.mark.parametrize("method", ("uccsd", "uccsd(t)"))
def test_native_unrestricted_rejects_unqualified_requests(method: str) -> None:
    """Do not advertise prototype forces or silently reroute unsupported work."""
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    calculator = _calculator(method)
    with pytest.raises((ValueError, NotImplementedError), match="forces"):
        calculator.singlepoint(atoms, properties=("energy", "forces"))
    with pytest.raises(NotImplementedError):
        calculator.prepare_batch([atoms])
    with pytest.raises(RuntimeError, match="budget"):
        _calculator(method, correlation_memory_budget_bytes=1024).singlepoint(atoms)
    with pytest.raises((RuntimeError, ValueError), match="converg"):
        _calculator(method, ccsd_max_iterations=1).singlepoint(atoms)
    with pytest.raises((ValueError, RuntimeError), match="denominator|energies"):
        _calculator(method, ccsd_denominator_threshold=100.0).singlepoint(atoms)
    with pytest.raises(RuntimeError, match="scalar-work bound"):
        _calculator(method, ccsd_max_iterations=2**32 - 1).singlepoint(atoms)


@pytest.mark.parametrize("method", ("uccsd", "uccsd(t)"))
@pytest.mark.parametrize("converges", (True, False))
def test_native_unrestricted_c_abi_owned_inputs_and_failure_publication(
    method: str, converges: bool
) -> None:
    """Destroy borrowed system input; publish diagnostics only after success."""
    calculator = _calculator(method, ccsd_max_iterations=200 if converges else 1)
    library = calculator._library
    context, calculation = ctypes.c_void_p(), ctypes.c_void_p()
    context_descriptor = calculator._context_descriptor()
    _native.check(
        library,
        library.generativeqc_context_create(
            ctypes.byref(context_descriptor), ctypes.byref(context)
        ),
    )
    try:
        atoms = (Atom(1, (0.0, 0.0, -0.7)), Atom(1, (0.0, 0.0, 0.7)))
        system = calculator._create_native_system(context, atoms, 0, 1)
        try:
            descriptor = calculator._method_descriptor()
            _native.check(
                library,
                library.generativeqc_calculation_prepare(
                    context, system, ctypes.byref(descriptor), ctypes.byref(calculation)
                ),
                context=context,
            )
        finally:
            library.generativeqc_system_destroy(system)
        for _attempt in range(2):
            output = _native.ResultDescriptor()
            output.struct_size, output.abi_version = (
                ctypes.sizeof(output),
                _native.ABI_VERSION,
            )
            status = library.generativeqc_calculation_execute(
                calculation, ctypes.byref(output)
            )
            assert status == (
                _native.STATUS_SUCCESS if converges else _native.STATUS_NOT_CONVERGED
            )
            diagnostic = _native.CorrelationDiagnostic()
            diagnostic.struct_size, diagnostic.abi_version = (
                ctypes.sizeof(diagnostic),
                _native.ABI_VERSION,
            )
            diagnostic_status = (
                library.generativeqc_calculation_get_correlation_diagnostic(
                    calculation, ctypes.byref(diagnostic)
                )
            )
            assert diagnostic_status == (
                _native.STATUS_SUCCESS if converges else _native.STATUS_NOT_IMPLEMENTED
            )
            assert bool(output.converged) == converges
            if converges:
                assert np.isfinite(output.energy)
                assert diagnostic.ccsd_replay_doubles_residual_max <= 1e-11
    finally:
        if calculation:
            library.generativeqc_calculation_destroy(calculation)
        library.generativeqc_context_destroy(context)


@pytest.mark.parametrize("method", ("uccsd", "uccsd(t)"))
def test_native_unrestricted_cli_without_interpreters(
    tmp_path: Path, method: str
) -> None:
    """The actual native executable evaluates energy without Python on PATH."""
    library_path = Path(_calculator(method)._library._name).resolve()
    candidates = (
        library_path.parent / "generativeqc",
        library_path.parent / "bin/generativeqc",
    )
    executable = next(
        (candidate for candidate in candidates if candidate.is_file()), None
    )
    if executable is None:
        pytest.skip("native CLI is not present in this library-only build")
    coordinates = tmp_path / "h2.xyz"
    coordinates.write_text("2\nH2 in bohr\nH 0 0 -0.7\nH 0 0 0.7\n")
    environment = dict(
        os.environ, PATH=str(tmp_path / "no-interpreters"), GENERATIVEQC_PROFILE="off"
    )
    command = [
        str(executable),
        "run",
        str(coordinates),
        "--method",
        method,
        "--basis",
        "sto-3g",
        "--units",
        "bohr",
        "--backend",
        "cpu",
        "--ccsd-max-iterations",
        "200",
        "--ccsd-energy-tolerance",
        "1e-13",
        "--ccsd-residual-tolerance",
        "1e-11",
        "--json",
    ]
    executed = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=60, check=True
    )
    result = json.loads(executed.stdout)
    expected = _calculator(method).singlepoint(
        [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    )
    np.testing.assert_allclose(
        result["energy_hartree"], expected.energy, atol=3e-9, rtol=0
    )


@pytest.mark.parametrize("method", ("uccsd", "uccsd(t)"))
def test_native_unrestricted_rejects_large_domain(method: str) -> None:
    """The public twelve-AO admission is explicit, before solving UHF."""
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, -1.43, 1.1)), ("H", (0.0, 1.43, 1.1))]
    with pytest.raises(NotImplementedError, match="twelve AOs"):
        _calculator(method, basis="def2-svp").singlepoint(atoms)


@pytest.mark.parametrize("method", ("uccsd", "uccsd(t)"))
def test_native_unrestricted_empty_beta_and_excitation_spaces(method: str) -> None:
    """The exact one-electron limit has no CC correlation or triples.

    PySCF UCCSD currently cannot construct this empty-domain HDF5 intermediate;
    use its independent UHF energy and the exact one-electron correlation zero.
    """
    pyscf = pytest.importorskip("pyscf")
    atoms = [("H", (0.0, 0.0, 0.0))]
    with NativeAO(atoms, basis="sto-3g", multiplicity=2) as source:
        shells = tuple(source.shells)
    basis = {
        "H": [
            [
                shell.angular_momentum,
                *[
                    (primitive.exponent, primitive.coefficient)
                    for primitive in shell.primitives
                ],
            ]
            for shell in shells
        ]
    }
    molecule = pyscf.gto.M(
        atom=atoms, basis=basis, spin=1, unit="Bohr", cart=True, verbose=0
    )
    reference = pyscf.scf.UHF(molecule).run(conv_tol=1e-13)
    result = _calculator(method).singlepoint(atoms, multiplicity=2)
    np.testing.assert_allclose(result.energy, reference.e_tot, atol=3e-9, rtol=0)
    assert result.correlation is not None
    assert result.correlation.ccsd_correlation_energy == 0.0
    assert result.correlation.ccsd_t_triples_energy == 0.0
    assert result.correlation.minimum_absolute_denominator == 0.0

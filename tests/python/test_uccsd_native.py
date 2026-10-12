"""Native UHF and shell-tile producer to UCCSD, without production PySCF work."""

from __future__ import annotations

import numpy as np
import pytest

from tools.generativeqc_cc import (
    PreparedUCCSD,
    PreparedUCCSDLambda,
    PreparedUCCSDT,
    SolverOptions,
)
from tools.generativeqc_posthf.export import export_uhf
from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_posthf.sources import NativeSource


@pytest.mark.parametrize(
    "atoms,spin",
    [
        ([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))], 0),
        ([("Li", (0.0, 0.0, 0.0))], 1),
        ([("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.83))], 1),
    ],
)
def test_native_producer_complete_uccsd_energy(atoms: list, spin: int) -> None:
    """Compare independent solvers using identical supplied shell primitives.

    Named STO-3G tables differ in their rounded decimal coefficients. Sharing
    only the shell definitions avoids comparing different Hamiltonians; PySCF
    still computes every integral, reference and correlated result independently.
    A symmetry-preserving AO core-Hamiltonian direction checks native response
    without identifying MO phases or degenerate electronic-state orientations
    between independently converged HF references. Both finite-difference CC
    solves keep their own orbitals fixed and start from independent oracle CC.
    """
    pyscf = pytest.importorskip("pyscf")
    cc = pytest.importorskip("pyscf.cc")
    with NativeSource(atoms, basis="sto-3g", multiplicity=spin + 1) as source:
        labels = [f"{symbol}{index}" for index, (symbol, _) in enumerate(atoms)]
        basis = {label: [] for label in labels}
        for shell in source.shells:
            basis[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *(
                        (primitive.exponent, primitive.coefficient)
                        for primitive in shell.primitives
                    ),
                ]
            )
        molecule = pyscf.gto.M(
            atom=[
                (label, position)
                for label, (_, position) in zip(labels, atoms, strict=True)
            ],
            basis=basis,
            unit="Bohr",
            spin=spin,
            cart=True,
            verbose=0,
        )
        reference = pyscf.scf.UHF(molecule)
        reference.conv_tol = 1e-13
        reference.kernel()
        assert reference.converged
        oracle = cc.UCCSD(reference)
        oracle.conv_tol, oracle.conv_tol_normt, oracle.max_cycle = 1e-13, 1e-10, 400
        correction, _, _ = oracle.kernel()
        assert oracle.converged
        triples = oracle.ccsd_t()

        snapshot, diagnostics = export_uhf(source, max_iterations=300, tolerance=1e-11)
        ao_direction = snapshot.hcore / np.linalg.norm(snapshot.hcore)
        with (
            ConventionalProvider(snapshot, source) as provider,
            PreparedUCCSD(
                snapshot, provider, options=SolverOptions(max_iterations=300)
            ) as prepared,
        ):
            result = prepared.solve()
            assert result.converged, result.history[-1]
            with PreparedUCCSDT(prepared, result) as triples_owner:
                correlated = triples_owner.energy()
            assert correlated.converged, correlated.reason
            assert correlated.triples_energy is not None
            assert correlated.total_energy is not None
            np.testing.assert_allclose(
                correlated.triples_energy, triples, atol=5e-10, rtol=0
            )
            np.testing.assert_allclose(
                correlated.total_energy,
                reference.e_tot + correction + triples,
                atol=3e-9,
                rtol=0,
            )
            with PreparedUCCSDLambda(prepared, result) as response_owner:
                multipliers = response_owner.solve()
                derivative = 0.0
                for spin_name, suffix, coefficients in (
                    ("alpha", "aa", snapshot.coefficients_alpha),
                    ("beta", "bb", snapshot.coefficients_beta),
                ):
                    mo_direction = coefficients.T @ ao_direction @ coefficients
                    occupied = snapshot.nocc(spin_name)
                    for domain, selected in (
                        ("oo", (slice(0, occupied), slice(0, occupied))),
                        ("ov", (slice(0, occupied), slice(occupied, None))),
                        ("vv", (slice(occupied, None), slice(occupied, None))),
                    ):
                        cotangent = response_owner.parameter_cotangent(
                            multipliers, f"f{domain}_{suffix}"
                        )
                        derivative += cotangent.contract(mo_direction[selected].copy())
                assert multipliers.provenance["tensor_backend"] == "native-cpu-tensorir"
                assert multipliers.stationarity_norm <= 1e-9

        independent_directions = [
            coefficients.T @ ao_direction @ coefficients
            for coefficients in reference.mo_coeff
        ]
        unperturbed_eris = oracle.ao2mo()
        energies = []
        for sign in (-1, 1):
            perturbed = cc.UCCSD(reference)
            perturbed.conv_tol, perturbed.conv_tol_normt, perturbed.max_cycle = (
                1e-13,
                1e-11,
                400,
            )
            perturbed_eris = perturbed.ao2mo()
            perturbed_eris.focka = (
                unperturbed_eris.focka + sign * 1e-4 * independent_directions[0]
            )
            perturbed_eris.fockb = (
                unperturbed_eris.fockb + sign * 1e-4 * independent_directions[1]
            )
            energies.append(
                perturbed.kernel(t1=oracle.t1, t2=oracle.t2, eris=perturbed_eris)[0]
            )
            assert perturbed.converged
        np.testing.assert_allclose(
            derivative, (energies[1] - energies[0]) / 2e-4, atol=2e-7, rtol=0
        )
        assert result.converged, result.history[-1]
        assert result.correlation_energy is not None and result.total_energy is not None
        assert snapshot.hf_backend == "native-cpu"
        assert result.provenance["source_backend"] == "cpu-reference-native-shell-tiles"
        assert result.provenance["provider_statistics"]["source_tiles"] > 0
        assert diagnostics["physical_residual"] < 1e-8
        np.testing.assert_allclose(
            snapshot.reference_energy, reference.e_tot, atol=2e-9, rtol=0
        )
        np.testing.assert_allclose(
            result.correlation_energy, correction, atol=3e-9, rtol=0
        )
        np.testing.assert_allclose(
            result.total_energy, reference.e_tot + correction, atol=3e-9, rtol=0
        )

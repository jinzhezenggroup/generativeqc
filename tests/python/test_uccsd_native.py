"""Native UHF and shell-tile producer to UCCSD, without production PySCF work."""

from __future__ import annotations

import numpy as np
import pytest

from tools.generativeqc_cc import PreparedUCCSD, PreparedUCCSDT, SolverOptions
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

"""Independent nuclear Fock directions and bounded spin-panel admission."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.hessian_basis import HessianBasis
from generativeqc.rks_hessian_integrals import (
    RKSIntegralTopology,
    generated_directional_semilocal_rks_integral_first_order,
    generated_directional_spin_integral_first_order,
)
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.integral.ir import FOUR_CENTER_ERI_OPERATOR
from test_integral_autodiff_second import require_native_library  # noqa: F401


@pytest.mark.parametrize("include_coulomb", (False, True))
def test_spin_panel_matches_independent_fixed_density_fock_directions(
    tmp_path: Path, include_coulomb: bool
) -> None:
    """Retain shared J, distinct same-spin K, moving nuclei and raw overlap."""
    gto = pytest.importorskip("pyscf.gto")
    atoms = [
        ("H", (-0.2, 0.1, -0.7)),
        ("He", (0.3, -0.2, 0.6)),
        ("Li", (1.4, 0.4, -0.1)),
    ]
    angular, exponents = (1, 0, 1), (0.71, 0.93, 0.58)
    record = BasisSet(
        "synthetic-first-spin",
        tuple(
            ElementBasis(number, (BasisShell(order, (str(exponent),), (("1.0",),)),))
            for number, order, exponent in zip(
                (1, 2, 3), angular, exponents, strict=True
            )
        ),
        BasisProvenance("synthetic-test", "1", "test", "0" * 64),
        representation="spherical",
    )
    vector = np.array(((0.04, -0.03, 0.01), (-0.02, 0.05, -0.04), (0.01, 0.02, -0.03)))
    coefficient = 0.23
    random = np.random.default_rng(2844)
    with NativeAO(atoms, basis=record, representation="spherical") as public:
        mapping = HessianBasis(public)
        try:
            topology = RKSIntegralTopology.from_basis(mapping.cartesian)
            raw = random.normal(size=(2, public.nao, public.nao))
            densities = (raw + raw.swapaxes(1, 2)) * 0.05
            cartesian = mapping.density(densities)
            original = cartesian.copy()
            operators = ((FOUR_CENTER_ERI_OPERATOR, coefficient),)
            results = []
            for spins in (1, 2):
                panel = cartesian[:spins]
                frozen, overlap, work = generated_directional_spin_integral_first_order(
                    topology,
                    panel,
                    vector,
                    cache=tmp_path,
                    exchange_operators=operators,
                    include_coulomb=include_coulomb,
                )
                np.testing.assert_array_equal(cartesian, original)
                for spin in range(spins):
                    separate, separate_overlap = (
                        generated_directional_semilocal_rks_integral_first_order(
                            topology,
                            panel.sum(axis=0),
                            vector,
                            cache=tmp_path,
                            exchange_operators=operators,
                            exchange_density=(2 if spins == 2 else 1) * panel[spin],
                            include_coulomb=include_coulomb,
                        )
                    )
                    np.testing.assert_allclose(
                        frozen[spin], separate, atol=2e-13, rtol=0
                    )
                    np.testing.assert_array_equal(overlap, separate_overlap)
                results.append((mapping.matrix(frozen), mapping.matrix(overlap), work))
        finally:
            mapping.close()

    def independent(
        displacement: np.ndarray, spins: int
    ) -> tuple[np.ndarray, np.ndarray]:
        molecule = gto.M(
            atom=[
                (symbol, np.asarray(position) + motion)
                for (symbol, position), motion in zip(atoms, displacement, strict=True)
            ],
            basis={
                symbol: [[order, [exponent, 1.0]]]
                for (symbol, _), order, exponent in zip(
                    atoms, angular, exponents, strict=True
                )
            },
            unit="Bohr",
            verbose=0,
        )
        core = molecule.intor("int1e_kin") + molecule.intor("int1e_nuc")
        integrals = molecule.intor("int2e")
        panel = densities[:spins]
        coulomb = np.einsum("ijkl,kl->ij", integrals, panel.sum(axis=0))
        exchange = np.einsum("ikjl,skl->sij", integrals, panel)
        fock = (
            core
            + (coulomb if include_coulomb else 0)
            - coefficient * (1 if spins == 2 else 0.5) * exchange
        )
        return fock, molecule.intor("int1e_ovlp")

    for spins, (actual, overlap, work) in enumerate(results, start=1):
        differences = []
        for step in (2e-4, 1e-4):
            minus, plus = [independent(sign * step * vector, spins) for sign in (-1, 1)]
            differences.append(
                tuple(
                    (right - left) / (2 * step)
                    for left, right in zip(minus, plus, strict=True)
                )
            )
        expected = tuple(
            (4 * fine - coarse) / 3 for coarse, fine in zip(*differences, strict=True)
        )
        np.testing.assert_allclose(actual, expected[0], atol=2e-9, rtol=2e-8)
        np.testing.assert_allclose(overlap, expected[1], atol=2e-10, rtol=2e-8)
        assert work["directional_calls"] > 0
        assert work["raw_first_calls"] > 0
    assert (
        results[0][2]["component_evaluations"] == results[1][2]["component_evaluations"]
    )


def test_spin_panel_is_admitted_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed spin panels or directions cannot start a compiler/native call."""
    topology = SimpleNamespace(nbf=2, atoms=(object(),), check_current=lambda: None)

    def unexpected_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid input reached a first-integral provider")

    monkeypatch.setattr(
        "generativeqc.rks_hessian_integrals._FirstDerivativeProvider",
        unexpected_provider,
    )
    invalid = (
        np.zeros((2, 2)),
        np.zeros((0, 2, 2)),
        np.zeros((3, 2, 2)),
        np.zeros((2, 3, 3)),
        np.full((2, 2, 2), np.nan),
        np.ones((2, 2, 2), dtype=complex),
        np.array([[[1.0, 1.0], [0.0, 1.0]]]),
    )
    for panel in invalid:
        with pytest.raises(ValueError, match="spin|symmetric"):
            generated_directional_spin_integral_first_order(
                topology, panel, np.zeros((1, 3))
            )
    for vector in (np.zeros(3), np.full((1, 3), np.inf)):
        with pytest.raises(ValueError, match="direction"):
            generated_directional_spin_integral_first_order(
                topology, np.zeros((2, 2, 2)), vector
            )
    with pytest.raises(FloatingPointError):
        generated_directional_spin_integral_first_order(
            topology, np.full((2, 2, 2), np.finfo(float).max), np.zeros((1, 3))
        )

"""Exact ordered-cotangent permutation schedules and independent molecular jets."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.hessian_basis import HessianBasis
from generativeqc.rks_hessian_integrals import (
    RKSIntegralTopology,
    generated_weighted_first_integral_gradient,
    generated_weighted_second_integral_hvp,
)
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.integral.eri_symmetry import (
    canonical_eri_quartets,
    distinct_eri_shell_permutations,
    fold_eri_shell_weights,
)
from test_integral_autodiff_second import require_native_library  # noqa: F401


@pytest.mark.parametrize("shell_count", (1, 2, 4, 23))
def test_distinct_shell_orbits_cover_the_ordered_domain(shell_count: int) -> None:
    """Include the real Ni fixture's shell count without evaluating integrals."""
    seen = set()
    for canonical in canonical_eri_quartets(shell_count):
        for ordered, permutation in distinct_eri_shell_permutations(canonical):
            assert ordered == tuple(canonical[axis] for axis in permutation)
            assert ordered not in seen
            seen.add(ordered)
    assert seen == set(product(range(shell_count), repeat=4))


def test_arbitrary_weight_folding_matches_a_dense_symmetric_tensor() -> None:
    """Repeated shells, unequal extents and nonsymmetric weights need no factors."""
    generator = np.random.default_rng(2843)
    sizes = (1, 2, 3)
    offsets = np.cumsum((0, *sizes))
    raw = generator.normal(size=(sum(sizes),) * 4)
    pair_symmetric = raw + raw.swapaxes(0, 1)
    pair_symmetric += pair_symmetric.swapaxes(2, 3)
    integrals = pair_symmetric + pair_symmetric.transpose(2, 3, 0, 1)
    weights = generator.normal(size=integrals.shape)

    def block(slots: tuple[int, ...]) -> np.ndarray:
        return weights[tuple(slice(offsets[slot], offsets[slot + 1]) for slot in slots)]

    folded = 0.0
    for slots in canonical_eri_quartets(len(sizes)):
        selection = tuple(slice(offsets[slot], offsets[slot + 1]) for slot in slots)
        cotangent = fold_eri_shell_weights(
            slots, tuple(sizes[slot] for slot in slots), block
        )
        folded += float(np.sum(cotangent * integrals[selection]))
    assert folded == pytest.approx(np.sum(weights * integrals), abs=3e-12)


@pytest.mark.parametrize("failure", ("shape", "nan", "complex"))
def test_noncanonical_invalid_weight_is_not_hidden(failure: str) -> None:
    """Validate every ordered callback, not only a representative's weights."""

    def block(slots: tuple[int, ...]) -> np.ndarray:
        if slots == (0, 1, 0, 0):
            return {
                "shape": np.ones((2,)),
                "nan": np.full((1,) * 4, np.nan),
                "complex": np.full((1,) * 4, 1j),
            }[failure]
        return np.ones((1,) * 4)

    with pytest.raises(ValueError, match="ordered"):
        fold_eri_shell_weights((1, 0, 0, 0), (1,) * 4, block)


@pytest.mark.parametrize("budget_bytes", (32 << 10, 64 << 20))
def test_complex_weights_reject_before_zero_translation_or_budget_fallback(
    tmp_path: Path, budget_bytes: int
) -> None:
    """Both schedules reject complex f-shell weights without discarding their phase."""
    record = BasisSet(
        "one-center-f-weight-validation",
        (ElementBasis(2, (BasisShell(3, ("0.83",), (("1.0",),)),)),),
        BasisProvenance("synthetic-test", "1", "test", "0" * 64),
        representation="cartesian",
    )
    with NativeAO([("He", (0.0, 0.0, 0.0))], basis=record) as basis:
        topology = RKSIntegralTopology.from_basis(basis)
        with pytest.raises(ValueError, match="finite real.*ordered"):
            generated_weighted_second_integral_hvp(
                topology,
                "coulomb",
                np.ones((1, 3)),
                eri_shell_weights=lambda slots: np.full((10,) * 4, 1j),
                budget_bytes=budget_bytes,
                cache=tmp_path,
            )


def test_spherical_df_ordered_weights_match_independent_first_and_second_jets(
    tmp_path: Path,
) -> None:
    """Freeze a physical spherical cotangent; do not substitute Cartesian space."""
    gto = pytest.importorskip("pyscf.gto")
    atoms = [("H", (0.13, -0.21, 0.31)), ("He", (-0.37, 0.29, 1.23))]
    angular_by_atom = ((0, 2), (1, 3))
    record = BasisSet(
        "spherical-df-permutation-oracle",
        tuple(
            ElementBasis(
                atomic_number,
                tuple(BasisShell(order, ("0.83",), (("1.0",),)) for order in angular),
            )
            for atomic_number, angular in zip((1, 2), angular_by_atom, strict=True)
        ),
        BasisProvenance("synthetic-test", "1", "test", "0" * 64),
        representation="spherical",
    )
    direction = np.array(((0.04, -0.03, 0.01), (-0.02, 0.05, -0.04)))
    generator = np.random.default_rng(2843)
    with NativeAO(atoms, basis=record, representation="spherical", charge=1) as public:
        public_size = public.nao
        mapping = HessianBasis(public)
        try:
            topology = RKSIntegralTopology.from_basis(mapping.cartesian)
            weights = generator.normal(size=(public.nao,) * 4) * 0.01
            transform = mapping.transform
            cartesian = np.einsum(
                "ai,bj,ck,dl,ijkl->abcd",
                transform,
                transform,
                transform,
                transform,
                weights,
                optimize=True,
            )
            offsets = np.cumsum((0, *topology.shell_sizes))

            def block(slots: tuple[int, ...]) -> np.ndarray:
                return cartesian[
                    tuple(slice(offsets[slot], offsets[slot + 1]) for slot in slots)
                ]

            first = generated_weighted_first_integral_gradient(
                topology, "coulomb", eri_shell_weights=block, cache=tmp_path
            )
            second, diagnostics = generated_weighted_second_integral_hvp(
                topology, "coulomb", direction, eri_shell_weights=block, cache=tmp_path
            )
        finally:
            mapping.close()

    def independent_gradient(displacement: np.ndarray) -> np.ndarray:
        molecule = gto.M(
            atom=[
                (symbol, np.asarray(position) + movement)
                for (symbol, position), movement in zip(
                    atoms, displacement, strict=True
                )
            ],
            basis={
                symbol: [[order, [0.83, 1.0]] for order in angular]
                for (symbol, _), angular in zip(atoms, angular_by_atom, strict=True)
            },
            charge=1,
            spin=0,
            cart=False,
            unit="Bohr",
            verbose=0,
        )
        owners = np.repeat(
            np.arange(molecule.natm),
            np.diff(molecule.aoslice_by_atom()[:, 2:4], axis=1).ravel(),
        )
        electron_gradient = molecule.intor("int2e_ip1", comp=3)
        gradient = np.zeros((len(atoms), 3))
        for slot, permutation in enumerate(
            ((0, 1, 2, 3, 4), (0, 2, 1, 3, 4), (0, 3, 4, 1, 2), (0, 3, 4, 2, 1))
        ):
            derivative = electron_gradient.transpose(permutation)
            for atom in range(len(atoms)):
                mask_shape = tuple(
                    public_size if axis == slot else 1 for axis in range(4)
                )
                mask = (owners == atom).reshape(mask_shape)
                gradient[atom] -= np.einsum("pqrs,xpqrs->x", weights * mask, derivative)
        return gradient

    np.testing.assert_allclose(
        first, independent_gradient(np.zeros_like(direction)), atol=3e-10, rtol=2e-9
    )
    differences = []
    for step in (2e-4, 1e-4):
        differences.append(
            (
                independent_gradient(step * direction)
                - independent_gradient(-step * direction)
            )
            / (2 * step)
        )
    np.testing.assert_allclose(
        second, (4 * differences[1] - differences[0]) / 3, atol=2e-8, rtol=2e-8
    )
    np.testing.assert_allclose(first.sum(axis=0), 0, atol=3e-11)
    np.testing.assert_allclose(second.sum(axis=0), 0, atol=3e-11)
    assert diagnostics["factorized_primitive_component_evaluations"] > 0
    assert diagnostics["symmetry_reduced_eri_shell_quartets"] > 0
    assert diagnostics["peak_host_bytes"] <= diagnostics["budget_bytes"]

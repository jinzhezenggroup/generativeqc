"""Ordered cotangents, bounded schedules and independent one-electron HVPs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.hessian_basis import HessianBasis
from generativeqc.rks_hessian_integrals import (
    RKSIntegralTopology,
    _one_electron_pair_blocks,
    generated_weighted_second_integral_hvp,
)
from generativeqc_compiler.dft import NativeAO
from test_integral_autodiff_second import require_native_library  # noqa: F401


def _schedule_data(backend: str, budget: int) -> dict:
    """Include unequal angular spaces and repeated angular/physical centers."""
    angular = (2, 0, 1, 2)
    sizes = tuple((order + 1) * (order + 2) // 2 for order in angular)
    return {
        "backend": backend,
        "budget_bytes": budget,
        "output_accumulator_bytes": 72,
        "peak_symmetry_weight_bytes": 0,
        "ordered_one_electron_shell_weight_blocks": 0,
        "symmetry_reduced_one_electron_shell_pairs": 0,
        "shells": tuple(SimpleNamespace(angular_momentum=order) for order in angular),
        "state": SimpleNamespace(offsets=np.cumsum((0, *sizes))),
    }


@pytest.mark.parametrize(
    "backend,budget", (("cpu", 1 << 20), ("cpu", 256), ("cuda", 1 << 20))
)
def test_pair_schedule_preserves_arbitrary_ordered_cotangents(
    backend: str, budget: int
) -> None:
    """Test schedule algebra only; the CUDA case does not execute a GPU kernel."""
    data = _schedule_data(backend, budget)
    offsets = data["state"].offsets
    generator = np.random.default_rng(2844)
    weights = generator.normal(size=(offsets[-1],) * 2)
    original = weights.copy()
    jets = generator.normal(size=(9, offsets[-1], offsets[-1]))
    jets += jets.swapaxes(1, 2)
    actual = np.zeros(9)
    for first, second, block, reuse in _one_electron_pair_blocks(data, weights):
        first_slice = slice(offsets[first], offsets[first + 1])
        second_slice = slice(offsets[second], offsets[second + 1])
        actual += np.einsum("xij,ij->x", jets[:, first_slice, second_slice], block)
        if reuse:
            assert (
                data["shells"][first].angular_momentum
                <= data["shells"][second].angular_momentum
            )
    np.testing.assert_allclose(
        actual, np.einsum("xij,ij->x", jets, weights), atol=3e-13
    )
    np.testing.assert_array_equal(weights, original)
    assert data["ordered_one_electron_shell_weight_blocks"] == len(data["shells"]) ** 2
    assert data["peak_symmetry_weight_bytes"] <= budget
    if backend == "cuda":
        assert data["symmetry_reduced_one_electron_shell_pairs"] == 0


def test_fold_overflow_is_not_published() -> None:
    """Finite ordered inputs cannot silently publish a nonfinite folded block."""
    data = _schedule_data("cpu", 1 << 20)
    size = data["state"].offsets[-1]
    weights = np.full((size, size), np.finfo(float).max)
    with pytest.raises(FloatingPointError):
        list(_one_electron_pair_blocks(data, weights))
    assert np.isfinite(weights).all()


@pytest.mark.parametrize("source", ("one_electron", "overlap_pulay"))
def test_generated_one_electron_hvp_matches_independent_moving_gradient(
    tmp_path: Path, source: str
) -> None:
    """Keep mixed charges, signed symmetric weights and repeated p-shell slots."""
    gto = pytest.importorskip("pyscf.gto")
    atoms = [
        ("H", (-0.2, 0.1, -0.7)),
        ("He", (0.3, -0.2, 0.6)),
        ("Li", (1.4, 0.4, -0.1)),
    ]
    angular = (1, 0, 1)
    exponents = (0.71, 0.93, 0.58)
    record = BasisSet(
        "synthetic-one-electron",
        tuple(
            ElementBasis(number, (BasisShell(order, (str(exponent),), (("1.0",),)),))
            for number, order, exponent in zip(
                (1, 2, 3), angular, exponents, strict=True
            )
        ),
        BasisProvenance("synthetic-test", "1", "test", "0" * 64),
        representation="spherical",
    )
    direction = np.array(
        ((0.04, -0.03, 0.01), (-0.02, 0.05, -0.04), (0.01, 0.02, -0.03))
    )
    generator = np.random.default_rng(2847)
    with NativeAO(atoms, basis=record, representation="spherical") as public:
        raw = generator.normal(size=(public.nao, public.nao))
        weights = (raw + raw.T) * 0.05
        original = weights.copy()
        mapping = HessianBasis(public)
        try:
            transform = mapping.transform
            assert transform is not None
            topology = RKSIntegralTopology.from_basis(mapping.cartesian)
            cartesian = transform @ weights @ transform.T
            with pytest.raises(ValueError, match="must be symmetric"):
                generated_weighted_second_integral_hvp(
                    topology,
                    source,
                    direction,
                    pair_weights=transform @ raw @ transform.T,
                    cache=tmp_path,
                )
            actual, diagnostic = generated_weighted_second_integral_hvp(
                topology, source, direction, pair_weights=cartesian, cache=tmp_path
            )
            reverse, _ = generated_weighted_second_integral_hvp(
                topology, source, -direction, pair_weights=cartesian, cache=tmp_path
            )
            np.testing.assert_array_equal(weights, original)
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
                symbol: [[order, [exponent, 1.0]]]
                for (symbol, _), order, exponent in zip(
                    atoms, angular, exponents, strict=True
                )
            },
            cart=False,
            unit="Bohr",
            verbose=0,
        )
        effective = weights + weights.T
        gradient = np.zeros((len(atoms), 3))
        operator = "kin" if source == "one_electron" else "ovlp"
        electronic = molecule.intor(f"int1e_ip{operator}_sph", comp=3)
        if source == "one_electron":
            for nucleus, charge in enumerate(molecule.atom_charges()):
                with molecule.with_rinv_origin(molecule.atom_coord(nucleus)):
                    attraction = molecule.intor("int1e_iprinv_sph", comp=3)
                electronic -= charge * attraction
                gradient[nucleus] -= charge * np.einsum(
                    "xij,ij->x", attraction, effective
                )
        for atom, (_, _, start, stop) in enumerate(molecule.aoslice_by_atom()):
            gradient[atom] -= np.einsum(
                "xij,ij->x", electronic[:, start:stop], effective[start:stop]
            )
        return gradient

    differences = [
        (
            independent_gradient(step * direction)
            - independent_gradient(-step * direction)
        )
        / (2 * step)
        for step in (2e-4, 1e-4)
    ]
    expected = (4 * differences[1] - differences[0]) / 3
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(reverse, -actual, atol=2e-12, rtol=2e-11)
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-12)
    assert diagnostic["symmetry_reduced_one_electron_shell_pairs"] > 0
    if source == "one_electron":
        assert diagnostic["unit_charge_nuclear_shell_pairs"] > 0
    else:
        assert diagnostic["unit_charge_nuclear_shell_pairs"] == 0

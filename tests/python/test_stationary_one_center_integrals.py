"""Exact translation oracle for physical one-center stationary contractions."""

import json
from pathlib import Path

import numpy as np
import pytest
from generativeqc import import_bse
from generativeqc.hessian_basis import HessianBasis
from generativeqc.rks_hessian_integrals import (
    RKSIntegralTopology,
    generated_directional_semilocal_rks_integral_first_order,
    generated_weighted_first_integral_gradient,
    generated_weighted_second_integral_hvp,
)
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.integral.ir import four_center_eri_operator
from generativeqc_compiler.integral.range_separation import CoulombKernel
from test_integral_autodiff_second import require_native_library  # noqa: F401


def test_multicenter_weighted_hvp_keeps_its_small_budget_fallback(
    tmp_path: Path,
) -> None:
    """A smaller admitted lease must preserve arbitrary ordered cotangents."""
    atoms = [("H", (0.1, -0.2, 0.3)), ("H", (-0.4, 0.3, 1.2))]
    direction = np.array(((0.1, -0.2, 0.05), (-0.06, 0.04, 0.03)))
    with NativeAO(atoms) as basis:
        topology = RKSIntegralTopology.from_basis(basis)
        coefficients = np.random.default_rng(2843).normal(size=(len(basis.shells),) * 4)

        def weights(slots: tuple[int, ...]) -> np.ndarray:
            return np.full(
                tuple(topology.shell_sizes[slot] for slot in slots), coefficients[slots]
            )

        common = {"eri_shell_weights": weights, "cache": tmp_path}
        fast, large = generated_weighted_second_integral_hvp(
            topology, "coulomb", direction, **common
        )
        bounded, small = generated_weighted_second_integral_hvp(
            topology, "coulomb", direction, budget_bytes=32 << 10, **common
        )
        np.testing.assert_allclose(fast, bounded, atol=2e-12, rtol=2e-11)
        assert large["factorized_primitive_component_evaluations"] > 0
        assert small["factorized_primitive_component_evaluations"] == 0
        assert small["peak_host_bytes"] <= 32 << 10


def test_one_center_f_sources_are_exact_translations_with_normal_guards(
    tmp_path: Path,
) -> None:
    """Arbitrary physical D/W weights cannot change a translated atomic integral.

    This is an independent analytic invariant, not Ni molecular qualification.
    Keep Cartesian storage only as the existing spherical cotangent workspace.
    """
    data = Path(__file__).resolve().parents[1] / "data/global_hybrid"
    manifest = json.loads((data / "manifest.json").read_text())
    basis = import_bse(
        data / manifest["file"],
        source=manifest["source_repository"],
        source_version=manifest["source_commit"],
        license=manifest["license"],
        representation="spherical",
    )
    with NativeAO(
        [("Ni", (0.1, -0.2, 0.3))], basis=basis, representation="spherical"
    ) as public:
        mapping = HessianBasis(public)
        try:
            topology = RKSIntegralTopology.from_basis(mapping.cartesian)
            assert max(shell.angular_momentum for shell in topology.shells) == 3
            generator = np.random.default_rng(2843)
            factors = generator.normal(size=(public.nao, public.nao)) * 0.01
            density = mapping.density(factors @ factors.T)
            direction = np.array(((0.17, -0.12, 0.08),))
            frozen, overlap = generated_directional_semilocal_rks_integral_first_order(
                topology,
                density,
                direction,
                cache=tmp_path,
                exchange_operators=(
                    (four_center_eri_operator(CoulombKernel("full_range")), 0.37),
                ),
            )
            np.testing.assert_array_equal(frozen, 0)
            np.testing.assert_array_equal(overlap, 0)

            def weights(slots: tuple[int, ...]) -> np.ndarray:
                return np.ones(tuple(topology.shell_sizes[slot] for slot in slots))

            for source in ("one_electron", "overlap_pulay", "coulomb"):
                inputs = (
                    {"eri_shell_weights": weights}
                    if source == "coulomb"
                    else {"pair_weights": density}
                )
                first = generated_weighted_first_integral_gradient(
                    topology, source, cache=tmp_path, **inputs
                )
                second, _ = generated_weighted_second_integral_hvp(
                    topology, source, direction, cache=tmp_path, **inputs
                )
                np.testing.assert_array_equal(first, 0)
                np.testing.assert_array_equal(second, 0)
            with pytest.raises(MemoryError, match="output"):
                generated_weighted_second_integral_hvp(
                    topology,
                    "one_electron",
                    direction,
                    pair_weights=density,
                    budget_bytes=1,
                    cache=tmp_path,
                )
            for order in (1, 2):
                with pytest.raises(ValueError, match="ordered"):
                    if order == 1:
                        generated_weighted_first_integral_gradient(
                            topology,
                            "coulomb",
                            eri_shell_weights=lambda slots: np.ones(1),
                            cache=tmp_path,
                        )
                    else:
                        generated_weighted_second_integral_hvp(
                            topology,
                            "coulomb",
                            direction,
                            eri_shell_weights=lambda slots: np.ones(1),
                            cache=tmp_path,
                        )
            mapping.close()
            with pytest.raises(RuntimeError, match="closed"):
                generated_directional_semilocal_rks_integral_first_order(
                    topology, density, direction, cache=tmp_path
                )
        finally:
            mapping.close()

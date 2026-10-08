"""Independent raw-J gates for the normal PBE0 J-only native integration."""

import os
from dataclasses import replace

import numpy as np
import pytest

from benchmarks._md_j_reference import reference_molecule

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_MD_J_NORMAL") != "1",
    reason="explicit finite Slurm normal-backend qualification required",
)

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize("screening", [0.0, 1e-12, 0.02])
def test_normal_md_j_independent_screened_oracle(
    representation: str, spin: str, screening: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nonsymmetric total densities retain native phases and exact AO masks."""
    from generativeqc.fock import FockBuildSpec, FockPlan, FockTerm
    from generativeqc_compiler.dft import NativeAO

    assert os.environ.get("SLURM_JOB_ID")
    with NativeAO(WATER, "def2-svp", representation=representation) as basis:
        molecule, scales = reference_molecule(basis)
    integrals = molecule.intor("int2e") * np.einsum(
        "i,j,k,l->ijkl", scales, scales, scales, scales
    )
    ao_count = molecule.nao_nr()
    bounds = np.sqrt(
        np.abs(integrals.reshape(ao_count * ao_count, ao_count * ao_count).diagonal())
    ).reshape(ao_count, ao_count)
    screened = integrals * (
        bounds[:, :, None, None] * bounds[None, None, :, :] >= screening
    )
    generator = np.random.default_rng(473)
    density = generator.normal(size=(ao_count, ao_count)) * 0.04
    if spin == "unrestricted":
        density = np.stack(
            (density, generator.normal(size=(ao_count, ao_count)) * 0.03)
        )
    total = density.sum(axis=0) if spin == "unrestricted" else density
    expected = np.einsum("ijkl,kl->ij", screened, total)
    spec = replace(FockBuildSpec.hf(spin, derivative_order=0), exchange=FockTerm(False))
    monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)
    with (
        NativeAO(WATER, "def2-svp", representation=representation) as basis,
        FockPlan(basis, spec, device="cuda", screening_tolerance=screening) as plan,
    ):
        assert plan.diagnostics["direct_schedule"].startswith("md-j-hermite/")
        np.testing.assert_allclose(
            plan.evaluate(density).coulomb, expected, atol=3e-11, rtol=3e-11
        )
        np.testing.assert_allclose(
            plan.evaluate(-0.7 * density).coulomb,
            -0.7 * expected,
            atol=3e-11,
            rtol=3e-11,
        )


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
def test_normal_k_and_bounded_fallback_are_preserved(
    spin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MD admission never displaces the generated K owner or its budget fallback."""
    from generativeqc.fock import FockBuildSpec, FockPlan
    from generativeqc_compiler.dft import NativeAO

    assert os.environ.get("SLURM_JOB_ID")
    spec = FockBuildSpec.hf(spin, derivative_order=0)
    with NativeAO(WATER, "def2-svp", representation="spherical") as basis:
        generator = np.random.default_rng(701)
        density = generator.normal(size=(basis.nao, basis.nao)) * 0.04
        if spin == "unrestricted":
            density = np.stack((density, -0.3 * density.T))
        monkeypatch.setenv("GENERATIVEQC_DISABLE_MD_J", "1")
        with FockPlan(basis, spec, device="cuda") as normal:
            reference = normal.evaluate(density)
            minimum = normal.diagnostics["device_bytes"]
        monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)
        with FockPlan(basis, spec, device="cuda") as md:
            result = md.evaluate(density)
            np.testing.assert_allclose(
                result.exchange, reference.exchange, atol=3e-11, rtol=3e-11
            )
            np.testing.assert_allclose(
                result.coulomb, reference.coulomb, atol=3e-11, rtol=3e-11
            )
        with FockPlan(
            basis, spec, device="cuda", device_budget_bytes=minimum
        ) as bounded:
            assert not bounded.diagnostics["direct_schedule"].startswith("md-j-")
            result = bounded.evaluate(density)
            np.testing.assert_allclose(
                result.exchange, reference.exchange, atol=3e-11, rtol=3e-11
            )
            np.testing.assert_allclose(
                result.coulomb, reference.coulomb, atol=3e-11, rtol=3e-11
            )

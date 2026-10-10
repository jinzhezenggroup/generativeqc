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
    from generativeqc.fock import FockBuildSpec, FockTerm
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe as FockPlan

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
    from generativeqc.fock import FockBuildSpec
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe as FockPlan

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


@pytest.mark.parametrize("screening", [0.0, 0.02])
def test_reciprocal_work_census_preserves_density_consumers(
    screening: float, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Count executed radial sources without confusing residual capacity/work."""
    from generativeqc.fock import FockBuildSpec, FockTerm
    from generativeqc_compiler.dft import NativeAO
    from md_j_resident_probe import ResidentMdJProbe as FockPlan

    monkeypatch.setenv("GENERATIVEQC_MD_J_WORK_COUNTS", "1")
    monkeypatch.delenv("GENERATIVEQC_DISABLE_MD_J", raising=False)
    spec = replace(
        FockBuildSpec.hf("restricted", derivative_order=0),
        exchange=FockTerm(False),
    )
    counts = []
    outputs = []
    with NativeAO(WATER, "def2-svp", representation="spherical") as basis:
        density = np.random.default_rng(5481).normal(size=(basis.nao, basis.nao)) * 0.04
        for mode in ("0", "1", None):
            if mode is None:
                monkeypatch.delenv("GENERATIVEQC_MD_J_RECIPROCAL", raising=False)
            else:
                monkeypatch.setenv("GENERATIVEQC_MD_J_RECIPROCAL", mode)
            capfd.readouterr()
            with FockPlan(
                basis, spec, device="cuda", screening_tolerance=screening
            ) as plan:
                outputs.append(plan.evaluate(density).coulomb)
                rows = [
                    {
                        name: int(value)
                        for name, value in (
                            field.split("=") for field in line.split()[1:]
                        )
                    }
                    for line in capfd.readouterr().err.splitlines()
                    if line.startswith("MD_J_WORK ")
                ]
                assert len(rows) == 25
                counts.append(rows)
                monkeypatch.setenv(
                    "GENERATIVEQC_MD_J_RECIPROCAL", "1" if mode == "0" else "0"
                )
                np.testing.assert_allclose(
                    plan.evaluate(density).coulomb, outputs[-1], atol=3e-11, rtol=3e-11
                )
                replay_rows = [
                    {
                        name: int(value)
                        for name, value in (
                            field.split("=") for field in line.split()[1:]
                        )
                    }
                    for line in capfd.readouterr().err.splitlines()
                    if line.startswith("MD_J_WORK ")
                ]
                assert len(replay_rows) == 25
                assert [row["uniform_roots"] for row in replay_rows] == [
                    row["uniform_roots"] for row in rows
                ]
                plan.evaluate(np.zeros_like(density))
                zero_rows = [
                    line
                    for line in capfd.readouterr().err.splitlines()
                    if line.startswith("MD_J_WORK ")
                ]
                assert len(zero_rows) == 25
                assert all(
                    "uniform_roots=0 " in line and line.endswith("residual_roots=0")
                    for line in zero_rows
                )
    np.testing.assert_allclose(outputs[0], outputs[1], atol=3e-11, rtol=3e-11)
    for normal, reciprocal, default in zip(*counts, strict=True):
        for name in (
            "uniform_directions",
            "uniform_summands",
            "residual_candidates",
            "residual_tasks",
            "residual_roots",
        ):
            assert normal[name] == reciprocal[name] == default[name]
        assert default["uniform_roots"] == reciprocal["uniform_roots"]
    assert sum(row["uniform_roots"] for row in counts[1]) < sum(
        row["uniform_roots"] for row in counts[0]
    )

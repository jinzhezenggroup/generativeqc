"""Independent Libcint gates for the prepared CUDA Direct J/K consumer."""

import os

import numpy as np
import pytest
from generativeqc.fock import FockBuildSpec, FockPlan
from generativeqc_compiler.dft import NativeAO

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_RUNTIME_TEST") != "1",
    reason="requires finite Slurm real-GPU qualification and PySCF oracle",
)


@pytest.mark.parametrize("representation", ("cartesian", "real_spherical"))
@pytest.mark.parametrize("unrestricted", (False, True))
def test_prepared_direct_pair_jk_matches_libcint(
    representation: str, unrestricted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare the production provider's raw J/K with independently computed ERIs.

    The d-shell basis exercises whole-shell recurrence reuse and the public
    spherical projection. The native stream test separately proves execution
    and fallback counts; parity alone cannot establish source selection.
    """
    from pyscf import gto

    assert os.environ.get("SLURM_JOB_ID")
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.7, 0.0, -0.6))]
    molecule = gto.M(
        atom=atoms,
        basis="def2-svp",
        unit="Bohr",
        cart=representation == "cartesian",
        charge=int(unrestricted),
        spin=int(unrestricted),
        verbose=0,
    )
    # Native Cartesian AOs are normalized component by component. Libcint
    # keeps angular normalization in d/f components; transform its ERIs to
    # the exact public native convention before contracting the test density.
    norms = np.sqrt(np.diag(molecule.intor("int1e_ovlp")))
    eri = molecule.intor("int2e") / np.einsum(
        "i,j,k,l->ijkl", norms, norms, norms, norms
    )
    random = (
        np.random.default_rng(1892).normal(size=(molecule.nao, molecule.nao)) * 0.01
    )
    density = 0.1 * np.eye(molecule.nao) + random + random.T
    if unrestricted:
        density = np.stack((density, 0.37 * density))
    total = density.sum(axis=0) if unrestricted else density
    expected_j = np.einsum("ijkl,kl->ij", eri, total)
    expected_k = np.einsum("ijkl,...jl->...ik", eri, density)
    spec = FockBuildSpec.hf("unrestricted" if unrestricted else "restricted")
    with NativeAO(
        atoms,
        basis="def2-svp",
        representation=representation,
        charge=int(unrestricted),
        multiplicity=2 if unrestricted else 1,
    ) as basis:
        for materialized in ("0", "1"):
            monkeypatch.setenv(
                "GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES", materialized
            )
            with FockPlan(
                basis,
                spec,
                device="cuda",
                screening_tolerance=1e-14,
                device_budget_bytes=256 << 20,
            ) as plan:
                # Requests are frozen at preparation. Replaying after an
                # environment change must preserve the prepared operator.
                monkeypatch.setenv(
                    "GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES",
                    "1" if materialized == "0" else "0",
                )
                for _ in range(2):
                    actual = plan.evaluate(density)
                    np.testing.assert_allclose(
                        actual.coulomb, expected_j, atol=3e-10, rtol=0
                    )
                    np.testing.assert_allclose(
                        actual.exchange, expected_k, atol=3e-10, rtol=0
                    )

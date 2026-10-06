"""Independent Libcint gates for the actual prepared pair-precontracted J owner."""

import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from generativeqc.fock import FockBuildSpec, FockPlan
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.fixtures import basis_arguments

from tools.generate_validation_references import pyscf_molecule
from tools.generativeqc_posthf.fixtures import load_fixture

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation and qualified native build",
)


@pytest.mark.parametrize("representation", ("cartesian", "real_spherical"))
@pytest.mark.parametrize("unrestricted", (False, True))
@pytest.mark.parametrize("screening", (0.0, 1e-14))
def test_precontracted_j_matches_libcint_and_freezes_owner(
    representation: str,
    unrestricted: bool,
    screening: float,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Signed primitive weights, nonsymmetric densities and empty replay matter.

    Public J depends on the symmetric part of D, while ordinary K preserves its
    existing unrestricted/non-Hermitian semantics. Both are independent oracle
    checks; trace receipts establish that preparation actually admitted MD.
    """
    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("GENERATIVEQC_DIRECT_J_FOCK_LOWERING", "md")
    monkeypatch.setenv("GENERATIVEQC_DIRECT_K_FOCK_LOWERING", "incumbent")
    receipt = tmp_path / "md-j.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(receipt))
    metadata, _ = load_fixture("water")
    metadata = deepcopy(metadata)
    inputs = metadata["inputs"]
    inputs["basis_representation"] = representation
    inputs["shells"].append(
        {
            "atom_index": 0,
            "angular_momentum": 2,
            "primitives": [[0.8, 0.7], [0.3, -0.2]],
        }
    )
    inputs["shells"].sort(key=lambda shell: shell["atom_index"])
    mol, scale, _ = pyscf_molecule(inputs)
    eri = mol.intor("int2e") * np.einsum("i,j,k,l->ijkl", scale, scale, scale, scale)
    spec = FockBuildSpec.hf("unrestricted" if unrestricted else "restricted")
    with (
        NativeAO(**basis_arguments(metadata)) as basis,
        FockPlan(basis, spec, device="cuda", screening_tolerance=screening) as plan,
    ):
        densities = np.random.default_rng(2026).normal(
            size=(2 if unrestricted else 1, len(scale), len(scale))
        )
        # Changes after preparation cannot silently select the incumbent and
        # make parity pass without testing the new owner.
        monkeypatch.setenv("GENERATIVEQC_DIRECT_J_FOCK_LOWERING", "incumbent")
        for factor in (1.0, -0.7, 0.0, 1.0):
            density = densities * factor
            actual = plan.evaluate(density if unrestricted else density[0])
            expected_j = np.einsum("ijkl,kl->ij", eri, density.sum(axis=0))
            expected_k = np.stack([np.einsum("ikjl,kl->ij", eri, d) for d in density])
            np.testing.assert_allclose(
                actual.coulomb, expected_j, atol=2e-10, rtol=2e-10
            )
            np.testing.assert_allclose(
                actual.exchange,
                expected_k if unrestricted else expected_k[0],
                atol=2e-10,
                rtol=2e-10,
            )
            assert actual.diagnostics["backend"] == "cuda"
    records = [json.loads(line) for line in receipt.read_text().splitlines()]
    j_builds = [record for record in records if record["operation"] == "direct_j"]
    assert len(j_builds) == 4
    assert all(record["counters"]["prepared_md_class_mask"] > 0 for record in j_builds)
    assert all(record["counters"]["md_primitive_pairs"] > 0 for record in j_builds)

"""Scheduled-device LR weighted force gates against independent Libcint."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite scheduler GPU allocation and independent Libcint",
)


@pytest.mark.parametrize("unrestricted", (False, True))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("weighted", ("0", "1"))
def test_lr_weighted_stationary_sources_match_libcint(
    unrestricted: bool,
    representation: str,
    weighted: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A physical fixed-density oracle checks J, SR and LR independently.

    The closed-shell dimer exercises distinct oxygen p/d shells; the charged
    open-shell monomer avoids a separate unstable dimer SCF trajectory. Host
    gates independently cover every class, spin, orientation and pair triangle.
    Repeating the export also protects the stationary source replay contract.
    """
    assert os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from pyscf import gto, lib
    from pyscf.grad import rhf

    lib.num_threads(1)
    monkeypatch.setenv("GENERATIVEQC_DIRECT_LR_WEIGHTED_FORCES", weighted)
    monkeypatch.setenv("GENERATIVEQC_KS_INTEGRAL_RESPONSE_REUSE", "0")
    monkeypatch.setenv("GENERATIVEQC_BOUNDED_ANGULAR_FORCE", "off")
    atoms = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.8)),
        ("H", (1.7, 0.0, -0.6)),
    ]
    if not unrestricted:
        atoms += [
            (element, (position[0] + 1.1, position[1] - 0.7, position[2] + 5.8))
            for element, position in atoms
        ]
    charge, multiplicity = (1, 2) if unrestricted else (0, 1)
    calculator = Calculator(
        method="wb97m-v-uks" if unrestricted else "wb97m-v",
        device="cuda",
        basis="def2-svp",
        basis_representation=representation,
        density_fitting=False,
        precision="fp64",
        ks_options=KsOptions(grid=GridSpec(24, 8, 16)),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    molecule = gto.M(
        atom=atoms,
        basis="def2-svp",
        unit="Bohr",
        charge=charge,
        spin=multiplicity - 1,
        cart=representation == "cartesian",
        verbose=0,
    )
    with (
        calculator.prepare_batch(
            [atoms], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(
            atoms,
            basis="def2-svp",
            representation="real_spherical"
            if representation == "spherical"
            else representation,
            charge=charge,
            multiplicity=multiplicity,
        ) as basis,
    ):
        batch.execute(properties=("energy",), strict=True)
        state = StationaryKsState.from_native(batch, basis)
        try:
            assert not state._source.density_fitted
            assert not state._source.range_density_fitted
            norms = np.sqrt(np.diag(molecule.intor("int1e_ovlp")))
            normalization = np.outer(norms, norms)
            density = np.asarray(state.density) / normalization
            total = density.sum(axis=0)
            coulomb, _ = rhf.get_jk(molecule, total)
            full = [rhf.get_jk(molecule, channel)[1] for channel in density]
            with molecule.with_range_coulomb(0.3):
                long = [rhf.get_jk(molecule, channel)[1] for channel in density]
            occupation = 1 if unrestricted else 2
            expected = np.zeros((3, len(atoms), 3))
            for atom, (_, _, begin, end) in enumerate(molecule.aoslice_by_atom()):
                expected[0, atom] = 2 * np.einsum(
                    "xij,ij->x", coulomb[:, begin:end], total[begin:end]
                )
                for source, coefficient, responses in (
                    (
                        1,
                        -0.15 / occupation,
                        [
                            full_response - long_response
                            for full_response, long_response in zip(
                                full, long, strict=True
                            )
                        ],
                    ),
                    (2, -1.0 / occupation, long),
                ):
                    expected[source, atom] = (
                        2
                        * coefficient
                        * sum(
                            np.einsum(
                                "xij,ij->x", response[:, begin:end], channel[begin:end]
                            )
                            for response, channel in zip(
                                responses, density, strict=True
                            )
                        )
                    )
            for _ in range(2):
                actual = state._source.cuda_integral_derivatives(
                    len(atoms), 512 << 20, range_exchange=True
                )
                assert actual is not None
                np.testing.assert_allclose(actual[0][2:], expected, atol=1e-8, rtol=0)
                np.testing.assert_allclose(
                    actual[0][2:].sum(axis=1), 0, atol=2e-10, rtol=0
                )
        finally:
            state._source.close()

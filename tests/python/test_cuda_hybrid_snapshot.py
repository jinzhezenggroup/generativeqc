"""Real CUDA hybrid exports must preserve the composition used by the SCF owner."""

import os

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._dft_gradient import StationaryKsState
from generativeqc_compiler.dft import NativeAO

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks", "b3lyp-rks", "b3lyp-uks"))
def test_cuda_hybrid_snapshot_matches_cpu_composition_and_state(method: str) -> None:
    """Exercise the native writer and Python reader after a converged solve."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    charge, multiplicity = (1, 2) if method.endswith("-uks") else (0, 1)
    options = KsOptions(
        grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
    )
    exported = {}
    for device in ("cpu", "cuda"):
        calculator = Calculator(
            method=method,
            device=device,
            ks_options=options,
            max_iterations=200,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
        )
        with (
            calculator.prepare_batch(
                [atoms], charges=[charge], multiplicities=[multiplicity]
            ) as batch,
            NativeAO(atoms, charge=charge, multiplicity=multiplicity) as basis,
        ):
            item = batch.execute(strict=True).items[0]
            state = StationaryKsState.from_native(batch, basis)
            try:
                snapshot = state._source
                assert snapshot.backend == device
                assert snapshot.metadata[0] == (6 if device == "cpu" else 8)
                assert snapshot.metadata[7] == (2 if method.startswith("b3lyp") else 1)
                assert item.ks_diagnostic.scf_domain == (
                    "b3lyp-vwn-rpa-tail-v1/density-vacuum-1e-18"
                    if method.startswith("b3lyp")
                    else "semilocal-scaled-v1/pbe-spin-c2-1e-18"
                )
                assert snapshot.coefficients == calculator.ks_options.coefficients
                assert state.identity.method == method
                exported[device] = (
                    item.energy,
                    np.array(state.density),
                    np.array(state.fock),
                )
            finally:
                state._source.close()
    assert exported["cuda"][0] == pytest.approx(exported["cpu"][0], abs=2e-9)
    for cpu, cuda in zip(exported["cpu"][1:], exported["cuda"][1:], strict=True):
        np.testing.assert_allclose(cuda, cpu, atol=2e-8, rtol=0)


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks", "pbe-rks"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("materialized_derivative", ("0", "1"))
@pytest.mark.parametrize("two_oxygens", (False, True))
@pytest.mark.parametrize("force_schedule", ("bounded", "angular", "resident"))
def test_separate_full_range_derivatives_match_libcint(
    method: str,
    representation: str,
    materialized_derivative: str,
    two_oxygens: bool,
    force_schedule: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One shell traversal must still publish independent J'/K' source channels.

    The fixed final density removes SCF differences from this derivative gate.
    Two oxygens exercise nonzero dddd responses across distinct atoms. Libcint
    independently evaluates both sources in Cartesian and spherical layouts.
    """
    from pyscf import gto
    from pyscf.grad import rhf

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    monkeypatch.setenv(
        "GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_DERIVATIVES", materialized_derivative
    )
    monkeypatch.setenv(
        "GENERATIVEQC_BOUNDED_ANGULAR_FORCE",
        "off" if force_schedule == "bounded" else "angular",
    )
    monkeypatch.setenv(
        "GENERATIVEQC_PSSS_RESIDENT_BRA", "1" if force_schedule == "resident" else "0"
    )
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.7, 0.0, -0.6))]
    if two_oxygens:
        atoms += [
            (element, (x + 1.1, y - 0.7, z + 5.8)) for element, (x, y, z) in atoms
        ]
    unrestricted = method.endswith("uks")
    charge, multiplicity = (1, 2) if unrestricted else (0, 1)
    calculator = Calculator(
        method=method,
        device="cuda",
        basis="def2-svp",
        basis_representation=representation,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
        ),
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
            actual = state._source.cuda_full_range_derivatives(len(atoms))
            assert actual is not None
            overlap = molecule.intor("int1e_ovlp")
            norms = np.sqrt(np.diag(overlap))
            normalization = np.outer(norms, norms)
            np.testing.assert_allclose(
                state.overlap, overlap / normalization, atol=2e-12
            )
            density = np.asarray(state.density) / normalization
            total_density = density.sum(axis=0)
            coulomb, _ = rhf.get_jk(molecule, total_density)
            exchange = [
                rhf.get_jk(molecule, spin_density)[1] for spin_density in density
            ]
            # The snapshot carries the signed Fock coefficient: -alpha/2 for
            # RKS and -alpha for UKS, not the positive exchange fraction.
            exchange_coefficient = state._source.coefficients[2]
            expected = np.zeros_like(actual)
            for atom, (_, _, begin, end) in enumerate(molecule.aoslice_by_atom()):
                expected[0, atom] = 2 * np.einsum(
                    "xij,ij->x", coulomb[:, begin:end], total_density[begin:end]
                )
                expected[1, atom] = (
                    2
                    * exchange_coefficient
                    * sum(
                        np.einsum(
                            "xij,ij->x", response[:, begin:end], spin_density[begin:end]
                        )
                        for response, spin_density in zip(
                            exchange, density, strict=True
                        )
                    )
                )
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
            # Total-force realization precontracts J/K together; raw exports
            # retain independent sources from the same final D/W frame.
            combined = state._source.cuda_integral_derivatives(
                len(atoms),
                512 * 1024 * 1024,
                range_exchange=False,
                combined_two_electron=True,
            )
            assert combined is not None
            values, _ = combined
            assert values.shape == (3, len(atoms), 3)
            np.testing.assert_allclose(
                values[2], expected.sum(axis=0), atol=1e-8, rtol=0
            )
            separate = state._source.cuda_full_range_derivatives(len(atoms))
            np.testing.assert_allclose(separate, expected, atol=1e-8, rtol=0)
            np.testing.assert_allclose(actual.sum(axis=1), 0, atol=2e-10, rtol=0)
            if not exchange_coefficient:
                np.testing.assert_array_equal(actual[1], 0)
        finally:
            state._source.close()


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("force_schedule", ("bounded", "angular"))
def test_cooperative_full_range_derivatives_match_libcint(
    method: str,
    representation: str,
    force_schedule: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cooperative opt-in retains both source contracts and bounded fallback.

    Two oxygens admit the four heavy order-six/seven s/p/d classes. Reuse the
    independent fixed-density Libcint oracle, including Separate-after-Combined
    replay, while keeping the materialized dddd opt-in independently disabled.
    """
    monkeypatch.setenv("GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES", "1")
    test_separate_full_range_derivatives_match_libcint(
        method, representation, "0", True, force_schedule, monkeypatch
    )

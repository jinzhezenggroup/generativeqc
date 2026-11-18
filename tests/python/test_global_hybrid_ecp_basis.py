"""Pinned def2-SVP/ECP input and genuine spherical-space gates for issue #2846."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tomllib
from generativeqc import import_bse
from generativeqc_compiler.dft import NativeAO
from test_integral_autodiff_second import require_native_library  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/global_hybrid_ecp"
CORES = {1: 0, 28: 0, 53: 28, 55: 46}


@pytest.fixture(scope="module")
def canonical_records() -> tuple:
    """Runtime inputs retain canonical data; BSE is an offline exporter only."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    payload = DATA / manifest["file"]
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == manifest["sha256"]
    record = import_bse(
        payload,
        source=manifest["source_repository"],
        source_version=manifest["source_commit"],
        license=manifest["license"],
        representation="spherical",
    )
    return manifest, json.loads(payload.read_text()), record


def _independent_ecp(raw: dict) -> dict:
    """Map BSE radial powers and the highest/local channel to PySCF's input ABI."""
    elements = pytest.importorskip("pyscf.data.elements")
    result = {}
    for number, element in raw["elements"].items():
        potentials = element.get("ecp_potentials", ())
        if not potentials:
            continue
        local = max(potential["angular_momentum"][0] for potential in potentials)
        channels = []
        for potential in potentials:
            assert len(potential["angular_momentum"]) == 1
            assert len(potential["coefficients"]) == 1
            power_blocks = [[] for _ in range(max(potential["r_exponents"]) + 1)]
            for power, exponent, coefficient in zip(
                potential["r_exponents"],
                potential["gaussian_exponents"],
                potential["coefficients"][0],
                strict=True,
            ):
                power_blocks[power].append([float(exponent), float(coefficient)])
            angular = potential["angular_momentum"][0]
            channels.append([-1 if angular == local else angular, power_blocks])
        result[elements.ELEMENTS[int(number)]] = [element["ecp_electrons"], channels]
    return result


def test_def2_ecp_records_bind_primary_source_and_nuclear_core_identities(
    canonical_records: tuple,
) -> None:
    """Core removal changes the electron/ionic model, never elemental identity."""
    manifest, raw, record = canonical_records
    sources = tomllib.loads((ROOT / "tools/reference_sources.toml").read_text())[
        "repository"
    ]
    source = next(item for item in sources if item["name"] == "basis_set_exchange")
    assert manifest["source_commit"] == source["commit"]
    assert manifest["license_sha256"] == source["license_sha256"]
    assert (
        hashlib.sha256((DATA / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    assert record.provenance.checksum == manifest["sha256"]
    assert manifest["ecp_core_electrons"] == {"53": 28, "55": 46}
    assert {int(number) for number in raw["elements"]} == set(CORES)
    for element in record.elements:
        assert element.nuclear_charge == element.atomic_number
        assert element.ecp_core_electrons == CORES[element.atomic_number]
        assert bool(element.ecp_data) == bool(element.ecp_core_electrons)
    nickel = next(element for element in record.elements if element.atomic_number == 28)
    assert max(shell.angular_momentum for shell in nickel.shells) == 3


@pytest.mark.parametrize("charge,multiplicity", ((0, 1), (1, 2)))
@pytest.mark.parametrize("case", ("ih", "csh", "niih"))
def test_def2_ecp_spherical_ao_jets_and_valence_counts_match_independent_inputs(
    case: str, charge: int, multiplicity: int, canonical_records: tuple
) -> None:
    """Qualify input-space semantics only, not B3LYP force or Hessian admission."""
    gto = pytest.importorskip("pyscf.gto")
    numint = pytest.importorskip("pyscf.dft.numint")
    from tools.generate_external_basis_references import pyscf_basis

    _, raw, record = canonical_records
    atoms = [("I", (0.03, -0.02, 0.01)), ("H", (0.17, 0.11, 3.1))]
    numbers = [53, 1]
    if case == "csh":
        atoms[0] = ("Cs", atoms[0][1])
        numbers[0] = 55
    elif case == "niih":
        atoms.insert(0, ("Ni", (-2.7, 0.2, -0.1)))
        numbers.insert(0, 28)
    reference = gto.M(
        atom=atoms,
        basis=pyscf_basis(raw),
        ecp=_independent_ecp(raw),
        unit="Bohr",
        cart=False,
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    assert (
        reference.nelectron
        == sum(number - CORES[number] for number in numbers) - charge
    )
    assert [reference.atom_nelec_core(atom) for atom in range(len(atoms))] == [
        CORES[number] for number in numbers
    ]
    points = np.array(
        ((0.2, 0.3, 0.4), (1.1, -0.4, 0.7), (4.4, 0.8, -0.6), (-0.2, 2.7, 0.1))
    )
    with NativeAO(
        atoms,
        basis=record,
        representation="spherical",
        charge=charge,
        multiplicity=multiplicity,
    ) as native:
        assert [atom.atomic_number for atom in native.atoms] == numbers
        assert native.nao == reference.nao_nr() < reference.nao_cart()
        np.testing.assert_allclose(
            native.evaluate(points, 2),
            numint.eval_ao(reference, points, deriv=2),
            atol=5e-11,
            rtol=2e-11,
        )


@pytest.mark.parametrize(
    "case", ("cs-rks", "cs-uks", "i-rks", "i-uks", "ni-i-stationary-rks")
)
def test_def2_ecp_active_cube_rotation_preserves_ao_metric_and_default_grid(
    case: str, canonical_records: tuple
) -> None:
    """Qualify the rotation input contract, not continuum Hessian invariance.

    A proper axis cycle preserves the finite angular rule. The independent AO
    transform takes passive axes, so its argument is the transpose of the active
    coordinate rotation. Both pointwise AOs and the overlap certify this choice.
    """
    gto = pytest.importorskip("pyscf.gto")
    spatial = pytest.importorskip("scipy.spatial")
    from generativeqc import GridSpec
    from generativeqc_compiler.dft.grid import MolecularGrid

    from tools.generate_external_basis_references import pyscf_basis

    specification = json.loads((DATA / "native_curvature_manifest.json").read_text())[
        "cases"
    ][case]
    atoms = specification["atoms_bohr"]
    rotation = np.array(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
    assert np.linalg.det(rotation) == 1
    rotated_atoms = [
        (symbol, tuple(rotation @ np.asarray(center))) for symbol, center in atoms
    ]
    _, raw, record = canonical_records
    reference = gto.M(
        atom=atoms,
        basis=pyscf_basis(raw),
        ecp=_independent_ecp(raw),
        unit="Bohr",
        cart=False,
        charge=specification["charge"],
        spin=specification["multiplicity"] - 1,
        verbose=0,
    )
    transform = gto.mole.ao_rotation_matrix(reference, rotation.T)
    rotated = reference.copy()
    rotated.set_geom_(rotated_atoms, unit="Bohr")
    np.testing.assert_allclose(
        rotated.intor_symmetric("int1e_ovlp"),
        transform @ reference.intor_symmetric("int1e_ovlp") @ transform.T,
        atol=1e-12,
        rtol=0,
    )
    probes = np.array(((0.21, -0.37, 0.68), (1.41, 0.23, -0.91), (-0.57, 1.81, 0.33)))
    with NativeAO(
        rotated_atoms,
        basis=record,
        representation="spherical",
        charge=specification["charge"],
        multiplicity=specification["multiplicity"],
    ) as native:
        np.testing.assert_allclose(
            native.evaluate(probes @ rotation.T, 0)[0],
            reference.eval_gto("GTOval_sph", probes) @ transform.T,
            atol=5e-11,
            rtol=2e-11,
        )
    grid = MolecularGrid(tuple(atoms), GridSpec.treutler(3), 2, 3).explicit(
        max_points=200_000
    )
    rotated_grid = MolecularGrid(
        tuple(rotated_atoms), GridSpec.treutler(3), 2, 3
    ).explicit(max_points=200_000)
    distances, indices = spatial.cKDTree(rotated_grid.points).query(
        grid.points @ rotation.T
    )
    assert len(np.unique(indices)) == len(indices) == len(rotated_grid.points)
    assert distances.max() < 1e-10
    np.testing.assert_allclose(
        grid.weights, rotated_grid.weights[indices], atol=1e-11, rtol=1e-12
    )

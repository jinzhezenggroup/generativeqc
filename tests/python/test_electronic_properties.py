"""Observable electronic properties against independent libcint/PySCF oracles."""

from __future__ import annotations

import ctypes
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator, Primitive, Shell, _native
from generativeqc._electronic_properties import _bind, _pointer

if TYPE_CHECKING:
    from pathlib import Path

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]
PROPERTIES = ("energy", "dipole", "charges", "density")
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
            or not os.environ.get("SLURM_JOB_ID"),
            reason="requires an explicitly Slurm-allocated GPU",
        ),
    ),
]


def _oracle_basis(basis: str) -> dict[str, Any]:
    """Share exact decimal inputs, not native normalization or integral outputs."""
    calculator = Calculator(basis=basis, device="cpu")
    atoms = tuple(map(Atom.from_value, WATER))
    shells = calculator._shells_for_atoms(atoms)
    return {
        element: [
            [
                shell.angular_momentum,
                *[
                    [primitive.exponent, primitive.coefficient]
                    for primitive in shell.primitives
                ],
            ]
            for shell in shells
            if shell.atom_index == atom
        ]
        for element, atom in (("O", 0), ("H", 1))
    }


def _oracle(basis: str, charge: int = 0, multiplicity: int = 1) -> Any:
    pyscf = pytest.importorskip("pyscf")
    return pyscf.gto.M(
        atom=WATER,
        unit="Bohr",
        basis=_oracle_basis(basis),
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )


@pytest.mark.parametrize("density_fitting", ["none", "cpu"])
@pytest.mark.parametrize("basis", ["sto-3g", "def2-svp"])
@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
@pytest.mark.parametrize("device", DEVICES)
def test_hf_properties_match_independent_scf(
    density_fitting: str,
    basis: str,
    method: str,
    charge: int,
    multiplicity: int,
    device: str,
) -> None:
    """Test the occupied state, operator normalization, partitions and AO samples."""
    molecule = _oracle(basis, charge, multiplicity)
    from pyscf import scf

    reference = scf.RHF(molecule) if method == "rhf" else scf.UHF(molecule)
    if density_fitting != "none":
        reference = reference.density_fit(auxbasis=_oracle_basis("def2-svp"))
    reference.conv_tol = 1e-12
    reference.kernel()
    assert reference.converged
    calculator = Calculator(
        method=method,
        basis=basis,
        basis_representation="spherical",
        device=device,
        density_fitting="cuda"
        if device == "cuda" and density_fitting != "none"
        else density_fitting,
        auxiliary_basis="def2-svp" if density_fitting != "none" else None,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    result = calculator.singlepoint(
        WATER,
        charge=charge,
        multiplicity=multiplicity,
        properties=PROPERTIES,
    )
    assert result.density.device == device
    assert result.executed_backend == ("cpu_reference" if device == "cpu" else "cuda")
    density = reference.make_rdm1()
    total = density if density.ndim == 2 else density.sum(axis=0)
    spin = np.zeros_like(total) if density.ndim == 2 else density[0] - density[1]
    overlap = molecule.intor("int1e_ovlp")
    position = molecule.intor("int1e_r")
    nuclei = np.einsum("i,ij->j", molecule.atom_charges(), molecule.atom_coords())
    expected_dipole = nuclei - np.einsum("ij,kji->k", total, position)
    np.testing.assert_allclose(result.energy, reference.e_tot, atol=2e-9, rtol=0)
    np.testing.assert_allclose(result.density.matrix, total, atol=3e-7, rtol=0)
    np.testing.assert_allclose(result.dipole, expected_dipole, atol=3e-7, rtol=0)
    np.testing.assert_allclose(
        result.dipole_debye, result.dipole * 2.541746473, atol=1e-13
    )
    np.testing.assert_allclose(
        result.charges, reference.mulliken_pop(verbose=0)[1], atol=3e-7, rtol=0
    )
    eigenvalues, vectors = np.linalg.eigh(overlap)
    root = (vectors * np.sqrt(eigenvalues)) @ vectors.T
    lowdin_population = np.diag(root @ total @ root)
    expected_charges = [
        molecule.atom_charge(atom) - lowdin_population[begin:end].sum()
        for atom, (_, _, begin, end) in enumerate(molecule.aoslice_by_atom())
    ]
    np.testing.assert_allclose(
        result.lowdin_charges, expected_charges, atol=3e-7, rtol=0
    )
    assert result.charges.sum() == pytest.approx(charge, abs=1e-10)
    assert result.lowdin_charges.sum() == pytest.approx(charge, abs=1e-10)
    points = np.array([[0.1, 0.2, 0.3], [-0.4, 0.8, 1.1], [1.5, -0.1, -0.2]])
    orbitals = molecule.eval_gto("GTOval_sph", points)
    expected = np.einsum("pi,ij,pj->p", orbitals, total, orbitals)
    expected_spin = np.einsum("pi,ij,pj->p", orbitals, spin, orbitals)
    np.testing.assert_allclose(
        result.density.evaluate(points, block_size=2), expected, atol=3e-7, rtol=0
    )
    np.testing.assert_allclose(
        result.density.evaluate(points, spin=True), expected_spin, atol=3e-7, rtol=0
    )
    assert result.forces is None
    assert not result.density.matrix.flags.writeable


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_position_integrals_spdf_match_libcint(representation: str) -> None:
    """The raised f polynomial uses g overlap without changing AO normalization."""
    gto = pytest.importorskip("pyscf.gto")
    atoms = [("He", (0.3, -0.4, 0.2)), ("He", (-0.5, 0.8, 0.9))]
    basis = [[angular, [0.7, 0.8], [0.2, -0.13]] for angular in range(4)]
    molecule = gto.M(
        atom=atoms,
        unit="Bohr",
        basis={"He": basis},
        cart=representation == "cartesian",
        verbose=0,
    )
    shells = [
        Shell(atom, angular, (Primitive(0.7, 0.8), Primitive(0.2, -0.13)))
        for atom in range(2)
        for angular in range(4)
    ]
    calculator = Calculator(basis=shells, basis_representation=representation)
    library = calculator._library
    _bind(library)
    context = ctypes.c_void_p()
    system = ctypes.c_void_p()
    _native.check(
        library,
        library.generativeqc_context_create(
            ctypes.byref(calculator._context_descriptor()),
            ctypes.byref(context),
        ),
    )
    try:
        from generativeqc import Atom

        system = calculator._create_native_system(
            context, tuple(map(Atom.from_value, atoms)), 0, 1
        )
        actual = np.empty((3, molecule.nao, molecule.nao))
        _native.check(
            library,
            library.generativeqc_system_position_integrals_cpu_v1(
                context,
                system,
                _pointer(actual),
                actual.size,
            ),
            context=context,
        )
        normalization = np.sqrt(np.diag(molecule.intor("int1e_ovlp")))
        expected = (
            molecule.intor("int1e_r")
            / normalization[None, :, None]
            / normalization[None, None, :]
        )
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
        status = library.generativeqc_system_position_integrals_cpu_v1(
            context, system, _pointer(actual), actual.size - 1
        )
        assert status == _native.STATUS_INVALID_ARGUMENT
    finally:
        library.generativeqc_system_destroy(system)
        library.generativeqc_context_destroy(context)


@pytest.mark.parametrize(
    "method,charge,multiplicity",
    [("pbe-rks", 0, 1), ("pbe-uks", 1, 2), ("pbe0-rks", 0, 1)],
)
@pytest.mark.parametrize("density_fitting", ["none", "cpu"])
@pytest.mark.parametrize("device", DEVICES)
def test_ks_outputs_use_the_accepted_state(
    method: str,
    charge: int,
    multiplicity: int,
    density_fitting: str,
    device: str,
) -> None:
    """Independent fixed-state operators verify the KS export and spin convention."""
    molecule = _oracle("sto-3g", charge, multiplicity)
    calculator = Calculator(
        method=method,
        basis="sto-3g",
        device=device,
        density_fitting="cuda"
        if device == "cuda" and density_fitting != "none"
        else density_fitting,
        auxiliary_basis="def2-svp" if density_fitting != "none" else None,
    )
    ordinary = calculator.singlepoint(
        WATER, charge=charge, multiplicity=multiplicity, properties=("energy",)
    )
    result = calculator.singlepoint(
        WATER, charge=charge, multiplicity=multiplicity, properties=PROPERTIES
    )
    assert result.energy == pytest.approx(ordinary.energy, abs=1e-12)
    total = result.density.matrix
    assert np.einsum("ij,ji", total, molecule.intor("int1e_ovlp")) == pytest.approx(
        10 - charge, abs=1e-10
    )
    nuclei = np.einsum("i,ij->j", molecule.atom_charges(), molecule.atom_coords())
    expected = nuclei - np.einsum("ij,kji->k", total, molecule.intor("int1e_r"))
    np.testing.assert_allclose(result.dipole, expected, atol=1e-11)
    if multiplicity == 2:
        assert np.einsum(
            "ij,ji", result.density.spin_matrix, molecule.intor("int1e_ovlp")
        ) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("charge,multiplicity,method", [(0, 1, "rhf"), (1, 2, "uhf")])
@pytest.mark.parametrize("device", DEVICES)
def test_dipole_translation_and_charge_conservation(
    charge: int, multiplicity: int, method: str, device: str
) -> None:
    calculator = Calculator(method=method, basis="sto-3g", device=device)
    shift = np.array([0.7, -0.4, 0.2])
    shifted = [(element, np.array(xyz) + shift) for element, xyz in WATER]
    first = calculator.singlepoint(
        WATER, charge=charge, multiplicity=multiplicity, properties=PROPERTIES
    )
    second = calculator.singlepoint(
        shifted, charge=charge, multiplicity=multiplicity, properties=PROPERTIES
    )
    np.testing.assert_allclose(
        second.dipole, first.dipole + charge * shift, atol=2e-9, rtol=0
    )
    np.testing.assert_allclose(second.charges, first.charges, atol=2e-9, rtol=0)
    np.testing.assert_allclose(
        second.lowdin_charges, first.lowdin_charges, atol=2e-9, rtol=0
    )


@pytest.mark.parametrize("device", DEVICES)
def test_density_lifetime_and_cube(tmp_path: Path, device: str) -> None:
    calculator = Calculator(method="rhf", basis="sto-3g", device=device)
    result = calculator.singlepoint(WATER, properties=PROPERTIES)
    calculator.clear_cache()
    del calculator
    path = tmp_path / "density.cube"
    origin, shape, spacing = (
        np.array([-0.4, 0.1, 0.3]),
        (2, 3, 5),
        np.array([0.2, 0.3, 0.4]),
    )
    result.density.write_cube(
        path, origin=origin, shape=shape, spacing=spacing, block_size=7
    )
    lines = path.read_text().splitlines()
    assert int(lines[2].split()[0]) == len(WATER)
    np.testing.assert_allclose(list(map(float, lines[2].split()[1:])), origin)
    for axis in range(3):
        record = list(map(float, lines[3 + axis].split()))
        assert record[0] == shape[axis]
        np.testing.assert_allclose(record[1:], np.eye(3)[axis] * spacing[axis])
    values = np.fromstring(" ".join(lines[6 + len(WATER) :]), sep=" ")
    indices = np.column_stack(np.unravel_index(np.arange(np.prod(shape)), shape))
    expected = result.density.evaluate(origin + indices * spacing)
    np.testing.assert_allclose(values, expected, rtol=5e-10, atol=1e-12)
    assert result.density.evaluate(np.empty((0, 3))).shape == (0,)
    assert result.density.evaluate([[0, 0, 0]], spin=True)[0] == 0


@pytest.mark.parametrize("points", [[[0, 1]], [[0, 0, np.inf]], [[0j, 0, 0]]])
def test_density_rejects_invalid_points(points: Any) -> None:
    density = Calculator().singlepoint(WATER, properties=("energy", "density")).density
    with pytest.raises(ValueError):
        density.evaluate(points)


def test_unrequested_fields_and_unsupported_domains(tmp_path: Path) -> None:
    calculator = Calculator(method="rhf", basis="sto-3g", device="cpu")
    result = calculator.singlepoint(WATER, properties=("energy",))
    assert (
        result.dipole
        is result.dipole_debye
        is result.charges
        is result.lowdin_charges
        is result.density
        is None
    )
    assert calculator.supported_electronic_properties == frozenset(
        {
            "dipole",
            "charges",
            "density",
        }
    )
    with calculator.prepare_batch([WATER]) as batch:
        assert "dipole" not in batch.capabilities.supported_properties
        with pytest.raises(ValueError):
            batch.execute(properties=("energy", "dipole"))
    density = calculator.singlepoint(WATER, properties=("energy", "density")).density
    for arguments in ({"shape": (2, 0, 2)}, {"shape": (2, 2, 2), "spacing": -0.1}):
        with pytest.raises(ValueError):
            density.write_cube(tmp_path / "invalid.cube", origin=(0, 0, 0), **arguments)
    assert not (tmp_path / "invalid.cube").exists()
    with pytest.raises(ValueError, match="correlated"):
        Calculator(method="mp2").singlepoint(WATER, properties=("energy", "density"))


@pytest.mark.parametrize("device", DEVICES)
def test_electronic_outputs_can_include_forces(device: str) -> None:
    calculator = Calculator(method="rhf", basis="sto-3g", device=device)
    expected = calculator.singlepoint(WATER)
    result = calculator.singlepoint(WATER, properties=(*PROPERTIES, "forces"))
    assert result.energy == pytest.approx(expected.energy, abs=1e-12)
    np.testing.assert_allclose(result.forces, expected.forces, atol=1e-12)
    assert result.dipole is not None


def test_nonconvergence_does_not_publish_density() -> None:
    calculator = Calculator(
        method="rhf", basis="sto-3g", device="cpu", max_iterations=1
    )
    with pytest.raises(RuntimeError):
        calculator.singlepoint(WATER, properties=PROPERTIES)


def test_native_capture_validates_buffers_and_recovers() -> None:
    """The additive C endpoint rejects invalid buffers and permits a later solve."""
    calculator = Calculator(method="rhf", basis="sto-3g", device="cpu")
    library = calculator._library
    _bind(library)
    context, system, calculation = (ctypes.c_void_p() for _ in range(3))
    _native.check(
        library,
        library.generativeqc_context_create(
            ctypes.byref(calculator._context_descriptor()),
            ctypes.byref(context),
        ),
    )
    try:
        system = calculator._create_native_system(
            context, tuple(map(Atom.from_value, WATER)), 0, 1
        )
        descriptor = calculator._method_descriptor(
            systems=(tuple(map(Atom.from_value, WATER)),)
        )
        _native.check(
            library,
            library.generativeqc_calculation_prepare(
                context,
                system,
                ctypes.byref(descriptor),
                ctypes.byref(calculation),
            ),
            context=context,
        )
        result = _native.ResultDescriptor()
        result.struct_size = ctypes.sizeof(result)
        result.abi_version = _native.ABI_VERSION
        density = np.full((7, 7), np.nan)
        for pointer, capacity in (
            (None, 0),
            (None, density.size),
            (_pointer(density), 0),
            (_pointer(density), density.size - 1),
        ):
            status = library.generativeqc_calculation_execute_density_v1(
                calculation,
                ctypes.byref(result),
                pointer,
                capacity,
            )
            assert status == _native.STATUS_INVALID_ARGUMENT
        _native.check(
            library,
            library.generativeqc_calculation_execute_density_v1(
                calculation,
                ctypes.byref(result),
                _pointer(density),
                density.size,
            ),
            context=context,
        )
        assert result.converged
        assert np.isfinite(density).all()
    finally:
        library.generativeqc_calculation_destroy(calculation)
        library.generativeqc_system_destroy(system)
        library.generativeqc_context_destroy(context)


def test_cuda_property_owner_rejects_cpu_context() -> None:
    """An explicitly CUDA consumer must not silently acquire a host fallback."""
    calculator = Calculator(method="rhf", basis="sto-3g", device="cpu")
    library = calculator._library
    _bind(library)
    context = ctypes.c_void_p()
    _native.check(
        library,
        library.generativeqc_context_create(
            ctypes.byref(calculator._context_descriptor()), ctypes.byref(context)
        ),
    )
    system = None
    try:
        system = calculator._create_native_system(
            context, tuple(map(Atom.from_value, WATER)), 0, 1
        )
        handle = ctypes.c_void_p(123)
        density = np.eye(7)
        status = library.generativeqc_electronic_properties_cuda_create_v1(
            context, system, _pointer(density), density.size, 3, ctypes.byref(handle)
        )
        assert status == _native.STATUS_NOT_IMPLEMENTED
        assert not handle.value
    finally:
        if system is not None:
            library.generativeqc_system_destroy(system)
        library.generativeqc_context_destroy(context)


def test_native_ks_density_requires_an_accepted_item() -> None:
    calculator = Calculator(method="pbe-rks", basis="sto-3g", device="cpu")
    library = calculator._library
    _bind(library)
    density = np.full((7, 7), np.nan)
    with calculator.prepare_batch([WATER], warm_start=False) as batch:
        status = library.generativeqc_ks_batch_copy_density_v1(
            batch._batch, 0, _pointer(density), density.size
        )
        assert status != _native.STATUS_SUCCESS
        batch.execute(strict=True, properties=("energy",))
        status = library.generativeqc_ks_batch_copy_density_v1(
            batch._batch, 1, _pointer(density), density.size
        )
        assert status == _native.STATUS_INVALID_ARGUMENT
        status = library.generativeqc_ks_batch_copy_density_v1(
            batch._batch, 0, _pointer(density), density.size - 1
        )
        assert status == _native.STATUS_INVALID_ARGUMENT
        _native.check(
            library,
            library.generativeqc_ks_batch_copy_density_v1(
                batch._batch,
                0,
                _pointer(density),
                density.size,
            ),
            context=batch._context,
        )
    assert np.isfinite(density).all()


def test_cartesian_density_samples_match_independent_aos() -> None:
    """Cartesian d-component normalization must match both property consumers."""
    molecule = _oracle("def2-svp")
    molecule.cart = True
    result = Calculator(basis="def2-svp", basis_representation="cartesian").singlepoint(
        WATER, properties=PROPERTIES
    )
    points = np.array([[0.1, 0.2, 0.3], [-0.4, 0.8, 1.1], [1.5, -0.1, -0.2]])
    normalization = np.sqrt(np.diag(molecule.intor("int1e_ovlp")))
    values = molecule.eval_gto("GTOval_cart", points) / normalization
    expected = np.einsum("pi,ij,pj->p", values, result.density.matrix, values)
    np.testing.assert_allclose(
        result.density.evaluate(points), expected, rtol=1e-11, atol=1e-11
    )


@pytest.mark.parametrize("property_name", ["dipole", "charges", "density"])
@pytest.mark.parametrize("device", DEVICES)
def test_electronic_output_selection_is_independent(
    property_name: str, device: str
) -> None:
    result = Calculator(device=device).singlepoint(
        WATER, properties=("energy", property_name)
    )
    assert (result.dipole is not None) == (property_name == "dipole")
    assert (result.charges is not None) == (property_name == "charges")
    assert (result.lowdin_charges is not None) == (property_name == "charges")
    assert (result.density is not None) == (property_name == "density")
    assert result.forces is None


@pytest.mark.parametrize("device", DEVICES)
def test_spin_cube_contains_alpha_minus_beta(tmp_path: Path, device: str) -> None:
    density = (
        Calculator(method="uhf", device=device)
        .singlepoint(
            WATER,
            charge=1,
            multiplicity=2,
            properties=("energy", "density"),
        )
        .density
    )
    path = tmp_path / "spin.cube"
    density.write_cube(
        path, origin=(0.1, 0.2, 0.3), shape=(2, 2, 2), spin=True, block_size=3
    )
    lines = path.read_text().splitlines()
    assert lines[1] == "Alpha minus beta spin density"
    indices = np.column_stack(np.unravel_index(np.arange(8), (2, 2, 2)))
    expected = density.evaluate(np.array([0.1, 0.2, 0.3]) + indices * 0.2, spin=True)
    values = np.fromstring(" ".join(lines[6 + len(WATER) :]), sep=" ")
    np.testing.assert_allclose(values, expected, rtol=5e-10, atol=1e-12)


def test_property_analysis_rejects_explicit_resource_budget() -> None:
    from generativeqc import ResourceBudget

    calculator = Calculator(resource_budget=ResourceBudget())
    with pytest.raises(NotImplementedError, match="resource budgets"):
        calculator.singlepoint(WATER, properties=PROPERTIES)

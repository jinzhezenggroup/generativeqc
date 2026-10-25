"""Independent, identically discretized CPU/CUDA semilocal SCF endpoints."""

import ctypes
import typing

import numpy as np
import pytest
from generativeqc import Atom, Calculator, _native, method_capabilities
from generativeqc_compiler.dft.grid import MolecularGrid


@pytest.fixture(params=("cpu", "cuda"))
def device(request: typing.Any, monkeypatch: typing.Any) -> typing.Any:
    """Only an unavailable CUDA context can skip; numerical failures must fail."""
    monkeypatch.setenv("GENERATIVEQC_PROFILE", "off")
    if request.param == "cuda":
        library = _native.load_library()
        descriptor = _native.ContextDescriptor(
            ctypes.sizeof(_native.ContextDescriptor),
            _native.ABI_VERSION,
            0,
            _native.BACKEND_CUDA,
        )
        context = ctypes.c_void_p()
        try:
            _native.check(
                library,
                library.generativeqc_context_create(
                    ctypes.byref(descriptor), ctypes.byref(context)
                ),
            )
        except RuntimeError as error:
            pytest.skip(f"CUDA context unavailable: {error}")
        library.generativeqc_context_destroy(context)
    return request.param


@pytest.mark.parametrize("method", ("lda-uks", "pbe-uks"))
def test_uks_public_energy_and_spin_contract(
    method: typing.Any, device: typing.Any
) -> None:
    capabilities = method_capabilities(method)
    assert capabilities.available
    assert capabilities.supported_properties == frozenset(("energy",))
    calculator = Calculator(method=method, basis="sto-3g", device=device)
    atoms = [("H", (0, 0, 0))]
    result = calculator.singlepoint(atoms, multiplicity=2, properties=("energy",))
    assert result.converged and result.forces is None
    assert result.executed_backend == ("cpu_reference" if device == "cpu" else "cuda")
    assert calculator.capabilities.supported_properties == frozenset(
        {"energy", "forces"}
    )
    forced = calculator.singlepoint(
        atoms, multiplicity=2, properties=("energy", "forces")
    )
    assert forced.converged and forced.energy == pytest.approx(result.energy, abs=2e-9)
    assert forced.forces is not None and forced.forces.shape == (1, 3)
    # An isolated atom has zero nuclear force by translational invariance.
    np.testing.assert_allclose(forced.forces, 0.0, atol=1e-9, rtol=0)
    if device == "cpu":
        assert calculator.singlepoint(atoms, multiplicity=2).forces is None


@pytest.mark.parametrize("method", ["lda-uks", "pbe-uks"])
def test_uks_s_p_basis_force_request_uses_qualified_native_owner(method: str) -> None:
    calculator = Calculator(method=method, basis="def2-tzvp", device="cpu")
    atoms = [("H", (0, 0, 0))]
    assert calculator.capabilities.supported_properties == frozenset(
        {"energy", "forces"}
    )
    energy = calculator.singlepoint(atoms, multiplicity=2, properties=("energy",))
    forced = calculator.singlepoint(
        atoms, multiplicity=2, properties=("energy", "forces")
    )
    assert energy.converged and forced.converged
    assert forced.energy == pytest.approx(energy.energy, abs=2e-9)
    assert forced.forces is not None and forced.forces.shape == (1, 3)
    np.testing.assert_allclose(forced.forces, 0.0, atol=1e-9, rtol=0)


@pytest.mark.parametrize("functional", ("lda", "pbe"))
@pytest.mark.parametrize(
    "raw_atoms,charge,multiplicity,basis_name",
    (
        ([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], 0, 1, "sto-3g"),
        ([("He", (0, 0, 0))], 0, 1, "sto-3g"),
        (
            [
                ("O", (0, 0, 0)),
                ("H", (0, -1.43233673, 1.10715266)),
                ("H", (0, 1.43233673, 1.10715266)),
            ],
            0,
            1,
            "sto-3g",
        ),
        ([("H", (0, 0, 0))], 0, 2, "sto-3g"),
        ([("Li", (0, 0, 0))], 0, 2, "sto-3g"),
        ([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], 1, 2, "sto-3g"),
        (
            [
                ("O", (0, 0, 0)),
                ("H", (0, -1.43233673, 1.10715266)),
                ("H", (0, 1.43233673, 1.10715266)),
            ],
            0,
            1,
            "def2-svp",
        ),
        ([("O", (0, 0, 0)), ("H", (0, 0, 1.8))], 0, 2, "sto-3g"),
        ([("O", (0, 0, 0)), ("H", (0, 0, 1.8))], 0, 2, "def2-svp"),
    ),
    ids=(
        "h2",
        "he",
        "water",
        "h",
        "li",
        "h2plus",
        "water_large_solver",
        "oh",
        "oh_large_solver",
    ),
)
def test_native_matches_independent_scf(
    functional: typing.Any,
    raw_atoms: typing.Any,
    charge: typing.Any,
    multiplicity: typing.Any,
    basis_name: typing.Any,
    device: typing.Any,
) -> None:
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    pyscf.lib.num_threads(1)
    atoms = tuple(Atom.from_value(a) for a in raw_atoms)
    restricted = multiplicity == 1
    calculator = Calculator(
        method=functional + ("-rks" if restricted else "-uks"),
        basis=basis_name,
        device=device,
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    # Copy exact primitive input numbers, rather than accepting differences
    # between the libraries' named-basis tables as numerical error.
    labels = [raw_atoms[i][0] + str(i) for i in range(len(atoms))]
    basis = {label: [] for label in labels}
    for shell in calculator._shells_for_atoms(atoms):
        basis[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )
    mol = gto.M(
        atom=[
            (label, atom.position) for label, atom in zip(labels, atoms, strict=True)
        ],
        basis=basis,
        charge=charge,
        spin=multiplicity - 1,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    hydroxyl = len(atoms) == 2 and atoms[0].atomic_number == 8
    if hydroxyl:
        # Resolve symmetry-related pi orientations in the exact molecular/grid
        # subgroup. The final gate still checks the full unrestricted AO
        # commutator, including directions excluded by the reference solver.
        mol.symmetry = "C2v"
        mol.build()
    if basis_name == "def2-svp":
        # This forces ordinary-stream execution of the larger shared native
        # eigensolver, above the small solver's 16-AO limit.
        assert mol.nao_nr() > 16
    grid = MolecularGrid(
        atoms,
        spec=calculator.ks_options.grid,
        charge=charge,
        multiplicity=multiplicity,
    ).explicit()
    # This gate compares independently converged SCF energies. Force defaults
    # and separately packaged stationary kernels have their own endpoint gates.
    native = calculator.singlepoint(
        atoms, charge=charge, multiplicity=multiplicity, properties=("energy",)
    )
    assert native.converged and native.density_rms < 1e-9
    assert native.physical_residual_rms is not None
    assert native.physical_residual_rms < 1e-9
    assert native.executed_backend == ("cpu_reference" if device == "cpu" else "cuda")
    # Axial OH has degenerate pi orientations in the continuum, but a finite
    # Lebedev grid distinguishes the axis and diagonal C2v subgroups. Enumerate
    # those two geometric branches before looking at the native energy; neither
    # ordinary SCF convergence nor this test promises the global minimum.
    angles = (
        (0.0, np.pi / 4)
        if hydroxyl and calculator.ks_options.grid.version == 4
        else (0.0,)
    )
    symmetry_orbitals = mol.symm_orb
    native_density = None
    if len(angles) > 1 and functional == "pbe":
        from generativeqc._dft_gradient import StationaryKsState
        from generativeqc_compiler.dft import NativeAO

        with (
            calculator.prepare_batch(
                [atoms], charges=[charge], multiplicities=[multiplicity]
            ) as batch,
            NativeAO(
                atoms, basis=calculator._basis, charge=charge, multiplicity=multiplicity
            ) as ao,
        ):
            replay = batch.execute(strict=True, properties=("energy",)).items[0]
            assert abs(replay.energy - native.energy) < 1e-10
            state = StationaryKsState.from_native(batch, ao)
            try:
                # Native Cartesian AOs are individually normalized; libcint's
                # Cartesian d functions retain their angular normalization.
                scale = np.sqrt(
                    np.diag(state.overlap) / np.diag(mol.intor("int1e_ovlp"))
                )
                np.testing.assert_allclose(
                    state.overlap,
                    mol.intor("int1e_ovlp") * scale[:, None] * scale[None, :],
                    atol=2e-13,
                    rtol=0,
                )
                native_density = state.density * scale[:, None] * scale[None, :]
            finally:
                state._source.close()
    branches = []
    for angle in angles:
        if hydroxyl:
            from pyscf.symm.geom import rotation_mat

            rotation = mol.ao_rotation_matrix(rotation_mat((0.0, 0.0, 1.0), angle))
            mol.symm_orb = [rotation @ block for block in symmetry_orbitals]
        energies = []
        for guess in ("minao", "1e"):
            reference = dft.RKS(mol) if restricted else dft.UKS(mol)
            reference.xc = "PBE" if functional == "pbe" else "LDA_X,LDA_C_PW"
            reference.grids.coords = np.array(grid.points)
            reference.grids.weights = np.array(grid.weights)
            reference.small_rho_cutoff = 0
            reference.conv_tol = 1e-12 if hydroxyl else 1e-13
            reference.conv_tol_grad = 1e-9
            reference.max_cycle = 150
            reference.init_guess = guess
            reference.kernel()
            assert reference.converged, f"independent {guess} solution is unverified"
            density = reference.make_rdm1()
            fock = reference.get_fock(dm=density)
            overlap = reference.get_ovlp()
            residual = fock @ density @ overlap - overlap @ density @ fock
            assert np.max(np.sqrt(np.mean(residual**2, axis=(-2, -1)))) < 1e-9
            energies.append(reference.e_tot)
            if native_density is not None:
                _assert_independent_stationary_state(
                    reference, native_density, native.energy
                )
                with pytest.raises(AssertionError, match="independent energy"):
                    _assert_independent_stationary_state(
                        reference, native_density, native.energy + 1e-6
                    )
                perturbed = native_density.copy()
                perturbed[0, 0, 1] += 1e-4
                perturbed[0, 1, 0] += 1e-4
                with pytest.raises(
                    AssertionError, match="independent (energy|residual)"
                ):
                    _assert_independent_stationary_state(
                        reference, perturbed, native.energy
                    )
        assert abs(energies[0] - energies[1]) < 1e-8
        branches.append(energies)
    assert any(
        all(abs(native.energy - energy) < 1e-8 for energy in branch)
        for branch in branches
    ), (native.energy, branches)
    assert not any(
        all(abs(native.energy + 1e-6 - energy) < 1e-8 for energy in branch)
        for branch in branches
    ), "off-branch energy corruption escaped the independent reference gate"


def _assert_independent_stationary_state(
    reference: typing.Any, density: np.ndarray, energy: float
) -> None:
    """Validate the published density without independently reconverging it."""
    assert abs(reference.energy_tot(dm=density) - energy) < 1e-8, "independent energy"
    fock = reference.get_fock(dm=density)
    overlap = reference.get_ovlp()
    residual = fock @ density @ overlap - overlap @ density @ fock
    assert np.max(np.sqrt(np.mean(residual**2, axis=(-2, -1)))) < 1e-9, (
        "independent residual"
    )

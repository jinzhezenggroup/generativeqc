"""Independent fixed-density science and transactional native audit contracts."""

from __future__ import annotations

import ctypes as ct
import typing
from contextlib import contextmanager
from hashlib import sha256

import numpy as np
import pytest
from generativeqc import Calculator, _native
from generativeqc import _hf_physical_native as native
from generativeqc.accuracy import ObservableTarget, TargetAccuracy
from generativeqc.calculator import Atom
from generativeqc.progressive_controller import TargetProblem

ACCURACY = TargetAccuracy(
    (ObservableTarget("energy", "absolute", "Eh", absolute=1e-8),)
)


@contextmanager
def inputs(calculator: Calculator, atoms: tuple) -> typing.Iterator[tuple]:
    """Borrow normalized SDK inputs while testing the native transport directly."""
    library = calculator._library
    if not native._bind(library):
        pytest.skip("requires native CPU physical HF audit")
    context, system = ct.c_void_p(), ct.c_void_p()
    _native.check(
        library,
        library.generativeqc_context_create(
            ct.byref(calculator._context_descriptor()), ct.byref(context)
        ),
    )
    try:
        system = calculator._create_native_system(context, atoms, 0, 1)
        controls = native._Controls(
            ct.sizeof(native._Controls),
            _native.ABI_VERSION,
            calculator._method == _native.METHOD_UHF,
            calculator._density_fitting_mode != _native.DENSITY_FITTING_NONE,
            calculator._diis_history,
            0,
            1 << 30,
            calculator._screening_tolerance,
            calculator._density_fitting_relative_threshold,
        )
        capacity = native._Capacity(ct.sizeof(native._Capacity), _native.ABI_VERSION)
        assert (
            library.generativeqc_hf_physical_audit_cpu_query_v1(
                system, None, ct.byref(controls), ct.byref(capacity)
            )
            == 0
        )
        yield library, context, system, controls, capacity
    finally:
        if system:
            library.generativeqc_system_destroy(system)
        library.generativeqc_context_destroy(context)


def execute(
    arguments: tuple,
    density: np.ndarray,
    overlap: np.ndarray,
    output: native._Result | None = None,
) -> tuple[int, native._Result]:
    """Call the ABI with caller-owned buffers, preserving observable errors."""
    library, context, system, controls, _ = arguments
    if output is None:
        output = native._Result(
            ct.sizeof(native._Result),
            _native.ABI_VERSION,
            overlap.ctypes.data_as(ct.POINTER(ct.c_double)),
            overlap.size,
        )
    status = library.generativeqc_hf_physical_audit_cpu_v1(
        context,
        system,
        None,
        ct.byref(controls),
        density.ctypes.data_as(ct.POINTER(ct.c_double)),
        density.size,
        -1.0,
        ct.byref(output),
    )
    return status, output


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("spherical", [False, True])
def test_fixed_density_against_independent_pyscf(
    fitted: bool,
    unrestricted: bool,
    spherical: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use d-containing Ne plus H2; neither physical audit may run an SCF solve."""
    pyscf = pytest.importorskip("pyscf")
    from generativeqc import progressive_controller as controller

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("admitted audit retried Python scientific orchestration")

    monkeypatch.setattr(controller, "_audit_target_physical_residual_python", forbidden)
    for offset in (0.0, 0.07):
        atoms = tuple(
            Atom.from_value(a)
            for a in (
                ("Ne", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, 3.5 + offset)),
                ("H", (0.1, offset, 4.9)),
            )
        )
        calculator = Calculator(
            basis="def2-svp",
            method="uhf" if unrestricted else "rhf",
            basis_representation="spherical" if spherical else "cartesian",
            density_fitting="cpu" if fitted else "none",
            screening_tolerance=1e-14,
            target_accuracy=ACCURACY,
        )
        if not native._bind(calculator._library):
            pytest.skip("requires native CPU physical HF audit")
        molecule = pyscf.gto.M(
            atom=[(a.atomic_number, a.position) for a in atoms],
            basis="def2-svp",
            unit="Bohr",
            cart=not spherical,
            verbose=0,
        )
        oracle = pyscf.scf.UHF(molecule) if unrestricted else pyscf.scf.RHF(molecule)
        if fitted:
            oracle = oracle.density_fit(auxbasis="def2-svp")
        n = molecule.nao_nr()
        rng = np.random.default_rng(31)
        raw = rng.normal(size=(2 if unrestricted else 1, n, n)) / n
        density = np.ascontiguousarray(raw + raw.transpose(0, 2, 1))
        raw_overlap = oracle.get_ovlp()
        # libcint Cartesian d+ components are not individually unit normalized.
        # Transform the independent oracle into the public normalized AO basis;
        # using the same raw density in both conventions changes the physics.
        scale = 1.0 / np.sqrt(np.diag(raw_overlap))
        oracle_density = density * scale[None, :, None] * scale[None, None, :]
        dm = oracle_density if unrestricted else oracle_density[0]
        overlap = raw_overlap * scale[:, None] * scale[None, :]
        hcore = oracle.get_hcore()
        fock = (hcore + oracle.get_veff(dm=dm)) * scale[:, None] * scale[None, :]
        focks = fock if unrestricted else fock[None, :, :]
        residuals = np.asarray(
            [
                f @ d @ overlap - overlap @ d @ f
                for f, d in zip(focks, density, strict=True)
            ]
        )
        energy = oracle.energy_tot(dm=dm)
        problem = TargetProblem.from_calculator(calculator, atoms)
        observed = controller._audit_target_physical_residual(
            problem,
            calculator,
            atoms,
            density,
            energy,
            charge=0,
            multiplicity=1,
            maximum_host_bytes=1 << 30,
        )
        assert observed.fixed_density_energy == pytest.approx(energy, abs=1e-8)
        assert observed.maximum_commutator == pytest.approx(
            np.max(np.abs(residuals)), abs=1e-8
        )
        assert observed.rms_commutator == pytest.approx(
            np.sqrt(np.mean(residuals**2)), abs=1e-8
        )
        assert observed.density_sha256 == sha256(density.tobytes()).hexdigest()
        assert observed.energy_difference < 1e-8


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_exact_and_below_capacity_preserve_outputs(
    fitted: bool, unrestricted: bool
) -> None:
    """Admission is inclusive; a refused provider publishes no scalar or array."""
    calculator = Calculator(
        basis="sto-3g",
        method="uhf" if unrestricted else "rhf",
        density_fitting="cpu" if fitted else "none",
    )
    atoms = tuple(Atom.from_value(a) for a in (("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))))
    with inputs(calculator, atoms) as args:
        _, _, _, controls, capacity = args
        assert (
            capacity.provider_bytes + capacity.workspace_bytes
            == capacity.required_host_bytes
        )
        density = np.full(capacity.density_count, 0.1)
        overlap = np.full(capacity.overlap_count + 3, -123.0)
        controls.maximum_host_bytes = capacity.required_host_bytes
        status, output = execute(args, density, overlap)
        assert status == 0 and output.fock_builds == 1
        assert np.all(overlap[-3:] == -123.0)
        before = bytes(output), overlap.copy()
        controls.maximum_host_bytes -= 1
        assert (
            execute(args, density, overlap, output)[0] == _native.STATUS_OUT_OF_MEMORY
        )
        assert bytes(output) == before[0]
        np.testing.assert_array_equal(overlap, before[1])


@pytest.mark.parametrize(
    "failure",
    ["nonfinite", "short_density", "short_overlap", "alias", "reserved", "abi"],
)
def test_transactional_invalid_inputs(failure: str) -> None:
    calculator = Calculator(basis="sto-3g")
    atoms = tuple(Atom.from_value(a) for a in (("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))))
    with inputs(calculator, atoms) as args:
        _, _, _, controls, capacity = args
        density = np.full(capacity.density_count, 0.1)
        overlap = np.full(capacity.overlap_count, -123.0)
        if failure == "nonfinite":
            density[0] = np.nan
        elif failure == "short_density":
            density = density[:-1]
        elif failure == "short_overlap":
            overlap = overlap[:-1]
        elif failure == "alias":
            overlap = density
        elif failure == "reserved":
            controls.reserved = 1
        elif failure == "abi":
            controls.abi_version += 1
        output = native._Result(
            ct.sizeof(native._Result),
            _native.ABI_VERSION,
            overlap.ctypes.data_as(ct.POINTER(ct.c_double)),
            overlap.size,
            -17,
            -18,
            -19,
            -20,
            21,
            22,
        )
        before = bytes(output), overlap.copy()
        assert execute(args, density, overlap, output)[0] == (
            2 if failure == "abi" else 1
        )
        assert bytes(output) == before[0]
        np.testing.assert_array_equal(overlap, before[1])


def test_native_refusal_never_retries_python_and_cleanup_allows_next_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An execution refusal is terminal; a later independent invocation is usable."""
    from generativeqc import progressive_controller as controller

    calculator = Calculator(basis="sto-3g", target_accuracy=ACCURACY)
    if not native._bind(calculator._library):
        pytest.skip("requires native CPU physical HF audit")
    atoms = tuple(Atom.from_value(a) for a in (("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))))
    problem = TargetProblem.from_calculator(calculator, atoms)
    density = np.full((1, 2, 2), 0.1)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("native audit refusal retried Python")

    def refused(*args: typing.Any) -> int:
        return _native.STATUS_NUMERICAL_FAILURE

    monkeypatch.setattr(controller, "_audit_target_physical_residual_python", forbidden)
    with monkeypatch.context() as injection:
        injection.setattr(
            calculator._library, "generativeqc_hf_physical_audit_cpu_v1", refused
        )
        with pytest.raises(RuntimeError):
            controller._audit_target_physical_residual(
                problem,
                calculator,
                atoms,
                density,
                -1,
                charge=0,
                multiplicity=1,
                maximum_host_bytes=1 << 28,
            )
    observed = controller._audit_target_physical_residual(
        problem,
        calculator,
        atoms,
        density,
        -1,
        charge=0,
        multiplicity=1,
        maximum_host_bytes=1 << 28,
    )
    assert np.isfinite(observed.fixed_density_energy)


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_native_admission_matches_full_planner_at_one_byte_boundary(
    monkeypatch: pytest.MonkeyPatch, fitted: bool, unrestricted: bool
) -> None:
    """Use one audit gate and do not construct a Fock owner below its cap."""
    from generativeqc import progressive_controller as controller

    calculator = Calculator(
        basis="sto-3g",
        method="uhf" if unrestricted else "rhf",
        density_fitting="cpu" if fitted else "none",
        initial_guess=None,
        target_accuracy=ACCURACY,
    )
    atoms = tuple(Atom.from_value(a) for a in (("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))))
    problem = TargetProblem.from_calculator(calculator, atoms)
    spins = 2 if unrestricted else 1
    density = np.full((spins, 2, 2), 0.1)
    planner_bytes = calculator.estimate_resources([atoms]).peak_bytes["host"]
    with inputs(calculator, atoms) as (_, _, _, _, capacity):
        assert capacity.provider_bytes == planner_bytes
        assert capacity.workspace_bytes == 8 * 2**2 * (12 * spins + 8)
        exact = capacity.provider_bytes + capacity.workspace_bytes
        assert capacity.required_host_bytes == exact

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("admitted audit called the compatibility planner or owner")

    monkeypatch.setattr(calculator, "estimate_resources", forbidden)
    monkeypatch.setattr(controller, "_audit_target_physical_residual_python", forbidden)
    with monkeypatch.context() as injection:
        injection.setattr(
            calculator._library, "generativeqc_hf_physical_audit_cpu_v1", forbidden
        )
        with pytest.raises(MemoryError, match="budget"):
            controller._audit_target_physical_residual(
                problem,
                calculator,
                atoms,
                density,
                -1,
                charge=0,
                multiplicity=1,
                maximum_host_bytes=exact - 1,
            )
    observed = controller._audit_target_physical_residual(
        problem,
        calculator,
        atoms,
        density,
        -1,
        charge=0,
        multiplicity=1,
        maximum_host_bytes=exact,
    )
    assert np.isfinite(observed.fixed_density_energy)


def test_preliminary_reservation_keeps_legacy_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pure fixed-density capacity may not waive the target's guess budget."""
    from generativeqc import InitialGuessSpec
    from generativeqc import progressive_controller as controller

    calculator = Calculator(
        basis="sto-3g", initial_guess=InitialGuessSpec("hf"), target_accuracy=ACCURACY
    )
    atoms = tuple(Atom.from_value(a) for a in (("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))))
    problem = TargetProblem.from_calculator(calculator, atoms)
    density = np.full((1, 2, 2), 0.1)
    with inputs(calculator, atoms) as (_, _, _, _, capacity):
        pure_audit_bytes = capacity.required_host_bytes

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("underfunded preliminary reservation reached native audit")

    monkeypatch.setattr(native, "audit", forbidden)
    with pytest.raises(MemoryError):
        controller._audit_target_physical_residual(
            problem,
            calculator,
            atoms,
            density,
            -1,
            charge=0,
            multiplicity=1,
            maximum_host_bytes=pure_audit_bytes,
        )


def test_missing_native_protocol_retains_explicit_compatibility() -> None:
    """An older selected library uses its bounded incumbent audit deliberately."""
    from generativeqc import progressive_controller as controller

    calculator = Calculator(basis="sto-3g", target_accuracy=ACCURACY)
    selected = calculator._library
    if not native._bind(selected):
        pytest.skip("requires native CPU physical HF audit")
    atoms = tuple(Atom.from_value(a) for a in (("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))))
    problem = TargetProblem.from_calculator(calculator, atoms)
    density = np.full((1, 2, 2), 0.1)
    keywords = {"charge": 0, "multiplicity": 1, "maximum_host_bytes": 1 << 28}
    actual = controller._audit_target_physical_residual(
        problem, calculator, atoms, density, -1, **keywords
    )

    class OlderLibrary:
        def __getattr__(self, name: str) -> typing.Any:
            if name.startswith("generativeqc_hf_physical_audit_cpu"):
                raise AttributeError(name)
            return getattr(selected, name)

    calculator._library = OlderLibrary()
    compatible = controller._audit_target_physical_residual(
        problem, calculator, atoms, density, -1, **keywords
    )
    assert compatible.fixed_density_energy == pytest.approx(
        actual.fixed_density_energy, abs=1e-12
    )
    assert compatible.maximum_commutator == pytest.approx(
        actual.maximum_commutator, abs=1e-12
    )
    assert compatible.rms_commutator == pytest.approx(actual.rms_commutator, abs=1e-12)
    assert compatible.density_sha256 == actual.density_sha256

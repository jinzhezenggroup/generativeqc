"""Fixed-model RKS/UKS derivative adapter for the native saddle corrector."""

from __future__ import annotations

import math
import typing
from contextlib import ExitStack
from numbers import Real
from time import perf_counter
from types import MappingProxyType

import numpy as np

from . import _native
from ._api_types import Atom
from .basis import BasisSet
from .saddle import MolecularSaddleResult, SaddleOptions, _real_array, correct_saddle

if typing.TYPE_CHECKING:
    from .calculator import Calculator
    from .saddle import SaddleProgress


def _occupied_blocks(reference: typing.Any, spin: str) -> tuple[np.ndarray, ...]:
    """Own each integer-occupied spin subspace without identifying alpha/beta."""
    spins, occupation = (1, 2.0) if spin == "unpolarized" else (2, 1.0)
    coefficients = np.asarray(reference.coefficients)
    occupations = np.asarray(reference.occupations)
    if (
        spin not in ("unpolarized", "polarized")
        or reference.identity.spin != spin
        or coefficients.ndim != 3
        or coefficients.shape[0] != spins
        or occupations.shape != (spins, coefficients.shape[-1])
        or not np.isfinite(coefficients).all()
        or not np.isin(occupations, (0.0, occupation)).all()
    ):
        raise ValueError("molecular saddle requires matching integer spin occupations")
    blocks = tuple(
        np.array(coefficients[index][:, populations == occupation], copy=True)
        for index, populations in enumerate(occupations)
    )
    if not any(block.shape[1] for block in blocks):
        raise ValueError("molecular saddle requires occupied orbitals")
    return blocks


def _occupied_overlaps(
    current: tuple[np.ndarray, ...],
    previous: tuple[np.ndarray, ...],
    metric: np.ndarray,
    minimum: float,
) -> tuple[float, ...]:
    """Track every nonempty spin space against the last accepted geometry.

    Empty channels contribute unit continuity only if they remain empty. A spin
    population change is not a rectangular-overlap success, and one good spin
    must never mask a root change in the other.
    """
    if len(current) != len(previous) or any(
        left.shape[1] != right.shape[1]
        for left, right in zip(current, previous, strict=True)
    ):
        raise ValueError("molecular saddle occupied spin populations changed")
    overlaps = tuple(
        float(np.min(np.linalg.svd(left.T @ metric @ right, compute_uv=False)))
        if left.shape[1]
        else 1.0
        for left, right in zip(current, previous, strict=True)
    )
    if not all(math.isfinite(value) and value >= minimum for value in overlaps):
        raise ValueError(
            f"occupied root continuity lost: minimum overlap {min(overlaps):.17g}"
        )
    return overlaps


def _registered_root_atoms(
    current: tuple[Atom, ...], previous: tuple[Atom, ...], masses: np.ndarray
) -> tuple[tuple[Atom, ...], np.ndarray]:
    """Remove only the relative centre-of-mass translation for root comparison.

    AO coefficients are unchanged by a common translation of their centres.
    Comparing compact core orbitals in different laboratory frames would reject
    an unchanged electronic state merely because its nuclei moved together.
    Rotation is deliberately not registered: the finite angular DFT grid fixes
    its orientation. Production geometries and derivative callbacks stay intact.
    """
    displacement = np.asarray([atom.position for atom in current]) - np.asarray(
        [atom.position for atom in previous]
    )
    weights = masses / np.max(masses)
    translation = np.average(displacement, axis=0, weights=weights)
    registered = tuple(
        Atom.from_value(
            (atom.atomic_number, tuple(np.asarray(atom.position) - translation))
        )
        for atom in current
    )
    return registered, translation


def run(
    calculator: Calculator, atoms: typing.Any, **request: typing.Any
) -> MolecularSaddleResult:
    """Bind one SCF and response lifetime per derivative geometry, with no oracle."""
    started = perf_counter()
    checked_atoms = []
    for atom in atoms:
        if isinstance(atom, Atom):
            checked_atoms.append(atom)
            continue
        element, position = atom
        position = tuple(position)
        # Validate scalars before Atom or NumPy can coerce mixed input types.
        if any(
            isinstance(component, (bool, np.bool_)) or not isinstance(component, Real)
            for component in position
        ):
            raise ValueError("atom coordinates must contain finite real numbers")
        checked_atoms.append(Atom.from_value((element, position)))
    atoms = tuple(checked_atoms)
    controls = request["options"]
    controls = (
        SaddleOptions(project_rigid_motions=True) if controls is None else controls
    )
    if not isinstance(controls, SaddleOptions):
        raise TypeError("options must be SaddleOptions")
    if not controls.project_rigid_motions:
        raise ValueError("molecular saddle correction requires rigid projection")
    if not atoms or 3 * len(atoms) > controls.max_coordinates:
        raise ValueError("molecular coordinates must fit max_coordinates")
    charge, multiplicity = request["charge"], request["multiplicity"]
    if type(charge) is not int or type(multiplicity) is not int:
        raise TypeError("charge and multiplicity must be integers")
    if multiplicity < 1:
        raise ValueError("multiplicity must be positive")
    basis = calculator._basis
    has_ecp = isinstance(basis, BasisSet) and any(
        element.ecp_core_electrons for element in basis.elements
    )
    if (
        calculator._device_name != "cpu"
        or calculator.ks_options is None
        or (multiplicity != 1 and calculator.ks_options.method_ir.spin == "unpolarized")
        or not calculator._composite_rks_second_order
        or not calculator._native_cpu_second_order
        or {primitive.kind for primitive in calculator.ks_options.method_ir.primitives}
        != {"semilocal_xc", "exact_exchange"}
        or has_ecp
        or calculator._density_fitting_mode != _native.DENSITY_FITTING_NONE
        or "forces" not in calculator.capabilities.supported_properties
    ):
        raise NotImplementedError(
            "molecular saddle requires qualified direct all-electron CPU full-range hybrid RKS/UKS"
        )
    calculator._checked_rks_second_order_atoms(
        atoms, multiplicity=multiplicity, endpoint="hessian"
    )
    masses = _real_array(request["masses"], "masses")
    if (
        masses.shape != (len(atoms),)
        or not np.isfinite(masses).all()
        or np.any(masses <= 0)
    ):
        raise ValueError("masses require one positive finite dalton mass per atom")
    minimum_overlap = request["minimum_occupied_overlap"]
    if isinstance(minimum_overlap, bool) or not isinstance(
        minimum_overlap, (float, int)
    ):
        raise TypeError("minimum_occupied_overlap must be a real number")
    if not math.isfinite(minimum_overlap) or not 0 < minimum_overlap <= 1:
        raise ValueError("minimum_occupied_overlap must be in (0, 1]")
    for name in (
        "root_overlap_budget_bytes",
        "integral_budget_bytes",
        "output_budget_bytes",
        "grid_budget_bytes",
    ):
        if type(request[name]) is not int or not 0 < request[name] < 1 << 63:
            raise ValueError(f"{name} must be a positive int64 byte count")
    dimension = 3 * len(atoms)
    if request["max_eri_primitives"] is not None:
        from ._native_hessian import _capacity

        _capacity(request["max_eri_primitives"], "max_eri_primitives")
    needed_output = 8 * (2 * dimension**2 + 3 * dimension + 2)
    if needed_output > request["output_budget_bytes"]:
        raise MemoryError("molecular Hessian exceeds output_budget_bytes")
    for name in ("progress", "evaluation_progress", "cancelled"):
        if request[name] is not None and not callable(request[name]):
            raise TypeError(f"{name} must be callable")

    from ._native_cpu_hessian import execute_batch, orbital_frame, preflight
    from .overlap import cross_overlap
    from .vibrations import analyze_harmonic_modes

    preflight(calculator)
    if not hasattr(calculator._library, "generativeqc_ks_orbitals_cpu_v1"):
        raise NotImplementedError(
            "molecular saddle requires native CPU occupied-root export"
        )

    signature = calculator._model_signature()
    method = calculator.method_ir
    electronic_method = calculator.ks_options.method_ir
    records: list[typing.Mapping[str, typing.Any]] = []
    work = {
        "scf_iterations": 0,
        "scf_fock_builds": 0,
        "nuclear_response_solves": 0,
        "response_iterations": 0,
        "induced_fock_actions": 0,
        "first_eri_primitives": 0,
        "curvature_eri_primitives": 0,
        "d3_pair_visits": 0,
    }
    state: dict[str, typing.Any] = {}
    accepted: dict[str, typing.Any] = {}

    def fixed_model() -> None:
        if (
            calculator._model_signature() != signature
            or calculator.method_ir.identity != method.identity
        ):
            raise ValueError(
                "molecular saddle calculator model changed during execution"
            )

    def geometry(positions: np.ndarray) -> tuple[Atom, ...]:
        return tuple(
            Atom(atom.atomic_number, tuple(position))
            for atom, position in zip(atoms, positions, strict=True)
        )

    def record(
        stage: str, status: str, positions: np.ndarray, **details: typing.Any
    ) -> None:
        event = MappingProxyType(
            {
                "stage": stage,
                "status": status,
                "elapsed_seconds": perf_counter() - started,
                "coordinates_bohr": tuple(
                    tuple(map(float, position)) for position in positions
                ),
                **details,
            }
        )
        records.append(event)
        if request["evaluation_progress"] is not None:
            request["evaluation_progress"](event)
        fixed_model()

    def close() -> None:
        owners = state.pop("owners", None)
        try:
            if owners is not None:
                owners.close()
        finally:
            state.pop("batch", None)

    def energy_gradient(positions: np.ndarray) -> tuple[float, np.ndarray]:
        fixed_model()
        close()
        state.clear()
        current_atoms = geometry(positions)
        state.update(
            atoms=current_atoms, positions=positions.copy(), owners=ExitStack()
        )
        record("energy_gradient", "start", positions)
        try:
            owners = state["owners"]
            batch = owners.enter_context(
                calculator.prepare_batch(
                    [current_atoms],
                    charges=[charge],
                    multiplicities=[multiplicity],
                    warm_start=False,
                )
            )
            if not batch._native_dft_force_eligible(0):
                raise NotImplementedError(
                    "molecular saddle requires a prepared native CPU force owner; no Python fallback"
                )
            primal = batch.execute(strict=True, properties=("energy", "forces")).items[
                0
            ]
            if not primal.converged or not primal.succeeded or primal.forces is None:
                raise RuntimeError("molecular saddle SCF/forces did not converge")
            work["scf_iterations"] += primal.iterations
            work["scf_fock_builds"] += primal.fock_builds
            reference = orbital_frame(
                calculator,
                batch,
                current_atoms,
                charge=charge,
                multiplicity=multiplicity,
                maximum_bytes=request["root_overlap_budget_bytes"],
            )
            occupied = _occupied_blocks(reference, electronic_method.spin)
            overlaps = (1.0,) * len(occupied)
            registration = np.zeros(3)
            if accepted:
                registered_atoms, registration = _registered_root_atoms(
                    current_atoms, accepted["atoms"], masses
                )
                metric = cross_overlap(
                    calculator,
                    calculator,
                    registered_atoms,
                    source_atoms=accepted["atoms"],
                    charge=charge,
                    multiplicity=multiplicity,
                    maximum_bytes=request["root_overlap_budget_bytes"],
                )
                overlaps = _occupied_overlaps(
                    occupied, accepted["occupied"], metric, minimum_overlap
                )
            state.update(batch=batch, occupied=occupied)
            record(
                "energy_gradient",
                "complete",
                positions,
                energy=primal.energy,
                scf_iterations=primal.iterations,
                scf_fock_builds=primal.fock_builds,
                scf_physical_residual=primal.physical_residual_rms,
                occupied_counts=tuple(block.shape[1] for block in occupied),
                minimum_occupied_overlap=min(overlaps),
                minimum_occupied_overlaps_by_spin=overlaps,
                occupied_overlap_coordinate_frame="mass-centroid-translation-registered",
                root_registration_translation_bohr=tuple(map(float, registration)),
                root_numeric_reservation_bytes=reference.numeric_reservation_bytes,
                numerical_runtime="native-cpp-cpu",
            )
            return primal.energy, -np.asarray(primal.forces)
        except Exception as error:
            close()
            record(
                "energy_gradient",
                "failed",
                positions,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

    def hessian(positions: np.ndarray) -> np.ndarray:
        fixed_model()
        if "batch" not in state or not np.array_equal(positions, state["positions"]):
            raise ValueError(
                "Hessian geometry lacks its prepared energy/gradient state"
            )
        record("hessian", "start", positions)
        try:
            result = execute_batch(
                calculator,
                state["batch"],
                state["atoms"],
                charge=charge,
                multiplicity=multiplicity,
                integral_budget_bytes=request["integral_budget_bytes"],
                grid_budget_bytes=request["grid_budget_bytes"],
                output_budget_bytes=request["output_budget_bytes"],
                solver_options=request["solver_options"],
                max_eri_primitives=request["max_eri_primitives"],
            )
            for name in work.keys() - {"scf_iterations", "scf_fock_builds"}:
                work[name] += int(result.diagnostics[name])
            close()
            record(
                "hessian",
                "complete",
                positions,
                nuclear_response_solves=result.diagnostics["nuclear_response_solves"],
                maximum_response_residual_norm=result.diagnostics[
                    "response_residual_norm"
                ],
                raw_symmetry=result.diagnostics["raw_symmetry_error"],
                numerical_runtime="native-cpp-cpu",
                total_method_identity=result.diagnostics["total_method_identity"],
                d3_pair_visits=result.diagnostics["d3_pair_visits"],
            )
            return result.matrix
        except Exception as error:
            close()
            record(
                "hessian",
                "failed",
                positions,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise

    def progress(observation: SaddleProgress) -> bool:
        fixed_model()
        if state.get("occupied") is not None and np.array_equal(
            observation.coordinates, state["positions"]
        ):
            accepted.update(
                atoms=state["atoms"],
                occupied=tuple(block.copy() for block in state["occupied"]),
            )
        stop = request["progress"] is not None and bool(
            request["progress"](observation)
        )
        fixed_model()
        return stop

    try:
        result = correct_saddle(
            [atom.position for atom in atoms],
            energy_gradient=energy_gradient,
            hessian=hessian,
            options=controls,
            initial_mode=request["initial_mode"],
            progress=progress,
            cancelled=request["cancelled"],
        )
    finally:
        close()
    harmonics = None
    harmonic_failure = None
    if result.receipt["stationary_verified"]:
        try:
            analyzed = analyze_harmonic_modes(
                result.hessian,
                coordinates=result.coordinates,
                masses=masses,
                hessian_units="hartree/bohr^2",
                coordinate_units="bohr",
                mass_units="dalton",
                symmetrize=True,
                raw_symmetry_tolerance=controls.raw_symmetry_tolerance,
            )
            if analyzed.rigid_mode_count != result.receipt["rigid_modes"]:
                raise ValueError(
                    "harmonic rigid-mode rank differs from the native optimizer"
                )
            if (
                result.converged
                and analyzed.negative_mode_count
                != result.receipt["negative_curvatures"]
            ):
                raise ValueError(
                    "harmonic negative-mode count differs from the verified native saddle"
                )
            harmonics = analyzed
        except Exception as error:  # noqa: BLE001 - Retain the verified geometry if mode publication fails.
            harmonic_failure = (type(error).__name__, str(error))
    return MolecularSaddleResult(
        result,
        geometry(result.coordinates),
        harmonics,
        MappingProxyType(
            {
                "method_identity": method.identity,
                "electronic_method_identity": electronic_method.identity,
                "model_signature": signature,
                "charge": charge,
                "multiplicity": multiplicity,
                "spin": electronic_method.spin,
                "root_policy": "occupied-spin-principal-overlap-to-last-accepted",
                "minimum_occupied_overlap": minimum_overlap,
                "evaluations": tuple(records),
                **work,
                "endpoint_seconds": perf_counter() - started,
                "final_status": "harmonic_analysis_failed"
                if harmonic_failure
                else result.receipt["status"],
                "harmonic_failure": harmonic_failure,
                "work_counters_complete": not result.failures
                and not any(record["status"] == "failed" for record in records),
                "work_counters_scope": "completed derivative receipts; partially failed work is not inferred",
                "displaced_scf_evaluations": 0,
                "numerical_runtime": "native-cpp-cpu",
                "runtime_compilation": False,
                "external_optimizer": False,
            }
        ),
    )

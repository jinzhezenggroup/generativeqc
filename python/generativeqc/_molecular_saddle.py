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
from .ecp import resolve_ecp
from .saddle import (
    MolecularSaddleResult,
    RksColdFockDamping,
    SaddleOptions,
    _real_array,
    correct_saddle,
)

if typing.TYPE_CHECKING:
    from .calculator import Calculator
    from .saddle import SaddleProgress


_DEF2_SADDLE_TERMS = {
    # Pinned BSE def2-ECP (tests/data/global_hybrid_ecp/manifest.json).
    # The resolved local channel is -1, regardless of the source channel label.
    (55, 46): (
        (-1, 2, 3.123269, -28.884309),
        (0, 2, 2.417406, 16.654173),
        (0, 2, 3.123269, 28.884309),
        (0, 2, 4.079750, 84.547730),
        (1, 2, 2.160316, 26.423307),
        (1, 2, 3.123269, 28.884309),
        (1, 2, 5.514080, 157.049059),
        (2, 2, 0.858182, 3.342833),
        (2, 2, 1.807410, 13.172753),
        (2, 2, 3.123269, 28.884309),
    ),
    (53, 28): (
        (-1, 2, 19.458609, -21.842040),
        (-1, 2, 19.349260, -28.468191),
        (-1, 2, 4.823767, -0.243713),
        (-1, 2, 4.884315, -0.320804),
        (0, 2, 40.015835, 49.994293),
        (0, 2, 17.429747, 281.025317),
        (0, 2, 9.005484, 61.573326),
        (0, 2, 19.458609, 21.842040),
        (0, 2, 19.349260, 28.468191),
        (0, 2, 4.823767, 0.243713),
        (0, 2, 4.884315, 0.320804),
        (1, 2, 15.355466, 67.442841),
        (1, 2, 14.971833, 134.881137),
        (1, 2, 8.960164, 14.675051),
        (1, 2, 8.259096, 29.375666),
        (1, 2, 19.458609, 21.842040),
        (1, 2, 19.349260, 28.468191),
        (1, 2, 4.823767, 0.243713),
        (1, 2, 4.884315, 0.320804),
        (2, 2, 15.068908, 35.439529),
        (2, 2, 14.555322, 53.176057),
        (2, 2, 6.718647, 9.067195),
        (2, 2, 6.456393, 13.206937),
        (2, 2, 1.191779, 0.089335),
        (2, 2, 1.291157, 0.052380),
        (2, 2, 19.458609, 21.842040),
        (2, 2, 19.349260, 28.468191),
        (2, 2, 4.823767, 0.243713),
        (2, 2, 4.884315, 0.320804),
    ),
}


def _saddle_ecp_family(basis: BasisSet, atoms: tuple[Atom, ...]) -> int | None:
    """Return the element only if every ECP center matches one qualified operator."""
    cores, terms = resolve_ecp(basis, atoms)
    center_terms: dict[int, list[tuple[int, int, float, float]]] = {}
    for center, channel, power, exponent, coefficient in terms:
        center_terms.setdefault(center, []).append(
            (channel, power, exponent, coefficient)
        )
    family = None
    for center, (atom, core) in enumerate(zip(atoms, cores, strict=True)):
        if not core:
            continue
        number = atom.atomic_number
        expected = _DEF2_SADDLE_TERMS.get((number, core))
        if (
            expected is None
            or (family is not None and family != number)
            or sorted(center_terms.get(center, ())) != sorted(expected)
        ):
            return None
        family = number
    return family


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
    for spin, value in enumerate(overlaps):
        if not math.isfinite(value) or value < minimum:
            raise ValueError(
                f"occupied root continuity lost in spin channel {spin}: "
                f"minimum overlap {value:.17g}, required {minimum:.17g}"
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
    if (
        calculator._device_name != "cpu"
        or calculator.ks_options is None
        or (multiplicity != 1 and calculator.ks_options.method_ir.spin == "unpolarized")
        or not calculator._composite_rks_second_order
        or not calculator._native_cpu_second_order
        or {primitive.kind for primitive in calculator.ks_options.method_ir.primitives}
        != {"semilocal_xc", "exact_exchange"}
        or calculator._density_fitting_mode != _native.DENSITY_FITTING_NONE
    ):
        raise NotImplementedError(
            "molecular saddle requires qualified direct CPU full-range hybrid RKS/UKS"
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
    warm = request["accepted_frame_warm_start"]
    if type(warm) is not bool:
        raise TypeError("accepted_frame_warm_start must be a bool")
    damping = request["cold_fock_damping"]
    if isinstance(basis, BasisSet):
        elements = basis.by_element
        uses_ecp = any(
            element.ecp_core_electrons
            for atom in atoms
            if (element := elements.get(atom.atomic_number)) is not None
        )
        if uses_ecp:
            from generativeqc_compiler.method import resolve_method

            family = _saddle_ecp_family(basis, atoms)
            qualified_method = resolve_method("B3LYP-D3(BJ)", spin="unpolarized")
            cs46 = family == 55 and damping is not None
            # Larger I28 trial steps failed occupied continuity even from a cold SCF.
            i28 = (
                family == 53
                and damping is None
                and minimum_overlap >= 0.99
                and controls.initial_radius <= 0.015
                and controls.maximum_radius <= 0.03
            )
            if (
                calculator._representation_name != "spherical"
                or calculator.method_ir.identity != qualified_method.identity
                or calculator._d3_second_order_spec is None
                or calculator.initial_guess is not None
                or not warm
                or not (cs46 or i28)
            ):
                raise NotImplementedError(
                    "molecular saddle ECP requires spherical B3LYP-D3(BJ) RKS, "
                    "accepted-frame warm starts and initial_guess=None; "
                    "use Cs46 def2-ECP with explicit cold Fock damping, or "
                    "I28 def2-ECP without damping, with initial/max radii at most "
                    "0.015/0.03 Bohr and minimum occupied overlap at least 0.99"
                )
    if damping is not None:
        if not isinstance(damping, RksColdFockDamping):
            raise TypeError("cold_fock_damping must be RksColdFockDamping or None")
        if (
            not warm
            or calculator.initial_guess is not None
            or damping.prefix_iterations >= calculator._max_iterations
        ):
            raise ValueError(
                "cold Fock damping requires accepted-frame warm starts, initial_guess=None "
                "and a prefix shorter than the SCF iteration budget"
            )
        if (
            calculator.ks_options.method_ir.spin != "unpolarized"
            or calculator._d3_second_order_spec is None
            or not hasattr(
                calculator._library, "generativeqc_batch_prepare_named_ks_cold_rks_v1"
            )
        ):
            raise NotImplementedError(
                "cold Fock damping requires native B3LYP-D3(BJ) RKS"
            )
    root_budget = request["root_overlap_budget_bytes"]
    dimension = 3 * len(atoms)
    from ._native_hessian import _capacity

    for name in ("max_eri_primitives", "max_map_terms"):
        if request[name] is not None:
            _capacity(request[name], name)
    needed_output = 8 * (2 * dimension**2 + 3 * dimension + 2)
    if needed_output > request["output_budget_bytes"]:
        raise MemoryError("molecular Hessian exceeds output_budget_bytes")
    for name in ("progress", "evaluation_progress", "cancelled"):
        if request[name] is not None and not callable(request[name]):
            raise TypeError(f"{name} must be callable")

    from ._native_cpu_hessian import (
        _numeric_budgets,
        execute_batch,
        orbital_frame,
        preflight,
    )
    from ._warm_seed_native import capture_snapshot, require_seed_protocol
    from .overlap import cross_overlap
    from .vibrations import analyze_harmonic_modes

    numeric_budget, response_budget = _numeric_budgets(
        request["numeric_budget_bytes"], request["response_budget_bytes"]
    )
    preflight(calculator)
    if warm:
        require_seed_protocol(calculator._library)
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
        "first_one_electron_primitives": 0,
        "curvature_one_electron_primitives": 0,
        "source_xc_point_visits": 0,
        "curvature_xc_point_visits": 0,
        "response_xc_point_visits": 0,
        "ecp_kernel_blocks": 0,
        "ecp_radial_layers": 0,
        "ecp_ao_primitive_points": 0,
        "ecp_contraction_terms": 0,
        "d3_pair_visits": 0,
        "d3_reference_terms": 0,
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

    def discard_seed(frame: dict[str, typing.Any]) -> None:
        seed = frame.pop("seed", None)
        if seed is not None:
            seed.close()

    def occupied_bytes(frame: dict[str, typing.Any]) -> int:
        return sum(block.nbytes for block in frame.get("occupied", ()))

    def root_capacity(*, extra_bytes: int = 0) -> int:
        if not warm:
            return root_budget
        seed = accepted.get("seed")
        retained = 0 if seed is None else seed.numeric_bytes
        remaining = root_budget - retained - occupied_bytes(accepted) - extra_bytes
        if remaining <= 0:
            raise MemoryError("occupied-root storage exceeds root_overlap_budget_bytes")
        return remaining

    def energy_gradient(positions: np.ndarray) -> tuple[float, np.ndarray]:
        fixed_model()
        close()
        discard_seed(state)
        state.clear()
        current_atoms = geometry(positions)
        state.update(
            atoms=current_atoms, positions=positions.copy(), owners=ExitStack()
        )
        record("energy_gradient", "start", positions)
        try:
            owners = state["owners"]
            cold = damping if not accepted else None
            if cold is None:
                prepared = calculator.prepare_batch(
                    [current_atoms],
                    charges=[charge],
                    multiplicities=[multiplicity],
                    warm_start=warm,
                )
            else:
                from .batch import PreparedBatch

                prepared = PreparedBatch(
                    calculator,
                    [current_atoms],
                    charges=[charge],
                    multiplicities=[multiplicity],
                    warm_start=warm,
                    _cold_rks_fock=cold._descriptor(),
                )
            batch = owners.enter_context(prepared)
            # Force support belongs to this prepared system, not the static
            # Calculator summary. Check before executing so saddle never uses
            # PreparedBatch's force compatibility path.
            if not batch._native_dft_force_eligible(0):
                raise NotImplementedError(
                    "molecular saddle requires a prepared native CPU force owner; no Python fallback"
                )
            if warm and accepted:
                accepted["seed"].restore(batch)
            # The opt-in path must inspect attempt flags even when the final
            # batch item failed or lacks a complete Fock-build census.
            primal = batch.execute(
                strict=not warm, properties=("energy", "forces")
            ).items[0]
            if warm:
                if primal.warm_start_fallback:
                    raise RuntimeError(
                        "accepted-frame SCF failed; cold fallback result refused"
                    )
                if primal.warm_start_used != bool(accepted):
                    raise RuntimeError(
                        "accepted-frame SCF seed policy was not satisfied"
                    )
            if not primal.converged or not primal.succeeded or primal.forces is None:
                raise RuntimeError(
                    f"molecular saddle SCF/forces did not converge: {primal.status_message}"
                )
            if primal.fock_builds is None:
                raise RuntimeError(
                    "molecular saddle SCF has no complete Fock work receipt"
                )
            work["scf_iterations"] += primal.iterations
            work["scf_fock_builds"] += primal.fock_builds
            reference = orbital_frame(
                calculator,
                batch,
                current_atoms,
                charge=charge,
                multiplicity=multiplicity,
                maximum_bytes=root_capacity(),
            )
            occupied = _occupied_blocks(reference, electronic_method.spin)
            orbital_bytes = reference.numeric_reservation_bytes
            del reference
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
                    maximum_bytes=root_capacity(
                        extra_bytes=sum(block.nbytes for block in occupied)
                    ),
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
                root_numeric_reservation_bytes=orbital_bytes,
                accepted_frame_warm_start=warm and primal.warm_start_used,
                cold_fock_prefix_iterations=0
                if cold is None
                else cold.prefix_iterations,
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
                max_map_terms=request["max_map_terms"],
                numeric_budget_bytes=numeric_budget,
                response_budget_bytes=response_budget,
            )
            # Only work from the completed Hessian receipt belongs in these counters.
            for name in work.keys() - {"scf_iterations", "scf_fock_builds"}:
                work[name] += int(result.diagnostics[name])
            if warm:
                seed = capture_snapshot(
                    state["batch"], root_capacity(extra_bytes=occupied_bytes(state))
                )
                state["seed"] = seed
                seed.require_geometry(0, positions)
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
                root_seed_snapshot_bytes=state["seed"].numeric_bytes if warm else 0,
            )
            return result.matrix
        except Exception as error:
            close()
            discard_seed(state)
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
            if warm:
                if "seed" in state:
                    # Only a frame with a completed Hessian may replace the seed.
                    discard_seed(accepted)
                    accepted["seed"] = state.pop("seed")
                elif not accepted or accepted["atoms"] != state["atoms"]:
                    raise RuntimeError("accepted derivative frame has no seed")
            accepted.update(
                atoms=state["atoms"],
                occupied=state["occupied"],
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
        try:
            close()
        finally:
            discard_seed(state)
            discard_seed(accepted)
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
                "accepted_frame_warm_start": warm,
                "cold_fock_damping": damping,
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

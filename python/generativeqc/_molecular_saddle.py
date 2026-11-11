"""Fixed-model RKS derivative adapter for the native saddle corrector."""

from __future__ import annotations

import math
import typing
from contextlib import ExitStack
from dataclasses import replace
from numbers import Real
from pathlib import Path
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
    basis = calculator._basis
    has_ecp = isinstance(basis, BasisSet) and any(
        element.ecp_core_electrons for element in basis.elements
    )
    if (
        calculator._device_name != "cpu"
        or calculator.ks_options is None
        or multiplicity != 1
        or calculator.ks_options.method_ir.spin != "unpolarized"
        or not calculator._composite_rks_second_order
        or {primitive.kind for primitive in calculator.ks_options.method_ir.primitives}
        != {"semilocal_xc", "exact_exchange"}
        or has_ecp
        or calculator._density_fitting_mode != _native.DENSITY_FITTING_NONE
        or "forces" not in calculator.capabilities.supported_properties
    ):
        raise NotImplementedError(
            "molecular saddle requires qualified direct all-electron CPU full-range hybrid RKS"
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
    needed_output = (
        (3 if calculator._d3_second_order_spec is not None else 2)
        * (3 * len(atoms)) ** 2
        * 8
    )
    if needed_output > request["output_budget_bytes"]:
        raise MemoryError("molecular Hessian exceeds output_budget_bytes")
    for name in ("progress", "evaluation_progress", "cancelled"):
        if request[name] is not None and not callable(request[name]):
            raise TypeError(f"{name} must be callable")

    # Imports remain after admission: rejected method domains do no SCF/JIT work.
    from generativeqc_compiler.dft import NativeAO

    from .composite_rks_hessian import CompositeRKSResponse, composite_rks_hessian
    from .overlap import cross_overlap
    from .vibrations import analyze_harmonic_modes

    signature = calculator._model_signature()
    method = calculator.method_ir
    electronic_method = calculator.ks_options.method_ir
    records: list[typing.Mapping[str, typing.Any]] = []
    work = {"scf_iterations": 0, "scf_fock_builds": 0, "nuclear_response_solves": 0}
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
            state.pop("operator", None)

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
            primal = batch.execute(strict=True, properties=("energy", "forces")).items[
                0
            ]
            if not primal.converged or not primal.succeeded or primal.forces is None:
                raise RuntimeError("molecular saddle SCF/forces did not converge")
            work["scf_iterations"] += primal.iterations
            work["scf_fock_builds"] += primal.fock_builds
            ao = owners.enter_context(
                NativeAO(
                    current_atoms,
                    basis=calculator._basis,
                    representation=calculator._representation_name,
                    charge=charge,
                    multiplicity=multiplicity,
                )
            )
            operator = owners.enter_context(
                CompositeRKSResponse.from_native(
                    batch,
                    ao,
                    grid_budget_bytes=request["grid_budget_bytes"],
                )
            )
            reference = operator.state
            if (
                reference.identity.spin != "unpolarized"
                or reference.coefficients.shape[0] != 1
            ):
                raise ValueError("molecular saddle state is not restricted")
            occupations = np.asarray(reference.occupations[0])
            if not np.isin(occupations, (0.0, 2.0)).all() or not np.any(
                occupations == 2
            ):
                raise ValueError(
                    "molecular saddle requires integer occupied RKS orbitals"
                )
            occupied = np.array(
                reference.coefficients[0][:, occupations == 2], copy=True
            )
            overlap = 1.0
            if accepted:
                metric = cross_overlap(
                    calculator,
                    calculator,
                    current_atoms,
                    source_atoms=accepted["atoms"],
                    charge=charge,
                    multiplicity=multiplicity,
                    maximum_bytes=request["root_overlap_budget_bytes"],
                )
                overlaps = np.linalg.svd(
                    occupied.T @ metric @ accepted["occupied"], compute_uv=False
                )
                overlap = float(np.min(overlaps))
                if not math.isfinite(overlap) or overlap < minimum_overlap:
                    raise ValueError(
                        f"occupied root continuity lost: minimum overlap {overlap:.17g}"
                    )
            state.update(operator=operator, occupied=occupied)
            record(
                "energy_gradient",
                "complete",
                positions,
                energy=primal.energy,
                scf_iterations=primal.iterations,
                scf_fock_builds=primal.fock_builds,
                scf_physical_residual=primal.physical_residual_rms,
                minimum_occupied_overlap=overlap,
                grid_identity=reference.grid.identity,
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
        if "operator" not in state or not np.array_equal(positions, state["positions"]):
            raise ValueError(
                "Hessian geometry lacks its prepared energy/gradient state"
            )
        record("hessian", "start", positions)
        try:
            operator = state["operator"]
            result = composite_rks_hessian(
                operator,
                cache=request["cache"],
                integral_budget_bytes=request["integral_budget_bytes"],
                output_budget_bytes=request["output_budget_bytes"],
                solver_options=request["solver_options"],
            )
            result = replace(
                result,
                _diagnostics=MappingProxyType(
                    {
                        **result.diagnostics,
                        "geometry_identity": operator.state.identity.geometry_identity,
                        "electronic_method_identity": electronic_method.identity,
                        "total_method_identity": method.identity,
                        "total_method_manifest_identity": method.manifest_identity,
                    }
                ),
            )
            work["nuclear_response_solves"] += int(
                result.diagnostics["nuclear_response_solves"]
            )
            residuals = tuple(map(float, result.diagnostics["response_residual_norms"]))
            # Geometry-only publication cannot retain live electronic resources.
            close()
            del operator
            if calculator._d3_second_order_spec is not None:
                from .d3_hessian import compose_second_order

                result = compose_second_order(
                    result,
                    method,
                    state["atoms"],
                    cache=Path(request["cache"]),
                    max_workspace_bytes=calculator._dispersion_memory_budget_bytes,
                    output_budget_bytes=request["output_budget_bytes"],
                )
            record(
                "hessian",
                "complete",
                positions,
                nuclear_response_solves=result.diagnostics["nuclear_response_solves"],
                response_residual_norms=residuals,
                raw_symmetry=result.diagnostics["raw_symmetry_error"],
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
            accepted.update(atoms=state["atoms"], occupied=state["occupied"].copy())
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
                "root_policy": "occupied-principal-overlap-to-last-accepted",
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
                "external_optimizer": False,
            }
        ),
    )

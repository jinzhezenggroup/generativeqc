"""Complete range-separated meta-GGA/VV10 stationary Hessians using compiler AD.

The compiler owns all feature, functional, pair, integral and stationary-weight
derivatives. This adapter binds those partials to one converged native state and
the installed implicit CPKS solver. WB97M-V admits physical RKS/UKS spin spaces,
Cartesian/spherical AOs, full-rank DF and scalar ECPs in FP64. CUDA uses a
device-resident numerical owner with no host derivative fallback. No
displaced SCF or reference engine enters production; quadrature/active-set
branches and occupations remain fixed during differentiation.
"""

from __future__ import annotations

import time
import typing
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import product
from types import MappingProxyType, SimpleNamespace

from generativeqc_compiler.common.array_execution import arrays as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.ao import directional_ao_jets
from generativeqc_compiler.integral.ir import four_center_eri_operator
from generativeqc_compiler.integral.range_separation import CoulombKernel
from generativeqc_compiler.method import StationaryHVPPlan, StationaryMeanField
from generativeqc_compiler.method.stationary_gradient import SCF_POINT_MODEL
from generativeqc_compiler.method.stationary_grid_second import (
    GRID_FIELD_WORKSPACE_PANELS,
    StationaryGridSecondOrder,
    features,
    nuclear_repulsion_hvp,
)
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.xc.grid_response import (
    partition_mixed_response,
    partition_response,
)
from generativeqc_compiler.xc.potential import (
    assemble_coefficients,
    assemble_coefficients_directional,
)

from ._dft_gradient import (
    StationaryDerivativeContract,
    StationaryKsState,
    _native_ao_atoms,
)
from .df_hessian import DFSecondOrder
from .ecp_hessian import ECPSecondOrder
from .fock import FockBuildSpec, FockPlan, FockTerm
from .hessian_basis import HessianBasis
from .ks import composite_stationary_second_order_eligible, resolve_ks_method
from .profiles import canonical_hash
from .response_operator import CPKSResponseOperator
from .response_solver import GMRESOptions, solve
from .rks_hessian import (
    RKSHessianResult,
    _checked_integral_budget,
)
from .rks_hessian_directional import _validate_partition_provenance
from .rks_hessian_integrals import (
    RKSIntegralTopology,
    checked_direction,
    generated_directional_semilocal_rks_integral_first_order,
    generated_weighted_first_integral_gradient,
    generated_weighted_second_integral_hvp,
)
from .rks_response import RKSResponseReference, _basis_hash, _geometry_hash
from .second_order import (
    StationaryHVPContributor,
    StationaryPerturbationProvider,
    StationaryResponseDriver,
    StationarySecondOrderExecutor,
)
from .spin_stationary import SpinCPKSOperator
from .stationary_nuclear import solve_stationary_nuclear_perturbation


@dataclass(frozen=True, eq=False)
class CompositeRKSHVPResult:
    """Detached complete AD HVP, with raw source contributions and diagnostics."""

    direction: np.ndarray
    value: np.ndarray
    components: typing.Mapping[str, np.ndarray]
    identity: str
    _diagnostics: typing.Mapping[str, typing.Any]

    @property
    def diagnostics(self) -> dict:
        """Return a detached copy of the completed endpoint's work record."""
        return deepcopy(dict(self._diagnostics))


class _CompositeJK:
    """Native linear J/SR-K/LR-K map, including physical exchange coefficients."""

    def __init__(
        self,
        state: StationaryKsState,
        basis: typing.Any,
        ecp: ECPSecondOrder | None = None,
    ) -> None:
        self.basis = basis
        self.plan_basis = basis
        self.auxiliary = None
        coulomb, exchange, threshold = state._source.fock_provider_proof()
        self.fitted = coulomb == "density-fitted"
        if self.fitted and exchange != "density-fitted":
            raise NotImplementedError(
                "composite DF Hessian requires fitted full-range J and K"
            )
        if not self.fitted and (coulomb, exchange) != ("exact", "exact"):
            raise NotImplementedError(
                "composite Hessian has no response rule for this J/K approximation"
            )
        self.metric_threshold = threshold
        self.spins = state.density.shape[0]
        self.device = state._source.backend
        self.device_id = (
            state._source._batch._calculator._device_id if self.device == "cuda" else 0
        )
        spin = "unrestricted" if self.spins == 2 else "restricted"
        self.plans = []
        self.host_workspace_bytes = 12 * basis.nao**2 * 8
        self.device_workspace_bytes = 0
        self.statistics = {"actions": 0, "seconds": 0.0, "peak_bytes": 0}
        absent = FockTerm(present=False, coefficient=0)
        try:
            if ecp is not None:
                self.plan_basis = NativeAO(
                    tuple(
                        replace(atom, atomic_number=atom.atomic_number - core)
                        for atom, core in zip(
                            basis.atoms, state._source.ecp_cores, strict=True
                        )
                    ),
                    basis.shells,
                    representation=basis.representation,
                    charge=basis.charge,
                    multiplicity=basis.multiplicity,
                )
            short_exchange = next(
                primitive.coefficient
                for primitive in state._source.method_ir.primitives
                if primitive.kind == "range_separated_exchange"
                and primitive.operator == "short-range"
            )
            if self.fitted:
                calculator = state._source._batch._calculator
                auxiliary_spec = calculator._auxiliary_basis
                auxiliary_shells = (
                    basis.shells
                    if auxiliary_spec is None
                    else calculator._shells_for_atoms(
                        basis.atoms, auxiliary_spec, operator="df_metric"
                    )
                )
                self.auxiliary = NativeAO(
                    self.plan_basis.atoms,
                    auxiliary_shells,
                    representation=getattr(
                        auxiliary_spec, "representation", basis.representation
                    ),
                    charge=basis.charge,
                    multiplicity=basis.multiplicity,
                )
            self.plans.append(
                FockPlan(
                    self.plan_basis,
                    FockBuildSpec(
                        spin=spin,
                        derivative_order=0,
                        coulomb=FockTerm(approximation="density_fitted")
                        if self.fitted
                        else FockTerm(),
                        exchange=FockTerm(
                            coefficient=(-1 if self.spins == 2 else -0.5)
                            * float(short_exchange),
                            approximation="density_fitted",
                        )
                        if self.fitted
                        else absent,
                    ),
                    auxiliary=self.auxiliary,
                    device=self.device,
                    device_id=self.device_id,
                    screening_tolerance=0,
                    metric_relative_threshold=threshold if self.fitted else 1e-10,
                )
            )
            for primitive in state._source.method_ir.primitives:
                if primitive.kind != "range_separated_exchange":
                    continue
                if self.fitted and primitive.operator != "long-range":
                    continue
                self.plans.append(
                    FockPlan(
                        self.plan_basis,
                        FockBuildSpec(
                            spin=spin,
                            derivative_order=0,
                            coulomb=absent,
                            exchange=FockTerm(
                                coefficient=(-1 if self.spins == 2 else -0.5)
                                * float(
                                    primitive.coefficient - short_exchange
                                    if self.fitted
                                    else primitive.coefficient
                                ),
                                operator=primitive.operator.replace("-", "_"),
                                omega=float(primitive.omega),
                            ),
                        ),
                        device=self.device,
                        device_id=self.device_id,
                        screening_tolerance=0,
                    )
                )
            zero = np.zeros(
                (2, basis.nao, basis.nao) if self.spins == 2 else (basis.nao, basis.nao)
            )
            hcore = self.plans[0].evaluate(zero).fock
            self.hcore = immutable(
                (hcore[0] if self.spins == 2 else hcore)
                + (0 if ecp is None else ecp.matrix)
            )
            self.device_workspace_bytes = sum(
                plan.diagnostics["device_bytes"] for plan in self.plans
            )
            self.identity = canonical_hash(
                {
                    "plans": [plan.identity for plan in self.plans],
                    "state": state.identity.to_payload(),
                }
            )
        except Exception:
            self.close()
            raise

    def validate_reference(self, reference: typing.Any) -> None:
        for plan in self.plans:
            plan._ensure_open()
        if reference.geometry_hash != _geometry_hash(
            self.basis
        ) or reference.basis_hash != _basis_hash(self.basis):
            raise ValueError("composite JK/reference identity mismatch")

    def coulomb_exchange(self, density: np.ndarray) -> tuple:
        started = time.perf_counter()
        value = np.zeros_like(density)
        for plan in self.plans:
            result = plan.evaluate(density)
            if result.coulomb is not None:
                value += plan.spec.coulomb.coefficient * result.coulomb
            if result.exchange is not None:
                value += plan.spec.exchange.coefficient * result.exchange
        self.statistics["actions"] += 1
        self.statistics["seconds"] += time.perf_counter() - started
        # CPKS has zero external exchange fraction: this backend already applies
        # each primitive's restricted -a/2 coefficient exactly once.
        return immutable(value), immutable(np.zeros_like(value))

    def close(self) -> None:
        for plan in self.plans:
            plan.close()
        if self.plan_basis is not self.basis:
            self.plan_basis.close()
        if self.auxiliary is not None:
            self.auxiliary.close()


def _ao_coefficients(values: np.ndarray) -> dict:
    spins = (values.shape[0] - 4) // 5
    fields = values[: 5 * spins].reshape(spins, 5, -1)
    return {
        "rho": fields[:, 0],
        "gradient": fields[:, 1:4].transpose(0, 2, 1),
        "tau": 0.5 * fields[:, 4],
    }


class _CompositeGrid:
    """Bind compiler grid partials to native AO jets and moving Becke weights."""

    def __init__(
        self,
        state: StationaryKsState,
        basis: typing.Any,
        *,
        grid_budget_bytes: int,
        max_nonlocal_pairs: int,
    ) -> None:
        if getattr(state._source, "nonlocal_grid", state.grid) is not state.grid:
            raise NotImplementedError(
                "independent XC/VV10 grids do not yet support coupled response"
            )
        self.state, self.basis, self.grid = state, basis, state.grid
        _, self.spec = resolve_ks_method(state.identity.method)
        self.functional_identity = self.spec.identity
        self.grid_identity = self.grid.identity
        self.basis_identity = _basis_hash(basis)
        self.natom = basis.natom
        self.owners = np.asarray(self.grid.owners, dtype=np.int64)
        self.centers = np.asarray([atom.position for atom in basis.atoms])
        self.ao_atoms = _native_ao_atoms(basis)
        self.atomic_weights = np.asarray(state._source.atomic_weights)
        self.grid_spec = state._source.grid_spec
        self.partition_radii = (
            self.grid_spec.partition_radii(basis.atoms) if self.grid_spec else None
        )
        if (
            self.grid_spec is None
            or self.atomic_weights.shape != self.grid.weights.shape
        ):
            raise ValueError(
                "composite Hessian requires retained native grid provenance"
            )
        nonlocal_primitive = next(
            primitive
            for primitive in state._source.method_ir.primitives
            if primitive.kind == "nonlocal_correlation"
        )
        ao_bytes = len(self.grid.points) * basis.nao * 8 * 64
        if ao_bytes >= grid_budget_bytes:
            raise MemoryError("AO jet/motion workspace exceeds grid_budget_bytes")
        self.model = StationaryGridSecondOrder(
            self.spec,
            nonlocal_primitive,
            budget_bytes=grid_budget_bytes - ao_bytes,
            max_pair_evaluations=max_nonlocal_pairs,
        )
        self.host_workspace_bytes = grid_budget_bytes
        self.feature_budget_bytes = (
            self.model.budget_bytes
            - GRID_FIELD_WORKSPACE_PANELS
            * self.model.field_count
            * 8
            * len(self.grid.points)
        )
        if self.feature_budget_bytes < 1:
            raise MemoryError("retained grid partials exceed grid_budget_bytes")
        self.jets = basis.evaluate(self.grid.points, 3)
        self.fields = np.vstack(
            (
                self.feature_fields(),
                self.grid.weights,
                self.grid.points.T,
            )
        )
        self.zero_jets = np.zeros_like(self.jets[:4])
        self.identity = canonical_hash(
            {"model": self.model.identity, "state": state.identity.to_payload()}
        )
        self.statistics = {
            "actions": 0,
            "seconds": 0.0,
            "peak_bytes": 0,
            "pair_evaluations": 0,
        }
        self.reference = self.evaluate(np.zeros_like(self.fields))

    def feature_fields(
        self, *, delta_density: np.ndarray | None = None, **options: typing.Any
    ) -> np.ndarray:
        """Evaluate each physical spin density with the same AO TensorIR program."""
        if delta_density is not None and delta_density.ndim == 2:
            delta_density = delta_density[None]
        return np.concatenate(
            [
                features(
                    self.jets,
                    density,
                    delta_density=None
                    if delta_density is None
                    else delta_density[spin],
                    max_bytes=self.feature_budget_bytes,
                    **options,
                )
                for spin, density in enumerate(self.state.density)
            ]
        )

    def public_fock(self, values: np.ndarray) -> np.ndarray:
        """Retain a spin axis only for a genuinely unrestricted reference."""
        return values if self.model.spins == 2 else values[0]

    def validate_reference(self, reference: typing.Any) -> None:
        self.state._source.check_current()
        if (
            reference.functional_identity != self.functional_identity
            or reference.grid_identity != self.grid_identity
            or reference.basis_hash != self.basis_identity
        ):
            raise ValueError("composite grid/reference identity mismatch")

    def evaluate(self, direction: np.ndarray) -> typing.Any:
        result = self.model.evaluate(self.fields, direction)
        self.statistics["pair_evaluations"] += result.pair_evaluations
        return result

    def apply(self, delta_density: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        direction = np.zeros_like(self.fields)
        direction[: self.model.feature_count] = self.feature_fields(
            left=self.zero_jets,
            delta_density=delta_density,
        )
        result = self.evaluate(direction)
        response = self.public_fock(
            assemble_coefficients(
                self.jets[:4],
                _ao_coefficients(result.directional_coefficients),
                np.ones(len(self.grid.points)),
            )
        )
        self.statistics["actions"] += 1
        self.statistics["seconds"] += time.perf_counter() - started
        return immutable(response)

    apply_transpose = apply

    def motion(self, centers: np.ndarray, points: np.ndarray) -> np.ndarray:
        return directional_ao_jets(
            self.jets,
            1,
            ao_atoms=self.ao_atoms,
            point_motion=points,
            center_motion=centers,
        )

    def partition(self, vector: np.ndarray) -> typing.Any:
        result = partition_response(
            self.grid.points,
            self.centers,
            point_motion=vector[self.owners],
            center_motion=vector,
            iterations=self.grid_spec.partition_iterations,
            coincident_tolerance=self.grid_spec.coincident_tolerance,
            atomic_radii=self.partition_radii,
        )
        selected = (np.arange(len(self.owners)), self.owners)
        _validate_partition_provenance(
            self.atomic_weights, self.grid.weights, result.weights[selected]
        )
        return result

    def direction(
        self, vector: np.ndarray, delta_density: np.ndarray | None = None
    ) -> tuple:
        point_motion = vector[self.owners]
        jets = self.motion(vector, point_motion)
        partition = self.partition(vector)
        selected = (np.arange(len(self.owners)), self.owners)
        weights = self.atomic_weights * partition.directional[selected]
        fields = np.vstack(
            (
                self.feature_fields(
                    left=jets,
                    delta_density=delta_density,
                ),
                weights,
                point_motion.T,
            )
        )
        return fields, jets, partition.branch_identity

    def frozen_fock(self, vector: np.ndarray) -> np.ndarray:
        direction, jets, _ = self.direction(vector)
        partials = self.evaluate(direction)
        return immutable(
            self.public_fock(
                assemble_coefficients_directional(
                    self.jets[:4],
                    jets,
                    _ao_coefficients(partials.coefficients),
                    _ao_coefficients(partials.directional_coefficients),
                    np.ones(len(self.grid.points)),
                    np.zeros(len(self.grid.points)),
                )
            )
        )

    def hvp(self, vector: np.ndarray, delta_density: np.ndarray) -> dict:
        direction, _, branch = self.direction(vector, delta_density)
        partials = self.evaluate(direction)
        zeros_points = np.zeros_like(self.grid.points)
        zeros_centers = np.zeros_like(vector)
        selected = (np.arange(len(self.owners)), self.owners)
        components = {
            f"{prefix}_{kind}": np.zeros_like(vector)
            for prefix in ("xc", "nonlocal")
            for kind in ("ao", "grid", "weight")
        }
        for coordinate in range(vector.size):
            left = np.zeros_like(vector)
            left.reshape(-1)[coordinate] = 1
            left_points = left[self.owners]
            partition = partition_mixed_response(
                self.grid.points,
                self.centers,
                left_point_motion=left_points,
                left_center_motion=left,
                right_point_motion=vector[self.owners],
                right_center_motion=vector,
                iterations=self.grid_spec.partition_iterations,
                coincident_tolerance=self.grid_spec.coincident_tolerance,
                atomic_radii=self.partition_radii,
            )
            if partition.branch_identity != branch:
                raise ValueError("Becke branch changed during Hessian assembly")
            for kind, centers, points in (
                ("ao", left, zeros_points),
                ("grid", zeros_centers, left_points),
                ("weight", zeros_centers, zeros_points),
            ):
                left_jets = self.motion(centers, points)
                left_extended = directional_ao_jets(
                    self.jets,
                    2,
                    ao_atoms=self.ao_atoms,
                    point_motion=points,
                    center_motion=centers,
                )
                mixed_jets = directional_ao_jets(
                    left_extended,
                    1,
                    ao_atoms=self.ao_atoms,
                    point_motion=vector[self.owners],
                    center_motion=vector,
                )
                right_jets = self.motion(vector, vector[self.owners])
                left_fields = np.zeros_like(self.fields)
                mixed_fields = np.zeros_like(self.fields)
                left_fields[: self.model.feature_count] = self.feature_fields(
                    left=left_jets,
                )
                mixed_fields[: self.model.feature_count] = self.feature_fields(
                    left=left_jets,
                    right=right_jets,
                    mixed=mixed_jets,
                    delta_density=delta_density,
                )
                left_fields[self.model.feature_count + 1 :] = points.T
                if kind == "weight":
                    left_fields[self.model.feature_count] = (
                        self.atomic_weights * partition.left[selected]
                    )
                    mixed_fields[self.model.feature_count] = (
                        self.atomic_weights * partition.mixed[selected]
                    )
                for prefix, coefficients in (
                    ("xc", partials.semilocal),
                    ("nonlocal", partials.nonlocal_correlation),
                ):
                    base, moving = coefficients
                    components[f"{prefix}_{kind}"].reshape(-1)[coordinate] = np.sum(
                        moving * left_fields + base * mixed_fields
                    )
        return {name: immutable(value) for name, value in components.items()}


class CompositeRKSResponse(CPKSResponseOperator):
    """Live implicit RKS response with all MethodIR interaction primitives."""

    @classmethod
    def from_native(
        cls,
        batch: typing.Any,
        basis: typing.Any,
        *,
        grid_budget_bytes: int = 128 << 20,
        max_nonlocal_pairs: int = 16_777_216,
    ) -> typing.Self:
        state = StationaryKsState.from_native(batch, basis)
        backend = None
        integral_basis = None
        try:
            if state._source.backend not in ("cpu", "cuda"):
                raise NotImplementedError(
                    "composite Hessian requires native CPU/CUDA KS"
                )
            if not composite_stationary_second_order_eligible(state._source.method_ir):
                raise NotImplementedError(
                    "composite molecular Hessian currently requires canonical WB97M-V"
                )
            if state._source.backend == "cuda":
                from .cuda_composite_hessian import CudaCompositeResponse

                return CudaCompositeResponse(
                    state,
                    basis,
                    batch,
                    grid_budget_bytes=grid_budget_bytes,
                    max_nonlocal_pairs=max_nonlocal_pairs,
                )
            plan = StationaryHVPPlan(
                state._source.method_ir,
                StationaryMeanField(
                    SCF_POINT_MODEL,
                    hamiltonian=state._source.hamiltonian,
                    coulomb="density-fitted-full-rank"
                    if state._source.fock_provider_proof()[0] == "density-fitted"
                    else "direct-full-range",
                ),
            )
            integral_basis = HessianBasis(basis)
            kernel = _CompositeGrid(
                state,
                basis,
                grid_budget_bytes=grid_budget_bytes,
                max_nonlocal_pairs=max_nonlocal_pairs,
            )
            persistent_grid = kernel.jets.nbytes + 7 * kernel.fields.nbytes
            ecp = (
                ECPSecondOrder(
                    state, basis, budget_bytes=grid_budget_bytes - persistent_grid
                )
                if state._source.hamiltonian == "scalar-semilocal-ecp"
                else None
            )
            backend = _CompositeJK(state, basis, ecp)
            interaction, _ = backend.coulomb_exchange(
                state.density if backend.spins == 2 else state.density[0]
            )
            grid_fock = kernel.public_fock(
                assemble_coefficients(
                    kernel.jets[:4],
                    _ao_coefficients(kernel.reference.coefficients),
                    np.ones(len(kernel.grid.points)),
                )
            )
            if (
                np.max(
                    abs(
                        backend.hcore
                        + interaction
                        + grid_fock
                        - (state.fock if backend.spins == 2 else state.fock[0])
                    )
                )
                > 2e-8
            ):
                raise ValueError("compiler AD/native SCF Fock provenance mismatch")
            if backend.spins == 2:
                result = CompositeUKSResponse(state, basis, backend, kernel)
            else:
                reference = RKSResponseReference(
                    overlap=state.overlap,
                    hcore=backend.hcore,
                    fock=state.fock[0],
                    coefficients=state.coefficients[0],
                    orbital_energies=state.orbital_energies[0],
                    occupations=state.occupations[0],
                    electron_count=int(np.sum(state.occupations[0])),
                    reference_energy=state._source.energy(),
                    scf_residual=state.physical_residual,
                    geometry_hash=_geometry_hash(basis),
                    basis_hash=_basis_hash(basis),
                    generation_id=canonical_hash(state.identity.to_payload()),
                    functional_identity=kernel.spec.identity,
                    grid_identity=state.grid.identity,
                    representation=basis.representation,
                    hf_backend=f"native-{backend.device}-composite-rks",
                    device_id=backend.device_id if backend.device == "cuda" else None,
                )
                result = cls(
                    cls.build_problem(reference, backend, kernel), backend, kernel
                )
            result.state, result.basis, result.plan = state, basis, plan
            result.contract = StationaryDerivativeContract(state.identity)
            result.integral_basis = integral_basis
            result.topology = RKSIntegralTopology.from_basis(integral_basis.cartesian)
            result.ecp = ecp
            result.df = None
            if ecp is not None:
                result.topology = replace(
                    result.topology,
                    atoms=tuple(
                        replace(atom, atomic_number=atom.atomic_number - core)
                        for atom, core in zip(
                            result.topology.atoms, state._source.ecp_cores, strict=True
                        )
                    ),
                )
            result.second_options = {
                "backend": backend.device,
                "compiler": batch._stationary_cuda_compiler()
                if backend.device == "cuda"
                else None,
                "device_id": backend.device_id,
            }
            return result
        except Exception:
            if backend is not None:
                backend.close()
            if integral_basis is not None:
                integral_basis.close()
            state._source.close()
            raise

    def validate_current(self) -> None:
        self.contract.validate(self.state)
        if not self.basis._handle:
            raise ValueError("composite Hessian AO owner is closed")
        self.backend.validate_reference(self.problem.reference)
        self.xc_kernel.validate_reference(self.problem.reference)

    def close(self) -> None:
        """Release only response-owned resources, not the borrowed AO/batch."""
        self.backend.close()
        if self.df is not None:
            self.df.close()
        self.integral_basis.close()
        self.state._source.close()

    def __enter__(self) -> typing.Self:
        self.validate_current()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class CompositeUKSResponse(SpinCPKSOperator, CompositeRKSResponse):
    """Coupled spin response sharing the composite integral/grid lifetime owner."""


def _composite_integral_hvp(
    operator: CompositeRKSResponse,
    response: typing.Any,
    source: str,
    direction: np.ndarray,
    cache: typing.Any,
    budget_bytes: int,
) -> np.ndarray:
    """Contract public-space stationary weights in a fixed Cartesian workspace."""
    density = operator.integral_basis.density(operator.state.density)
    moving = operator.integral_basis.density(response.density_derivative)
    if moving.ndim == 2:
        moving = moving[None]
    weighted = operator.integral_basis.density(operator.state.weighted_density)
    dweighted = operator.integral_basis.density(
        response.energy_weighted_density_derivative
    )
    if dweighted.ndim == 2:
        dweighted = dweighted[None]
    offsets = np.cumsum((0, *operator.topology.shell_sizes))

    def weights(slots: tuple | None = None) -> tuple:
        if slots is None:
            rows, columns = np.indices((operator.topology.nbf,) * 2).reshape(2, -1)
            feeds = (
                {
                    "weighted_density": weighted[:, rows, columns],
                    "d_weighted_density": dweighted[:, rows, columns],
                }
                if source == "overlap_pulay"
                else {
                    "density_left": density[:, rows, columns],
                    "d_density_left": moving[:, rows, columns],
                }
            )
            shape = (operator.topology.nbf,) * 2
        else:
            indices = np.asarray(
                tuple(
                    product(
                        *(range(offsets[slot], offsets[slot + 1]) for slot in slots)
                    )
                )
            )
            first, second, third, fourth = indices.T
            feeds = {
                "density_left": density[:, first, second],
                "density_right": density[:, third, fourth],
                "d_density_left": moving[:, first, second],
                "d_density_right": moving[:, third, fourth],
            }
            shape = tuple(operator.topology.shell_sizes[slot] for slot in slots)
        block = operator.plan.integral_block(
            source, terms=int(np.prod(shape)), coordinates=1
        )
        return (
            execute(block.weights, feeds).outputs["weights"].reshape(shape),
            execute(block.response_weights, feeds)
            .outputs["response_weights"]
            .reshape(shape),
        )

    if source == "coulomb":
        fixed_options = {"eri_shell_weights": lambda slots: weights(slots)[0]}
        moving_options = {"eri_shell_weights": lambda slots: weights(slots)[1]}
    else:
        fixed, delta = weights()
        fixed_options = {"pair_weights": fixed}
        moving_options = {"pair_weights": delta}
    first = generated_weighted_first_integral_gradient(
        operator.topology, source, cache=cache, **moving_options
    )
    second, _ = generated_weighted_second_integral_hvp(
        operator.topology,
        source,
        direction,
        cache=cache,
        budget_bytes=budget_bytes,
        **operator.second_options,
        **fixed_options,
    )
    return immutable(first + second)


def _exchange_weights(
    operator: CompositeRKSResponse, response: typing.Any, source: str, slots: tuple
) -> tuple:
    offsets = np.cumsum((0, *operator.topology.shell_sizes))
    indices = np.asarray(
        tuple(product(*(range(offsets[slot], offsets[slot + 1]) for slot in slots)))
    )
    first, second, third, fourth = indices.T
    density = operator.integral_basis.density(operator.state.density)
    moving = operator.integral_basis.density(response.density_derivative)
    if moving.ndim == 2:
        moving = moving[None]
    feeds = {
        "density_left": density[:, first, third],
        "density_right": density[:, second, fourth],
        "d_density_left": moving[:, first, third],
        "d_density_right": moving[:, second, fourth],
    }
    block = operator.plan.integral_block(source, terms=len(indices), coordinates=1)
    shape = tuple(operator.topology.shell_sizes[slot] for slot in slots)
    return (
        execute(block.weights, feeds).outputs["weights"].reshape(shape),
        execute(block.response_weights, feeds)
        .outputs["response_weights"]
        .reshape(shape),
    )


def composite_rks_hvp(
    operator: CompositeRKSResponse,
    direction: typing.Any,
    *,
    cache: typing.Any = ".artifacts",
    integral_budget_bytes: int = 64 << 20,
    solver_options: GMRESOptions | None = None,
) -> CompositeRKSHVPResult:
    """Execute a complete compiler-AD HVP using one implicit nuclear response."""
    if getattr(operator, "cuda_resident", False):
        return operator.hvp(
            direction,
            cache=cache,
            integral_budget_bytes=integral_budget_bytes,
            solver_options=solver_options,
        )
    if operator.backend.device != "cpu":
        raise ValueError(
            "CUDA Hessians require the resident owner; host AD fallback is disabled"
        )
    started = time.perf_counter()
    operator.validate_current()
    vector = checked_direction(direction, len(operator.topology.atoms))
    if solver_options is not None and not isinstance(solver_options, GMRESOptions):
        raise TypeError("solver_options must be GMRESOptions")
    _checked_integral_budget(
        SimpleNamespace(_source=operator.topology), integral_budget_bytes
    )
    initial_pairs = operator.xc_kernel.statistics["pair_evaluations"]
    if operator.backend.fitted and operator.df is None:
        fitting = DFSecondOrder(
            operator.state,
            operator.integral_basis,
            operator.backend.auxiliary,
            operator.backend.metric_threshold,
            budget_bytes=integral_budget_bytes,
        )
        try:
            fitting.prepare(cache)
            primary = operator.backend.plans[0]
            value = primary.evaluate(
                operator.state.density
                if fitting.spins == 2
                else operator.state.density[0]
            )
            expected = (
                primary.spec.coulomb.coefficient * value.coulomb
                + primary.spec.exchange.coefficient * value.exchange
            )
            if np.max(abs(fitting.fock() - expected)) > 2e-8:
                raise ValueError("compiler DF/native fitted Fock provenance mismatch")
            operator.df = fitting
        except Exception:
            fitting.close()
            raise
    if operator.df is not None:
        operator.df.check_budget(integral_budget_bytes)
    exchange = tuple(
        (
            four_center_eri_operator(
                CoulombKernel(
                    primitive.operator.replace("-", "_"), float(primitive.omega)
                )
            ),
            primitive.coefficient,
        )
        for primitive in operator.plan.method.primitives
        if primitive.kind == "range_separated_exchange"
    )
    exchange_sources = {
        "exchange_" + primitive.operator.replace("-", "_"): entry[0]
        for primitive, entry in zip(
            (
                primitive
                for primitive in operator.plan.method.primitives
                if primitive.kind == "range_separated_exchange"
            ),
            exchange,
            strict=True,
        )
    }
    if operator.df is not None:
        short_fraction = next(
            primitive.coefficient
            for primitive in operator.plan.method.primitives
            if primitive.kind == "range_separated_exchange"
            and primitive.operator == "short-range"
        )
        long_operator = exchange_sources["exchange_long_range"]
        exchange = tuple(
            (eri, coefficient - short_fraction)
            for eri, coefficient in exchange
            if eri == long_operator
        )
        exchange_sources["exchange_short_range"] = long_operator

    def perturbation(vector: np.ndarray) -> tuple:
        densities = operator.integral_basis.density(operator.state.density)
        frozen_blocks = []
        for density in densities:
            frozen, overlap = generated_directional_semilocal_rks_integral_first_order(
                operator.topology,
                densities.sum(axis=0),
                vector,
                cache=cache,
                exchange_operators=exchange,
                exchange_density=(2 if len(densities) == 2 else 1) * density,
                include_coulomb=operator.df is None,
            )
            frozen_blocks.append(operator.integral_basis.matrix(frozen))
        return (
            operator.xc_kernel.public_fock(np.asarray(frozen_blocks))
            + operator.xc_kernel.frozen_fock(vector)
            + (0 if operator.ecp is None else operator.ecp.frozen_fock(vector))
            + (0 if operator.df is None else operator.df.frozen_fock(vector)),
            operator.integral_basis.matrix(overlap),
        )

    def response_driver(values: tuple) -> typing.Any:
        if isinstance(operator, CompositeUKSResponse):
            return operator.solve_nuclear(
                *values, GMRESOptions() if solver_options is None else solver_options
            )
        return solve_stationary_nuclear_perturbation(
            operator,
            *values,
            solver=solve,
            options=GMRESOptions() if solver_options is None else solver_options,
        )

    grid_components = {}
    ecp_components = {}
    df_components = {}

    def contribution(context: typing.Any, source: str) -> np.ndarray:
        response = context.response
        if operator.df is not None and source in ("coulomb", "exchange_short_range"):
            if not df_components:
                df_components.update(
                    operator.df.hvp(context.direction, response.density_derivative)
                )
            if source == "coulomb":
                return df_components["coulomb"]
        if source.startswith("ecp_"):
            if not ecp_components:
                moving = response.density_derivative
                ecp_components.update(
                    operator.ecp.hvp(
                        context.direction,
                        operator.state.density.sum(axis=0),
                        moving.sum(axis=0) if moving.ndim == 3 else moving,
                    )
                )
            return ecp_components[source]
        if source.startswith(("xc_", "nonlocal_")):
            if not grid_components:
                grid_components.update(
                    operator.xc_kernel.hvp(
                        context.direction, response.density_derivative
                    )
                )
            return grid_components[source]
        if source in ("one_electron", "coulomb", "overlap_pulay"):
            return _composite_integral_hvp(
                operator,
                response,
                source,
                context.direction,
                cache,
                integral_budget_bytes,
            )
        if source in exchange_sources:

            def fixed(slots: tuple) -> np.ndarray:
                return _exchange_weights(operator, response, source, slots)[0]

            def moving(slots: tuple) -> np.ndarray:
                return _exchange_weights(operator, response, source, slots)[1]

            first = generated_weighted_first_integral_gradient(
                operator.topology,
                "coulomb",
                eri_shell_weights=moving,
                eri_operator=exchange_sources[source],
                cache=cache,
            )
            second, _ = generated_weighted_second_integral_hvp(
                operator.topology,
                "coulomb",
                context.direction,
                eri_shell_weights=fixed,
                eri_operator=exchange_sources[source],
                cache=cache,
                budget_bytes=integral_budget_bytes,
                **operator.second_options,
            )
            value = first + second
            if operator.df is not None and source == "exchange_short_range":
                value = df_components["exchange"] - value
            return immutable(value)
        if source == "nuclear":
            return nuclear_repulsion_hvp(
                np.asarray([atom.position for atom in operator.topology.atoms]),
                np.asarray([atom.atomic_number for atom in operator.topology.atoms]),
                context.direction,
            )
        raise ValueError(f"unbound composite RKS source {source!r}")

    executor = StationarySecondOrderExecutor(
        operator.plan,
        natoms=len(operator.topology.atoms),
        perturbation=StationaryPerturbationProvider(operator.identity, perturbation),
        response=StationaryResponseDriver(operator.identity, response_driver),
        contributors=tuple(
            StationaryHVPContributor(
                source,
                canonical_hash({"plan": operator.plan.identity, "source": source}),
                lambda context, source=source: contribution(context, source),
            )
            for source in operator.plan.source_names
        ),
    )
    executed = executor.apply(vector)
    response, components, value = (
        executed.response,
        executed.components,
        immutable(executed.value),
    )
    operator.validate_current()
    identity = canonical_hash(
        {
            "plan": operator.plan.identity,
            "state": operator.state.identity.to_payload(),
            "direction": vector.tolist(),
            "value": value.tolist(),
        }
    )
    diagnostics = {
        **executed.diagnostics,
        "derivative_method": "compiler-automatic-differentiation",
        "public_calculator_endpoint": True,
        "backend": operator.backend.device,
        "spin": operator.state.identity.spin,
        "representation": operator.basis.representation,
        "first_integral_backend": "cpu",
        "second_integral_backend": operator.backend.device,
        "grid_ad_backend": "cpu",
        "ecp_ad_backend": "cpu" if operator.ecp is not None else None,
        "df_integral_ad_backend": "cpu" if operator.df is not None else None,
        "density_fitting": "full-rank-full-range-jk-plus-exact-lr"
        if operator.df is not None
        else "none",
        "df_metric_threshold": operator.backend.metric_threshold
        if operator.df is not None
        else None,
        "execution_residency": "host",
        "nuclear_response_solves": 1,
        "displaced_scf_evaluations": 0,
        "response_residual_norm": response.solve_result.residual_norm,
        "response_iterations": response.solve_result.iterations,
        "source_names": operator.plan.source_names,
        "pair_evaluations": operator.xc_kernel.statistics["pair_evaluations"]
        - initial_pairs,
        "assembly_seconds": time.perf_counter() - started,
    }
    return CompositeRKSHVPResult(
        immutable(vector),
        value,
        MappingProxyType(components),
        identity,
        MappingProxyType(diagnostics),
    )


def composite_rks_hessian(
    operator: CompositeRKSResponse,
    *,
    cache: typing.Any = ".artifacts",
    integral_budget_bytes: int = 64 << 20,
    output_budget_bytes: int = 64 << 20,
    solver_options: GMRESOptions | None = None,
) -> RKSHessianResult:
    """Assemble raw Cartesian columns; never symmetrize or project errors away."""
    if getattr(operator, "cuda_resident", False):
        return operator.matrix(
            cache=cache,
            integral_budget_bytes=integral_budget_bytes,
            output_budget_bytes=output_budget_bytes,
            solver_options=solver_options,
        )
    if operator.backend.device != "cpu":
        raise ValueError(
            "CUDA Hessians require the resident owner; host AD fallback is disabled"
        )
    started = time.perf_counter()
    natom = len(operator.topology.atoms)
    coordinates = 3 * natom
    if type(output_budget_bytes) is not int or not 0 < output_budget_bytes < 2**63:
        raise ValueError("output_budget_bytes must be a positive int64 byte count")
    if 2 * coordinates**2 * 8 > output_budget_bytes:
        raise ValueError("full Hessian publication exceeds output_budget_bytes")
    matrix = np.empty((coordinates, coordinates))
    residuals = []
    for coordinate in range(coordinates):
        direction = np.zeros((natom, 3))
        direction.reshape(-1)[coordinate] = 1
        result = composite_rks_hvp(
            operator,
            direction,
            cache=cache,
            integral_budget_bytes=integral_budget_bytes,
            solver_options=solver_options,
        )
        matrix[:, coordinate] = result.value.reshape(-1)
        residuals.append(result.diagnostics["response_residual_norm"])
    diagnostics = {
        "derivative_method": "compiler-automatic-differentiation",
        "public_calculator_endpoint": True,
        "backend": operator.backend.device,
        "spin": operator.state.identity.spin,
        "representation": operator.basis.representation,
        "first_integral_backend": "cpu",
        "second_integral_backend": operator.backend.device,
        "grid_ad_backend": "cpu",
        "ecp_ad_backend": "cpu" if operator.ecp is not None else None,
        "df_integral_ad_backend": "cpu" if operator.df is not None else None,
        "density_fitting": "full-rank-full-range-jk-plus-exact-lr"
        if operator.df is not None
        else "none",
        "df_metric_threshold": operator.backend.metric_threshold
        if operator.df is not None
        else None,
        "execution_residency": "host",
        "posthoc_symmetrization": False,
        "nuclear_response_solves": coordinates,
        "displaced_scf_evaluations": 0,
        "raw_symmetry_error": float(np.max(abs(matrix - matrix.T))),
        "response_residual_norms": tuple(residuals),
        "source_names": operator.plan.source_names,
        "pair_evaluations": operator.xc_kernel.statistics["pair_evaluations"],
        "assembly_seconds": time.perf_counter() - started,
    }
    identity = canonical_hash(
        {
            "plan": operator.plan.identity,
            "state": operator.state.identity.to_payload(),
            "operator": operator.identity,
            "matrix": matrix.tolist(),
        }
    )
    return RKSHessianResult(immutable(matrix), identity, MappingProxyType(diagnostics))

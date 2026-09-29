"""Installed live native LDA/PBE RKS response adapter.

This module binds a current native KS snapshot to the production closed-shell
CPKS equations, semilocal XC response kernel, exact J-only FockPlan provider,
and installed Krylov solver.  It owns no SCF implementation and does not infer
UKS, meta-GGA, hybrid, ECP, or density-fitted response capability.
"""

from __future__ import annotations

import math
import typing
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from time import perf_counter

import numpy as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.features import density_features, spin_densities
from generativeqc_compiler.xc.potential import assemble_coefficients

from ._dft_gradient import StationaryDerivativeContract, StationaryKsState
from .fock import FockBuildSpec, FockPlan, FockTerm
from .ks import resolve_ks_method
from .profiles import canonical_hash
from .response_operator import CPKSResponseOperator, cpks_operator_identity
from .response_problem import ResponseUnsupported
from .response_solver import solve, solve_many
from .response_xc import FixedDensityXCDerivativeKernel

__all__ = [
    "NativeRKSResponse",
    "RKSResponseReference",
]


def _geometry_hash(basis: NativeAO) -> str:
    return canonical_hash([asdict(atom) for atom in basis.atoms])


def _basis_hash(basis: NativeAO) -> str:
    return canonical_hash(
        {
            "shells": [asdict(shell) for shell in basis.shells],
            "representation": basis.representation,
        }
    )


@dataclass(frozen=True, eq=False)
class RKSResponseReference:
    """Immutable closed-shell KS view consumed by the production CPKS core."""

    overlap: np.ndarray
    hcore: np.ndarray
    fock: np.ndarray
    coefficients: np.ndarray
    orbital_energies: np.ndarray
    occupations: np.ndarray
    electron_count: int
    reference_energy: float
    scf_residual: float
    geometry_hash: str
    basis_hash: str
    generation_id: str
    functional_identity: str
    grid_identity: str
    representation: str = "cartesian"
    hf_backend: str = "native-cpu-rks"
    device_id: int | None = None
    hamiltonian_id: str = "conventional-unscreened"
    algorithm: str = "KS"
    precision: str = "float64"
    screening_tolerance: float = 0.0
    converged: bool = True
    validation_tolerance: float = 1e-8
    overlap_threshold: float = 1e-10
    identity: str = field(init=False)
    diagnostics: tuple[tuple[str, float], ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.algorithm != "KS" or self.precision != "float64":
            raise ValueError("RKS response requires a real FP64 KS reference")
        if self.representation not in ("cartesian", "real_spherical"):
            raise ValueError("unknown RKS response AO representation")
        if not self.converged:
            raise ValueError("unconverged KS state cannot enter response")
        for name in (
            "geometry_hash",
            "basis_hash",
            "generation_id",
            "functional_identity",
            "grid_identity",
            "hamiltonian_id",
            "hf_backend",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty identity")
        for name in (
            "reference_energy",
            "scf_residual",
            "screening_tolerance",
            "validation_tolerance",
            "overlap_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or (name != "reference_energy" and value < 0.0):
                raise ValueError(f"invalid {name}")
        if (
            not 0 < self.validation_tolerance <= 1e-6
            or not 0 < self.overlap_threshold <= 1e-6
        ):
            raise ValueError("validation tolerances must be positive and at most 1e-6")

        coefficients = immutable(self.coefficients)
        if coefficients.ndim != 2 or coefficients.shape[0] != coefficients.shape[1]:
            raise ValueError("RKS response requires a square canonical MO frame")
        n = coefficients.shape[0]
        if (
            type(self.electron_count) is not int
            or self.electron_count <= 0
            or self.electron_count % 2
            or self.electron_count >= 2 * n
        ):
            raise ValueError("RKS response requires occupied and virtual orbitals")
        object.__setattr__(self, "coefficients", coefficients)

        for name in ("overlap", "hcore", "fock"):
            value = immutable(getattr(self, name), shape=(n, n))
            if np.max(np.abs(value - value.T)) > self.validation_tolerance:
                raise ValueError(f"{name} is not symmetric")
            object.__setattr__(self, name, value)
        for name in ("orbital_energies", "occupations"):
            object.__setattr__(
                self,
                name,
                immutable(getattr(self, name), shape=(n,)),
            )

        expected = np.zeros(n)
        expected[: self.electron_count // 2] = 2.0
        if not np.array_equal(self.occupations, expected):
            raise ValueError(
                "RKS response occupations must be ordered closed-shell 2/0"
            )
        if np.any(np.diff(self.orbital_energies) < -self.validation_tolerance):
            raise ValueError("canonical RKS orbital energies must be ascending")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                overlap_eigenvalues = np.linalg.eigvalsh(self.overlap)
        except np.linalg.LinAlgError as error:
            raise ValueError("RKS response overlap eigensystem is invalid") from error
        if not np.isfinite(overlap_eigenvalues).all():
            raise ValueError("RKS response overlap eigenvalues must be finite")
        smallest = float(overlap_eigenvalues[0])
        if smallest <= self.overlap_threshold:
            raise ValueError(
                "linearly dependent RKS response AO overlap is unsupported"
            )

        with np.errstate(over="ignore", invalid="ignore"):
            orthogonality = float(
                np.max(np.abs(coefficients.T @ self.overlap @ coefficients - np.eye(n)))
            )
            canonicality = float(
                np.max(
                    np.abs(
                        coefficients.T @ self.fock @ coefficients
                        - np.diag(self.orbital_energies)
                    )
                )
            )
            generalized = float(
                np.max(
                    np.abs(
                        self.fock @ coefficients
                        - (self.overlap @ coefficients) * self.orbital_energies
                    )
                )
            )
        residuals = (orthogonality, canonicality, generalized, self.scf_residual)
        if not all(math.isfinite(value) for value in residuals):
            raise ValueError("RKS response validation produced a nonfinite residual")
        if max(residuals) > self.validation_tolerance:
            raise ValueError(
                "invalid RKS response reference: "
                f"orthogonality={orthogonality}, canonicality={canonicality}, "
                f"eigen_residual={generalized}, scf_residual={self.scf_residual}"
            )
        object.__setattr__(
            self,
            "diagnostics",
            (
                ("orthogonality", orthogonality),
                ("canonicality", canonicality),
                ("eigen_residual", generalized),
                ("overlap_min_eigenvalue", smallest),
            ),
        )

        metadata = {
            name: getattr(self, name)
            for name in (
                "electron_count",
                "reference_energy",
                "scf_residual",
                "geometry_hash",
                "basis_hash",
                "generation_id",
                "functional_identity",
                "grid_identity",
                "representation",
                "hf_backend",
                "device_id",
                "hamiltonian_id",
                "algorithm",
                "precision",
                "screening_tolerance",
                "validation_tolerance",
                "overlap_threshold",
            )
        }
        for name in (
            "overlap",
            "hcore",
            "fock",
            "coefficients",
            "orbital_energies",
            "occupations",
        ):
            metadata[name] = sha256(
                getattr(self, name).astype("<f8", copy=False).tobytes()
            ).hexdigest()
        object.__setattr__(self, "identity", canonical_hash(metadata))

    @property
    def nmo(self) -> int:
        return len(self.orbital_energies)

    @property
    def nocc(self) -> int:
        return self.electron_count // 2


class _RKSIntegralSourceView:
    """Borrowed NativeAO topology for pre-cutover Hessian compatibility.

    This is not a post-HF NativeSource and owns no integral provider. It exists
    only so a stacked intermediate commit can feed the older generated Hessian
    consumers until their NativeAO-owned production migration lands.
    """

    def __init__(self, basis: NativeAO) -> None:
        self.basis = basis
        self.atoms = tuple(basis.atoms)
        self.shells = tuple(basis.shells)
        self.shell_sizes = tuple(
            (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
            for shell in basis.shells
        )
        if sum(self.shell_sizes) != basis.nao:
            raise ValueError("RKS integral topology does not match NativeAO")
        self.nbf = basis.nao
        self.representation = basis.representation
        self.auxiliary_shells: tuple[typing.Any, ...] = ()

    def _check_open(self) -> None:
        if not self.basis._handle:
            raise RuntimeError("RKS response AO basis is closed")


class _PreparedRKSJBackend:
    """Exact J-only response through the installed method-neutral FockPlan."""

    def __init__(
        self,
        state: StationaryKsState,
        basis: NativeAO,
        *,
        device_budget_bytes: int,
    ) -> None:
        if not isinstance(basis, NativeAO):
            raise TypeError("RKS response J backend requires NativeAO")
        if type(device_budget_bytes) is not int or not 0 <= device_budget_bytes < 2**64:
            raise ValueError("RKS response J budget must fit uint64")
        device = state._source.backend
        device_id = 0 if device == "cpu" else int(state._source.metadata[12])
        specification = FockBuildSpec(
            spin="restricted",
            derivative_order=0,
            coulomb=FockTerm(
                present=True,
                coefficient=1.0,
                operator="full_range",
                approximation="exact",
            ),
            exchange=FockTerm(
                present=False,
                coefficient=0.0,
                operator="full_range",
                approximation="exact",
            ),
        )
        self._plan = FockPlan(
            basis,
            specification,
            device=device,
            device_id=device_id,
            screening_tolerance=0.0,
            device_budget_bytes=device_budget_bytes,
        )
        self.basis = basis
        self.geometry_hash = _geometry_hash(basis)
        self.basis_hash = _basis_hash(basis)
        diagnostics = self._plan.diagnostics
        if diagnostics["resolved"]["exchange"]["present"]:
            self._plan.close()
            raise RuntimeError("RKS response J owner unexpectedly enabled exchange")
        if diagnostics["resolved"]["coulomb"]["approximation"] != "exact":
            self._plan.close()
            raise RuntimeError("RKS response J owner changed Coulomb approximation")
        zero = self._plan.evaluate(np.zeros((basis.nao, basis.nao)))
        if zero.coulomb is None or np.max(np.abs(zero.coulomb)) > 1e-14:
            self._plan.close()
            raise RuntimeError("zero-density RKS response Coulomb is not zero")
        self.hcore = immutable(zero.fock)
        self.identity = canonical_hash(
            {
                "schema": "generativeqc.native-rks-j-response/v1",
                "plan": self._plan.identity,
                "execution": self._plan.execution_identity,
                "state_model": state.identity.model_identity,
                "state_basis": state.identity.basis_identity,
                "geometry": self.geometry_hash,
                "basis": self.basis_hash,
            }
        )
        self.host_workspace_bytes = 4 * basis.nao * basis.nao * 8
        self.device_workspace_bytes = int(diagnostics["device_bytes"])
        self.device_resident_bytes = self.device_workspace_bytes
        self.statistics = {
            "actions": 0,
            "seconds": 0.0,
            "peak_bytes": self.device_workspace_bytes,
            "host_input_bytes": 0,
            "host_jk_result_bytes": 0,
        }

    def validate_reference(self, reference: typing.Any) -> typing.Self:
        for name, expected in (
            ("geometry_hash", self.geometry_hash),
            ("basis_hash", self.basis_hash),
            ("representation", self.basis.representation),
            ("hamiltonian_id", "conventional-unscreened"),
            ("algorithm", "KS"),
            ("nmo", self.basis.nao),
        ):
            if getattr(reference, name, None) != expected:
                raise ValueError(f"native RKS J/reference {name} mismatch")
        return self

    def coulomb_exchange(self, density: typing.Any) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(density)
        if (
            value.shape != (self.basis.nao, self.basis.nao)
            or np.iscomplexobj(value)
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"density response must have shape ({self.basis.nao},{self.basis.nao})"
            )
        started = perf_counter()
        result = self._plan.evaluate(value)
        if result.coulomb is None:
            raise RuntimeError("RKS response J owner returned no Coulomb matrix")
        coulomb = immutable(result.coulomb)
        exchange = immutable(np.zeros_like(coulomb))
        self.statistics["actions"] += 1
        self.statistics["seconds"] += perf_counter() - started
        self.statistics["host_input_bytes"] += value.nbytes
        self.statistics["host_jk_result_bytes"] += coulomb.nbytes
        return coulomb, exchange

    def close(self) -> None:
        self._plan.close()


class _NativeRKSXCKernel(FixedDensityXCDerivativeKernel):
    """Bind the common feature-Hessian contraction to the native SCF point model."""

    def __init__(
        self,
        state: StationaryKsState,
        spec: typing.Any,
        basis: NativeAO,
        *,
        tile_points: int,
    ) -> None:
        self.state = state
        super().__init__(
            spec,
            basis,
            state.grid,
            state.density[0],
            tile_points=tile_points,
        )
        self.identity = canonical_hash(
            {
                "kernel": self.identity,
                "point_model": state.identity.regularization_identity,
                "native_state": state.identity.to_payload(),
                "derivative": "unpolarized-cartesian-directional-v1",
            }
        )

    def _response_tile(
        self,
        jets: typing.Any,
        density: typing.Any,
        direction: typing.Any,
        weights: typing.Any,
    ) -> typing.Any:
        ingredients = (
            ("rho", "gradient") if "sigma" in self.spec.ingredients else ("rho",)
        )
        features = density_features(jets, density, ingredients=ingredients)
        delta = density_features(jets, direction, ingredients=ingredients)
        zero = np.zeros((2, jets.shape[1], 3))
        coefficients = self.state._source.evaluate_rks_response_points(
            "sigma" in self.spec.ingredients,
            features["rho"].sum(axis=0),
            features.get("gradient", zero).sum(axis=0),
            delta["rho"].sum(axis=0),
            delta.get("gradient", zero).sum(axis=0),
        )
        if "sigma" not in self.spec.ingredients:
            coefficients.pop("gradient")
        return assemble_coefficients(jets, coefficients, weights)


class _NativeCudaRKSXCKernel(_NativeRKSXCKernel):
    """Execute the same native RKS XC response contract on the snapshot CUDA owner."""

    def __init__(
        self,
        state: StationaryKsState,
        spec: typing.Any,
        basis: NativeAO,
        *,
        tile_points: int,
        device_budget_bytes: int,
    ) -> None:
        super().__init__(state, spec, basis, tile_points=tile_points)
        self._cuda = state._source.prepare_cuda_response(
            tile_points=tile_points,
            budget_bytes=device_budget_bytes,
        )
        self.identity = canonical_hash(
            {"kernel": self.identity, "cuda_owner": self._cuda.identity}
        )

    def apply_spin(self, delta_density: typing.Any) -> typing.Any:
        self.state._source.check_current()
        if not self.basis._handle:
            raise ValueError("native CPKS AO basis is closed")
        direction = spin_densities(delta_density, self.basis.nao)
        if not np.array_equal(direction[0], direction[1]):
            raise ResponseUnsupported(
                "unpolarized response requires equal spin directions"
            )
        direction = direction.sum(axis=0, keepdims=True)
        started = perf_counter()
        result = self._cuda.apply(direction)
        self.statistics["actions"] += 1
        self.statistics["tiles"] += (
            len(self.grid.points) + self.tile_points - 1
        ) // self.tile_points
        self.statistics["seconds"] += perf_counter() - started
        self.statistics["peak_bytes"] = self._cuda.diagnostics["device_bytes"]
        return result

    def close(self) -> None:
        self._cuda.close()

    def validate_reference(self, reference: typing.Any) -> typing.Any:
        self._cuda._ensure_open()
        return super().validate_reference(reference)


class NativeRKSResponse(CPKSResponseOperator):
    """Live all-electron LDA/PBE RKS response with installed scientific owners."""

    @classmethod
    def from_native(
        cls,
        batch: typing.Any,
        basis: typing.Any,
        grid: typing.Any = None,
        *,
        index: int = 0,
        functional: typing.Any = None,
        tile_points: int = 256,
        axis_tile: int = 2,
        device_budget_bytes: int = 128 << 20,
        perturbation_labels: tuple[str, ...] = (),
    ) -> typing.Self:
        """Bind one current native RKS state without rerunning SCF."""
        if not isinstance(basis, NativeAO):
            raise TypeError("native RKS response requires NativeAO")
        if type(axis_tile) is not int or axis_tile < 1:
            raise ValueError("axis_tile must be a positive integer")
        if type(device_budget_bytes) is not int or not 0 < device_budget_bytes < 2**64:
            raise ValueError("native RKS response budget must be a positive uint64")

        state = StationaryKsState.from_native(batch, basis, grid, index=index)
        backend = kernel = None
        try:
            if (
                state.identity.method not in ("lda-rks", "pbe-rks")
                or state.identity.spin != "unpolarized"
                or state._source.hamiltonian != "all-electron"
            ):
                raise ResponseUnsupported(
                    "native CPKS requires all-electron RKS LDA/PBE"
                )
            if state._source.coefficients != (1.0, 1.0, 0.0):
                raise ResponseUnsupported("native CPKS requires unscaled LDA/PBE RKS")
            _, expected = resolve_ks_method(state.identity.method)
            spec = expected if functional is None else functional
            if spec.identity != state.identity.functional_identity:
                raise ValueError("native CPKS functional identity mismatch")
            if basis.identity != state.identity.basis_identity:
                raise ValueError("native CPKS basis identity mismatch")

            if state._source.backend == "cuda":
                kernel = _NativeCudaRKSXCKernel(
                    state,
                    spec,
                    basis,
                    tile_points=tile_points,
                    device_budget_bytes=device_budget_bytes,
                )
                remaining = (
                    device_budget_bytes - kernel._cuda.diagnostics["device_bytes"]
                )
                if remaining <= 0:
                    raise MemoryError("native CPKS budget cannot hold Coulomb after XC")
                backend = _PreparedRKSJBackend(
                    state,
                    basis,
                    device_budget_bytes=remaining,
                )
            else:
                backend = _PreparedRKSJBackend(
                    state,
                    basis,
                    device_budget_bytes=0,
                )
                kernel = _NativeRKSXCKernel(
                    state,
                    spec,
                    basis,
                    tile_points=tile_points,
                )

            occupations = np.asarray(state.occupations[0])
            electrons = float(np.sum(occupations))
            if (
                not math.isfinite(electrons)
                or electrons <= 0
                or not electrons.is_integer()
            ):
                raise ValueError("native RKS response has invalid electron count")
            reference = RKSResponseReference(
                overlap=state.overlap,
                hcore=backend.hcore,
                fock=state.fock[0],
                coefficients=state.coefficients[0],
                orbital_energies=state.orbital_energies[0],
                occupations=state.occupations[0],
                electron_count=int(electrons),
                reference_energy=state._source.energy(),
                scf_residual=state.physical_residual,
                geometry_hash=_geometry_hash(basis),
                basis_hash=_basis_hash(basis),
                generation_id=canonical_hash(state.identity.to_payload()),
                functional_identity=spec.identity,
                grid_identity=state.grid.identity,
                representation=basis.representation,
                hf_backend=f"native-{state._source.backend}-rks",
            )
            problem = cls.build_problem(
                reference,
                backend,
                kernel,
                perturbation_labels=perturbation_labels,
            )
            result = cls(problem, backend, kernel)
            result.state = state
            result._basis = basis
            result._source = _RKSIntegralSourceView(basis)
            result._reference_identity = reference.identity
            result._backend_identity = backend.identity
            result._contract = StationaryDerivativeContract(state.identity)
            result.validate_current()
            return result
        except Exception:
            if kernel is not None and hasattr(kernel, "close"):
                kernel.close()
            if backend is not None:
                backend.close()
            state._source.close()
            raise

    def validate_current(self) -> None:
        """Reject a stale SCF state, basis, J owner, XC owner or model identity."""
        self._contract.validate(self.state)
        if not self._basis._handle:
            raise ValueError("native CPKS AO basis is closed")
        if (
            self.problem.reference.identity != self._reference_identity
            or self.backend.identity != self._backend_identity
            or self.backend.basis is not self._basis
            or self.xc_kernel.state is not self.state
            or self.xc_kernel.spec.identity != self.state.identity.functional_identity
            or self.xc_kernel.grid.identity != self.state.identity.grid_identity
            or self.xc_kernel.basis.identity != self.state.identity.basis_identity
            or self.problem.operator_identity
            != cpks_operator_identity(self.backend, self.xc_kernel)
        ):
            raise ValueError("native CPKS state/provider/kernel identity mismatch")
        self.backend.validate_reference(self.problem.reference)
        self.xc_kernel.validate_reference(self.problem.reference)

    @property
    def diagnostics(self) -> dict[str, typing.Any]:
        self.validate_current()
        cuda = self.state._source.backend == "cuda"
        xc = self.xc_kernel._cuda.diagnostics if cuda else None
        return {
            "execution": "cuda" if cuda else "cpu",
            "coulomb_execution": "cuda" if cuda else "cpu",
            "xc_ao_features_points_assembly": "cuda" if cuda else "cpu",
            "orbital_transforms": "host",
            "krylov_execution": "host",
            "owned_device_bytes": (
                self.backend.device_resident_bytes + xc["device_bytes"]
            )
            if cuda
            else 0,
            "xc": xc,
            "snapshot_export": dict(self.state._source.export_work),
            "memory_scope": (
                "retained response J/XC allocations only; excludes borrowed SCF, "
                "preparation temporaries, host arrays/Krylov, context and "
                "library-private memory"
            ),
        }

    def solve(
        self,
        rhs: typing.Any,
        *,
        options: typing.Any = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Run the installed shared scalar GMRES on this live response owner."""
        return solve(self, rhs, options=options, **kwargs)

    def solve_many(
        self,
        rhs: typing.Any,
        *,
        strategy: str = "recycled",
        options: typing.Any = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Run the installed shared multi-RHS GMRES on this live response owner."""
        return solve_many(
            self,
            rhs,
            strategy=strategy,
            options=options,
            **kwargs,
        )

    def induced_fock(
        self,
        delta_density: typing.Any,
        *,
        transpose: bool = False,
    ) -> typing.Any:
        self.validate_current()
        result = super().induced_fock(delta_density, transpose=transpose)
        self.validate_current()
        return result

    def _base_action(
        self,
        vector: typing.Any,
        *,
        transpose: bool = False,
    ) -> typing.Any:
        self.validate_current()
        result = super()._base_action(vector, transpose=transpose)
        self.validate_current()
        return result

    def close(self) -> None:
        for owner in (
            getattr(self, "xc_kernel", None),
            getattr(self, "backend", None),
        ):
            if owner is not None and hasattr(owner, "close"):
                owner.close()
        if hasattr(self, "state"):
            self.state._source.close()

    def __enter__(self) -> typing.Self:
        self.validate_current()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "state") or hasattr(self, "backend"):
            self.close()

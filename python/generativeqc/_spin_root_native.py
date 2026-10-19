"""Native CPU UHF/UKS root model over existing SCF, J/K, XC and response owners."""

from __future__ import annotations

import ctypes as ct
import math
import typing
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.dft.features import density_features
from generativeqc_compiler.xc.potential import assemble_coefficients
from scipy.linalg import eigh

from . import _native
from ._api_types import Result
from ._dft_gradient import StationaryDerivativeContract, StationaryKsState
from .checkpoint import _controls, _descriptor, _pointer
from .fock import FockBuildSpec, FockPlan, FockTerm
from .overlap import cross_overlap
from .profiles import canonical_hash
from .rks_response import _NativeRKSXCKernel
from .spin_roots import (
    SpinRootPolicy,
    SpinRootReference,
    SpinRootUnqualified,
    _has_uhf_reference,
)
from .spin_stationary import SpinCPKSOperator

if typing.TYPE_CHECKING:
    from generativeqc_compiler.dft import NativeAO


class _SpinKernel:
    """Preserve two physical spin outputs of the common point-response adapter."""

    def __init__(self, kernel: typing.Any = None) -> None:
        self.kernel = kernel
        self.identity = "no-xc-spin-response-v1" if kernel is None else kernel.identity
        self.functional_identity = (
            "none" if kernel is None else kernel.functional_identity
        )
        self.grid_identity = "none" if kernel is None else kernel.grid_identity

    def apply(self, density: np.ndarray) -> np.ndarray:
        return (
            np.zeros_like(density)
            if self.kernel is None
            else self.kernel.apply_spin(density)
        )


class _Interaction:
    """The method-neutral native fixed-density Fock plan owns J/K algebra."""

    def __init__(self, model: typing.Any) -> None:
        self.model = model
        self.identity = canonical_hash(
            {"plan": model.plan.identity, "execution": model.plan.execution_identity}
        )

    def coulomb_exchange(self, density: np.ndarray) -> tuple[np.ndarray, None]:
        value = self.model.plan.evaluate(density)
        return value.fock - self.model.hcore[None], None


class _BoundSpinOperator(SpinCPKSOperator):
    """Bind existing coupled OV equations to the complete current native model."""

    def __init__(self, model: typing.Any) -> None:
        self.model = model
        super().__init__(model.state, model.basis, _Interaction(model), model.kernel)

    def validate_current(self) -> None:
        self.model.validate_current()


def _read_density(batch: typing.Any, nbf: int) -> np.ndarray:
    state = _descriptor()
    _native.check(
        batch._library,
        batch._library.generativeqc_batch_get_hf_warm_state(
            batch._batch, 0, ct.byref(state)
        ),
    )
    if not state.present or state.density_count != 2 * nbf * nbf:
        raise SpinRootUnqualified("native result has no two-spin occupied density")
    density = np.empty((2, nbf, nbf))
    coordinates = np.empty(state.coordinate_count)
    state.density, state.coordinates = _pointer(density), _pointer(coordinates)
    _native.check(
        batch._library,
        batch._library.generativeqc_batch_get_hf_warm_state(
            batch._batch, 0, ct.byref(state)
        ),
    )
    return immutable(density)


def _install_seed(
    batch: typing.Any, basis: NativeAO, density: np.ndarray, energy: float
) -> None:
    """Use the existing atomic seed import, with exact target model/topology."""
    if batch.system_count != 1 or not math.isfinite(energy):
        raise SpinRootUnqualified("one finite native root seed is required")
    value = np.ascontiguousarray(density, dtype=np.float64)
    coordinates = np.ascontiguousarray(
        [a.position for a in basis.atoms], dtype=np.float64
    )
    if value.shape != (2, basis.nao, basis.nao) or not np.isfinite(value).all():
        raise SpinRootUnqualified("invalid native root seed")
    states = (_native.HfWarmState * 1)(_descriptor())
    state = states[0]
    state.density, state.density_count = _pointer(value), value.size
    state.coordinates, state.coordinate_count = _pointer(coordinates), coordinates.size
    state.energy, state.energy_change, state.density_rms = energy, 0.0, 0.0
    state.iterations, state.present = 0, 1
    # This low-level transaction validates source-metric trace, spin and
    # projector/Hermiticity before mutation. Its metadata is a seed proposal,
    # never a claim of target SCF convergence or stability.
    _native.check(
        batch._library,
        batch._library.generativeqc_batch_restore_hf_warm_states(
            batch._batch, states, 1
        ),
    )
    batch._warm_state.metadata[0] = {
        "controls": _controls(batch._calculator),
        "backend": "cpu",
    }
    batch._warm_state.restart_indices.add(0)


class _NativeSpinModel:
    """One independently live native SCF result and fixed-density response model."""

    def __init__(
        self,
        calculator: typing.Any,
        basis: NativeAO,
        model_identity: str,
        *,
        charge: int,
        multiplicity: int,
        properties: tuple[str, ...],
        policy: SpinRootPolicy,
        seed: np.ndarray | None = None,
        seed_energy: float | None = None,
    ) -> None:
        self.calculator, self.basis = calculator, basis
        self.batch: typing.Any = None
        self.plan: typing.Any = None
        self._snapshot: typing.Any = None
        self.state: typing.Any = None
        self.item: typing.Any = None
        self._closed = False
        self._signature = calculator._model_signature()
        n = basis.nao
        if n > policy.maximum_ao:
            raise SpinRootUnqualified("native root AO domain exceeds policy")
        try:
            self.batch = calculator.prepare_batch(
                [basis.atoms],
                charges=[charge],
                multiplicities=[multiplicity],
                warm_start=True,
            )
            if seed is not None:
                if seed_energy is None:
                    raise SpinRootUnqualified(
                        "native root seed lacks evaluated source energy"
                    )
                _install_seed(self.batch, basis, seed, seed_energy)
            self.item = self.batch.execute(strict=True, properties=properties).items[0]
            if not self.item.succeeded or not self.item.converged:
                raise SpinRootUnqualified("native root SCF did not converge")
            if seed is not None and (
                not self.item.warm_start_used or self.item.warm_start_fallback
            ):
                raise SpinRootUnqualified(
                    "native SCF did not consume the requested spin seed"
                )
            hf = _has_uhf_reference(calculator._method)
            spec = (
                FockBuildSpec.hf("unrestricted", derivative_order=0)
                if hf
                else FockBuildSpec(
                    spin="unrestricted",
                    exchange=FockTerm(present=False, coefficient=0.0),
                )
            )
            self.plan = FockPlan(basis, spec, device="cpu", screening_tolerance=0.0)
            zero = self.plan.evaluate(np.zeros((2, n, n)))
            self.hcore = immutable(zero.fock[0])
            overlap = cross_overlap(
                calculator,
                calculator,
                basis.atoms,
                charge=charge,
                multiplicity=multiplicity,
                maximum_bytes=policy.memory_budget_bytes,
            )
            if hf:
                density = _read_density(self.batch, n)
                physical = self.plan.evaluate(density)
                counts = (
                    (
                        sum(a.atomic_number for a in basis.atoms)
                        - charge
                        + multiplicity
                        - 1
                    )
                    // 2,
                    (
                        sum(a.atomic_number for a in basis.atoms)
                        - charge
                        - multiplicity
                        + 1
                    )
                    // 2,
                )
                frames = [
                    eigh(fock, overlap, check_finite=True) for fock in physical.fock
                ]
                coefficients = np.asarray([item[1] for item in frames])
                energies = np.asarray([item[0] for item in frames])
                occupations = np.asarray(
                    [np.r_[np.ones(count), np.zeros(n - count)] for count in counts]
                )
                identity = SimpleNamespace(
                    spin="polarized",
                    to_payload=lambda: {
                        "native_uhf": self.plan.execution_identity,
                        "density": self.item.energy,
                    },
                )
                self.state = SimpleNamespace(
                    identity=identity,
                    density=density,
                    fock=physical.fock,
                    coefficients=coefficients,
                    orbital_energies=energies,
                    occupations=occupations,
                )
                self.kernel = _SpinKernel()
                source_identity = canonical_hash(
                    {
                        "plan": self.plan.identity,
                        "execution": self.plan.execution_identity,
                        "geometry": [asdict(a) for a in basis.atoms],
                        "density": density.tolist(),
                    }
                )
                residual = self.item.density_rms
            else:
                self.state = StationaryKsState.from_native(self.batch, basis)
                self._snapshot = self.state._source
                if (
                    self.state.identity.spin != "polarized"
                    or self._snapshot.backend != "cpu"
                    or self._snapshot.coefficients != (1.0, 1.0, 0.0)
                    or self.state.identity.ingredients
                    not in (("rho",), ("rho", "sigma"))
                ):
                    raise SpinRootUnqualified(
                        "native UKS stability requires qualified unscaled CPU LDA/GGA point response"
                    )
                self.contract = StationaryDerivativeContract(self.state.identity)
                self.kernel = _SpinKernel(
                    _NativeRKSXCKernel(
                        self.state, self._snapshot.functional, basis, tile_points=128
                    )
                )
                source_identity = canonical_hash(
                    {
                        "snapshot": self.state.identity.to_payload(),
                        "plan": self.plan.execution_identity,
                    }
                )
                residual = self.state.physical_residual
            reference_energy = (
                self.item.energy
                if self.item.correlation is None
                else self.item.correlation.reference_energy
            )
            if self.item.correlation is not None:
                residual = max(residual, self.item.correlation.reference_residual)
            self.reference = SpinRootReference(
                basis.atoms,
                immutable(overlap),
                immutable(self.state.density),
                immutable(self.state.coefficients),
                immutable(self.state.orbital_energies),
                immutable(self.state.occupations),
                immutable(self.state.fock),
                float(reference_energy),
                float(residual),
                model_identity,
                source_identity,
            )
            self.operator = _BoundSpinOperator(self)
            grid_bytes = (
                0
                if hf
                else sum(
                    np.asarray(value).nbytes
                    for value in (
                        self.state.grid.points,
                        self.state.grid.weights,
                        self.state.grid.owners,
                    )
                )
            )
            self.workspace_bytes = 8 * (64 * n * n + 128 * (16 * n + 128)) + grid_bytes
            self.validate_current()
        except Exception:
            self.close()
            raise

    def validate_current(self) -> None:
        if self._closed or not self.basis._handle:
            raise SpinRootUnqualified("native spin-root model is closed")
        self.batch._ensure_open()
        self.plan._ensure_open()
        if self.calculator._model_signature() != self._signature:
            raise SpinRootUnqualified(
                "calculator model changed during root qualification"
            )
        if self._snapshot is not None:
            self.contract.validate(self.state)
            self._snapshot.check_current()

    def evaluate(self, density: np.ndarray) -> typing.Any:
        """Complete nonlinear Fock/energy; never insert a linear XC Hessian here."""
        self.validate_current()
        native = self.plan.evaluate(density)
        fock = np.array(native.fock, copy=True)
        energy = float(native.energy)
        if self._snapshot is not None:
            spec, grid = self._snapshot.functional, self.state.grid
            ingredients = (
                ("rho", "gradient") if "sigma" in spec.ingredients else ("rho",)
            )
            for begin in range(0, len(grid.points), 128):
                end = min(begin + 128, len(grid.points))
                jets = self.basis.evaluate(
                    grid.points[begin:end], 1 if "sigma" in spec.ingredients else 0
                )
                # Feature/assembly owners require the common first-jet ABI.
                if jets.shape[0] == 1:
                    jets = np.concatenate(
                        (jets, np.zeros((3, end - begin, self.basis.nao)))
                    )
                features = density_features(jets, density, ingredients=ingredients)
                gradient = features.get("gradient", np.zeros((2, end - begin, 3)))
                point = self._snapshot.evaluate_xc_points(
                    spec, features["rho"], gradient
                )
                coefficients = {
                    name: point[name]
                    for name in ("rho", "gradient")
                    if name == "rho" or "sigma" in spec.ingredients
                }
                fock += assemble_coefficients(
                    jets, coefficients, grid.weights[begin:end]
                )
                energy += float(np.dot(point["energy"], grid.weights[begin:end]))
        self.validate_current()
        return SimpleNamespace(energy=energy, fock=immutable(fock))

    def result(self) -> Result:
        """Detach only the chosen native output after all root gates passed."""
        self.validate_current()
        item = self.item
        return Result(
            energy=item.energy,
            forces=None if item.forces is None else immutable(item.forces),
            converged=item.converged,
            iterations=item.iterations,
            energy_change=item.energy_change,
            density_rms=item.density_rms,
            executed_backend=item.executed_backend,
            basis_metadata=item.basis_metadata,
            accuracy=item.accuracy,
            correlation=item.correlation,
            cc_performance=item.cc_performance,
            resource_diagnostics=self.batch.resource_diagnostics,
            precision=item.precision,
            physical_residual_rms=item.physical_residual_rms,
            ks_diagnostic=item.ks_diagnostic,
            initial_guess=item.initial_guess,
            ks_transport_diagnostic=self.batch.ks_transport_diagnostics[0],
        )

    def close(self) -> None:
        self._closed = True
        for owner in (self._snapshot, self.plan, self.batch):
            if owner is not None:
                owner.close()

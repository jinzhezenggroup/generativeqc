"""Fully device-numerical WB97M-V Hessians with explicit resident resource gates.

The native converged state is uploaded once. Generated integral/AO/scalar AD,
TensorIR cotangents, moving grids, DF, ECP, coupled CPKS and D/W reconstruction
then run on one CUDA stream. Only control metadata and completed public results
are downloaded. Host shell enumeration and fixed quadrature construction do not
evaluate any molecular derivative. A missing capability fails, never falls back.
"""

from __future__ import annotations

import time
import typing
from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
from generativeqc_compiler.common.array_execution import array_execution
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.method import StationaryHVPPlan, StationaryMeanField
from generativeqc_compiler.method.stationary_gradient import SCF_POINT_MODEL
from generativeqc_compiler.method.stationary_grid_second import nuclear_repulsion_hvp
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.xc.grid_response import partition_response
from generativeqc_compiler.xc.potential import assemble_coefficients

from ._cuda_hessian_execution import CudaAO, CudaHessianExecution
from ._cuda_hessian_integrals import CudaIntegralSources
from ._cuda_hessian_response import CudaSpinResponse
from .composite_rks_hessian import (
    CompositeRKSHVPResult,
    _ao_coefficients,
    _CompositeGrid,
)
from .df_hessian import DFSecondOrder
from .ecp_hessian import ECPSecondOrder
from .hessian_basis import HessianBasis
from .profiles import canonical_hash
from .response_solver import GMRESOptions
from .rks_hessian import RKSHessianResult
from .rks_hessian_directional import _validate_partition_provenance
from .rks_hessian_integrals import RKSIntegralTopology, checked_direction


class _CudaGrid(_CompositeGrid):
    """Reuse the authoritative grid AD model with a resident AO/array owner."""

    def partition(self, vector: typing.Any) -> typing.Any:
        result = partition_response(
            self.grid.points,
            self.centers,
            point_motion=vector[self.owners],
            center_motion=vector,
            iterations=self.grid_spec.partition_iterations,
            coincident_tolerance=self.grid_spec.coincident_tolerance,
        )
        selected = (self.execution.cp.arange(len(self.owners)), self.owners)
        _validate_partition_provenance(
            self.atomic_weights, self.grid.weights, result.weights[selected]
        )
        return result


class _CudaDF(DFSecondOrder):
    """The CPU and CUDA consumers share all fitted stationary TensorIR graphs."""

    def _stream(self, family: str, direction: typing.Any) -> typing.Iterator[tuple]:
        return self.sources.stream_df(
            family, direction, self.orbital.cartesian, self.auxiliary_map.cartesian
        )


class CudaCompositeResponse:
    """Own a complete resident RKS/UKS Hessian, including spherical/DF/ECP.

    The bounded dense primal ERI option is intentional, not a scaling claim.
    Derivatives stream shell tiles. Insufficient resident storage raises before
    publication; it never selects the old mixed host/device implementation.
    """

    cuda_resident = True

    def __init__(
        self,
        state: typing.Any,
        basis: typing.Any,
        batch: typing.Any,
        *,
        grid_budget_bytes: int,
        max_nonlocal_pairs: int,
    ) -> None:
        self.state, self.basis = state, basis
        self.grid_budget_bytes, self.max_nonlocal_pairs = (
            grid_budget_bytes,
            max_nonlocal_pairs,
        )
        self.execution = CudaHessianExecution(
            batch._stationary_cuda_compiler(), device_id=batch._calculator._device_id
        )
        self.integral_basis = None
        self.auxiliary = None
        self.df = None
        self.ready = False
        self.closed = False
        self.integral_basis = HessianBasis(basis)
        try:
            self.topology = RKSIntegralTopology.from_basis(
                self.integral_basis.cartesian
            )
            self.has_ecp = state._source.hamiltonian == "scalar-semilocal-ecp"
            if self.has_ecp:
                self.topology = replace(
                    self.topology,
                    atoms=tuple(
                        replace(atom, atomic_number=atom.atomic_number - core)
                        for atom, core in zip(
                            self.topology.atoms, state._source.ecp_cores, strict=True
                        )
                    ),
                )
            coulomb, exchange, self.metric_threshold = (
                state._source.fock_provider_proof()
            )
            self.fitted = coulomb == "density-fitted"
            if (coulomb, exchange) not in (
                ("exact", "exact"),
                ("density-fitted", "density-fitted"),
            ):
                raise NotImplementedError(
                    "resident Hessian has no rule for this J/K approximation"
                )
            if self.fitted:
                calculator = batch._calculator
                auxiliary_spec = calculator._auxiliary_basis
                shells = (
                    basis.shells
                    if auxiliary_spec is None
                    else calculator._shells_for_atoms(
                        basis.atoms, auxiliary_spec, operator="df_metric"
                    )
                )
                self.auxiliary = NativeAO(
                    basis.atoms,
                    shells,
                    representation=getattr(
                        auxiliary_spec, "representation", basis.representation
                    ),
                    charge=basis.charge,
                    multiplicity=basis.multiplicity,
                )
            self.plan = StationaryHVPPlan(
                state._source.method_ir,
                StationaryMeanField(
                    SCF_POINT_MODEL,
                    hamiltonian=state._source.hamiltonian,
                    coulomb="density-fitted-full-rank"
                    if self.fitted
                    else "direct-full-range",
                ),
            )
            self.identity = canonical_hash(
                {
                    "schema": "generativeqc.cuda-composite-hessian/v1",
                    "state": state.identity.to_payload(),
                    "plan": self.plan.identity,
                }
            )
        except Exception:
            self.close()
            raise

    @contextmanager
    def scope(self) -> typing.Iterator[None]:
        """Bind private storage, device, stream and AD execution as one scope."""
        with (
            self.execution.device,
            self.execution.stream,
            self.execution.cp.cuda.using_allocator(self.execution.pool.malloc),
            array_execution(self.execution),
        ):
            yield

    def validate_current(self) -> None:
        if self.closed or not self.basis._handle:
            raise ValueError("resident Hessian owner is closed")
        self.state._source.check_current()
        self.topology.check_current()

    def _density_cart(self, value: typing.Any) -> typing.Any:
        return (
            value
            if self.transform is None
            else self.transform @ value @ self.transform.T
        )

    def _matrix_public(self, value: typing.Any) -> typing.Any:
        return (
            value
            if self.transform is None
            else self.transform.T @ value @ self.transform
        )

    def initialize(self, cache: typing.Any, budget_bytes: int) -> None:
        """Admit all retained sources, then verify the reconstructed SCF Fock.

        Dense ERI reuse is optional. The full-rank DF reserve is protected before
        choosing it, so a smaller allowance can select device streaming without
        changing the electronic model or consuming DF response workspace.
        """
        cp = self.execution.cp
        count = self.topology.nbf
        minimum = 8 * (64 * count**2 + 16384)
        if type(budget_bytes) is not int or not minimum <= budget_bytes < 2**63:
            raise ValueError(
                "resident integral sources/cotangents exceed integral_budget_bytes"
            )
        if self.ready:
            if budget_bytes < self.integral_reserve:
                raise ValueError(
                    "retained resident integral sources exceed integral_budget_bytes"
                )
            if self.df is not None:
                self.df.check_budget(budget_bytes - self.integral_reserve)
            return
        fitting_reserve = 0
        if self.fitted:
            auxiliary_count = sum(
                (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
                for shell in self.auxiliary.shells
            )
            fitting_reserve = 8 * (
                64 * count**2 * auxiliary_count + 80 * auxiliary_count**2 + 4096
            )
        dense_reserve = 8 * count**4 * (6 if self.fitted else 12)
        self.resident_eri = dense_reserve <= budget_bytes - minimum - fitting_reserve
        self.integral_reserve = minimum + (dense_reserve if self.resident_eri else 0)
        self.execution.initialize(cache)
        self.device_state = SimpleNamespace(
            **{
                name: getattr(self.state, name)
                for name in ("identity", "grid", "_source")
            },
            **{
                name: cp.asarray(getattr(self.state, name))
                for name in (
                    "density",
                    "coefficients",
                    "orbital_energies",
                    "occupations",
                    "fock",
                    "overlap",
                )
            },
        )
        self.device_state.weighted_density = cp.einsum(
            "smi,si,sni->smn",
            self.device_state.coefficients,
            self.device_state.orbital_energies * self.device_state.occupations,
            self.device_state.coefficients,
        )
        self.transform = (
            None
            if self.integral_basis.transform is None
            else cp.asarray(self.integral_basis.transform)
        )
        self.sources = CudaIntegralSources(self.topology, self.execution)
        self.ao = CudaAO(self.basis, self.execution)
        self.grid = _CudaGrid.__new__(_CudaGrid)
        self.grid.execution = self.execution
        _CudaGrid.__init__(
            self.grid,
            self.device_state,
            self.ao,
            grid_budget_bytes=self.grid_budget_bytes,
            max_nonlocal_pairs=self.max_nonlocal_pairs,
        )
        persistent_grid = self.grid.jets.nbytes + 7 * self.grid.fields.nbytes
        self.ecp = (
            ECPSecondOrder(
                self.device_state,
                self.ao,
                budget_bytes=self.grid_budget_bytes - persistent_grid,
            )
            if self.has_ecp
            else None
        )
        zero = cp.zeros((self.basis.natom, 3))
        self.zero_direction = zero
        self.omega = next(
            float(primitive.omega)
            for primitive in self.plan.method.primitives
            if primitive.kind == "range_separated_exchange"
        )
        self.short_exchange = float(
            next(
                primitive.coefficient
                for primitive in self.plan.method.primitives
                if primitive.kind == "range_separated_exchange"
                and primitive.operator == "short-range"
            )
        )
        self.long_exchange = float(
            next(
                primitive.coefficient
                for primitive in self.plan.method.primitives
                if primitive.kind == "range_separated_exchange"
                and primitive.operator == "long-range"
            )
        )
        if self.fitted:
            with array_execution(None):
                self.df = _CudaDF(
                    self.state,
                    self.integral_basis,
                    self.auxiliary,
                    self.metric_threshold,
                    budget_bytes=budget_bytes - self.integral_reserve,
                )
            self.df.state, self.df.sources = self.device_state, self.sources
            for mapping in (self.df.orbital, self.df.auxiliary_map):
                if mapping.transform is not None:
                    mapping.transform = cp.asarray(mapping.transform)
            self.df.prepare(cache)
            self.eri_full = None
        elif self.resident_eri:
            self.eri_full = self.sources.tensor("eri", zero)
        else:
            self.eri_full = None
        self.eri_long = (
            self.sources.tensor("eri", zero, radial="long_range", omega=self.omega)
            if self.resident_eri
            else None
        )
        hcore = self.sources.tensor("kinetic", zero) + self.sources.tensor(
            "nuclear_attraction", zero
        )
        hcore = self._matrix_public(hcore) + (
            0 if self.ecp is None else self.ecp.matrix
        )
        grid_fock = assemble_coefficients(
            self.grid.jets[:4],
            _ao_coefficients(self.grid.reference.coefficients),
            cp.ones(len(self.grid.grid.points)),
        )
        expected = hcore + self.jk(self.device_state.density) + grid_fock
        if float(cp.max(abs(expected - self.device_state.fock))) > 2e-8:
            raise ValueError("resident compiler-AD/native SCF Fock provenance mismatch")
        self.response = CudaSpinResponse(self)
        self.execution.check()
        self.ready = True

    def jk(self, density: typing.Any) -> typing.Any:
        """Apply the declared native approximation, not range-separated RI-K.

        Fitting uses a_SR times fitted full exchange plus (a_LR-a_SR) times
        exact LR exchange. Physical spin occupations fix the exchange factor.
        """
        cp = self.execution.cp
        cart = self._density_cart(density)
        occupation = 1 if len(density) == 2 else 2
        long_exchange = (
            cp.einsum("ikjl,skl->sij", self.eri_long, cart)
            if self.resident_eri
            else self.sources.jk(
                cart, self.zero_direction, radial="long_range", omega=self.omega
            )[1]
        )
        if self.fitted:
            fitted = execute(
                self.df.programs[4],
                {
                    "density": density,
                    "three_center": self.df.three_center,
                    "inverse": self.df.inverse,
                    "bar_energy": cp.asarray(1.0),
                },
                max_bytes=self.df.ad_budget_bytes,
            ).outputs["bar_density"]
            return fitted - (
                self.long_exchange - self.short_exchange
            ) / occupation * self._matrix_public(long_exchange)
        if self.resident_eri:
            coulomb = cp.einsum("ijkl,kl->ij", self.eri_full, cart.sum(axis=0))
            exchange = cp.einsum("ikjl,skl->sij", self.eri_full, cart)
        else:
            coulomb, exchange = self.sources.jk(cart, self.zero_direction)
        exchange = exchange * self.short_exchange
        exchange += long_exchange * (self.long_exchange - self.short_exchange)
        return self._matrix_public(coulomb[None] - exchange / occupation)

    def induced_fock(self, density: typing.Any) -> typing.Any:
        grid = self.grid.apply(density)
        return self.jk(density) + (grid if grid.ndim == 3 else grid[None])

    def frozen(self, vector: typing.Any) -> tuple:
        """Build the nuclear Fock/overlap RHS with streamed directional sources."""
        overlap = self._matrix_public(
            self.sources.tensor("overlap", vector, tangent=True)
        )
        one = self._matrix_public(
            self.sources.tensor("kinetic", vector, tangent=True)
            + self.sources.tensor("nuclear_attraction", vector, tangent=True)
        )
        density = self._density_cart(self.device_state.density)
        occupation = 1 if len(density) == 2 else 2
        if self.fitted:
            fitted = self.df.frozen_fock(vector)
            interaction = fitted if fitted.ndim == 3 else fitted[None]
        else:
            coulomb, exchange = self.sources.jk(density, vector, tangent=True)
            interaction = self._matrix_public(
                coulomb[None] - self.short_exchange / occupation * exchange
            )
        long_exchange = self.sources.jk(
            density, vector, tangent=True, radial="long_range", omega=self.omega
        )[1]
        interaction -= (
            (self.long_exchange - self.short_exchange)
            / occupation
            * self._matrix_public(long_exchange)
        )
        grid = self.grid.frozen_fock(vector)
        return (
            one
            + interaction
            + (grid if grid.ndim == 3 else grid[None])
            + (0 if self.ecp is None else self.ecp.frozen_fock(vector)),
            overlap,
        )

    def weights(
        self, source: str, response: typing.Any, coordinates: tuple | None = None
    ) -> tuple:
        """Evaluate compiler stationary cotangents/JVPs on one shell component tile."""
        density, moving = (
            response.cartesian_density,
            response.cartesian_density_derivative,
        )
        count = self.topology.nbf
        if source in ("one_electron", "overlap_pulay"):
            if source == "overlap_pulay":
                feeds = {
                    "weighted_density": self._density_cart(
                        self.device_state.weighted_density
                    ).reshape(len(density), -1),
                    "d_weighted_density": self._density_cart(
                        response.energy_weighted_density_derivative
                    ).reshape(len(density), -1),
                }
            else:
                feeds = {
                    "density_left": density.reshape(len(density), -1),
                    "d_density_left": moving.reshape(len(density), -1),
                }
            shape = (count, count)
        else:
            first, second, third, fourth = coordinates
            if source.startswith("exchange_"):
                second, third = third, second
            feeds = {
                "density_left": density[:, first, second],
                "density_right": density[:, third, fourth],
                "d_density_left": moving[:, first, second],
                "d_density_right": moving[:, third, fourth],
            }
            shape = (len(first),)
        block = self.plan.integral_block(
            source, terms=int(np.prod(shape)), coordinates=1
        )
        return (
            execute(block.weights, feeds).outputs["weights"].reshape(shape),
            execute(block.response_weights, feeds)
            .outputs["response_weights"]
            .reshape(shape),
        )

    def _apply(self, vector: typing.Any, options: GMRESOptions) -> tuple:
        cp = self.execution.cp
        response = self.response.solve(*self.frozen(vector), options)
        response.cartesian_density = self._density_cart(self.device_state.density)
        response.cartesian_density_derivative = self._density_cart(
            response.density_derivative
        )
        components = self.grid.hvp(vector, response.density_derivative)
        if self.ecp is not None:
            components.update(
                self.ecp.hvp(
                    vector,
                    self.device_state.density.sum(axis=0),
                    response.density_derivative.sum(axis=0),
                )
            )
        fitted = (
            self.df.hvp(vector, response.density_derivative)
            if self.df is not None
            else None
        )
        for source, family in (
            ("one_electron", "kinetic"),
            ("overlap_pulay", "overlap"),
        ):
            fixed, moving = self.weights(source, response)
            value = self.sources.contract(family, vector, fixed, moving)
            if source == "one_electron":
                value += self.sources.contract(
                    "nuclear_attraction", vector, fixed, moving
                )
            components[source] = value
        if fitted is None:
            components["coulomb"] = self.sources.contract(
                "eri",
                vector,
                lambda coordinates: self.weights("coulomb", response, coordinates),
            )
        else:
            components["coulomb"] = fitted["coulomb"]
        for source, radial in (
            ("exchange_short_range", "short_range"),
            ("exchange_long_range", "long_range"),
        ):
            value = self.sources.contract(
                "eri",
                vector,
                lambda coordinates, source=source: self.weights(
                    source, response, coordinates
                ),
                radial="long_range" if fitted is not None else radial,
                omega=self.omega,
            )
            components[source] = (
                fitted["exchange"] - value
                if fitted is not None and source == "exchange_short_range"
                else value
            )
        components["nuclear"] = nuclear_repulsion_hvp(
            cp.asarray([atom.position for atom in self.topology.atoms]),
            cp.asarray([atom.atomic_number for atom in self.topology.atoms]),
            vector,
        )
        if set(components) != set(self.plan.source_names):
            raise RuntimeError("resident Hessian source coverage mismatch")
        value = sum(components.values())
        self.execution.check()
        return value, components, response

    def _work_snapshot(self) -> dict:
        return {
            **self.execution.statistics,
            **(self.sources.statistics if self.ready else {}),
            "pair_evaluations": self.grid.statistics["pair_evaluations"]
            if self.ready
            else 0,
        }

    def diagnostics(self, response: typing.Any, initial: dict) -> dict:
        return {
            "derivative_method": "compiler-automatic-differentiation",
            "public_calculator_endpoint": True,
            "backend": "cuda",
            "spin": self.state.identity.spin,
            "representation": self.basis.representation,
            "execution_residency": "device-resident",
            "first_integral_backend": "cuda",
            "second_integral_backend": "cuda",
            "grid_ad_backend": "cuda",
            "response_backend": "cuda",
            "ecp_ad_backend": "cuda" if self.ecp is not None else None,
            "df_integral_ad_backend": "cuda" if self.df is not None else None,
            "density_fitting": "full-rank-full-range-jk-plus-exact-lr"
            if self.fitted
            else "none",
            "df_metric_threshold": self.metric_threshold if self.fitted else None,
            "displaced_scf_evaluations": 0,
            "nuclear_response_solves": 1,
            "response_residual_norm": response.solve_result.residual_norm,
            "response_iterations": response.solve_result.iterations,
            "response_operator_actions": response.solve_result.operator_actions,
            "source_names": self.plan.source_names,
            "integral_reserve_bytes": self.integral_reserve,
            "device_pool_used_bytes": self.execution.pool.used_bytes(),
            "device_pool_reserved_bytes": self.execution.pool.total_bytes(),
            "eri_primal_storage": "resident-dense"
            if self.resident_eri
            else "device-streamed",
            **{
                name: value - initial.get(name, 0)
                for name, value in self._work_snapshot().items()
            },
        }

    def hvp(
        self,
        direction: typing.Any,
        *,
        cache: typing.Any,
        integral_budget_bytes: int,
        solver_options: GMRESOptions | None,
    ) -> CompositeRKSHVPResult:
        """Publish a detached complete HVP only after every device/error gate."""
        started = time.perf_counter()
        vector = checked_direction(direction, self.basis.natom)
        options = GMRESOptions() if solver_options is None else solver_options
        if not isinstance(options, GMRESOptions):
            raise TypeError("solver_options must be GMRESOptions")
        self.validate_current()
        with self.scope():
            initial = self._work_snapshot()
            self.execution.pool.set_limit(
                size=integral_budget_bytes
                + self.grid_budget_bytes
                + options.max_workspace_bytes
                + 8 * vector.size * (len(self.plan.source_names) + 2)
            )
            self.initialize(cache, integral_budget_bytes)
            value, components, response = self._apply(
                self.execution.cp.asarray(vector), options
            )
            packed = self.execution.cp.stack(
                (value, *(components[name] for name in self.plan.source_names))
            )
            host = self.execution.publish(packed)
            diagnostics = self.diagnostics(response, initial)
        self.validate_current()
        diagnostics["assembly_seconds"] = time.perf_counter() - started
        return CompositeRKSHVPResult(
            immutable(vector),
            immutable(host[0]),
            MappingProxyType(
                {
                    name: immutable(host[index + 1])
                    for index, name in enumerate(self.plan.source_names)
                }
            ),
            canonical_hash(
                {
                    "operator": self.identity,
                    "direction": vector.tolist(),
                    "value": host[0].tolist(),
                }
            ),
            MappingProxyType(diagnostics),
        )

    def matrix(
        self,
        *,
        cache: typing.Any,
        integral_budget_bytes: int,
        output_budget_bytes: int,
        solver_options: GMRESOptions | None,
    ) -> RKSHessianResult:
        """Retain raw columns on the device; never repair their symmetry."""
        started = time.perf_counter()
        coordinates = 3 * self.basis.natom
        if (
            type(output_budget_bytes) is not int
            or not 0 < output_budget_bytes < 2**63
            or output_budget_bytes < 2 * coordinates**2 * 8
        ):
            raise ValueError("full Hessian publication exceeds output_budget_bytes")
        options = GMRESOptions() if solver_options is None else solver_options
        if not isinstance(options, GMRESOptions):
            raise TypeError("solver_options must be GMRESOptions")
        self.validate_current()
        with self.scope():
            initial = self._work_snapshot()
            self.execution.pool.set_limit(
                size=integral_budget_bytes
                + self.grid_budget_bytes
                + options.max_workspace_bytes
                + output_budget_bytes
            )
            self.initialize(cache, integral_budget_bytes)
            matrix = self.execution.cp.empty((coordinates, coordinates))
            residuals, iterations, actions = [], [], []
            for column in range(coordinates):
                direction = self.execution.cp.zeros((self.basis.natom, 3))
                direction.reshape(-1)[column] = 1
                value, _, response = self._apply(direction, options)
                matrix[:, column] = value.reshape(-1)
                residuals.append(response.solve_result.residual_norm)
                iterations.append(response.solve_result.iterations)
                actions.append(response.solve_result.operator_actions)
            symmetry = float(self.execution.cp.max(abs(matrix - matrix.T)))
            host = self.execution.publish(matrix)
            diagnostics = self.diagnostics(response, initial)
        diagnostics.update(
            nuclear_response_solves=coordinates,
            posthoc_symmetrization=False,
            raw_symmetry_error=symmetry,
            response_residual_norms=tuple(residuals),
            response_iterations=sum(iterations),
            response_operator_actions=sum(actions),
            assembly_seconds=time.perf_counter() - started,
        )
        self.validate_current()
        return RKSHessianResult(
            immutable(host),
            canonical_hash({"operator": self.identity, "matrix": host.tolist()}),
            MappingProxyType(diagnostics),
        )

    def close(self) -> None:
        """Release response-owned storage, preserving the caller's AO/batch owners."""
        if self.closed:
            return
        self.closed = True
        self.execution.stream.synchronize()
        if self.df is not None:
            self.df.close()
        if self.auxiliary is not None:
            self.auxiliary.close()
        if self.integral_basis is not None:
            self.integral_basis.close()
        self.state._source.close()
        for name in (
            "response",
            "device_state",
            "grid",
            "ecp",
            "ao",
            "eri_full",
            "eri_long",
            "zero_direction",
            "transform",
        ):
            if hasattr(self, name):
                setattr(self, name, None)
        self.df = None
        if self.integral_basis is not None:
            self.integral_basis.transform = None
        self.execution.status = None
        with self.execution.device:
            self.execution.pool.free_all_blocks()

    def __enter__(self) -> typing.Self:
        self.validate_current()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

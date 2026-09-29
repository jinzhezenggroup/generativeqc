"""Complete WB97M-V CUDA stationary composition from a live native KS state.

The native owner evaluates integral derivatives; generated CUDA contracts AO
jets, semilocal/nonlocal feature adjoints, partition motion and the final sum.
Host work is explicit snapshot validation and bounded tile scheduling. Nonlocal
features and force seeds remain on the device throughout composition.
"""

from __future__ import annotations

import ctypes as ct
import typing
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np
from generativeqc_compiler.dft.cuda import CudaGrid
from generativeqc_compiler.dft.nonlocal_policy import (
    MOLECULAR_VV10_DENSITY_POLICY,
    MOLECULAR_VV10_DENSITY_THRESHOLD,
)
from generativeqc_compiler.dft.plan import plan_tiles
from generativeqc_compiler.integral.first_derivative_native import (
    emit_first_derivative_cuda,
)
from generativeqc_compiler.method.nonlocal_correlation import (
    NonlocalCorrelationPrimitive,
)
from generativeqc_compiler.method.stationary_cuda import compile_stationary_cuda
from generativeqc_compiler.method.stationary_feature_lease import (
    plan_stationary_feature_leases,
)
from generativeqc_compiler.method.stationary_gradient import (
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.method.stationary_prepared import (
    compile_stationary_prepared_plan,
)
from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from generativeqc_compiler.tensor.cuda_plan import plan_cuda

from . import _native
from ._dft_gradient import StationaryDerivativeContract, native_ao_geometry_identity
from ._stationary_cuda import _DOUBLE, _CudaSources, _native_grid_artifact, _ptr
from ._stationary_nonlocal_cuda import resident_nonlocal_geometry
from .nonlocal_runtime import _ResidentNonlocalForceOwner


class PreparedWb97mvCudaGradient:
    """Retain geometry-bound CUDA owners, rebuilding explicitly on geometry change.

    Numeric capacity is checked before constructing any derivative owner.
    Driver modules/compiler objects and the pre-existing SCF snapshot are
    reported separately from this additional host/device allowance.
    """

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._identity = None
        self._nonlocal = None
        self.executions = 0
        self.last_work = None

    def close(self) -> None:
        if self._nonlocal is not None:
            # The native nonlocal owner drains its bound stream before freeing
            # seeds, including a failed downstream geometry enqueue.
            self._nonlocal.close()
            self._nonlocal = None
        self._stack.close()
        self._identity = None
        self.last_work = None

    def execute(
        self,
        state: typing.Any,
        basis: typing.Any,
        *,
        compiler: typing.Any,
        cache: Path,
        library: Path,
        tile_points: int = 256,
        max_device_bytes: int = 1 << 30,
        max_host_bytes: int = 2 << 30,
    ) -> tuple[np.ndarray, dict[str, typing.Any]]:
        """Publish only a complete result; failed executions discard retained scratch."""
        try:
            return self._execute(
                state,
                basis,
                compiler=compiler,
                cache=cache,
                library=library,
                tile_points=tile_points,
                max_device_bytes=max_device_bytes,
                max_host_bytes=max_host_bytes,
            )
        except BaseException:
            self.close()
            raise

    def _execute(
        self,
        state: typing.Any,
        basis: typing.Any,
        *,
        compiler: typing.Any,
        cache: Path,
        library: Path,
        tile_points: int,
        max_device_bytes: int,
        max_host_bytes: int,
    ) -> tuple[np.ndarray, dict[str, typing.Any]]:
        """Contract all twelve gradients under the live SCF token and publish forces."""
        started = perf_counter()
        if not callable(getattr(_CudaSources, "geometry_external_device", None)):
            raise NotImplementedError(
                "resident nonlocal force composition requires the stationary seed consumer"
            )
        contract = StationaryDerivativeContract(state.identity)
        contract.validate(state)
        source = state._source
        if (
            source.backend != "cuda"
            or source.metadata[0] != 8
            or source.metadata[6] != 4
            or source.hamiltonian != "all-electron"
            or source.nonlocal_density_policy != MOLECULAR_VV10_DENSITY_POLICY
        ):
            raise NotImplementedError(
                "WB97M-V forces require its complete FP64 CUDA owner"
            )
        if (
            basis.identity != state.identity.basis_identity
            or native_ao_geometry_identity(basis) != state.identity.geometry_identity
        ):
            raise ValueError("WB97M-V stationary basis/geometry mismatch")
        n, na, npnt = basis.nao, basis.natom, len(state.grid.points)
        if not (1 <= n <= 1024 and 1 <= na <= 128 and 1 <= npnt <= 4_000_000):
            raise ValueError("WB97M-V CUDA stationary shape exceeds its bounded domain")
        if any(shell.angular_momentum > 2 for shell in basis.shells):
            raise NotImplementedError("WB97M-V CUDA forces currently qualify s/p/d AOs")
        device = int(source.metadata[12])
        plan = StationaryGradientPlan(
            source.method_ir,
            StationaryMeanField("libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16"),
        )
        prepared_plan = compile_stationary_prepared_plan(plan)
        feature_plan = plan_stationary_feature_leases(prepared_plan)
        grid_features = feature_plan.features
        if "rho" not in grid_features or "gradient" not in grid_features:
            raise RuntimeError(
                "stationary execution planner omitted nonlocal density features"
            )
        nlc = next(
            p
            for p in source.method_ir.primitives
            if isinstance(p, NonlocalCorrelationPrimitive)
        )
        gp = plan_tiles(
            basis,
            backend="cuda",
            order=2,
            tile_points=tile_points,
            active_ao_capacity=n,
            budget_bytes=max_device_bytes,
        )
        # Integral work belongs to the native source, never an AO^4 host loop.
        capacity = 1
        source_bytes = (
            8
            * (
                22 * capacity
                + 2 * basis.nprimitive
                + 4 * n
                + 600 * na
                + 3 * tile_points
                + 2 * plan.spin_blocks * n * n
            )
            + 256
        )
        nlc_budget = min(
            source._batch._calculator.ks_options.nonlocal_memory_budget_bytes,
            max_host_bytes // 4,
            max_device_bytes // 4,
        )
        # The Direct derivative source is retained by the prepared SCF owner and
        # is already charged to that owner's resource ledger. This allowance is
        # only for force-time one-electron/transient native work; do not reserve
        # a second Direct owner here. The matrix term also covers final-state
        # revalidation/export on host.
        native_budget = 256 * n * n + 1024 * (
            na + n + basis.nprimitive + len(basis.shells)
        )
        device_bound = (
            gp.peak_bytes + 2 * source_bytes + 48 * tile_points + nlc_budget + native_budget
        )
        host_bound = (
            gp.host_bytes
            + 8 * (64 * npnt + 8 * n * n + 128 * na)
            + nlc_budget
            + native_budget
        )
        if device_bound > max_device_bytes or host_bound > max_host_bytes:
            raise ValueError("WB97M-V stationary numeric capacity budget exceeded")
        cache = Path(cache)
        identity = (
            basis.identity,
            state.identity.geometry_identity,
            prepared_plan.identity,
            feature_plan.identity,
            source.grid_spec,
            device,
            tile_points,
            str(library),
            str(compiler.target),
            max_device_bytes,
            max_host_bytes,
            nlc_budget,
        )
        reused = identity == self._identity
        if not reused:
            self.close()
            self._stack = ExitStack()
            try:
                primitive = emit_first_derivative_cuda((("nuclear", ()),))
                artifact = compile_stationary_cuda(
                    primitive,
                    functional=4,
                    plan=plan,
                    iterations=source.grid_spec.partition_iterations,
                    compiler=compiler,
                    cache=cache,
                )
                self.sources = self._stack.enter_context(
                    _CudaSources(
                        basis,
                        artifact,
                        compiler,
                        device,
                        tile_points,
                        capacity,
                        source_bytes,
                        spin_blocks=plan.spin_blocks,
                        # This retained owner executes one nuclear primitive per
                        # native call. Total pair coverage is bounded separately
                        # by the admitted atom domain and explicit pair loop.
                        page_work_budget=1,
                    )
                )
                self.sources.kinds[("nuclear", ())] = 0
                # Keep semilocal and nonlocal source accounting distinct while
                # both consumers borrow the same GridTask lease. This second
                # bounded accumulator replaces the former second AO/grid pass.
                self.nonlocal_sources = self._stack.enter_context(
                    _CudaSources(
                        basis,
                        artifact,
                        compiler,
                        device,
                        tile_points,
                        capacity,
                        source_bytes,
                        spin_blocks=plan.spin_blocks,
                        page_work_budget=1,
                    )
                )
                self.grid = self._stack.enter_context(
                    CudaGrid(
                        basis,
                        _native_grid_artifact(library, compiler.target.architecture),
                        order=2,
                        tile_points=tile_points,
                        active_ao_capacity=n,
                        budget_bytes=gp.peak_bytes,
                        device_id=device,
                        ingredients=grid_features,
                    )
                )
                self._nonlocal = _ResidentNonlocalForceOwner(
                    nlc.spec,
                    state.grid.points,
                    state.grid.weights,
                    coefficient=nlc.coefficient,
                    tile_points=tile_points,
                    maximum_bytes=nlc_budget,
                    density_threshold=float(MOLECULAR_VV10_DENSITY_THRESHOLD),
                    device_id=device,
                    context=source._batch._context,
                    library=source._library,
                )
                rp = plan_cuda(
                    plan.reduction_program(atoms=na),
                    compiler.target,
                    max_bytes=max_device_bytes - device_bound,
                )
                self.reduction = self._stack.enter_context(
                    PreparedCuda(rp, compile_cuda(rp, compiler, cache), device=device)
                )
                self._reduction_device_bytes = rp.peak_bytes
                self._reduction_host_bytes = rp.host_bytes
                if host_bound + rp.host_bytes > max_host_bytes:
                    raise ValueError(
                        "WB97M-V stationary reduction exceeds host capacity"
                    )
                self._identity = identity
            except BaseException:
                self.close()
                raise
        device_bound += self._reduction_device_bytes
        host_bound += self._reduction_host_bytes
        component_seconds = {"prepare": perf_counter() - started}
        component_start = perf_counter()
        evaluate = source._library.generativeqc_ks_snapshot_cuda_integral_gradient_v1
        evaluate.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            _DOUBLE,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(ct.c_uint64),
            ct.c_size_t,
        ]
        evaluate.restype = ct.c_int
        integral = np.empty((5, na, 3))
        native_usage = np.zeros(9, dtype=np.uint64)
        _native.check(
            source._library,
            evaluate(
                source._batch._batch,
                source._handle,
                _ptr(integral),
                integral.size,
                native_budget,
                native_usage.ctypes.data_as(ct.POINTER(ct.c_uint64)),
                native_usage.size,
            ),
            context=source._batch._context,
        )
        names = (
            "one_electron",
            "overlap_pulay",
            "coulomb",
            "exchange_short_range",
            "exchange_long_range",
        )
        components = dict(zip(names, integral, strict=True))
        component_seconds["integral_derivatives"] = perf_counter() - component_start
        component_start = perf_counter()
        density = state.density if plan.spin_blocks == 2 else state.density[0]
        self.grid.set_density(density)
        self.sources.reset(
            source.grid_spec.coincident_tolerance, state.density, state.weighted_density
        )
        self.nonlocal_sources.reset(
            source.grid_spec.coincident_tolerance, state.density, state.weighted_density
        )
        charges = np.array([a.atomic_number for a in basis.atoms], dtype=float)
        for atom in range(na):
            for other in range(atom):
                self.sources._call(
                    "stationary_nuclear",
                    self.sources.handle,
                    0,
                    atom,
                    other,
                    float(charges[atom]),
                    float(charges[other]),
                )
        component_seconds["density_and_nuclear_setup"] = (
            perf_counter() - component_start
        )
        resident_parts, resident_seconds, resident_work = resident_nonlocal_geometry(
            grid=self.grid,
            sources=self.sources,
            nonlocal_sources=self.nonlocal_sources,
            nonlocal_owner=self._nonlocal,
            state=state,
            raw_weights=source.atomic_weights,
            tile_points=tile_points,
            ao_count=n,
            functional=4,
            ingredients=grid_features,
        )
        components.update(resident_parts)
        component_seconds.update(resident_seconds)
        component_start = perf_counter()
        plan.reduction_program(atoms=na, sources=components)
        result = self.reduction.execute(components)
        contract.validate(state)
        component_seconds["reduction_and_validation"] = perf_counter() - component_start
        self.executions += 1
        work = {
            "execution": "cuda-complete-wb97mv",
            "plan_identity": plan.identity,
            "prepared_plan_identity": prepared_plan.identity,
            "execution_graph_identity": prepared_plan.graph.identity,
            "lifetime_plan_identity": prepared_plan.lifetimes.identity,
            "feature_lease_identity": feature_plan.identity,
            "grid_features": list(feature_plan.features),
            "retained_grid_features": list(feature_plan.retained_features),
            "source_names": list(plan.source_names),
            "grid_points": npnt,
            **resident_work,
            "partition_pair_visits": 2 * npnt * na * (na - 1),
            "symmetry_unique_quartets_per_integral_source": (
                (n * (n + 1) // 2) * (n * (n + 1) // 2 + 1) // 2
            ),
            "two_electron_quartet_traversals": 1,
            "maximum_center_dual3_evaluations_total": (
                6 * (n * (n + 1) // 2) * (n * (n + 1) // 2 + 1) // 2
            ),
            "range_recurrences_per_participating_center": 2,
            "additional_device_peak_bound": device_bound,
            "additional_host_numeric_bound": host_bound,
            "native_integral_resources": dict(
                zip(
                    (
                        "retained_device_bytes",
                        "source_host_preparation_bytes",
                        "one_electron_device_peak_bytes",
                        "one_electron_host_peak_bytes",
                        "one_electron_h2d_bytes",
                        "one_electron_d2h_bytes",
                        "final_state_export_d2h_bytes",
                        "final_state_export_reads",
                        "final_state_export_synchronizations",
                    ),
                    map(int, native_usage),
                    strict=True,
                )
            ),
            "prepared_execution_reused": reused,
            "execution_index": self.executions,
            "snapshot_export_work": dict(source.export_work),
            "host_scope": "snapshot validation and bounded tile scheduling",
            "endpoint_seconds": perf_counter() - started,
            "component_seconds": component_seconds,
        }
        self.last_work = work
        return -np.asarray(result.outputs["gradient"]).copy(), work

"""Complete compiler-planned composite CUDA stationary forces from a live KS state.

The shared stationary CUDA arena owns semilocal sources. This module composes
the additional providers selected by StationaryGradientPlan, currently the
qualified range-separated-exchange plus nonlocal-correlation source inventory.
No method-name dispatch or method-specific scientific kernel lives here.

The native owner evaluates integral derivatives; generated CUDA contracts AO
jets, semilocal/nonlocal feature adjoints and partition motion. Components that
are already published on the host are summed there exactly once, avoiding a
redundant host->device->host final-add round trip. Nonlocal features and force
seeds remain on the device throughout composition.
"""

from __future__ import annotations

import ctypes as ct
import os
import typing
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np
from generativeqc_compiler.dft.ao_map_plan import ExactAoMapResources
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
from generativeqc_compiler.method.stationary_composite_resources import (
    plan_composite_stationary_cuda_resources,
)
from generativeqc_compiler.method.stationary_cuda import (
    COMPOSITE_STATIONARY_CUDA_EXTERNAL_SOURCES,
    compile_stationary_cuda,
    load_stationary_aot_artifact,
    stationary_aot_profile_for_plan,
    stationary_external_provider_sources,
    stationary_runtime_sources,
)
from generativeqc_compiler.method.stationary_feature_lease import (
    plan_stationary_feature_leases,
)
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.method.stationary_prepared import (
    compile_stationary_prepared_plan,
)
from generativeqc_compiler.method.stationary_resources import (
    stationary_cuda_requires_native_integrals,
)

from . import _native
from ._dft_gradient import StationaryDerivativeContract, native_ao_geometry_identity
from ._geometry_response_cache import ExactGeometryResponseCache
from ._ks_snapshot import _integral_source_resources
from ._resident_ao_maps import (
    ResidentAoMapCache,
    ResidentAoMapDomain,
    ResidentDeviceAoMapOwner,
)
from ._stationary_cuda import (
    _DOUBLE,
    _CudaSources,
    _Fp32GeometryUnavailable,
    _metric_delta,
    _native_grid_artifact,
    _ptr,
    _resolve_becke_primitive_policy,
    _resolve_phased_becke_policy,
)
from ._stationary_nonlocal_cuda import (
    resident_nonlocal_geometry,
    resident_split_nonlocal_geometry,
)
from .nonlocal_runtime import _ResidentNonlocalForceOwner

_COMPOSITE_EXTERNAL_SOURCES = COMPOSITE_STATIONARY_CUDA_EXTERNAL_SOURCES


def _plan_for_state(state: typing.Any) -> StationaryGradientPlan:
    source = state._source
    point_model = (
        source._batch._calculator._ks_options.scf_domain
        if getattr(source, "nonlocal_density_policy", None)
        == MOLECULAR_VV10_DENSITY_POLICY
        else SCF_POINT_MODEL
    )
    return StationaryGradientPlan(
        source.method_ir,
        StationaryMeanField(point_model, hamiltonian=source.hamiltonian),
    )


def requires_composite_stationary_cuda(state: typing.Any) -> bool:
    """Select the qualified composite owner or fail closed on unowned sources."""
    plan = _plan_for_state(state)
    external_sources = stationary_external_provider_sources(plan)
    if not external_sources:
        return False
    if external_sources != _COMPOSITE_EXTERNAL_SOURCES:
        raise NotImplementedError(
            "stationary CUDA external-provider source inventory is not qualified: "
            + ", ".join(external_sources)
        )
    return True


def composite_stationary_aot_profile(state: typing.Any) -> typing.Any:
    """Select an exact packaged point model without discovering a compiler.

    Custom Hamiltonians, point models and partition rules keep bounded JIT.
    A selected package must pass the ordinary binary/plan integrity loader;
    missing or corrupt declared packages never silently compile a replacement.
    """
    source = state._source
    if (
        source.hamiltonian != "all-electron"
        or source.grid_spec.partition_iterations != 3
        or source.nonlocal_grid_spec.partition_iterations != 3
    ):
        return None
    return stationary_aot_profile_for_plan(
        int(source.functional_code), state.identity.spin, _plan_for_state(state)
    )


def _canonical_gradient_sum(
    plan: StationaryGradientPlan,
    components: typing.Mapping[str, typing.Any],
    natom: int,
) -> np.ndarray:
    """Validate complete source coverage and sum already-host-resident gradients."""
    if type(natom) is not int or natom < 1:
        raise ValueError("atoms must be a positive integer")
    # The host sum consumes already-scattered components. Reuse the compiler's
    # coverage contract without constructing an unused TensorIR program at runtime.
    plan.validate_source_coverage(sources=components)
    gradient = np.zeros((natom, 3), dtype=np.float64)
    for name in plan.source_names:
        component = np.asarray(components[name])
        if component.dtype != np.float64 or component.shape != (natom, 3):
            raise ValueError(
                "composite stationary gradient component shape/dtype mismatch"
            )
        if not np.all(np.isfinite(component)):
            raise ValueError("composite stationary gradient component is nonfinite")
        np.add(gradient, component, out=gradient)
    if not np.all(np.isfinite(gradient)):
        raise ValueError("composite stationary final gradient is nonfinite")
    return gradient


class PreparedCompositeStationaryCudaGradient:
    """Retain geometry-bound CUDA owners, rebuilding explicitly on geometry change.

    Numeric capacity is checked before constructing any derivative owner.
    Driver modules/compiler objects and the pre-existing SCF snapshot are
    reported separately from this additional host/device allowance.
    """

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._identity = None
        self._nonlocal = None
        self._ao_maps = None
        self._geometry_response = ExactGeometryResponseCache()
        self.executions = 0
        self.last_work = None
        self._attempt_source_metrics = None

    def close(self) -> None:
        self._geometry_response.clear()
        if self._nonlocal is not None:
            # The native nonlocal owner drains its bound stream before freeing
            # seeds, including a failed downstream geometry enqueue.
            self._nonlocal.close()
            self._nonlocal = None
        self._ao_maps = None
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
        target: typing.Any = None,
        tile_points: int | None = None,
        max_device_bytes: int = 1 << 30,
        max_host_bytes: int = 2 << 30,
        active_ao_cutoff: float | None = None,
        active_ao_cache_bytes: int = 64 << 20,
        active_ao_producer: str = "sampled-jets",
        active_ao_max_active_fraction: float = 1.0,
        fp32_semilocal_geometry: bool = False,
    ) -> tuple[np.ndarray, dict[str, typing.Any]]:
        """Publish complete forces; an explicit semilocal candidate is not admission.

        The diagnostic FP32 option changes only semilocal point/projection/Becke
        arithmetic. SCF, AO production, integral derivatives and the independent
        nonlocal seed consumer stay strict. A numerical candidate failure retries
        the complete gradient once at the same live state; driver, allocation and
        identity failures never trigger a retry. Failed calls discard scratch.
        """
        started = perf_counter()
        options = {
            "compiler": compiler,
            "cache": cache,
            "library": library,
            "target": target,
            "tile_points": tile_points,
            "max_device_bytes": max_device_bytes,
            "max_host_bytes": max_host_bytes,
            "active_ao_cutoff": active_ao_cutoff,
            "active_ao_cache_bytes": active_ao_cache_bytes,
            "active_ao_producer": active_ao_producer,
            "active_ao_max_active_fraction": active_ao_max_active_fraction,
            "fp32_semilocal_geometry": fp32_semilocal_geometry,
        }
        try:
            try:
                return self._execute(state, basis, **options)
            except _Fp32GeometryUnavailable as error:
                if not fp32_semilocal_geometry:
                    raise
                attempts = (
                    {}
                    if self._attempt_source_metrics is None
                    else _metric_delta(
                        self.sources.metrics(), self._attempt_source_metrics
                    )
                )
                self.close()
                options["fp32_semilocal_geometry"] = False
                forces, work = self._execute(state, basis, **options)
                work = dict(work)
                work.update(
                    fp32_semilocal_geometry_requested=True,
                    fp32_geometry_unavailable=str(error),
                    rejected_semilocal_work=attempts,
                    endpoint_seconds=perf_counter() - started,
                )
                self.last_work = work
                return forces, work
            except ArithmeticError:
                if not fp32_semilocal_geometry or self._attempt_source_metrics is None:
                    raise
                attempts = _metric_delta(
                    self.sources.metrics(), self._attempt_source_metrics
                )
                if not attempts.get("fp32_geometry_points", 0):
                    raise
                self.close()
                options["fp32_semilocal_geometry"] = False
                forces, work = self._execute(state, basis, **options)
                work = dict(work)
                work.update(
                    fp32_semilocal_geometry_requested=True,
                    fp64_geometry_retries=1,
                    rejected_semilocal_work=attempts,
                    endpoint_seconds=perf_counter() - started,
                )
                self.last_work = work
                return forces, work
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
        target: typing.Any,
        tile_points: int | None,
        max_device_bytes: int,
        max_host_bytes: int,
        active_ao_cutoff: float | None,
        active_ao_cache_bytes: int,
        active_ao_producer: str,
        active_ao_max_active_fraction: float,
        fp32_semilocal_geometry: bool,
    ) -> tuple[np.ndarray, dict[str, typing.Any]]:
        """Contract all twelve gradients under the live SCF token and publish forces."""
        self._attempt_source_metrics = None
        started = perf_counter()
        target = compiler.target if target is None else target
        if type(fp32_semilocal_geometry) is not bool:
            raise TypeError("semilocal FP32 geometry selection must be boolean")
        if type(active_ao_producer) is not str or active_ao_producer not in {
            "sampled-jets",
            "pre-ao-envelope",
            "pre-ao-envelope-native-csr",
            "exact-jets-native-bitmask",
        }:
            raise ValueError("unsupported resident AO domain producer")
        if active_ao_cutoff is not None and (
            type(active_ao_cutoff) not in (int, float)
            or not np.isfinite(active_ao_cutoff)
            or active_ao_cutoff <= 0
        ):
            raise ValueError("active AO cutoff must be finite and positive")
        if type(active_ao_cache_bytes) is not int or active_ao_cache_bytes < 0:
            raise ValueError("active AO cache budget must be nonnegative")
        if not callable(getattr(_CudaSources, "geometry_external_device", None)):
            raise NotImplementedError(
                "resident nonlocal force composition requires the stationary seed consumer"
            )
        contract = StationaryDerivativeContract.for_native_force(state)
        contract.validate(state)
        source = state._source
        if (
            source.backend != "cuda"
            or source.metadata[0] != 8
            or source.hamiltonian != "all-electron"
            or source.nonlocal_density_policy != MOLECULAR_VV10_DENSITY_POLICY
        ):
            raise NotImplementedError(
                "composite stationary forces require its complete FP64 CUDA owner"
            )
        if (
            basis.identity != state.identity.basis_identity
            or native_ao_geometry_identity(basis) != state.identity.geometry_identity
        ):
            raise ValueError("composite stationary stationary basis/geometry mismatch")
        functional = int(source.functional_code)
        n, na, npnt = basis.nao, basis.natom, len(state.grid.points)
        nonlocal_grid = source.nonlocal_grid
        nonlocal_spec = source.nonlocal_grid_spec
        nonlocal_points = len(nonlocal_grid.points)
        split_grid = nonlocal_grid is not state.grid
        # Share the compiler's shape contract, including its primitive bound.
        # This composition always uses native integral derivatives, even for
        # shapes where the small diagnostic AO-descriptor route is admissible.
        stationary_cuda_requires_native_integrals(
            atoms=na, aos=n, primitives=basis.nprimitive
        )
        if not 1 <= npnt <= 4_000_000 or not 1 <= nonlocal_points <= 4_000_000:
            raise ValueError(
                "composite stationary CUDA stationary shape exceeds its bounded domain"
            )
        if any(shell.angular_momentum > 3 for shell in basis.shells):
            raise NotImplementedError(
                "composite stationary CUDA forces currently qualify through-f AOs"
            )
        device = int(source.metadata[12])
        plan = _plan_for_state(state)
        external_sources = stationary_external_provider_sources(plan)
        if external_sources != _COMPOSITE_EXTERNAL_SOURCES:
            raise NotImplementedError(
                "composite stationary CUDA owner does not cover this source inventory"
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
        # Integral work belongs to the native source, never an AO^4 host loop.
        capacity = 1
        # The resident pair/seed arena scales with the complete grid, while the
        # AO and geometry owners scale with one tile. Reserve its exact native
        # capacity, then admit all simultaneously live owners under both totals.
        # An explicit user cap still fails before any force allocation or JIT.
        nlc_budget = _ResidentNonlocalForceOwner.required_device_bytes(
            source._library,
            nonlocal_points,
            256 if tile_points is None else tile_points,
        )
        if (
            nlc_budget
            > source._batch._calculator.ks_options.nonlocal_memory_budget_bytes
        ):
            raise ValueError(
                "resident nonlocal force exceeds nonlocal_memory_budget_bytes"
            )
        native_derivative_bytes = (
            source.cuda_integral_gradient_additional_device_bytes()
            if bool(getattr(source, "density_fitted", False))
            else 0
        )
        # Match the source owner's existing large-system policy in the dry
        # inventory. Reserving only the ordinary arena silently forces both
        # owners back to the slow schedule even when the complete cap has room.
        phased_becke = _resolve_phased_becke_policy(na, None)
        becke_primitive = _resolve_becke_primitive_policy()
        layout = plan_composite_stationary_cuda_resources(
            basis,
            phased_becke=phased_becke,
            becke_primitive=bool(becke_primitive),
            grid_plan=lambda points: plan_tiles(
                basis,
                backend="cuda",
                order=2,
                tile_points=points,
                active_ao_capacity=n,
                budget_bytes=max_device_bytes,
            ),
            grid_points=npnt + (nonlocal_points if split_grid else 0),
            spins=plan.spin_blocks,
            source_count=len(stationary_runtime_sources(plan)),
            nonlocal_bytes=nlc_budget,
            target=target,
            max_device_bytes=max_device_bytes,
            max_host_bytes=max_host_bytes,
            tile_points=tile_points,
            native_derivative_bytes=native_derivative_bytes,
        )
        gp = layout.grid
        tile_points = gp.tile_points
        source_bytes = layout.sources.allocation_bytes
        native_budget = layout.native_bytes
        device_bound, host_bound = layout.device_bound, layout.host_bound
        # Admit dense scratch first. Optional maps use only remaining host
        # capacity, with a zero-budget dense fallback when no room remains.
        ao_cache_allowance = (
            min(active_ao_cache_bytes, max(0, max_host_bytes - host_bound))
            if active_ao_cutoff is not None
            else 0
        )
        if active_ao_producer in {
            "pre-ao-envelope-native-csr",
            "exact-jets-native-bitmask",
        }:
            # Native CSR/staging must coexist with every complete force owner.
            # Reserve only dense-path headroom, retaining its bounded fallback.
            ao_cache_allowance = min(
                ao_cache_allowance, max(0, max_device_bytes - device_bound)
            )
            if active_ao_producer == "exact-jets-native-bitmask":
                ao_cache_allowance = ExactAoMapResources(
                    basis.nao, npnt, tile_points
                ).admitted_bytes(ao_cache_allowance)
            device_bound += ao_cache_allowance
        host_bound += ao_cache_allowance
        cache = Path(cache)
        identity = (
            basis.identity,
            state.identity.geometry_identity,
            prepared_plan.identity,
            feature_plan.identity,
            source.grid_spec,
            nonlocal_spec,
            device,
            tile_points,
            str(library),
            str(target),
            max_device_bytes,
            max_host_bytes,
            nlc_budget,
            phased_becke,
            becke_primitive,
            active_ao_cutoff,
            ao_cache_allowance,
            active_ao_producer,
            active_ao_max_active_fraction,
            fp32_semilocal_geometry,
        )
        from ._stationary_cuda import _prepare_force_memory

        reused = identity == self._identity
        if not reused:
            self.close()
            force_memory = _prepare_force_memory(source, device_bound)
            self._stack = ExitStack()
            try:
                packaged = composite_stationary_aot_profile(state) is not None
                primitive = None
                if packaged:
                    artifact = load_stationary_aot_artifact(
                        Path(library).parent,
                        functional=functional,
                        spin=state.identity.spin,
                        plan=plan,
                        iterations=source.grid_spec.partition_iterations,
                        architecture=target.architecture,
                    )
                else:
                    primitive = emit_first_derivative_cuda((("nuclear", ()),))
                    artifact = compile_stationary_cuda(
                        primitive,
                        functional=functional,
                        plan=plan,
                        iterations=source.grid_spec.partition_iterations,
                        compiler=compiler,
                        cache=cache,
                    )
                # Accumulators borrow this producer's stream. Register the
                # producer first so exception cleanup drains/destroys its
                # consumers before releasing that stream, including pre-drain
                # numerical failures and the complete strict retry.
                self.grid = self._stack.enter_context(
                    CudaGrid(
                        basis,
                        _native_grid_artifact(library, target.architecture),
                        order=2,
                        tile_points=tile_points,
                        active_ao_capacity=n,
                        budget_bytes=gp.peak_bytes,
                        device_id=device,
                        ingredients=grid_features,
                    )
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
                        target=target,
                        integral_derivatives=False,
                        cooperative_becke=True,
                        phased_becke=phased_becke,
                        becke_primitive=becke_primitive,
                        fp32_geometry=fp32_semilocal_geometry,
                        # The batched nuclear call owns every unordered atom
                        # pair in one deterministic native page.
                        page_work_budget=max(1, na * (na - 1) // 2),
                    )
                )
                nonlocal_artifact = (
                    compile_stationary_cuda(
                        primitive,
                        functional=functional,
                        plan=plan,
                        iterations=nonlocal_spec.partition_iterations,
                        compiler=compiler,
                        cache=cache,
                    )
                    if nonlocal_spec.partition_iterations
                    != source.grid_spec.partition_iterations
                    else artifact
                )
                # Keep semilocal and nonlocal source accounting distinct while
                # both consumers borrow the same GridTask lease. This second
                # bounded accumulator replaces the former second AO/grid pass.
                self.nonlocal_sources = self._stack.enter_context(
                    _CudaSources(
                        basis,
                        nonlocal_artifact,
                        compiler,
                        device,
                        tile_points,
                        capacity,
                        source_bytes,
                        spin_blocks=plan.spin_blocks,
                        target=target,
                        integral_derivatives=False,
                        cooperative_becke=True,
                        phased_becke=phased_becke,
                        becke_primitive=becke_primitive,
                        page_work_budget=1,
                    )
                )
                self._nonlocal = _ResidentNonlocalForceOwner(
                    nlc.spec,
                    nonlocal_grid.points,
                    nonlocal_grid.weights,
                    coefficient=nlc.coefficient,
                    tile_points=tile_points,
                    maximum_bytes=nlc_budget,
                    density_threshold=float(MOLECULAR_VV10_DENSITY_THRESHOLD),
                    device_id=device,
                    context=source._batch._context,
                    library=source._library,
                )
                self._identity = identity
            except BaseException:
                self.close()
                raise
        component_seconds = {"prepare": perf_counter() - started}
        # Both retained owners have independent streams and cumulative counters.
        # Snapshot before nuclear/grid work; never merge their device durations
        # into a clean host-wall endpoint or omit the nonlocal owner's traffic.
        if reused:
            force_memory = _prepare_force_memory(
                source, native_budget + native_derivative_bytes
            )
        source_metrics_before = self.sources.metrics()
        self._attempt_source_metrics = source_metrics_before
        nonlocal_metrics_before = self.nonlocal_sources.metrics()
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
        range_sources = tuple(source.name for source in plan.range_exchange_sources)
        integral_names = (
            "one_electron",
            "overlap_pulay",
            "coulomb",
            *range_sources,
        )
        integral = np.empty((len(integral_names), na, 3))
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
        components = dict(zip(integral_names, integral, strict=True))
        component_seconds["integral_derivatives"] = perf_counter() - component_start
        component_start = perf_counter()
        resident_density = source.cuda_resident_density()
        if resident_density is None:
            raise NotImplementedError(
                "composite stationary CUDA force requires the resident final-density bridge"
            )
        # Geometry/XC response is a function of D at this immutable owner,
        # while the current one-electron/Pulay sources above still run anew.
        # Optional AO experiments retain their own observable execution path.
        reuse_geometry = (
            os.environ.get("GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE") == "1"
            # Fast precision receipts require this invocation's executed work.
            # Keep their capability/strict retry path observable.
            and not fp32_semilocal_geometry
            and active_ao_cutoff is None
            and not bool(getattr(source, "density_fitted", False))
            and not any(
                name.startswith("GENERATIVEQC_STATIONARY_")
                and name != "GENERATIVEQC_STATIONARY_CACHE"
                for name in os.environ
            )
        )
        geometry_allowance = min(
            64 << 20, max_host_bytes // 4, max(0, max_host_bytes - host_bound)
        )
        if not reuse_geometry:
            self._geometry_response.clear()
        retained = (
            self._geometry_response.lookup(state.density, geometry_allowance)
            if reuse_geometry
            else None
        )
        geometry_hit = retained is not None
        source.check_current()
        if geometry_hit:
            resident_parts = retained
            # These are actual zero-work receipts for this invocation. Never
            # replay the producing call's sweep/traffic counters or timings.
            resident_work = {
                "nonlocal_execution": "exact-density-geometry-reuse",
                "nonlocal_feature_source": "retained-density-only-response",
                "nonlocal_feature_d2d_bytes": 0,
                "nonlocal_feature_collection_point_visits": 0,
                "nonlocal_feature_d2h_bytes": 0,
                "nonlocal_seed_h2d_bytes": 0,
                "nonlocal_dense_pair_capacity": npnt * npnt,
                "nonlocal_seed_generation": None,
                "nonlocal_active_count_scope": "not evaluated on exact response reuse",
                "grid_owner_source": "retained-density-only-response",
                "grid_owner_h2d_bytes": 0,
                "grid_point_source": "retained-density-only-response",
                "grid_point_h2d_bytes": 0,
                "grid_weight_source": "retained-density-only-response",
                "grid_weight_h2d_bytes": 0,
                "grid_atomic_measure_source": "retained-density-only-response",
                "grid_atomic_measure_h2d_bytes": 0,
                "nonlocal_pair_cache_probes": 0,
                "nonlocal_pair_cache_hits": 0,
                "nonlocal_pair_sweeps": 0,
                "ao_collocation_point_visits": 0,
                "geometry_point_visits": 0,
                "stationary_source_d2h_policy": "retained-density-only-response",
                "semilocal_stationary_source_d2h_bytes": 0,
                "nonlocal_stationary_source_d2h_bytes": 0,
            }
            resident_seconds = {
                "exact_density_geometry_reuse": perf_counter() - component_start
            }
            component_seconds["density_and_nuclear_setup"] = 0.0
        else:
            self.grid.set_density_device(
                device_id=resident_density.device,
                alpha=resident_density.alpha,
                beta=resident_density.beta,
                matrix_elements=resident_density.matrix_elements,
                spins=resident_density.spins,
                source_stream=resident_density.source_stream,
            )
            # A concurrent replacement can only occur before the producer-stream
            # copy is enqueued; reject it before any stationary source publication.
            source.check_current()
            # Integral derivatives already come from the token-bound native prepared
            # owner above. These retained CUDA accumulators execute only nuclear and
            # grid-geometry sources, so uploading detached host D/W here is redundant.
            self.sources.configure_grid(source.grid_spec, basis.atoms)
            self.nonlocal_sources.configure_grid(nonlocal_spec, basis.atoms)
            self.sources.reset_geometry(source.grid_spec.coincident_tolerance)
            self.nonlocal_sources.reset_geometry(nonlocal_spec.coincident_tolerance)
            charges = np.array([a.atomic_number for a in basis.atoms], dtype=float)
            self.sources.nuclear_all(charges)
            component_seconds["density_and_nuclear_setup"] = (
                perf_counter() - component_start
            )
            ao_domain = None
            if active_ao_cutoff is not None:
                lease = source.cuda_resident_grid()
                if lease is None:
                    raise NotImplementedError(
                        "active AO maps require the resident molecular grid"
                    )
                ao_domain = ResidentAoMapDomain(
                    self.grid.basis_identity,
                    state.identity.geometry_identity,
                    state.grid.identity,
                    lease.device,
                    lease.points,
                    lease.point_count,
                    tile_points,
                    self.grid.plan.order,
                )
                if self._ao_maps is None or self._ao_maps.domain != ao_domain:
                    self._ao_maps = (
                        ResidentDeviceAoMapOwner(
                            self.grid,
                            ao_domain,
                            cutoff=active_ao_cutoff,
                            budget_bytes=ao_cache_allowance,
                            max_active_fraction=active_ao_max_active_fraction,
                            **(
                                {"producer": active_ao_producer}
                                if active_ao_producer == "exact-jets-native-bitmask"
                                else {}
                            ),
                        )
                        if active_ao_producer
                        in {"pre-ao-envelope-native-csr", "exact-jets-native-bitmask"}
                        else ResidentAoMapCache(
                            self.grid,
                            ao_domain,
                            cutoff=active_ao_cutoff,
                            budget_bytes=ao_cache_allowance,
                            producer=active_ao_producer,
                        )
                    )
            resident_parts, resident_seconds, resident_work = (
                resident_split_nonlocal_geometry
                if split_grid
                else resident_nonlocal_geometry
            )(
                grid=self.grid,
                sources=self.sources,
                nonlocal_sources=self.nonlocal_sources,
                nonlocal_owner=self._nonlocal,
                state=state,
                raw_weights=source.atomic_weights,
                tile_points=tile_points,
                ao_count=n,
                functional=functional,
                ingredients=grid_features,
                ao_maps=self._ao_maps,
                ao_domain=ao_domain,
            )
        components.update(resident_parts)
        component_seconds.update(resident_seconds)
        component_start = perf_counter()
        # Every component is already host-resident at this boundary. Keep the
        # compiler coverage gate, but do not upload them solely to add and
        # download the same 3*Natom result again.
        gradient = _canonical_gradient_sum(plan, components, na)
        contract.validate(state)
        if reuse_geometry and not geometry_hit:
            # Only a complete, independently owned result can replace the key.
            # Resource refusal leaves the computed endpoint untouched.
            self._geometry_response.store(
                state.density, resident_parts, geometry_allowance
            )
        host_bound += self._geometry_response.peak_bytes
        component_seconds["reduction_and_validation"] = perf_counter() - component_start
        self.executions += 1
        work = {
            "execution": "cuda-complete-composite",
            "stationary_artifact_kind": self.sources.artifact.metadata.get(
                "artifact_kind", "runtime-jit"
            ),
            "fp32_semilocal_geometry_requested": fp32_semilocal_geometry,
            "fp64_geometry_retries": 0,
            "plan_identity": plan.identity,
            "prepared_plan_identity": prepared_plan.identity,
            "execution_graph_identity": prepared_plan.graph.identity,
            "lifetime_plan_identity": prepared_plan.lifetimes.identity,
            "feature_lease_identity": feature_plan.identity,
            "grid_features": list(feature_plan.features),
            "retained_grid_features": list(feature_plan.retained_features),
            "source_names": list(plan.source_names),
            "grid_points": npnt,
            "grid_tile_points": tile_points,
            "grid_tile_count": (npnt + tile_points - 1) // tile_points,
            "geometry_planned_lanes": layout.sources.geometry_lanes,
            "becke_planned_threads_per_point": layout.sources.becke_threads_per_point,
            "grid_density_source": (
                "retained-density-only-response"
                if geometry_hit
                else "exact-final-scf-device-binding"
            ),
            "grid_density_h2d_bytes": 0,
            "final_reduction": "host-canonical-source-sum",
            "final_reduction_h2d_bytes": 0,
            "final_reduction_d2h_bytes": 0,
            **resident_work,
            "partition_pair_visits": 0
            if geometry_hit
            else (npnt + nonlocal_points) * na * (na - 1),
            "geometry_response_cache_hits": int(geometry_hit),
            "geometry_response_evaluations": int(not geometry_hit),
            "geometry_response_retained_host_bytes": self._geometry_response.retained_bytes,
            "geometry_response_host_peak_bytes": self._geometry_response.peak_bytes,
            # The native v1 result does not identify whether optional shell
            # capability or its public-AO fallback executed. Do not infer work
            # from the method name or turn unavailable evidence into zero.
            "symmetry_unique_quartets_per_integral_source": None,
            "two_electron_quartet_traversals": None,
            "maximum_center_dual3_evaluations_total": None,
            "two_electron_shell_traversals": None,
            "two_electron_radial_operators": [
                "full-range",
                *(primitive.operator for primitive in plan.range_exchange_primitives),
            ],
            "range_recurrences_per_participating_center": None,
            "two_electron_work_scope": (
                "unavailable: native execution route is not exported; screened shell "
                "scheduler and public-AO capability fallback are both supported"
            ),
            "scf_force_memory": force_memory,
            "additional_device_peak_bound": device_bound,
            "additional_host_numeric_bound": host_bound,
            "active_ao_cache_allowance_bytes": ao_cache_allowance,
            "active_ao_cutoff": active_ao_cutoff,
            "active_ao_producer": active_ao_producer,
            "active_ao_max_active_fraction": active_ao_max_active_fraction,
            "native_integral_additional_device_reserve_bytes": native_derivative_bytes,
            "native_integral_resources": dict(
                _integral_source_resources(
                    native_usage,
                    mixed_rsh_df=bool(getattr(source, "density_fitted", False)),
                    lazy_lr_device_owner=not bool(
                        getattr(source, "range_density_fitted", False)
                    ),
                )
            ),
            **(
                {
                    "density_fitted_response_resources_included": 0,
                    "additional_bounds_exclude_df_response": True,
                }
                if bool(getattr(source, "density_fitted", False))
                else {}
            ),
            "prepared_execution_reused": reused,
            "execution_index": self.executions,
            "snapshot_export_work": dict(source.export_work),
            "stationary_source_work": {
                "semilocal": _metric_delta(
                    self.sources.metrics(), source_metrics_before
                ),
                "nonlocal": _metric_delta(
                    self.nonlocal_sources.metrics(), nonlocal_metrics_before
                ),
            },
            "host_scope": "snapshot validation, bounded tile scheduling, and canonical host source sum",
            "endpoint_seconds": perf_counter() - started,
            "component_seconds": component_seconds,
        }
        self.last_work = work
        return -gradient.copy(), work
